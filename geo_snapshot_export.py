"""
Geo Insights — Static HTML Export
==================================

Imports the EXACT chart functions from geo_dashboard.py, generates every
Plotly figure with real data, and writes a self-contained index.html.

The HTML uses inline JavaScript (per R&D "EXTRA FROM THE DEVS") to toggle
between distance groupings and chart variants — all figures are pre-rendered
and toggled via JS visibility.

Usage (run from the data_analyst_repository folder, or pass --dashboard-dir):

    python geo_snapshot_export.py                  # uses cached parquet
    python geo_snapshot_export.py --fresh          # re-queries BigQuery
    python geo_snapshot_export.py --out /path/to/repo/index.html

Output:
    index.html (ready for GitHub Pages push)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock


# =============================================================================
# 1. Mock Streamlit so geo_dashboard.py can be imported without running the app
# =============================================================================

def _create_streamlit_mock():
    """Return a mock that absorbs all Streamlit calls at import time."""
    mock = MagicMock()
    # cache_data decorator: return the function unchanged
    mock.cache_data = lambda f=None, **kw: f if f else (lambda fn: fn)
    # set_page_config, markdown, etc. — all silently absorbed by MagicMock
    return mock


def _import_dashboard(dashboard_path: Path):
    """Import geo_dashboard.py as a module, with Streamlit mocked."""
    # Inject mock before import
    st_mock = _create_streamlit_mock()
    sys.modules['streamlit'] = st_mock

    spec = importlib.util.spec_from_file_location('geo_dashboard', dashboard_path)
    mod = importlib.util.module_from_spec(spec)

    # The module-level code calls st.set_page_config, st.markdown, st.sidebar, etc.
    # Our mock absorbs all of these. But it also calls load_data() etc. which need
    # real data — we'll suppress those by catching errors in module-level code.
    # Actually, module-level code after the function defs will fail because it
    # tries to call load_data() which needs parquet/BQ. We need to stop execution
    # at that point. We do this by patching sys.argv to NOT include --refresh,
    # and making sure parquet caches exist.

    # Actually, simpler: just exec only the function/class definitions, not the
    # app code. We'll load the source, split at the "# App" marker, and exec
    # only the top portion.

    source = dashboard_path.read_text(encoding='utf-8')
    marker = '# =============================================================================\n# App\n# ============================================================================='
    if marker in source:
        source = source[:source.index(marker)]

    # Execute the truncated source in the module namespace
    mod.__file__ = str(dashboard_path)
    mod.__name__ = 'geo_dashboard'
    exec(compile(source, str(dashboard_path), 'exec'), mod.__dict__)

    return mod


# =============================================================================
# 2. Generate all chart figures using the real chart functions
# =============================================================================

def generate_figures(geo, refresh: bool = False):
    """Call every chart function from geo_dashboard and return named figures."""
    figures = {}

    # Load data using the dashboard's own loaders
    df = geo.load_data(refresh=refresh)
    all_houses_df = geo.load_all_houses(refresh=refresh)
    house_pop_df = geo.load_house_population(refresh=refresh)

    # No filters applied (full dataset) — mirrors dashboard defaults
    dff = df.copy()
    dff_all = df.copy()
    dff_base = df.copy()

    # Maps (Section 1) — may be None if parquet caches don't exist
    plz_map_df = geo.load_plz_map_data()
    city_map_df = geo.load_city_map_data()

    if city_map_df is not None:
        figures['city_map_pop'] = geo.chart_city_map(city_map_df, all_houses_df, metric='population')

    if plz_map_df is not None:
        for m in ['customers', 'rev_per_1k', 'activation']:
            figures[f'plz_map_{m}'] = geo.chart_population_map(
                plz_map_df, all_houses_df, metric=m, filtered_df=dff
            )

    # Section 2: Local Impact — both color modes × multiple metrics
    for cm in ['6 Distance Bands', '3 Geo Tiers']:
        tag = '6band' if '6' in cm else '3tier'

        figures[f'comp_pct_{tag}'] = geo.chart_1_pct(dff, cm, df_all=dff_all, df_base=dff_base)

        for metric in ['customers', 'revenue', 'spc']:
            figures[f'comp_{metric}_{tag}'] = geo.chart_1_abs(
                dff, cm, metric=metric, df_all=dff_all, df_base=dff_base
            )

        figures[f'fts_{tag}'] = geo.chart_acq_metric(dff, cm, metric='fts_pct', df_all=dff_all, df_base=dff_base)
        figures[f'ltv_{tag}'] = geo.chart_acq_metric(dff, cm, metric='ltv', df_all=dff_all, df_base=dff_base)

    # Section 3: Region Activation
    for cm in ['6 Distance Bands', '3 Geo Tiers']:
        tag = '6band' if '6' in cm else '3tier'
        figures[f'pop_house_{tag}'] = geo.chart_2_1(dff, cm, house_pop_df=house_pop_df)
        figures[f'act_weekly_{tag}'] = geo.chart_2_2(dff, cm, cumulative=False)
        figures[f'act_cumul_{tag}'] = geo.chart_2_2(dff, cm, cumulative=True)

    figures['act_by_house'] = geo.chart_activation_by_house(dff)

    # Section 4: Geo Mix by House
    for cm in ['6 Distance Bands', '3 Geo Tiers']:
        tag = '6band' if '6' in cm else '3tier'
        figures[f'geo_mix_{tag}'] = geo.chart_geo_mix_by_house(dff, cm)

    # Section 5: Platform Insights — top platforms
    dff_cp = dff.dropna(subset=['channel_type', 'platform']).copy()
    if not dff_cp.empty:
        dff_cp['channel_platform'] = dff_cp['channel_type'] + ' | ' + dff_cp['platform']
        default_platforms = [p for p in ['Meta', 'Google', 'Direct', 'Organic']
                            if p in dff_cp['platform'].unique()]
        if default_platforms:
            cp_filtered = dff_cp[dff_cp['platform'].isin(default_platforms)]
            cp_options = (cp_filtered.groupby('channel_platform')['customer_id']
                          .nunique().sort_values(ascending=False).index.tolist())
            for cm in ['6 Distance Bands', '3 Geo Tiers']:
                tag = '6band' if '6' in cm else '3tier'
                figures[f'platform_{tag}'] = geo.chart_platform_geo_composition(
                    dff, cm, selected_platforms=cp_options
                )

    return figures


# =============================================================================
# 3. Build the HTML page
# =============================================================================

def fig_to_div(fig, div_id: str, hidden: bool = False) -> str:
    """Convert a Plotly figure to an HTML div (no full page, CDN plotly)."""
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
    display = 'none' if hidden else 'block'
    return f'<div id="wrap_{div_id}" style="display:{display}">{inner}</div>'


def build_html(figures: dict, geo_module) -> str:
    """Assemble the full index.html with all chart divs and JS toggles."""

    BG = geo_module.BG_COLOR
    TEXT = geo_module.TEXT_COLOR
    TILE_BG = geo_module.SECTION_TILE_BG
    BORDER = geo_module.GRID_COLOR
    MUTED = '#6b6ba3'
    as_of = date.today().isoformat()

    def section(title, subtitle=''):
        sub = f'<p style="color:{TEXT};opacity:0.65;margin:3px 0 0;font-size:0.82rem">{subtitle}</p>' if subtitle else ''
        return (f'<div style="background:{TILE_BG};padding:10px 16px;border-radius:8px;'
                f'margin:28px 0 14px"><h2 style="color:{TEXT};margin:0;font-size:1.2rem;'
                f'font-weight:600">{title}</h2>{sub}</div>')

    def row_2(*divs):
        cols = ''.join(f'<div style="flex:1;min-width:0">{d}</div>' for d in divs)
        return f'<div style="display:flex;gap:14px;flex-wrap:wrap">{cols}</div>'

    def panel(content, title='', sub=''):
        hdr = ''
        if title:
            hdr = (f'<div style="font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:2px">{title}</div>'
                   f'<div style="font-size:0.72rem;color:{MUTED};margin-bottom:8px">{sub}</div>')
        return (f'<div style="background:#fff;border-radius:12px;border:1px solid {BORDER};'
                f'padding:14px 16px;box-shadow:0 2px 8px rgba(56,57,119,0.06)">{hdr}{content}</div>')

    def get(key, hidden=False):
        if key in figures:
            return fig_to_div(figures[key], key, hidden=hidden)
        return f'<p style="color:{MUTED};font-size:0.8rem">Chart not available (missing cache data)</p>'

    # Toggle buttons builder
    def toggle_bar(group_name, options):
        btns = []
        for i, (label, value) in enumerate(options):
            cls = 'active' if i == 0 else ''
            btns.append(f'<button class="toggle-btn {cls}" data-group="{group_name}" '
                        f'data-value="{value}" onclick="toggle(\'{group_name}\',\'{value}\')">{label}</button>')
        return f'<div class="toggle-bar">{"".join(btns)}</div>'

    # ── Assemble sections ──

    parts = []

    # Header
    parts.append(f'''
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
         padding-bottom:14px;border-bottom:2px solid {BORDER}">
      <div>
        <div style="font-size:1.6rem;font-weight:700;color:{TEXT}">Geo Insights</div>
        <div style="font-size:0.85rem;color:{MUTED};margin-top:4px">
          Geographic performance across houses, distance bands &amp; channels
        </div>
      </div>
      <div style="font-size:0.75rem;color:{MUTED};white-space:nowrap">
        Snapshot exported {as_of}
      </div>
    </div>''')

    # Section 1: Geographic Distribution (maps)
    if any(k.startswith('city_map') or k.startswith('plz_map') for k in figures):
        parts.append(section('Geographic Distribution', 'Population density and customer reach by PLZ area'))
        # Map metric toggle
        parts.append(toggle_bar('map_metric', [
            ('Customers', 'customers'), ('Rev / 1k Pop', 'rev_per_1k'), ('Activation', 'activation')
        ]))
        map_left = get('city_map_pop')
        map_right = ''.join([
            get('plz_map_customers', hidden=False),
            get('plz_map_rev_per_1k', hidden=True),
            get('plz_map_activation', hidden=True),
        ])
        parts.append(row_2(panel(map_left, 'Population by City'), panel(map_right, 'PLZ Area Map')))

    # Section 2: Local Impact
    parts.append(section('Local Impact', 'Geo composition, FTS% and LTV by distance over time'))
    parts.append(toggle_bar('color_mode', [('6 Distance Bands', '6band'), ('3 Geo Tiers', '3tier')]))
    parts.append(toggle_bar('vol_metric', [('Customers', 'customers'), ('Revenue', 'revenue'), ('SpC', 'spc')]))

    # Row 1: Composition % + Volume metric
    pct_html = get('comp_pct_6band') + get('comp_pct_3tier', hidden=True)
    vol_html = ''
    for metric in ['customers', 'revenue', 'spc']:
        for tag in ['6band', '3tier']:
            hidden = not (metric == 'customers' and tag == '6band')
            vol_html += get(f'comp_{metric}_{tag}', hidden=hidden)
    parts.append(row_2(panel(pct_html), panel(vol_html)))

    # Row 2: FTS% + LTV
    fts_html = get('fts_6band') + get('fts_3tier', hidden=True)
    ltv_html = get('ltv_6band') + get('ltv_3tier', hidden=True)
    parts.append(row_2(panel(fts_html), panel(ltv_html)))

    # Section 3: Region Activation
    parts.append(section('Region Activation', 'Population catchment and activation rate per house'))
    parts.append(toggle_bar('act_mode', [('Weekly', 'weekly'), ('Cumulative', 'cumul')]))

    pop_html = get('pop_house_6band') + get('pop_house_3tier', hidden=True)
    act_html = ''
    for mode in ['weekly', 'cumul']:
        for tag in ['6band', '3tier']:
            hidden = not (mode == 'weekly' and tag == '6band')
            act_html += get(f'act_{mode}_{tag}', hidden=hidden)
    parts.append(row_2(panel(pop_html), panel(act_html)))
    parts.append(panel(get('act_by_house')))

    # Section 4: Geo Mix by House
    parts.append(section('Geo Mix by House', 'Distance-band composition evolves as each house matures'))
    mix_html = get('geo_mix_6band') + get('geo_mix_3tier', hidden=True)
    parts.append(panel(mix_html))

    # Section 5: Platform Insights
    if any(k.startswith('platform_') for k in figures):
        parts.append(section('Platform Insights', 'Geo composition per acquisition channel'))
        plat_html = get('platform_6band') + get('platform_3tier', hidden=True)
        parts.append(panel(plat_html))

    body = '\n'.join(parts)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Geo Insights — Snapshot</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: system-ui, -apple-system, sans-serif;
      background: {BG}; color: {TEXT};
    }}
    .page {{ max-width: 1300px; margin: 0 auto; padding: 24px 16px 48px; }}
    .toggle-bar {{ display: flex; gap: 6px; margin-bottom: 12px; }}
    .toggle-btn {{
      padding: 6px 14px; border-radius: 6px; border: 1px solid {BORDER};
      background: #fff; color: {TEXT}; font-size: 0.78rem; cursor: pointer;
      transition: all 0.15s;
    }}
    .toggle-btn.active {{ background: {TEXT}; color: #fff; border-color: {TEXT}; }}
    .toggle-btn:hover:not(.active) {{ background: {TILE_BG}; }}
    .js-plotly-plot {{ width: 100% !important; }}
    @media (max-width: 900px) {{
      div[style*="display:flex"] {{ flex-direction: column !important; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    {body}
  </div>
  <script>
    // ── Toggle logic (inline JS per DEVS guidance) ──
    // Each toggle group controls which chart divs are visible.

    const TOGGLE_RULES = {{
      // group -> which div ID prefixes to show/hide
      color_mode: {{
        '6band': {{show: ['6band'], hide: ['3tier']}},
        '3tier': {{show: ['3tier'], hide: ['6band']}},
      }},
      vol_metric: {{
        'customers': {{show: ['comp_customers'], hide: ['comp_revenue', 'comp_spc']}},
        'revenue':   {{show: ['comp_revenue'],   hide: ['comp_customers', 'comp_spc']}},
        'spc':       {{show: ['comp_spc'],       hide: ['comp_customers', 'comp_revenue']}},
      }},
      act_mode: {{
        'weekly': {{show: ['act_weekly'], hide: ['act_cumul']}},
        'cumul':  {{show: ['act_cumul'],  hide: ['act_weekly']}},
      }},
      map_metric: {{
        'customers':  {{show: ['plz_map_customers'],  hide: ['plz_map_rev_per_1k', 'plz_map_activation']}},
        'rev_per_1k': {{show: ['plz_map_rev_per_1k'], hide: ['plz_map_customers', 'plz_map_activation']}},
        'activation': {{show: ['plz_map_activation'],  hide: ['plz_map_customers', 'plz_map_rev_per_1k']}},
      }},
    }};

    // Track active state
    const activeState = {{
      color_mode: '6band',
      vol_metric: 'customers',
      act_mode: 'weekly',
      map_metric: 'customers',
    }};

    function toggle(group, value) {{
      activeState[group] = value;

      // Update button styles
      document.querySelectorAll(`.toggle-btn[data-group="${{group}}"]`).forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.value === value);
      }});

      // Apply visibility rules
      applyVisibility();
    }}

    function applyVisibility() {{
      // For color_mode + vol_metric, the compound key determines visibility
      // e.g. comp_customers_6band is visible when vol_metric=customers AND color_mode=6band

      const cm = activeState.color_mode;   // '6band' or '3tier'
      const vm = activeState.vol_metric;   // 'customers', 'revenue', 'spc'
      const am = activeState.act_mode;     // 'weekly' or 'cumul'
      const mm = activeState.map_metric;   // 'customers', 'rev_per_1k', 'activation'

      // Iterate all wrap_ divs and decide visibility
      document.querySelectorAll('[id^="wrap_"]').forEach(el => {{
        const id = el.id.replace('wrap_', '');

        // Map charts
        if (id.startsWith('plz_map_')) {{
          const metric = id.replace('plz_map_', '');
          el.style.display = (metric === mm) ? 'block' : 'none';
          return;
        }}

        // Charts that depend on color_mode only
        const cmOnly = ['comp_pct_', 'fts_', 'ltv_', 'pop_house_', 'geo_mix_', 'platform_'];
        for (const prefix of cmOnly) {{
          if (id.startsWith(prefix)) {{
            const tag = id.slice(prefix.length);
            el.style.display = (tag === cm) ? 'block' : 'none';
            return;
          }}
        }}

        // Volume charts: depend on both vol_metric AND color_mode
        if (id.startsWith('comp_')) {{
          // e.g. comp_customers_6band
          const rest = id.replace('comp_', '');
          // But NOT comp_pct_ (handled above)
          if (rest.startsWith('pct_')) return;
          const parts = rest.split('_');
          const metric = parts[0];
          const tag = parts[1];
          el.style.display = (metric === vm && tag === cm) ? 'block' : 'none';
          return;
        }}

        // Activation charts: depend on act_mode AND color_mode
        if (id.startsWith('act_weekly_') || id.startsWith('act_cumul_')) {{
          const parts = id.split('_');
          const mode = parts[1];   // 'weekly' or 'cumul'
          const tag = parts[2];    // '6band' or '3tier'
          el.style.display = (mode === am && tag === cm) ? 'block' : 'none';
          return;
        }}

        // act_by_house — always visible
        if (id === 'act_by_house') {{
          el.style.display = 'block';
          return;
        }}

        // city_map_pop — always visible
        if (id === 'city_map_pop') {{
          el.style.display = 'block';
          return;
        }}
      }});

      // Trigger Plotly relayout on now-visible charts (fixes sizing)
      setTimeout(() => {{
        document.querySelectorAll('[id^="wrap_"]').forEach(el => {{
          if (el.style.display !== 'none') {{
            const plotDiv = el.querySelector('.js-plotly-plot');
            if (plotDiv) Plotly.Plots.resize(plotDiv);
          }}
        }});
      }}, 50);
    }}

    // Initial visibility pass
    window.addEventListener('load', applyVisibility);
  </script>
</body>
</html>'''


# =============================================================================
# 4. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export geo_dashboard.py charts to static HTML')
    parser.add_argument('--fresh', '--refresh', action='store_true', help='Re-query BigQuery instead of using cached parquet')
    parser.add_argument('--dashboard-dir', type=str, default=None,
                        help='Path to the directory containing geo_dashboard.py (default: same as this script)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output path for index.html (default: ./index.html next to this script)')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dashboard_dir = Path(args.dashboard_dir) if args.dashboard_dir else here
    dashboard_path = dashboard_dir / 'geo_dashboard.py'

    if not dashboard_path.exists():
        # Try one level up (if this script is in a subfolder)
        alt = dashboard_dir.parent / 'geo_dashboard.py'
        if alt.exists():
            dashboard_path = alt
        else:
            print(f"ERROR: Cannot find geo_dashboard.py at {dashboard_path}")
            sys.exit(1)

    print(f"Importing chart functions from: {dashboard_path}")
    geo = _import_dashboard(dashboard_path)

    print("Generating figures (this may take a moment)...")
    figures = generate_figures(geo, refresh=args.fresh)
    print(f"  Generated {len(figures)} chart(s)")

    print("Building HTML...")
    html = build_html(figures, geo)

    out_path = Path(args.out) if args.out else here / 'index.html'
    out_path.write_text(html, encoding='utf-8')
    print(f"Written to: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
