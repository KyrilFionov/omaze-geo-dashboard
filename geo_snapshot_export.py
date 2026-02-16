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

import pandas as pd


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
# 2. Build map parquet caches if missing (from seed CSVs + transaction data)
# =============================================================================

def _find_seeds_dir(dashboard_dir: Path) -> Path | None:
    """Locate the dbt seeds directory (contains dim_plz_*.csv)."""
    candidates = [
        dashboard_dir.parent / 'seeds',
        dashboard_dir.parent.parent / 'seeds',
    ]
    for d in candidates:
        if (d / 'dim_plz_coordinates.csv').exists():
            return d
    return None


def ensure_map_caches(geo, df, dashboard_dir: Path):
    """Build plz_map_data.parquet and city_map_data.parquet from seed CSVs + transaction data."""
    import numpy as np
    plz_cache = Path(geo.PLZ_MAP_CACHE)
    city_cache = Path(geo.CITY_MAP_CACHE)

    if plz_cache.exists() and city_cache.exists():
        print("  Map caches already exist — skipping rebuild")
        return

    seeds_dir = _find_seeds_dir(dashboard_dir)
    if seeds_dir is None:
        print("  WARNING: Cannot find seeds directory (dim_plz_coordinates.csv) — maps will be skipped")
        return

    print(f"  Building map caches from seeds at {seeds_dir}")

    coords = pd.read_csv(seeds_dir / 'dim_plz_coordinates.csv', dtype={'plz_three_digits': str})
    pop = pd.read_csv(seeds_dir / 'dim_plz_population.csv', dtype={'plz_gebiet': str})
    pop = pop.rename(columns={'plz_gebiet': 'plz_three_digits', 'ort': 'plz_ort', 'einwohner': 'population'})

    # Transaction aggregates by PLZ
    txn_plz = df.groupby('plz_three_digits').agg(
        customers=('customer_id', 'nunique'),
        revenue=('variant_price_after_discount', 'sum'),
    ).reset_index()

    # PLZ map: merge coords + population + transaction data
    plz_df = coords.merge(pop[['plz_three_digits', 'plz_ort', 'population']], on='plz_three_digits', how='left')
    plz_df = plz_df.merge(txn_plz, on='plz_three_digits', how='left')
    plz_df['customers'] = plz_df['customers'].fillna(0).astype(int)
    plz_df['revenue'] = plz_df['revenue'].fillna(0)
    plz_df['population'] = plz_df['population'].fillna(0).astype(int)
    plz_df.to_parquet(plz_cache, index=False)
    print(f"    Written {plz_cache.name} ({len(plz_df)} rows)")

    # City map: aggregate population + transactions by city (plz_ort)
    city_pop = pop.groupby('plz_ort').agg(population=('population', 'sum')).reset_index().rename(columns={'plz_ort': 'city'})
    # City coords = mean of PLZ coords within each city
    plz_with_city = coords.merge(pop[['plz_three_digits', 'plz_ort']], on='plz_three_digits', how='left')
    city_coords = plz_with_city.groupby('plz_ort').agg(latitude=('latitude', 'mean'), longitude=('longitude', 'mean')).reset_index().rename(columns={'plz_ort': 'city'})

    txn_city = df.groupby('city').agg(
        customers=('customer_id', 'nunique'),
        revenue=('variant_price_after_discount', 'sum'),
    ).reset_index()

    city_df = city_pop.merge(city_coords, on='city', how='left')
    city_df = city_df.merge(txn_city, on='city', how='left')
    city_df['customers'] = city_df['customers'].fillna(0).astype(int)
    city_df['revenue'] = city_df['revenue'].fillna(0)
    city_df['spc'] = (city_df['revenue'] / city_df['customers'].replace(0, np.nan)).fillna(0).round(2)
    city_df.to_parquet(city_cache, index=False)
    print(f"    Written {city_cache.name} ({len(city_df)} rows)")


# =============================================================================
# 3. Generate all chart figures using the real chart functions
# =============================================================================

def generate_figures(geo, refresh: bool = False, dashboard_dir: Path | None = None):
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

    # Build map caches if missing
    if dashboard_dir:
        ensure_map_caches(geo, df, dashboard_dir)

    # Maps (Section 1)
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

    # Build city performance table (Top 20)
    import numpy as np
    city_table_html = ''
    if 'city' in df.columns and 'customer_id' in df.columns:
        seeds_dir = _find_seeds_dir(dashboard_dir) if dashboard_dir else None
        if seeds_dir and (seeds_dir / 'dim_plz_population.csv').exists():
            pop_csv = pd.read_csv(seeds_dir / 'dim_plz_population.csv', dtype={'plz_gebiet': str})
            city_pop = pop_csv.groupby('ort').agg(population=('einwohner', 'sum')).reset_index().rename(columns={'ort': 'city'})
            city_cust = df.groupby('city').agg(
                customers=('customer_id', 'nunique'),
                revenue=('variant_price_after_discount', 'sum'),
            ).reset_index()
            city_tbl = city_pop.merge(city_cust, on='city', how='inner')
            city_tbl['activation'] = (city_tbl['customers'] / (city_tbl['population'] / 1000)).round(2)
            city_tbl['rev_per_1k'] = (city_tbl['revenue'] / (city_tbl['population'] / 1000)).round(2)
            city_tbl = city_tbl.sort_values('population', ascending=False).head(20)
            city_table_html = city_tbl.to_dict(orient='records')

    return figures, city_table_html


# =============================================================================
# 4. Build the HTML page
# =============================================================================

def fig_to_div(fig, div_id: str, hidden: bool = False) -> str:
    """Convert a Plotly figure to an HTML div (no full page, CDN plotly)."""
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
    display = 'none' if hidden else 'block'
    return f'<div id="wrap_{div_id}" style="display:{display}">{inner}</div>'


def build_html(figures: dict, geo_module, city_table_data=None) -> str:
    """Assemble the full index.html with all chart divs, control panel, and city table."""
    import json as _json

    BG = geo_module.BG_COLOR
    TEXT = geo_module.TEXT_COLOR
    TILE_BG = geo_module.SECTION_TILE_BG
    BORDER = geo_module.GRID_COLOR
    SIDEBAR_BG = geo_module.SIDEBAR_BG
    MUTED = '#6b6ba3'
    as_of = date.today().isoformat()

    def section(title, subtitle=''):
        sub = f'<p style="color:{TEXT};opacity:0.65;margin:3px 0 0;font-size:0.82rem">{subtitle}</p>' if subtitle else ''
        return (f'<div style="background:{TILE_BG};padding:10px 16px;border-radius:8px;'
                f'margin:28px 0 14px"><h2 style="color:{TEXT};margin:0;font-size:1.2rem;'
                f'font-weight:600">{title}</h2>{sub}</div>')

    def row_2(*divs):
        cols = ''.join(f'<div style="flex:1;min-width:0">{d}</div>' for d in divs)
        return f'<div class="chart-row">{cols}</div>'

    def panel(content, title='', sub=''):
        hdr = ''
        if title:
            hdr = (f'<div style="font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:2px">{title}</div>'
                   f'<div style="font-size:0.72rem;color:{MUTED};margin-bottom:8px">{sub}</div>')
        return (f'<div class="chart-panel">{hdr}{content}</div>')

    def get(key, hidden=False):
        if key in figures:
            return fig_to_div(figures[key], key, hidden=hidden)
        return f'<p style="color:{MUTED};font-size:0.8rem">Chart not available (missing cache data)</p>'

    # ── Assemble main content sections ──
    parts = []

    # Section 1: Geographic Distribution (maps)
    if any(k.startswith('city_map') or k.startswith('plz_map') for k in figures):
        parts.append(section('Geographic Distribution', 'Population density and customer reach by PLZ area'))
        map_left = get('city_map_pop')
        map_right = ''.join([
            get('plz_map_customers', hidden=False),
            get('plz_map_rev_per_1k', hidden=True),
            get('plz_map_activation', hidden=True),
        ])
        parts.append(row_2(panel(map_left, 'Population by City'), panel(map_right, 'PLZ Area Map')))

    # City Performance Table (Top 20)
    if city_table_data:
        parts.append(f'''<div class="chart-panel" style="margin-top:14px">
          <div style="font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:2px">Performance by City (Top 20)</div>
          <div style="font-size:0.72rem;color:{MUTED};margin-bottom:8px">Sorted by population</div>
          <div style="overflow-x:auto">
          <table class="city-table">
            <thead><tr>
              <th>#</th><th>City</th><th>Population</th><th>Customers</th>
              <th>Revenue</th><th>Activation</th><th>Rev / 1k Pop</th>
            </tr></thead>
            <tbody>''' + ''.join(
            f'<tr><td>{i+1}</td><td>{r["city"]}</td>'
            f'<td>{r["population"]:,.0f}</td><td>{r["customers"]:,.0f}</td>'
            f'<td>&euro;{r["revenue"]:,.0f}</td><td>{r["activation"]:.2f}</td>'
            f'<td>&euro;{r["rev_per_1k"]:,.2f}</td></tr>'
            for i, r in enumerate(city_table_data)
        ) + '''</tbody></table></div></div>''')

    # Section 2: Local Impact
    parts.append(section('Local Impact', 'Geo composition, FTS% and LTV by distance over time'))

    pct_html = get('comp_pct_6band') + get('comp_pct_3tier', hidden=True)
    vol_html = ''
    for metric in ['customers', 'revenue', 'spc']:
        for tag in ['6band', '3tier']:
            hidden = not (metric == 'customers' and tag == '6band')
            vol_html += get(f'comp_{metric}_{tag}', hidden=hidden)
    parts.append(row_2(panel(pct_html), panel(vol_html)))

    fts_html = get('fts_6band') + get('fts_3tier', hidden=True)
    ltv_html = get('ltv_6band') + get('ltv_3tier', hidden=True)
    parts.append(row_2(panel(fts_html), panel(ltv_html)))

    # Section 3: Region Activation
    parts.append(section('Region Activation', 'Population catchment and activation rate per house'))

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

    /* ── Layout: sidebar + main ── */
    .layout {{ display: flex; min-height: 100vh; }}

    .sidebar {{
      width: 260px; min-width: 260px; background: {SIDEBAR_BG};
      padding: 20px 16px; position: sticky; top: 0; height: 100vh;
      overflow-y: auto; border-right: 1px solid {BORDER};
    }}
    .sidebar h1 {{
      font-size: 1.3rem; font-weight: 700; color: {TEXT}; margin: 0 0 4px;
    }}
    .sidebar .subtitle {{
      font-size: 0.78rem; color: {MUTED}; margin: 0 0 18px; line-height: 1.4;
    }}
    .sidebar .as-of {{
      font-size: 0.7rem; color: {MUTED}; margin-bottom: 20px;
    }}
    .sidebar hr {{
      border: none; border-top: 1px solid {BORDER}; margin: 14px 0;
    }}

    .ctrl-group {{ margin-bottom: 16px; }}
    .ctrl-label {{
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: {MUTED}; margin-bottom: 6px; font-weight: 600;
    }}
    .ctrl-btns {{ display: flex; flex-wrap: wrap; gap: 4px; }}
    .toggle-btn {{
      padding: 5px 11px; border-radius: 6px; border: 1px solid {BORDER};
      background: #fff; color: {TEXT}; font-size: 0.74rem; cursor: pointer;
      transition: all 0.15s; white-space: nowrap;
    }}
    .toggle-btn.active {{ background: {TEXT}; color: #fff; border-color: {TEXT}; }}
    .toggle-btn:hover:not(.active) {{ background: {TILE_BG}; }}

    .main {{ flex: 1; min-width: 0; padding: 24px 20px 48px; }}

    .chart-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }}
    .chart-row > div {{ flex: 1; min-width: 0; }}
    .chart-panel {{
      background: #fff; border-radius: 12px; border: 1px solid {BORDER};
      padding: 14px 16px; box-shadow: 0 2px 8px rgba(56,57,119,0.06);
      margin-bottom: 14px;
    }}

    .js-plotly-plot {{ width: 100% !important; }}

    /* City table */
    .city-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
    .city-table th, .city-table td {{
      padding: 6px 8px; border-bottom: 1px solid {BORDER}; text-align: left;
    }}
    .city-table th {{
      font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em;
      color: {MUTED}; font-weight: 600;
    }}
    .city-table tr:last-child td {{ border-bottom: none; }}
    .city-table td:nth-child(n+3) {{ text-align: right; }}
    .city-table th:nth-child(n+3) {{ text-align: right; }}

    /* Sidebar toggle (mobile) */
    .sidebar-toggle {{
      display: none; position: fixed; top: 12px; left: 12px; z-index: 1000;
      background: {TEXT}; color: #fff; border: none; border-radius: 8px;
      padding: 8px 12px; font-size: 0.8rem; cursor: pointer;
    }}

    @media (max-width: 1000px) {{
      .sidebar {{ display: none; position: fixed; z-index: 999; left: 0; top: 0; }}
      .sidebar.open {{ display: block; }}
      .sidebar-toggle {{ display: block; }}
      .chart-row {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <button class="sidebar-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">
    Controls
  </button>
  <div class="layout">
    <!-- ── Control Panel (Sidebar) ── -->
    <div class="sidebar">
      <h1>Geo Insights</h1>
      <div class="subtitle">Geographic performance across houses, distance bands &amp; channels</div>
      <div class="as-of">Snapshot exported {as_of}</div>
      <hr>

      <div class="ctrl-group">
        <div class="ctrl-label">Distance Grouping</div>
        <div class="ctrl-btns">
          <button class="toggle-btn active" data-group="color_mode" data-value="6band"
                  onclick="toggle('color_mode','6band')">6 Distance Bands</button>
          <button class="toggle-btn" data-group="color_mode" data-value="3tier"
                  onclick="toggle('color_mode','3tier')">3 Geo Tiers</button>
        </div>
      </div>

      <div class="ctrl-group">
        <div class="ctrl-label">Volume Metric</div>
        <div class="ctrl-btns">
          <button class="toggle-btn active" data-group="vol_metric" data-value="customers"
                  onclick="toggle('vol_metric','customers')">Customers</button>
          <button class="toggle-btn" data-group="vol_metric" data-value="revenue"
                  onclick="toggle('vol_metric','revenue')">Revenue</button>
          <button class="toggle-btn" data-group="vol_metric" data-value="spc"
                  onclick="toggle('vol_metric','spc')">SpC</button>
        </div>
      </div>

      <div class="ctrl-group">
        <div class="ctrl-label">Activation Mode</div>
        <div class="ctrl-btns">
          <button class="toggle-btn active" data-group="act_mode" data-value="weekly"
                  onclick="toggle('act_mode','weekly')">Weekly</button>
          <button class="toggle-btn" data-group="act_mode" data-value="cumul"
                  onclick="toggle('act_mode','cumul')">Cumulative</button>
        </div>
      </div>

      <div class="ctrl-group">
        <div class="ctrl-label">Map Metric</div>
        <div class="ctrl-btns">
          <button class="toggle-btn active" data-group="map_metric" data-value="customers"
                  onclick="toggle('map_metric','customers')">Customers</button>
          <button class="toggle-btn" data-group="map_metric" data-value="rev_per_1k"
                  onclick="toggle('map_metric','rev_per_1k')">Rev / 1k Pop</button>
          <button class="toggle-btn" data-group="map_metric" data-value="activation"
                  onclick="toggle('map_metric','activation')">Activation</button>
        </div>
      </div>

      <hr>
      <div style="font-size:0.68rem;color:{MUTED};line-height:1.4">
        Static snapshot of <code style="font-size:0.66rem">geo_dashboard.py</code>.<br>
        Charts are pre-rendered — toggles switch between variants.
      </div>
    </div>

    <!-- ── Main Content ── -->
    <div class="main">
      {body}
    </div>
  </div>

  <script>
    // ── Toggle logic (inline JS per DEVS guidance) ──
    const activeState = {{
      color_mode: '6band',
      vol_metric: 'customers',
      act_mode: 'weekly',
      map_metric: 'customers',
    }};

    function toggle(group, value) {{
      activeState[group] = value;
      document.querySelectorAll(`.toggle-btn[data-group="${{group}}"]`).forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.value === value);
      }});
      applyVisibility();
    }}

    function applyVisibility() {{
      const cm = activeState.color_mode;
      const vm = activeState.vol_metric;
      const am = activeState.act_mode;
      const mm = activeState.map_metric;

      document.querySelectorAll('[id^="wrap_"]').forEach(el => {{
        const id = el.id.replace('wrap_', '');

        // Map charts
        if (id.startsWith('plz_map_')) {{
          el.style.display = (id.replace('plz_map_', '') === mm) ? 'block' : 'none';
          return;
        }}

        // Charts depending on color_mode only
        const cmOnly = ['comp_pct_', 'fts_', 'ltv_', 'pop_house_', 'geo_mix_', 'platform_'];
        for (const prefix of cmOnly) {{
          if (id.startsWith(prefix)) {{
            el.style.display = (id.slice(prefix.length) === cm) ? 'block' : 'none';
            return;
          }}
        }}

        // Volume charts: vol_metric + color_mode
        if (id.startsWith('comp_') && !id.startsWith('comp_pct_')) {{
          const rest = id.replace('comp_', '');
          const parts = rest.split('_');
          el.style.display = (parts[0] === vm && parts[1] === cm) ? 'block' : 'none';
          return;
        }}

        // Activation charts: act_mode + color_mode
        if (id.startsWith('act_weekly_') || id.startsWith('act_cumul_')) {{
          const parts = id.split('_');
          el.style.display = (parts[1] === am && parts[2] === cm) ? 'block' : 'none';
          return;
        }}

        // Always visible
        if (id === 'act_by_house' || id === 'city_map_pop') {{
          el.style.display = 'block';
        }}
      }});

      // Resize newly visible Plotly charts
      setTimeout(() => {{
        document.querySelectorAll('[id^="wrap_"]').forEach(el => {{
          if (el.style.display !== 'none') {{
            const p = el.querySelector('.js-plotly-plot');
            if (p) Plotly.Plots.resize(p);
          }}
        }});
      }}, 50);
    }}

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
    figures, city_table_data = generate_figures(geo, refresh=args.fresh, dashboard_dir=dashboard_path.parent)
    print(f"  Generated {len(figures)} chart(s)")
    if city_table_data:
        print(f"  City table: {len(city_table_data)} rows")

    print("Building HTML...")
    html = build_html(figures, geo, city_table_data=city_table_data)

    out_path = Path(args.out) if args.out else here / 'index.html'
    out_path.write_text(html, encoding='utf-8')
    print(f"Written to: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
