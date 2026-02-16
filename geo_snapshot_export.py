"""
Geo Insights — Static HTML Export
==================================

Imports the EXACT chart functions from geo_dashboard.py, generates every
Plotly figure with real data, and writes a self-contained index.html.

The HTML uses inline JavaScript (per R&D "EXTRA FROM THE DEVS") to toggle
between distance groupings and chart variants — all figures are pre-rendered
and toggled via JS visibility.

Supported interactive toggles (matching geo_dashboard.py):
  - Distance Grouping: 6 Distance Bands / 3 Geo Tiers
  - Volume Metric: Customers / Revenue / SpC
  - Activation Mode: Weekly / Cumulative
  - Map Metric: Customers / Rev per 1k Pop / Activation
  - Customer Type: All / FTB / RB
  - Renewals: Include / Exclude

Informational (static — shown but not interactive):
  - House selection, Event Category, Date Range

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
    """Build plz_map_data.parquet and city_map_data.parquet using PLZ_MERGE_MAP from geo_dashboard.py."""
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

    # Use PLZ_MERGE_MAP from the imported geo_dashboard module
    merge_map = getattr(geo, 'PLZ_MERGE_MAP', {})

    coords = pd.read_csv(seeds_dir / 'dim_plz_coordinates.csv', dtype={'plz_three_digits': str})
    pop = pd.read_csv(seeds_dir / 'dim_plz_population.csv', dtype={'plz_gebiet': str})
    pop = pop.rename(columns={'plz_gebiet': 'plz_three_digits', 'ort': 'plz_ort', 'einwohner': 'population'})

    # ── PLZ map (with Berlin/Hamburg merged) ──
    plz_df = coords.merge(pop[['plz_three_digits', 'plz_ort', 'population']], on='plz_three_digits', how='left')
    plz_df['group_key'] = plz_df['plz_three_digits'].map(merge_map).fillna(plz_df['plz_three_digits'])
    plz_df['w_lat'] = plz_df['latitude'] * plz_df['population'].fillna(0)
    plz_df['w_lon'] = plz_df['longitude'] * plz_df['population'].fillna(0)

    agg_plz = plz_df.groupby('group_key').agg(
        w_lat=('w_lat', 'sum'), w_lon=('w_lon', 'sum'),
        population=('population', 'sum'), plz_ort=('plz_ort', 'first'),
    ).reset_index()
    agg_plz['latitude'] = agg_plz['w_lat'] / agg_plz['population'].replace(0, np.nan)
    agg_plz['longitude'] = agg_plz['w_lon'] / agg_plz['population'].replace(0, np.nan)
    agg_plz['plz_three_digits'] = agg_plz['group_key']
    for city_name in merge_map.values():
        agg_plz.loc[agg_plz['group_key'] == city_name, 'plz_ort'] = city_name

    # Transaction aggregates — normalize PLZ codes before grouping
    txn = df.copy()
    txn['plz_three_digits'] = txn['plz_three_digits'].replace(merge_map)
    txn_plz = txn.groupby('plz_three_digits').agg(
        customers=('customer_id', 'nunique'),
        revenue=('variant_price_after_discount', 'sum'),
    ).reset_index()

    result_plz = agg_plz[['plz_three_digits', 'plz_ort', 'latitude', 'longitude', 'population']].merge(
        txn_plz, on='plz_three_digits', how='left')
    result_plz['customers'] = result_plz['customers'].fillna(0).astype(int)
    result_plz['revenue'] = result_plz['revenue'].fillna(0)
    result_plz['population'] = result_plz['population'].fillna(0).astype(int)
    result_plz.to_parquet(plz_cache, index=False)
    print(f"    Written {plz_cache.name} ({len(result_plz)} rows)")

    # ── City map (reads city_population.csv if available, same as dashboard) ──
    city_pop_path = Path(geo.CITY_POP_PATH)
    if city_pop_path.exists():
        city_pop = pd.read_csv(city_pop_path)
    else:
        # Fallback: aggregate from seed using merge_map
        pop['city'] = pop['plz_three_digits'].map(merge_map).fillna(pop['plz_ort'])
        city_pop = pop.groupby('city').agg(population=('population', 'sum')).reset_index()

    plz_df['city'] = plz_df['group_key']
    plz_df.loc[~plz_df['plz_three_digits'].isin(merge_map), 'city'] = plz_df['plz_ort']
    city_coords = plz_df.groupby('city').agg(
        w_lat=('w_lat', 'sum'), w_lon=('w_lon', 'sum'), _pop=('population', 'sum'),
    ).reset_index()
    city_coords['latitude'] = city_coords['w_lat'] / city_coords['_pop'].replace(0, np.nan)
    city_coords['longitude'] = city_coords['w_lon'] / city_coords['_pop'].replace(0, np.nan)

    txn_city = df.groupby('city').agg(
        customers=('customer_id', 'nunique'),
        revenue=('variant_price_after_discount', 'sum'),
    ).reset_index()

    city_df = city_pop.merge(city_coords[['city', 'latitude', 'longitude']], on='city', how='left')
    city_df = city_df.merge(txn_city, on='city', how='left')
    city_df['customers'] = city_df['customers'].fillna(0).astype(int)
    city_df['revenue'] = city_df['revenue'].fillna(0)
    city_df['spc'] = (city_df['revenue'] / city_df['customers'].replace(0, float('nan'))).fillna(0).round(2)
    city_df.to_parquet(city_cache, index=False)
    print(f"    Written {city_cache.name} ({len(city_df)} rows)")


# =============================================================================
# 3. Generate all chart figures for every filter combination
# =============================================================================

FTB_OPTIONS = [('All', 'all'), ('FTB', 'ftb'), ('RB', 'rb')]
RENEWAL_OPTIONS = [('Include', 'incl'), ('Exclude', 'excl')]


def generate_figures(geo, refresh: bool = False, dashboard_dir: Path | None = None):
    """Call every chart function from geo_dashboard for all filter combos and return named figures."""
    import numpy as np

    # Load data using the dashboard's own loaders
    df = geo.load_data(refresh=refresh)
    all_houses_df = geo.load_all_houses(refresh=refresh)
    house_pop_df = geo.load_house_population(refresh=refresh)

    # Build map caches if missing
    if dashboard_dir:
        ensure_map_caches(geo, df, dashboard_dir)

    plz_map_df = geo.load_plz_map_data()
    city_map_df = geo.load_city_map_data()

    # Metadata for sidebar info display
    house_list = geo.get_house_order(df)
    event_cats = sorted(df['event_category'].dropna().unique().tolist())
    date_min = df['created_at'].min().strftime('%d %b %Y')
    date_max = df['created_at'].max().strftime('%d %b %Y')
    meta = {
        'houses': house_list,
        'event_cats': event_cats,
        'date_min': date_min,
        'date_max': date_max,
    }

    figures = {}
    city_tables = {}

    # ── Maps (Section 1) — generated once, shared across all filter combos ──
    # Maps use the full dataset (no FTB/RB or renewals filter) to keep
    # the page lightweight (4 mapbox GL instances instead of 24).
    print("  Generating maps (shared across filters)...")
    if city_map_df is not None:
        figures['city_map_pop'] = geo.chart_city_map(
            city_map_df, all_houses_df, metric='population')

    if plz_map_df is not None:
        for m in ['customers', 'rev_per_1k', 'activation']:
            figures[f'plz_map_{m}'] = geo.chart_population_map(
                plz_map_df, all_houses_df, metric=m, filtered_df=df)

    # ── Charts per filter combination ──
    for ftb_label, ftb_tag in FTB_OPTIONS:
        for ren_label, ren_tag in RENEWAL_OPTIONS:
            suffix = f'__{ftb_tag}__{ren_tag}'
            print(f"  Generating charts for Customer Type={ftb_label}, Renewals={ren_label}...")

            # Apply filters — mirrors geo_dashboard.py lines 1200-1210
            dff = df.copy()
            dff_all = df.copy()      # no FTB/RB filter — for FTS% & LTV cohort
            dff_base = df.copy()     # no FTB/RB, no event — for LTV revenue

            if ftb_label != 'All':
                dff = dff[dff['ftb_rb'] == ftb_label]

            if ren_label == 'Exclude':
                dff = dff[dff['is_renewal'] != 1]
                dff_all = dff_all[dff_all['is_renewal'] != 1]
                dff_base = dff_base[dff_base['is_renewal'] != 1]

            if dff.empty:
                print(f"    Skipping (no data for {ftb_label} / {ren_label})")
                continue

            # ── Section 2: Local Impact ──
            for cm in ['6 Distance Bands', '3 Geo Tiers']:
                tag = '6band' if '6' in cm else '3tier'
                figures[f'comp_pct_{tag}{suffix}'] = geo.chart_1_pct(
                    dff, cm, df_all=dff_all, df_base=dff_base)
                for metric in ['customers', 'revenue', 'spc']:
                    figures[f'comp_{metric}_{tag}{suffix}'] = geo.chart_1_abs(
                        dff, cm, metric=metric, df_all=dff_all, df_base=dff_base)
                figures[f'fts_{tag}{suffix}'] = geo.chart_acq_metric(
                    dff, cm, metric='fts_pct', df_all=dff_all, df_base=dff_base)
                figures[f'ltv_{tag}{suffix}'] = geo.chart_acq_metric(
                    dff, cm, metric='ltv', df_all=dff_all, df_base=dff_base)

            # ── Section 3: Region Activation ──
            for cm in ['6 Distance Bands', '3 Geo Tiers']:
                tag = '6band' if '6' in cm else '3tier'
                figures[f'pop_house_{tag}{suffix}'] = geo.chart_2_1(
                    dff, cm, house_pop_df=house_pop_df)
                figures[f'act_weekly_{tag}{suffix}'] = geo.chart_2_2(
                    dff, cm, cumulative=False)
                figures[f'act_cumul_{tag}{suffix}'] = geo.chart_2_2(
                    dff, cm, cumulative=True)

            figures[f'act_by_house{suffix}'] = geo.chart_activation_by_house(dff)

            # ── Section 4: Geo Mix by House ──
            for cm in ['6 Distance Bands', '3 Geo Tiers']:
                tag = '6band' if '6' in cm else '3tier'
                figures[f'geo_mix_{tag}{suffix}'] = geo.chart_geo_mix_by_house(dff, cm)

            # ── Section 5: Platform Insights ──
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
                        figures[f'platform_{tag}{suffix}'] = geo.chart_platform_geo_composition(
                            dff, cm, selected_platforms=cp_options)

            # ── City Performance Table ──
            city_pop_path = Path(geo.CITY_POP_PATH)
            if (city_pop_path.exists() and 'city' in dff.columns
                    and 'customer_id' in dff.columns and not dff.empty):
                city_pop = pd.read_csv(city_pop_path)
                city_cust = dff.groupby('city').agg(
                    customers=('customer_id', 'nunique'),
                    revenue=('variant_price_after_discount', 'sum'),
                ).reset_index()
                city_tbl = city_pop.merge(city_cust, on='city', how='inner')
                city_tbl['activation'] = (city_tbl['customers'] / (city_tbl['population'] / 1000)).round(2)
                city_tbl['rev_per_1k'] = (city_tbl['revenue'] / (city_tbl['population'] / 1000)).round(2)
                city_tbl = city_tbl.sort_values('population', ascending=False).head(20)
                city_tables[suffix] = city_tbl.to_dict(orient='records')

    return figures, city_tables, meta


# =============================================================================
# 4. Build the HTML page
# =============================================================================

def _is_mapbox_fig(fig) -> bool:
    """Check if a Plotly figure contains mapbox traces."""
    for trace in fig.data:
        if hasattr(trace, 'type') and 'mapbox' in str(trace.type):
            return True
    return False


def fig_to_div(fig, div_id: str, hidden: bool = False) -> str:
    """Convert a Plotly figure to an HTML div (no full page, CDN plotly).

    For mapbox charts, uses visibility:hidden + position:absolute instead of
    display:none so the GL context can still initialize (mapbox requires a
    non-zero-dimension container to load tiles).
    """
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
    is_map = _is_mapbox_fig(fig)

    if hidden:
        if is_map:
            # Keep in the flow at 0 opacity so mapbox GL can init
            style = 'visibility:hidden;position:absolute;left:-9999px;width:100%;height:600px'
        else:
            style = 'display:none'
    else:
        style = 'display:block'

    data_attr = ' data-mapbox="1"' if is_map else ''
    return f'<div id="wrap_{div_id}" style="{style}"{data_attr}>{inner}</div>'


def build_html(figures: dict, geo_module, city_tables: dict = None, meta: dict = None) -> str:
    """Assemble the full index.html with all chart divs, control panel, and city table."""
    import json as _json

    BG = geo_module.BG_COLOR
    TEXT = geo_module.TEXT_COLOR
    TILE_BG = geo_module.SECTION_TILE_BG
    BORDER = geo_module.GRID_COLOR
    SIDEBAR_BG = geo_module.SIDEBAR_BG
    MUTED = '#6b6ba3'
    as_of = date.today().isoformat()

    meta = meta or {}
    city_tables = city_tables or {}

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

    def get_all(base_key, base_hidden=False):
        """Generate all filter variants of a chart.
        base_hidden: if True, even the default filter combo starts hidden.
        """
        html = ''
        for _, ft in FTB_OPTIONS:
            for _, rn in RENEWAL_OPTIONS:
                suffix = f'__{ft}__{rn}'
                full_key = f'{base_key}{suffix}'
                is_default_filter = (ft == 'all' and rn == 'incl')
                hidden = not (is_default_filter and not base_hidden)
                if full_key in figures:
                    html += fig_to_div(figures[full_key], full_key, hidden=hidden)
        if not html:
            return f'<p style="color:{MUTED};font-size:0.8rem">Chart not available (missing cache data)</p>'
        return html

    # ── Assemble main content sections ──
    parts = []

    # Helper for single (non-filtered) charts like maps
    def get(key, hidden=False):
        if key in figures:
            return fig_to_div(figures[key], key, hidden=hidden)
        return f'<p style="color:{MUTED};font-size:0.8rem">Chart not available (missing cache data)</p>'

    # Section 1: Geographic Distribution (maps — shared across filter combos)
    has_maps = any(k.startswith('city_map') or k.startswith('plz_map') for k in figures)
    if has_maps:
        parts.append(section('Geographic Distribution', 'Population density and customer reach by PLZ area'))
        map_left = get('city_map_pop')
        map_right = ''
        for m in ['customers', 'rev_per_1k', 'activation']:
            map_right += get(f'plz_map_{m}', hidden=(m != 'customers'))
        parts.append(row_2(panel(map_left, 'Population by City'), panel(map_right, 'PLZ Area Map')))

    # City Performance Tables (one per filter combo, toggled by JS)
    if city_tables:
        table_parts = []
        for suffix, rows in city_tables.items():
            is_default = suffix == '__all__incl'
            display = 'block' if is_default else 'none'
            tbl_html = f'''<div data-city-table="{suffix}" style="display:{display}">
              <div class="chart-panel" style="margin-top:14px">
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
                for i, r in enumerate(rows)
            ) + '''</tbody></table></div></div></div>'''
            table_parts.append(tbl_html)
        parts.append('\n'.join(table_parts))

    # Section 2: Local Impact
    parts.append(section('Local Impact', 'Geo composition, FTS% and LTV by distance over time'))

    pct_html = get_all('comp_pct_6band') + get_all('comp_pct_3tier', base_hidden=True)
    vol_html = ''
    for metric in ['customers', 'revenue', 'spc']:
        for tag in ['6band', '3tier']:
            hidden = not (metric == 'customers' and tag == '6band')
            vol_html += get_all(f'comp_{metric}_{tag}', base_hidden=hidden)
    parts.append(row_2(panel(pct_html), panel(vol_html)))

    fts_html = get_all('fts_6band') + get_all('fts_3tier', base_hidden=True)
    ltv_html = get_all('ltv_6band') + get_all('ltv_3tier', base_hidden=True)
    parts.append(row_2(panel(fts_html), panel(ltv_html)))

    # Section 3: Region Activation
    parts.append(section('Region Activation', 'Population catchment and activation rate per house'))

    pop_html = get_all('pop_house_6band') + get_all('pop_house_3tier', base_hidden=True)
    act_html = ''
    for mode in ['weekly', 'cumul']:
        for tag in ['6band', '3tier']:
            hidden = not (mode == 'weekly' and tag == '6band')
            act_html += get_all(f'act_{mode}_{tag}', base_hidden=hidden)
    parts.append(row_2(panel(pop_html), panel(act_html)))
    parts.append(panel(get_all('act_by_house')))

    # Section 4: Geo Mix by House
    parts.append(section('Geo Mix by House', 'Distance-band composition evolves as each house matures'))
    mix_html = get_all('geo_mix_6band') + get_all('geo_mix_3tier', base_hidden=True)
    parts.append(panel(mix_html))

    # Section 5: Platform Insights
    if any(k.startswith('platform_') for k in figures):
        parts.append(section('Platform Insights', 'Geo composition per acquisition channel'))
        plat_html = get_all('platform_6band') + get_all('platform_3tier', base_hidden=True)
        parts.append(panel(plat_html))

    body = '\n'.join(parts)

    # ── Sidebar info items ──
    houses_list = meta.get('houses', [])
    events_list = meta.get('event_cats', [])
    date_min = meta.get('date_min', '—')
    date_max = meta.get('date_max', '—')

    houses_info = ', '.join(houses_list) if houses_list else '—'
    events_info = ', '.join(events_list) if events_list else '—'

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
      width: 280px; min-width: 280px; background: {SIDEBAR_BG};
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

    .info-group {{ margin-bottom: 14px; }}
    .info-label {{
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: {MUTED}; margin-bottom: 4px; font-weight: 600;
    }}
    .info-value {{
      font-size: 0.74rem; color: {TEXT}; background: #fff;
      border: 1px solid {BORDER}; border-radius: 6px;
      padding: 5px 10px; line-height: 1.4; opacity: 0.85;
    }}

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
        <div class="ctrl-label">Customer Type</div>
        <div class="ctrl-btns">
          <button class="toggle-btn active" data-group="customer_type" data-value="all"
                  onclick="toggle('customer_type','all')">All</button>
          <button class="toggle-btn" data-group="customer_type" data-value="ftb"
                  onclick="toggle('customer_type','ftb')">FTB</button>
          <button class="toggle-btn" data-group="customer_type" data-value="rb"
                  onclick="toggle('customer_type','rb')">RB</button>
        </div>
      </div>

      <div class="ctrl-group">
        <div class="ctrl-label">Renewals</div>
        <div class="ctrl-btns">
          <button class="toggle-btn active" data-group="renewals" data-value="incl"
                  onclick="toggle('renewals','incl')">Include</button>
          <button class="toggle-btn" data-group="renewals" data-value="excl"
                  onclick="toggle('renewals','excl')">Exclude</button>
        </div>
      </div>

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

      <div class="info-group">
        <div class="info-label">Houses</div>
        <div class="info-value">{houses_info}</div>
      </div>

      <div class="info-group">
        <div class="info-label">Event Category</div>
        <div class="info-value">All ({', '.join(events_list[:3])}{'...' if len(events_list) > 3 else ''})</div>
      </div>

      <div class="info-group">
        <div class="info-label">Date Range</div>
        <div class="info-value">{date_min} — {date_max}</div>
      </div>

      <hr>
      <div style="font-size:0.68rem;color:{MUTED};line-height:1.4">
        Static snapshot of <code style="font-size:0.66rem">geo_dashboard.py</code>.<br>
        Charts are pre-rendered for each Customer Type &amp; Renewals combo.<br>
        House, Event &amp; Date filters show snapshot defaults.
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
      customer_type: 'all',
      renewals: 'incl',
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
      const ct = activeState.customer_type;
      const rn = activeState.renewals;
      const filterSuffix = `__${{ct}}__${{rn}}`;

      // Helpers for show/hide that respect mapbox divs
      function showDiv(el) {{
        if (el.dataset.mapbox) {{
          el.style.visibility = 'visible';
          el.style.position = 'relative';
          el.style.left = '0';
          el.style.height = '';
        }} else {{
          el.style.display = 'block';
        }}
      }}
      function hideDiv(el) {{
        if (el.dataset.mapbox) {{
          el.style.visibility = 'hidden';
          el.style.position = 'absolute';
          el.style.left = '-9999px';
          el.style.height = '600px';
        }} else {{
          el.style.display = 'none';
        }}
      }}
      function isVisible(el) {{
        if (el.dataset.mapbox) return el.style.visibility !== 'hidden';
        return el.style.display !== 'none';
      }}

      // Toggle chart divs
      document.querySelectorAll('[id^="wrap_"]').forEach(el => {{
        const id = el.id.replace('wrap_', '');

        // ── Maps: no filter suffix, only toggled by map_metric ──
        if (id === 'city_map_pop') {{
          showDiv(el);
          return;
        }}
        if (id.startsWith('plz_map_')) {{
          (id.replace('plz_map_', '') === mm) ? showDiv(el) : hideDiv(el);
          return;
        }}

        // ── All other charts have filter suffix __<ftb>__<ren> ──
        if (!id.includes('__')) {{
          // Unknown chart without suffix — leave visible
          return;
        }}

        // Check filter match
        if (!id.endsWith(filterSuffix)) {{
          hideDiv(el);
          return;
        }}

        // Strip filter suffix to get the base chart key
        const base = id.slice(0, id.length - filterSuffix.length);

        // Charts depending on color_mode only
        const cmOnly = ['comp_pct_', 'fts_', 'ltv_', 'pop_house_', 'geo_mix_', 'platform_'];
        for (const prefix of cmOnly) {{
          if (base.startsWith(prefix)) {{
            (base.slice(prefix.length) === cm) ? showDiv(el) : hideDiv(el);
            return;
          }}
        }}

        // Volume charts: vol_metric + color_mode
        if (base.startsWith('comp_') && !base.startsWith('comp_pct_')) {{
          const rest = base.replace('comp_', '');
          const parts = rest.split('_');
          (parts[0] === vm && parts[1] === cm) ? showDiv(el) : hideDiv(el);
          return;
        }}

        // Activation charts: act_mode + color_mode
        if (base.startsWith('act_weekly_') || base.startsWith('act_cumul_')) {{
          const parts = base.split('_');
          (parts[1] === am && parts[2] === cm) ? showDiv(el) : hideDiv(el);
          return;
        }}

        // Always visible (within active filter group)
        if (base === 'act_by_house') {{
          showDiv(el);
        }}
      }});

      // Toggle city tables
      document.querySelectorAll('[data-city-table]').forEach(el => {{
        el.style.display = (el.dataset.cityTable === filterSuffix) ? 'block' : 'none';
      }});

      // Resize and relayout newly visible Plotly charts (especially maps)
      setTimeout(() => {{
        document.querySelectorAll('[id^="wrap_"]').forEach(el => {{
          if (isVisible(el)) {{
            const p = el.querySelector('.js-plotly-plot');
            if (p) {{
              Plotly.Plots.resize(p);
              // Force mapbox tile reload
              if (el.dataset.mapbox) {{
                Plotly.relayout(p, {{}});
              }}
            }}
          }}
        }});
      }}, 150);
    }}

    window.addEventListener('load', applyVisibility);
  </script>
</body>
</html>'''


# =============================================================================
# 5. CLI
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

    print("Generating figures for all filter combinations...")
    figures, city_tables, meta = generate_figures(geo, refresh=args.fresh, dashboard_dir=dashboard_path.parent)
    print(f"  Total: {len(figures)} chart(s), {len(city_tables)} city table(s)")

    print("Building HTML...")
    html = build_html(figures, geo, city_tables=city_tables, meta=meta)

    out_path = Path(args.out) if args.out else here / 'index.html'
    out_path.write_text(html, encoding='utf-8')
    print(f"Written to: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
