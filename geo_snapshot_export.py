"""
Geo Insights — Static Snapshot Export
=====================================

Reads the same BigQuery table that powers `geo_dashboard.py` and produces
a **self-contained** `index.html` with the JSON data embedded inline.

Per the R&D team guidance (html_dashboard_load.md, "EXTRA FROM THE DEVS"):
the HTML includes inline JavaScript that reads an embedded JSON datasource,
so the page works as a single file on GitHub Pages / Violet without needing
a separate fetch.

Usage (run from any directory — paths are resolved relative to THIS file):

    python geo_snapshot_export.py          # uses cached parquet if present
    python geo_snapshot_export.py --fresh  # force re-query from BigQuery

Output:
    - index.html  (same directory as this script — ready for git push)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from string import Template

import pandas as pd
from google.cloud import bigquery

# =============================================================================
# Paths & constants (kept in sync with `geo_dashboard.py`)
# =============================================================================

HERE = Path(__file__).resolve().parent

BQ_TABLE = "ex-omaze-de.bi_development.geo_report"
CACHE_PATH = HERE / "interim_geo_report.parquet"
OUT_PATH = HERE / "index.html"

POP_COLUMNS = [
    "house_population_fifty_km",
    "house_population_hundred_km",
    "house_population_two_hundred_km",
    "house_population_three_hundred_km",
    "house_population_five_hundred_km",
    "house_population_total",
]


# =============================================================================
# Data loading
# =============================================================================

def load_data(refresh: bool = False) -> pd.DataFrame:
    if not refresh and CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
    else:
        client = bigquery.Client()
        df = client.query(f"SELECT * FROM `{BQ_TABLE}`").to_dataframe()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)

    for col in ["created_at", "week_start_date", "house_launch_date_sku", "house_close_date_sku"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def get_house_order(df: pd.DataFrame) -> list[str]:
    if "calendar_house_name" not in df.columns or "calendar_house_rank" not in df.columns:
        return []
    return (
        df[["calendar_house_name", "calendar_house_rank"]]
        .drop_duplicates()
        .sort_values("calendar_house_rank")
        ["calendar_house_name"]
        .tolist()
    )


def build_incremental_pop(df: pd.DataFrame) -> pd.DataFrame:
    house_pop = df[["calendar_house_name"] + POP_COLUMNS].drop_duplicates("calendar_house_name")
    rows = []
    for _, r in house_pop.iterrows():
        h = r["calendar_house_name"]
        p50  = r["house_population_fifty_km"] or 0
        p100 = r["house_population_hundred_km"] or 0
        p200 = r["house_population_two_hundred_km"] or 0
        p300 = r["house_population_three_hundred_km"] or 0
        p500 = r["house_population_five_hundred_km"] or 0
        ptot = r["house_population_total"] or 0
        rows.append({"house": h, "band": "0-50 km",    "pop": p50})
        rows.append({"house": h, "band": "50-100 km",  "pop": p100 - p50})
        rows.append({"house": h, "band": "100-200 km", "pop": p200 - p100})
        rows.append({"house": h, "band": "200-300 km", "pop": p300 - p200})
        rows.append({"house": h, "band": "300-500 km", "pop": p500 - p300})
        rows.append({"house": h, "band": "500+ km",    "pop": ptot - p500})
    return pd.DataFrame(rows)


# =============================================================================
# Snapshot construction
# =============================================================================

def build_snapshot(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Dataframe is empty — cannot build snapshot.")

    # Recent 90-day window
    if "created_at" in df.columns:
        max_created = df["created_at"].max()
        if pd.notna(max_created):
            df = df[df["created_at"] >= max_created - pd.Timedelta(days=90)]

    as_of = None
    if "created_at" in df.columns:
        v = df["created_at"].max()
        as_of = v.date().isoformat() if pd.notna(v) else None

    total_customers = int(df["customer_id"].nunique()) if "customer_id" in df.columns else 0
    total_revenue = float(df["variant_price_after_discount"].sum()) if "variant_price_after_discount" in df.columns else 0.0
    total_houses = int(df["calendar_house_name"].nunique()) if "calendar_house_name" in df.columns else 0

    # Distance bands
    try:
        pop_df = build_incremental_pop(df)
        dist = pop_df.groupby("band")["pop"].sum().reset_index()
        total_pop = float(dist["pop"].sum()) or 1.0
        dist["pct"] = (dist["pop"] / total_pop * 100).round(1)
        # Ensure sorted order
        band_order = ["0-50 km", "50-100 km", "100-200 km", "200-300 km", "300-500 km", "500+ km"]
        dist["_sort"] = dist["band"].apply(lambda b: band_order.index(b) if b in band_order else 99)
        dist = dist.sort_values("_sort").drop(columns="_sort")
        distance_bands = dist.rename(columns={"band": "label", "pop": "population", "pct": "population_pct"}).to_dict(orient="records")
    except Exception:
        distance_bands = []

    # Top cities
    top_cities = []
    if "city" in df.columns and "customer_id" in df.columns:
        city_agg = df.groupby("city").agg(
            customers=("customer_id", "nunique"),
            revenue=("variant_price_after_discount", "sum"),
        ).reset_index().sort_values("customers", ascending=False).head(10)
        top_cities = city_agg.to_dict(orient="records")

    # Geo composition over time (weekly)
    geo_weekly = []
    if {"week_start_date", "distance_to_house_bucket_km", "customer_id", "calendar_house_name"}.issubset(df.columns):
        meta = (
            df.groupby(["week_start_date", "calendar_house_name", "map_event_house"])
            .size().reset_index(name="_n")
            .sort_values("_n", ascending=False)
            .drop_duplicates("week_start_date")
        )
        wk = df.groupby(["week_start_date", "distance_to_house_bucket_km"])["customer_id"].nunique().reset_index(name="customers")
        totals = wk.groupby("week_start_date")["customers"].transform("sum")
        wk["pct"] = (wk["customers"] / totals * 100).round(1)
        wk = wk.merge(meta[["week_start_date", "calendar_house_name", "map_event_house"]], on="week_start_date", how="left")
        wk["week_str"] = wk["week_start_date"].dt.strftime("%d %b")
        wk["label"] = wk["calendar_house_name"].fillna("") + " | " + wk["map_event_house"].fillna("") + " | " + wk["week_str"]
        for _, row in wk.iterrows():
            geo_weekly.append({
                "week": row["week_start_date"].isoformat(),
                "label": row["label"],
                "band": row["distance_to_house_bucket_km"],
                "customers": int(row["customers"]),
                "pct": float(row["pct"]),
            })

    # Houses
    houses = []
    if {"calendar_house_name", "customer_id", "variant_price_after_discount"}.issubset(df.columns):
        house_order = get_house_order(df)
        h_agg = df.groupby("calendar_house_name").agg(
            customers=("customer_id", "nunique"),
            revenue=("variant_price_after_discount", "sum"),
        ).reset_index()
        if house_order:
            h_agg["_o"] = h_agg["calendar_house_name"].apply(lambda h: house_order.index(h) if h in house_order else 999)
            h_agg = h_agg.sort_values("_o").drop(columns="_o")
        houses = h_agg.rename(columns={"calendar_house_name": "house_name"}).to_dict(orient="records")

    return {
        "as_of": as_of,
        "kpis": {"total_customers": total_customers, "total_revenue": total_revenue, "total_houses": total_houses},
        "distance_bands": distance_bands,
        "top_cities": top_cities,
        "geo_weekly": geo_weekly,
        "houses": houses,
    }


# =============================================================================
# HTML template (self-contained with inline JS + embedded JSON)
# =============================================================================

HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Geo Insights — Snapshot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {
      --bg: #f0f2fc; --card: #ffffff; --border: #dfe1f0;
      --text: #383977; --muted: #6b6ba3; --accent: #836DFF;
      --navy: #383977; --teal: #5CB8B2; --gold: #E6A832;
      --red: #D4544A; --grey: #B8B8B8; --orange: #FF6B00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text);
    }
    .page { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
    .header {
      display: flex; justify-content: space-between; align-items: flex-start;
      gap: 16px; padding-bottom: 14px; border-bottom: 2px solid var(--border);
    }
    .title { font-size: 1.6rem; font-weight: 700; color: var(--navy); }
    .subtitle { font-size: 0.85rem; color: var(--muted); margin-top: 4px; }
    .as-of { font-size: 0.75rem; color: var(--muted); white-space: nowrap; }

    .kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 20px 0 24px; }
    .kpi-card {
      background: var(--card); border-radius: 12px; border: 1px solid var(--border);
      padding: 14px 16px; box-shadow: 0 2px 8px rgba(56,57,119,0.06);
    }
    .kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 4px; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; color: var(--navy); }
    .kpi-meta { font-size: 0.7rem; color: var(--muted); margin-top: 3px; }

    .section { margin-top: 24px; }
    .section-tile {
      background: #e4e6f4; padding: 10px 16px; border-radius: 8px; margin-bottom: 14px;
    }
    .section-tile h2 { margin: 0; font-size: 1.15rem; font-weight: 600; color: var(--navy); }
    .section-tile p { margin: 3px 0 0; font-size: 0.8rem; color: var(--muted); }

    .grid-2 { display: grid; grid-template-columns: 1.3fr 1fr; gap: 14px; }
    .panel {
      background: var(--card); border-radius: 12px; border: 1px solid var(--border);
      padding: 14px 16px; box-shadow: 0 2px 8px rgba(56,57,119,0.06);
    }
    .panel-title { font-size: 0.85rem; font-weight: 600; color: var(--navy); margin-bottom: 2px; }
    .panel-sub { font-size: 0.72rem; color: var(--muted); margin-bottom: 8px; }

    /* View toggle */
    .view-toggle { display: flex; gap: 6px; margin-bottom: 12px; }
    .view-toggle button {
      padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
      background: var(--card); color: var(--navy); font-size: 0.78rem; cursor: pointer;
      transition: all 0.15s;
    }
    .view-toggle button.active { background: var(--navy); color: #fff; border-color: var(--navy); }
    .view-toggle button:hover:not(.active) { background: #e4e6f4; }

    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    th, td { padding: 7px 6px; border-bottom: 1px solid var(--border); text-align: left; }
    th { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
    tr:last-child td { border-bottom: none; }

    @media (max-width: 900px) {
      .kpi-row { grid-template-columns: 1fr 1fr; }
      .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <div class="title">Geo Insights — Snapshot</div>
        <div class="subtitle">Static export of the Streamlit geo_dashboard.py for embedding into Violet via GitHub Pages</div>
      </div>
      <div class="as-of" id="asOfLabel">Loading…</div>
    </div>

    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Customers</div>
        <div class="kpi-value" id="kpiCustomers">–</div>
        <div class="kpi-meta">Unique customers (last 90 days)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Revenue</div>
        <div class="kpi-value" id="kpiRevenue">–</div>
        <div class="kpi-meta">Total revenue after discount</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Active Houses</div>
        <div class="kpi-value" id="kpiHouses">–</div>
        <div class="kpi-meta">Distinct houses in period</div>
      </div>
    </div>

    <!-- Distance Bands + Top Cities -->
    <div class="section grid-2">
      <div class="panel">
        <div class="panel-title">Population by Distance Band</div>
        <div class="panel-sub">Aggregated across all houses</div>
        <div id="distChart" style="width:100%;height:280px"></div>
      </div>
      <div class="panel">
        <div class="panel-title">Top 10 Cities</div>
        <div class="panel-sub">By unique customers</div>
        <table id="cityTable"><thead><tr><th>City</th><th>Customers</th><th>Revenue</th></tr></thead><tbody></tbody></table>
      </div>
    </div>

    <!-- Geo Composition Over Time -->
    <div class="section">
      <div class="section-tile">
        <h2>Geo Composition Over Time</h2>
        <p>Weekly distance-band mix by house and event</p>
      </div>
      <div class="panel">
        <div class="view-toggle">
          <button id="btnPct" class="active" onclick="setGeoView('pct')">Percentage</button>
          <button id="btnAbs" onclick="setGeoView('abs')">Absolute</button>
        </div>
        <div id="geoChart" style="width:100%;height:420px"></div>
      </div>
    </div>

    <!-- House Summary -->
    <div class="section">
      <div class="section-tile">
        <h2>House Summary</h2>
        <p>Customers and revenue per house</p>
      </div>
      <div class="panel">
        <div id="houseChart" style="width:100%;height:320px"></div>
      </div>
    </div>
  </div>

  <script>
    // ── Embedded snapshot data (generated by geo_snapshot_export.py) ──
    const DATA = $snapshot_json;

    // ── Palette (matches geo_dashboard.py theme) ──
    const PAL = {
      '1. 0-50 km':    '#383977', '2. 50-100 km':  '#836DFF',
      '3. 100-200 km': '#5CB8B2', '4. 200-300 km': '#E6A832',
      '5. 300-500 km': '#D4544A', '6. 500+ km':    '#B8B8B8',
      // short labels for distance_bands
      '0-50 km': '#383977',   '50-100 km': '#836DFF',
      '100-200 km': '#5CB8B2','200-300 km': '#E6A832',
      '300-500 km': '#D4544A','500+ km': '#B8B8B8',
    };
    const BAND_ORDER = ['1. 0-50 km','2. 50-100 km','3. 100-200 km','4. 200-300 km','5. 300-500 km','6. 500+ km'];
    const PLOTLY_BG = '#f0f2fc';
    const FONT_COLOR = '#383977';

    function fmt(x) { return x.toLocaleString('en-GB'); }
    function fmtMoney(x) { return '€' + x.toLocaleString('en-GB', {minimumFractionDigits: 0, maximumFractionDigits: 0}); }

    // ── KPIs ──
    function renderKpis() {
      document.getElementById('asOfLabel').textContent = 'Snapshot as of ' + (DATA.as_of || 'unknown');
      const k = DATA.kpis || {};
      document.getElementById('kpiCustomers').textContent = fmt(k.total_customers || 0);
      document.getElementById('kpiRevenue').textContent = fmtMoney(k.total_revenue || 0);
      document.getElementById('kpiHouses').textContent = fmt(k.total_houses || 0);
    }

    // ── Distance Bands Chart ──
    function renderDistChart() {
      const bands = DATA.distance_bands || [];
      if (!bands.length) { document.getElementById('distChart').innerHTML = '<p style="color:var(--muted)">No data</p>'; return; }
      const trace = {
        x: bands.map(d => d.label), y: bands.map(d => d.population_pct),
        type: 'bar',
        marker: { color: bands.map(d => PAL[d.label] || '#836DFF') },
        hovertemplate: '%{x}<br>%{y:.1f}% of population<br>Pop: %{customdata:,.0f}<extra></extra>',
        customdata: bands.map(d => d.population),
      };
      Plotly.newPlot('distChart', [trace], {
        margin: {l:42, r:10, t:10, b:90},
        paper_bgcolor: PLOTLY_BG, plot_bgcolor: PLOTLY_BG,
        font: {color: FONT_COLOR}, xaxis: {tickangle: -35}, yaxis: {title: '% of population'},
      }, {displayModeBar: false, responsive: true});
    }

    // ── Top Cities Table ──
    function renderCityTable() {
      const rows = DATA.top_cities || [];
      const tbody = document.querySelector('#cityTable tbody');
      tbody.innerHTML = '';
      if (!rows.length) { tbody.innerHTML = '<tr><td colspan="3" style="color:var(--muted)">No data</td></tr>'; return; }
      rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + (r.city||'—') + '</td><td>' + fmt(r.customers||0) + '</td><td>' + fmtMoney(r.revenue||0) + '</td>';
        tbody.appendChild(tr);
      });
    }

    // ── Geo Composition (interactive toggle % / absolute) ──
    let currentGeoView = 'pct';

    function setGeoView(mode) {
      currentGeoView = mode;
      document.getElementById('btnPct').className = mode === 'pct' ? 'active' : '';
      document.getElementById('btnAbs').className = mode === 'abs' ? 'active' : '';
      renderGeoChart();
    }

    function renderGeoChart() {
      const weekly = DATA.geo_weekly || [];
      if (!weekly.length) { document.getElementById('geoChart').innerHTML = '<p style="color:var(--muted)">No weekly data</p>'; return; }

      // Build x-axis order (unique labels in week order)
      const seen = new Set();
      const xOrder = [];
      weekly.forEach(d => { if (!seen.has(d.label)) { seen.add(d.label); xOrder.push(d.label); } });

      // Group by band
      const byBand = {};
      weekly.forEach(d => {
        if (!byBand[d.band]) byBand[d.band] = {};
        byBand[d.band][d.label] = d;
      });

      const yCol = currentGeoView === 'pct' ? 'pct' : 'customers';
      const yTitle = currentGeoView === 'pct' ? '% of Customers' : 'Customers';
      const chartType = currentGeoView === 'pct' ? 'bar' : 'scatter';

      const traces = BAND_ORDER.filter(b => byBand[b]).map(band => {
        const vals = byBand[band];
        const x = xOrder;
        const y = xOrder.map(lbl => vals[lbl] ? vals[lbl][yCol] : 0);
        const base = { x, y, name: band, marker: {color: PAL[band]} };
        if (chartType === 'bar') {
          return { ...base, type: 'bar', hovertemplate: '%{x}<br>%{y:.1f}%<extra>' + band + '</extra>' };
        }
        return { ...base, type: 'scatter', mode: 'lines+markers',
          line: {color: PAL[band], shape: 'spline'}, marker: {color: PAL[band], size: 5},
          hovertemplate: '%{x}<br>%{y:,.0f} customers<extra>' + band + '</extra>' };
      });

      const layout = {
        barmode: currentGeoView === 'pct' ? 'stack' : undefined,
        margin: {l:50, r:120, t:10, b:130},
        paper_bgcolor: PLOTLY_BG, plot_bgcolor: PLOTLY_BG,
        font: {color: FONT_COLOR},
        xaxis: {tickangle: -45, type: 'category', categoryorder: 'array', categoryarray: xOrder},
        yaxis: {title: yTitle, range: currentGeoView === 'pct' ? [0, 100] : undefined},
        legend: {orientation: 'v', yanchor: 'top', y: 1, xanchor: 'left', x: 1.02},
      };

      Plotly.newPlot('geoChart', traces, layout, {displayModeBar: false, responsive: true});
    }

    // ── Houses Chart ──
    function renderHouseChart() {
      const houses = DATA.houses || [];
      if (!houses.length) { document.getElementById('houseChart').innerHTML = '<p style="color:var(--muted)">No data</p>'; return; }
      const names = houses.map(h => h.house_name);
      const t1 = { x: names, y: houses.map(h => h.customers||0), name: 'Customers', type: 'bar', marker: {color: '#383977'}, yaxis: 'y1' };
      const t2 = { x: names, y: houses.map(h => h.revenue||0), name: 'Revenue', type: 'scatter', mode: 'lines+markers',
        line: {color: '#E6A832'}, marker: {color: '#E6A832'}, yaxis: 'y2' };
      Plotly.newPlot('houseChart', [t1, t2], {
        margin: {l:50, r:60, t:10, b:90},
        paper_bgcolor: PLOTLY_BG, plot_bgcolor: PLOTLY_BG,
        font: {color: FONT_COLOR}, xaxis: {tickangle: -35},
        yaxis: {title: 'Customers'}, yaxis2: {title: 'Revenue (€)', overlaying: 'y', side: 'right', showgrid: false},
        legend: {orientation: 'h', y: -0.22},
      }, {displayModeBar: false, responsive: true});
    }

    // ── Init ──
    renderKpis();
    renderDistChart();
    renderCityTable();
    renderGeoChart();
    renderHouseChart();
  </script>
</body>
</html>""")


# =============================================================================
# Main
# =============================================================================

def export(refresh: bool = False) -> Path:
    df = load_data(refresh=refresh)
    snapshot = build_snapshot(df)
    snapshot_json = json.dumps(snapshot, default=str, ensure_ascii=False)
    html = HTML_TEMPLATE.safe_substitute(snapshot_json=snapshot_json)
    OUT_PATH.write_text(html, encoding="utf-8")
    return OUT_PATH


def main() -> None:
    refresh = "--fresh" in sys.argv or "--refresh" in sys.argv
    out = export(refresh=refresh)
    print(f"Self-contained HTML snapshot written to: {out}")


if __name__ == "__main__":
    main()
