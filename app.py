
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

CLEAN_FILE = PROCESSED_DIR / "california_acs_2010_2016_clean.csv"
TARGET_FILE = RAW_DIR / "housing_cost_target.csv"
SHAPEFILE = RAW_DIR / "tract10" / "tl_2010_06_tract10.shp"

st.set_page_config(
    page_title="California Gentrification Early Warning System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
        max-width: 1900px;
    }

    [data-testid="stTabs"] button {
        font-weight: 600;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        gap: 0.8rem;
    }

    .hero {
        padding: 0.35rem 0 0.55rem 0;
    }

    .hero h1 {
        font-size: 1.9rem;
        line-height: 1.2;
        font-weight: 750;
        margin: 0;
        color: #f8fafc;
    }

    .hero p {
        display: none;
    }

    .metric-card {
        background: #171c28;
        border: 1px solid #2b3240;
        border-radius: 12px;
        padding: 18px;
        min-height: 115px;
    }

    .metric-label {
        color: #a5b0c0;
        font-size: 0.88rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .metric-caption {
        color: #a5b0c0;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    .notice {
        background: #172a3d;
        border: 1px solid #254967;
        border-radius: 8px;
        padding: 14px 16px;
        color: #d7e9fa;
        margin: 10px 0 18px 0;
    }

    .section-subtitle {
        color: #a5b0c0;
        font-size: 0.9rem;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    div[data-testid="stMetric"] {
        background: #171c28;
        border: 1px solid #2b3240;
        border-radius: 10px;
        padding: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a5b0c0;
    }

    hr {
        border-color: #2b3240;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e5e7eb"),
    margin=dict(l=35, r=25, t=55, b=45),
    height=390,
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_id(series):
    """Normalize Census tract IDs without converting them to floats."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(11)
    )


def money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def number(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_target():
    """Load and validate the historical target dataset."""
    if not TARGET_FILE.exists():
        return None, (
            f"Target dataset not found: {TARGET_FILE.relative_to(ROOT)}"
        )

    df = pd.read_csv(
        TARGET_FILE,
        dtype={"tract_id": "string"},
        low_memory=False,
    )

    required = {
        "tract_id",
        "housing_cost_2012",
        "housing_cost_2016",
        "housing_cost_change",
        "gentrified",
    }

    missing = required.difference(df.columns)

    if missing:
        return None, (
            "The target CSV is missing these columns: "
            + ", ".join(sorted(missing))
        )

    df["tract_id"] = normalize_id(df["tract_id"])

    for col in [
        "housing_cost_2012",
        "housing_cost_2016",
        "housing_cost_change",
        "gentrified",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required)).copy()

    # Remove invalid or impossible cost observations.
    # ACS missing-value sentinels should already have been cleaned
    # by the target-building script.
    df = df[
        df["housing_cost_2012"].between(1, 100000)
        & df["housing_cost_2016"].between(1, 100000)
    ].copy()

    df["gentrified"] = (
        df["housing_cost_change"] > 0
    ).astype(int)

    # Recalculate change to keep all three columns consistent.
    df["housing_cost_change"] = (
        df["housing_cost_2016"] - df["housing_cost_2012"]
    )

    df = df.drop_duplicates("tract_id", keep="first")

    return df, None


@st.cache_data(show_spinner=False)
def load_acs():
    """Load cleaned historical ACS features if available."""
    if not CLEAN_FILE.exists():
        return None

    df = pd.read_csv(
        CLEAN_FILE,
        dtype={"GEOID": "string"},
        low_memory=False,
    )

    if "GEOID" in df.columns:
        df["tract_id"] = normalize_id(df["GEOID"])
    elif "tract_id" in df.columns:
        df["tract_id"] = normalize_id(df["tract_id"])
    else:
        return None

    return df.drop_duplicates("tract_id", keep="first")


@st.cache_data(show_spinner=False)
def load_map_data(target_signature):
    """Load tract boundaries and join historical targets."""
    try:
        import geopandas as gpd
    except ImportError:
        return None, "GeoPandas is not installed."

    if not SHAPEFILE.exists():
        return None, (
            "The census-tract shapefile was not found at "
            f"{SHAPEFILE.relative_to(ROOT)}."
        )

    target, error = load_target()

    if error or target is None:
        return None, error or "Could not load the target dataset."

    try:
        tracts = gpd.read_file(SHAPEFILE)
    except Exception as exc:
        return None, f"Could not read the shapefile: {exc}"

    # Prefer standard Census GEOID fields.
    candidates = [
        "GEOID10",
        "GEOID",
        "GEOID_10",
        "GEOID20",
    ]

    geo_col = next(
        (col for col in candidates if col in tracts.columns),
        None,
    )

    if geo_col is not None:
        tracts["tract_id"] = normalize_id(tracts[geo_col])

    elif all(
        col in tracts.columns
        for col in ["STATEFP10", "COUNTYFP10", "TRACTCE10"]
    ):
        tracts["tract_id"] = (
            tracts["STATEFP10"].astype(str).str.zfill(2)
            + tracts["COUNTYFP10"].astype(str).str.zfill(3)
            + tracts["TRACTCE10"].astype(str).str.zfill(6)
        )

    elif all(
        col in tracts.columns
        for col in ["STATEFP", "COUNTYFP", "TRACTCE"]
    ):
        tracts["tract_id"] = (
            tracts["STATEFP"].astype(str).str.zfill(2)
            + tracts["COUNTYFP"].astype(str).str.zfill(3)
            + tracts["TRACTCE"].astype(str).str.zfill(6)
        )

    else:
        return None, (
            "Could not identify a census tract GEOID in the shapefile. "
            f"Available columns: {tracts.columns.tolist()}"
        )

    # The join is on the 11-digit Census tract GEOID.
    map_gdf = tracts.merge(
        target,
        on="tract_id",
        how="left",
        validate="one_to_one",
    )

    # Standardize the display CRS.
    if map_gdf.crs is not None:
        map_gdf = map_gdf.to_crs(epsg=4326)

    if map_gdf["gentrified"].notna().sum() == 0:
        return None, (
            "The map boundaries loaded, but no target rows matched. "
            "Check the GEOID format and geographic vintage."
        )

    return map_gdf, None


def available_metrics(df):
    """Return only map metrics present in the joined dataset."""
    definitions = {
        "Housing-cost change (2012–2016)": "housing_cost_change",
        "Housing cost in 2012": "housing_cost_2012",
        "Housing cost in 2016": "housing_cost_2016",
        "Historical housing-cost increase label": "gentrified",
    }

    return {
        label: col
        for label, col in definitions.items()
        if col in df.columns
    }


def find_result_files():
    """Find saved model evaluation CSVs."""
    if not RESULTS_DIR.exists():
        return []

    return sorted(RESULTS_DIR.rglob("*.csv"))


def find_model_files():
    """Find saved model artifacts."""
    if not MODELS_DIR.exists():
        return []

    extensions = {".pkl", ".joblib", ".sav", ".keras", ".h5"}

    return sorted(
        path for path in MODELS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

target_df, target_error = load_target()

if target_error or target_df is None:
    st.error(target_error or "Unable to load historical data.")
    st.info(
        "Check that data/raw/housing_cost_target.csv exists "
        "and contains the expected columns."
    )
    st.stop()

acs_df = load_acs()

valid_tract_count = len(target_df)
rising_count = int(target_df["gentrified"].sum())
non_rising_count = valid_tract_count - rising_count
rising_pct = 100 * rising_count / valid_tract_count if valid_tract_count else 0

state_median_2012 = target_df["housing_cost_2012"].median()
state_median_2016 = target_df["housing_cost_2016"].median()
median_change = target_df["housing_cost_change"].median()

# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>California Gentrification Early Warning</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, explorer_tab, map_tab, models_tab, methodology_tab = st.tabs(
    [
        "Historical Overview",
        "Tract Explorer",
        "California Map",
        "Model Results",
        "Methodology",
    ]
)

# ============================================================
# 7. HISTORICAL OVERVIEW
# ============================================================

with overview_tab:
    st.header("Historical housing-cost analysis")
    st.markdown(
        '<p class="section-subtitle">'
        "Summary of the matched California census tracts."
        "</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Valid census tracts",
            number(valid_tract_count),
            "Matched historical target rows",
        )

    with c2:
        metric_card(
            "Median cost in 2012",
            money(state_median_2012),
            "Selected monthly housing-cost measure",
        )

    with c3:
        metric_card(
            "Median cost in 2016",
            money(state_median_2016),
            f"Median change: {money(median_change)}",
        )

    with c4:
        metric_card(
            "Tracts with rising costs",
            number(rising_count),
            f"{rising_pct:.1f}% of matched tracts",
        )

    st.subheader("Housing-cost comparison")

    cost_summary = pd.DataFrame(
        {
            "Year": ["2012", "2016"],
            "Median housing cost": [
                state_median_2012,
                state_median_2016,
            ],
        }
    )

    fig = px.bar(
        cost_summary,
        x="Year",
        y="Median housing cost",
        text_auto=".0f",
        title="Median housing cost across valid census tracts",
    )
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_yaxes(title="Housing cost (source units)")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Historical target distribution")

        label_df = pd.DataFrame(
            {
                "Historical label": [
                    "No increase",
                    "Increase",
                ],
                "Census tracts": [
                    non_rising_count,
                    rising_count,
                ],
            }
        )

        fig = px.bar(
            label_df,
            x="Historical label",
            y="Census tracts",
            text="Census tracts",
            title="Tracts by housing-cost change label",
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Housing-cost change distribution")

        fig = px.histogram(
            target_df,
            x="housing_cost_change",
            nbins=55,
            title="Change in housing costs, 2012–2016",
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            annotation_text="No change",
        )
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_xaxes(title="Housing-cost change (source units)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Largest historical housing-cost increases")

    top_increases = (
        target_df.sort_values("housing_cost_change", ascending=False)
        .head(10)
        .copy()
    )

    top_increases["Housing cost, 2012"] = (
        top_increases["housing_cost_2012"].map(money)
    )
    top_increases["Housing cost, 2016"] = (
        top_increases["housing_cost_2016"].map(money)
    )
    top_increases["Change"] = (
        top_increases["housing_cost_change"].map(money)
    )

    st.dataframe(
        top_increases[
            [
                "tract_id",
                "Housing cost, 2012",
                "Housing cost, 2016",
                "Change",
            ]
        ].rename(columns={"tract_id": "Census tract GEOID"}),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download validated historical target CSV",
        data=target_df.to_csv(index=False).encode("utf-8"),
        file_name="california_historical_target.csv",
        mime="text/csv",
        key="download_historical_target_csv",
    )


# ============================================================
# 8. TRACT EXPLORER
# ============================================================

with explorer_tab:
    st.header("Explore a census tract")
    st.markdown(
        '<p class="section-subtitle">'
        "Select a tract by its geographic ID and compare it with "
        "statewide historical values."
        "</p>",
        unsafe_allow_html=True,
    )

    tract_ids = sorted(target_df["tract_id"].dropna().unique().tolist())

    if not tract_ids:
        st.warning("No valid census tract IDs are available.")
    else:
        selected_tract = st.selectbox(
            "Census tract GEOID",
            tract_ids,
            key="explorer_tract",
        )

        selected_row = target_df[
            target_df["tract_id"] == selected_tract
        ].iloc[0]

        a, b, c, d = st.columns(4)

        with a:
            metric_card(
                "Housing cost in 2012",
                money(selected_row["housing_cost_2012"]),
            )

        with b:
            metric_card(
                "Housing cost in 2016",
                money(selected_row["housing_cost_2016"]),
            )

        with c:
            metric_card(
                "Absolute change",
                money(selected_row["housing_cost_change"]),
            )

        with d:
            label_text = (
                "Increase"
                if selected_row["gentrified"] == 1
                else "No increase"
            )
            metric_card(
                "Historical target",
                label_text,
                "Housing-cost proxy only",
            )

        comparison = pd.DataFrame(
            {
                "Year": ["2012", "2016"],
                "Selected tract": [
                    selected_row["housing_cost_2012"],
                    selected_row["housing_cost_2016"],
                ],
                "Statewide median": [
                    state_median_2012,
                    state_median_2016,
                ],
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=comparison["Year"],
                y=comparison["Selected tract"],
                name="Selected tract",
            )
        )

        fig.add_trace(
            go.Bar(
                x=comparison["Year"],
                y=comparison["Statewide median"],
                name="Statewide median",
            )
        )

        fig.update_layout(
            barmode="group",
            title="Selected tract versus statewide median",
            **PLOT_LAYOUT,
        )
        fig.update_yaxes(title="Housing cost (source units)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Available historical indicators")

        if acs_df is not None:
            tract_acs = acs_df[
                acs_df["tract_id"] == selected_tract
            ]

            if not tract_acs.empty:
                row = tract_acs.iloc[0]

                indicator_cols = [
                    col for col in tract_acs.columns
                    if col not in {"tract_id", "GEOID"}
                    and (
                        col.endswith("_2010")
                        or col.endswith("_2011")
                        or col.endswith("_2012")
                        or col.endswith("_2013")
                        or col.endswith("_2014")
                        or col.endswith("_2015")
                        or col.endswith("_2016")
                    )
                ]

                indicator_df = pd.DataFrame(
                    {
                        "Indicator": indicator_cols,
                        "Value": [
                            row[col] for col in indicator_cols
                        ],
                    }
                )

                st.dataframe(
                    indicator_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No cleaned ACS feature row was found for this tract."
                )
        else:
            st.info(
                "The cleaned ACS feature dataset was not found. "
                "The housing-cost comparison is still available."
            )

        st.download_button(
            "Download selected tract data",
            data=pd.DataFrame([selected_row]).to_csv(
                index=False
            ).encode("utf-8"),
            file_name=f"tract_{selected_tract}_history.csv",
            mime="text/csv",
            key="download_explorer_tract_csv",
        )


# ============================================================
# 9. CALIFORNIA INTERACTIVE MAP
# ============================================================

with map_tab:
    st.markdown(
        """
        <div style="padding:0.15rem 0 0.5rem 0;">
          <h2 style="margin-bottom:0.2rem;">California map</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    map_gdf, map_error = load_map_data(
        f"{TARGET_FILE.stat().st_mtime if TARGET_FILE.exists() else 0}"
    )

    if map_error or map_gdf is None:
        st.warning(map_error or "The map could not be loaded.")
        st.info("Verify the target CSV, California tract shapefile, and mapping dependencies.")
        st.code("python -m pip install geopandas folium streamlit-folium")
        st.code("data/raw/tract10/tl_2010_06_tract10.shp")
    else:
        try:
            import folium
            from streamlit_folium import st_folium

            map_gdf = map_gdf[
                map_gdf.geometry.notna() & ~map_gdf.geometry.is_empty
            ].copy()
            if map_gdf.crs is not None:
                map_gdf = map_gdf.to_crs(epsg=4326)
            map_gdf = map_gdf.reset_index(drop=True)

            valid_map_rows = int(map_gdf["gentrified"].notna().sum())
            total_geo_tracts = len(map_gdf)
            avg_change = map_gdf["housing_cost_change"].mean()
            rising_map_count = int((map_gdf["gentrified"] == 1).sum())
            rising_map_pct = 100 * rising_map_count / valid_map_rows if valid_map_rows else 0

            # Top summary cards follow the reference dashboard, using values from this dataset.
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Total census tracts (CA)", number(total_geo_tracts), "2010 Census geography")
            with k2:
                metric_card("Valid matched tracts", number(valid_map_rows), "After housing-cost validation")
            with k3:
                metric_card("Tracts with rising costs", number(rising_map_count), f"{rising_map_pct:.1f}% of valid matched tracts")
            with k4:
                avg_pct = (
                    100 * (map_gdf["housing_cost_2016"] / map_gdf["housing_cost_2012"] - 1)
                ).replace([np.inf, -np.inf], np.nan).dropna().mean()
                avg_display = pct(avg_pct) if pd.notna(avg_pct) else "N/A"
                metric_card("Average housing-cost change", avg_display, "2012–2016 across matched tracts")

            left_controls, right_controls = st.columns([1.15, 4.85])
            tract_ids = sorted(map_gdf.loc[map_gdf["gentrified"].notna(), "tract_id"].astype(str).unique())
            with left_controls:
                st.markdown("#### Map controls")
                map_metric_label = st.selectbox(
                    "Color tracts by",
                    [
                        "Historical housing-cost increase label",
                        "Housing-cost change (2012–2016)",
                        "Housing cost in 2012",
                        "Housing cost in 2016",
                    ],
                    index=0,
                    key="map_color_metric",
                )
                metric_col = {
                    "Historical housing-cost increase label": "gentrified",
                    "Housing-cost change (2012–2016)": "housing_cost_change",
                    "Housing cost in 2012": "housing_cost_2012",
                    "Housing cost in 2016": "housing_cost_2016",
                }[map_metric_label]
                initial_tract = st.session_state.get("selected_map_tract")
                if initial_tract not in tract_ids:
                    initial_tract = tract_ids[0] if tract_ids else None
                if tract_ids:
                    selected_by_search = st.selectbox(
                        "Search/select census tract",
                        tract_ids,
                        index=tract_ids.index(initial_tract) if initial_tract in tract_ids else 0,
                    )
                    st.session_state["selected_map_tract"] = selected_by_search
                else:
                    selected_by_search = None
                    st.warning("No matched tract IDs are available.")

                st.markdown("#### Historical label legend")
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:8px;margin:8px 0;'>"
                    "<span style='width:13px;height:13px;background:#ef4444;border-radius:3px;display:inline-block;'></span>"
                    "<span>Housing costs increased</span></div>"
                    "<div style='display:flex;align-items:center;gap:8px;margin:8px 0;'>"
                    "<span style='width:13px;height:13px;background:#10b981;border-radius:3px;display:inline-block;'></span>"
                    "<span>No increase</span></div>"
                    "<div style='display:flex;align-items:center;gap:8px;margin:8px 0;'>"
                    "<span style='width:13px;height:13px;background:#475569;border-radius:3px;display:inline-block;'></span>"
                    "<span>No matched data</span></div>",
                    unsafe_allow_html=True,
                )
                

            map_gdf[metric_col] = pd.to_numeric(map_gdf[metric_col], errors="coerce")
            values = map_gdf[metric_col].dropna()
            if not values.empty:
                vmin = float(values.quantile(0.02))
                vmax = float(values.quantile(0.98))
                if vmin == vmax:
                    vmax = vmin + 1
            else:
                vmin, vmax = 0.0, 1.0

            def feature_color(value):
                if value is None or pd.isna(value):
                    return "#475569"
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return "#475569"
                if not np.isfinite(value):
                    return "#475569"
                if metric_col == "gentrified":
                    return "#ef4444" if value == 1 else "#10b981"
                scaled = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
                # Blue -> yellow -> red ramp for continuous measures.
                if scaled < 0.5:
                    t = scaled * 2
                    a, b = np.array([37, 99, 235]), np.array([250, 204, 21])
                else:
                    t = (scaled - 0.5) * 2
                    a, b = np.array([250, 204, 21]), np.array([239, 68, 68])
                rgb = (a + (b - a) * t).astype(int)
                return "#{:02x}{:02x}{:02x}".format(*rgb)

            def style_function(feature):
                props = feature.get("properties", {})
                fill = feature_color(props.get(metric_col))
                return {
                    "fillColor": fill,
                    "color": "#94a3b8",
                    "weight": 0.25,
                    "fillOpacity": 0.88,
                }

            # Reduce geometry payload for a more responsive interactive map.
            map_for_display = map_gdf.copy()
            try:
                map_for_display["geometry"] = map_for_display.geometry.simplify(0.0008, preserve_topology=True)
            except Exception:
                pass

            # No third-party basemap tiles: this avoids the API-key-required
            # tile placeholders and keeps the map focused on California tracts.
            folium_map = folium.Map(
                location=[36.7, -119.5],
                zoom_start=5,
                tiles="OpenStreetMap",
                control_scale=False,
                prefer_canvas=True,
                zoom_control=True,
                scrollWheelZoom=True,
                attributionControl=True,
            )
            folium_map.get_root().html.add_child(folium.Element("""
                <style>
                  html, body { background: #0e1117 !important; }
                  .leaflet-container { background: #0e1117 !important; }
                </style>
            """))
            # Fit to the actual California tract geometry rather than a fixed extent.
            minx, miny, maxx, maxy = map_for_display.total_bounds
            folium_map.fit_bounds([[miny, minx], [maxy, maxx]], padding=(32, 32), max_zoom=5)

            folium.GeoJson(
                map_for_display.to_json(),
                name="California census tracts",
                style_function=style_function,
                highlight_function=lambda feature: {
                    "weight": 1.6,
                    "color": "#f8fafc",
                    "fillOpacity": 0.95,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["tract_id", metric_col],
                    aliases=["Census tract:", f"{map_metric_label}:"] ,
                    localize=True,
                    sticky=False,
                ),
                popup=folium.GeoJsonPopup(
                    fields=["tract_id", "housing_cost_2012", "housing_cost_2016", "housing_cost_change", "gentrified"],
                    aliases=["Census tract:", "Housing cost 2012:", "Housing cost 2016:", "Cost change:", "Historical increase label:"],
                    localize=True,
                ),
            ).add_to(folium_map)

            with right_controls:
                
                
                map_result = st_folium(
                    folium_map,
                    height=690,
                    use_container_width=True,
                    returned_objects=["last_clicked"],
                    key="california_tract_map_v2",
                )

            # A map click returns coordinates. Resolve them to the tract polygon and persist selection.
            click = map_result.get("last_clicked") if isinstance(map_result, dict) else None
            if click and click.get("lat") is not None and click.get("lng") is not None:
                try:
                    from shapely.geometry import Point
                    clicked_point = Point(float(click["lng"]), float(click["lat"]))
                    hits = map_gdf[map_gdf.geometry.contains(clicked_point)]
                    if not hits.empty:
                        clicked_id = str(hits.iloc[0]["tract_id"])
                        st.session_state["selected_map_tract"] = clicked_id
                except Exception:
                    pass

            selected_tract = st.session_state.get("selected_map_tract")
            if selected_tract not in tract_ids and tract_ids:
                selected_tract = tract_ids[0]
                st.session_state["selected_map_tract"] = selected_tract

            selected_rows = map_gdf[map_gdf["tract_id"].astype(str) == str(selected_tract)] if selected_tract else map_gdf.iloc[0:0]
            if not selected_rows.empty:
                selected_map_row = selected_rows.iloc[0]
                selected_target = target_df[target_df["tract_id"].astype(str) == str(selected_tract)]
                target_row = selected_target.iloc[0] if not selected_target.empty else None

                st.markdown("---")
                st.markdown("### Selected tract details")
                identity, cost12, cost16, change, label_col = st.columns([1.5, 1, 1, 1, 1.2])
                with identity:
                    st.markdown(f"**Census tract**\n\n`{selected_tract}`")
                    county = selected_map_row.get("COUNTYFP10", selected_map_row.get("COUNTYFP", ""))
                    st.caption(f"California · County code {county}" if pd.notna(county) and str(county) else "California census tract")
                with cost12:
                    st.markdown("**Housing cost 2012**")
                    st.markdown(f"### {money(target_row['housing_cost_2012']) if target_row is not None else 'N/A'}")
                with cost16:
                    st.markdown("**Housing cost 2016**")
                    st.markdown(f"### {money(target_row['housing_cost_2016']) if target_row is not None else 'N/A'}")
                with change:
                    st.markdown("**Change (2012–2016)**")
                    st.markdown(f"### {money(target_row['housing_cost_change']) if target_row is not None else 'N/A'}")
                with label_col:
                    st.markdown("**Historical label**")
                    if target_row is not None and int(target_row["gentrified"]) == 1:
                        st.markdown("<span style='background:#7f1d1d;color:#fecaca;padding:5px 10px;border-radius:8px;font-weight:700;'>Rising costs</span>", unsafe_allow_html=True)
                    elif target_row is not None:
                        st.markdown("<span style='background:#064e3b;color:#a7f3d0;padding:5px 10px;border-radius:8px;font-weight:700;'>No increase</span>", unsafe_allow_html=True)
                    else:
                        st.write("No matched data")

                detail_tabs = st.tabs(["Housing & Demographics", "Model Predictions", "Nearby Tracts", "Feature Snapshot"])
                with detail_tabs[0]:
                    chart_col, demo_col = st.columns([1.15, 1])
                    with chart_col:
                        if target_row is not None:
                            chart_df = pd.DataFrame({"Year": ["2012", "2016"], "Housing cost": [target_row["housing_cost_2012"], target_row["housing_cost_2016"]]})
                            fig = px.bar(chart_df, x="Year", y="Housing cost", text_auto=".0f", title="Housing costs over time", color="Year", color_discrete_sequence=["#3b82f6", "#f97360"])
                            fig.update_layout(**{**PLOT_LAYOUT, "height": 330, "margin": dict(l=20, r=15, t=50, b=30), "showlegend": False})
                            fig.update_yaxes(title="Housing cost (source units)")
                            st.plotly_chart(fig, use_container_width=True)
                    with demo_col:
                        st.markdown("#### Key historical indicators")
                        if acs_df is not None:
                            tract_acs = acs_df[acs_df["tract_id"].astype(str) == str(selected_tract)]
                            if not tract_acs.empty:
                                row = tract_acs.iloc[0]
                                cols = [c for c in tract_acs.columns if c not in {"tract_id", "GEOID"} and any(c.endswith(f"_{year}") for year in range(2010, 2017))]
                                if cols:
                                    latest_cols = [c for c in cols if c.endswith("_2016")]
                                    show_cols = latest_cols[:12] if latest_cols else cols[-12:]
                                    indicator_df = pd.DataFrame({"Indicator": show_cols, "Value": [row[c] for c in show_cols]})
                                    st.dataframe(indicator_df, use_container_width=True, hide_index=True, height=300)
                                else:
                                    st.info("No year-suffixed ACS indicators were found for this tract.")
                            else:
                                st.info("No cleaned ACS feature row was found for this tract.")
                        else:
                            st.info("Cleaned ACS feature data is unavailable.")
                with detail_tabs[1]:
                    st.info("Model probabilities are not calculated by the current map code. This panel intentionally avoids presenting invented predictions; connect the saved model pipeline to generate per-tract probabilities.")
                    model_paths = find_model_files()
                    if model_paths:
                        st.markdown("Saved model artifacts detected:")
                        for model_path in model_paths:
                            st.write(f"- `{model_path.relative_to(ROOT)}`")
                    else:
                        st.caption("No saved model artifacts were detected in models/.")
                with detail_tabs[2]:
                    st.markdown("#### Nearby census tracts")
                    try:
                        selected_centroid = selected_map_row.geometry.representative_point()
                        distances = map_gdf.geometry.representative_point().distance(selected_centroid)
                        nearby = map_gdf.loc[distances.index[distances > 0]].copy()
                        nearby["distance_degrees"] = distances[distances > 0]
                        nearby = nearby.nsmallest(6, "distance_degrees")
                        nearby["Housing cost change"] = nearby["housing_cost_change"].map(money)
                        nearby["Historical label"] = nearby["gentrified"].map({1.0: "Rising costs", 0.0: "No increase"}).fillna("No matched data")
                        st.dataframe(nearby[["tract_id", "Housing cost change", "Historical label"]].rename(columns={"tract_id": "Census tract"}), use_container_width=True, hide_index=True)
                    except Exception:
                        st.info("Nearby tract comparison could not be calculated for this geometry.")
                with detail_tabs[3]:
                    st.markdown("#### Feature snapshot")
                    if acs_df is not None:
                        snapshot = acs_df[acs_df["tract_id"].astype(str) == str(selected_tract)]
                        if not snapshot.empty:
                            st.dataframe(snapshot.drop(columns=["tract_id"], errors="ignore").T.rename(columns={snapshot.index[0]: "Value"}), use_container_width=True, height=350)
                        else:
                            st.info("No ACS feature snapshot is available for the selected tract.")
                    else:
                        st.info("Cleaned ACS feature data is unavailable.")

                st.download_button(
                    "Download selected tract data",
                    data=(selected_target if not selected_target.empty else selected_rows.drop(columns="geometry", errors="ignore")).to_csv(index=False).encode("utf-8"),
                    file_name=f"tract_{selected_tract}_history.csv",
                    mime="text/csv",
                    key="download_map_selected_tract_csv",
                )

            st.caption("Historical analysis: 2010–2016. The map does not establish causation, displacement, or future gentrification.")

        except ImportError:
            st.error("Interactive map dependencies are missing. Install them in the same environment used to run Streamlit.")
            st.code("python -m pip install geopandas folium streamlit-folium")
        except Exception as exc:
            st.error(f"The interactive map could not be rendered: {exc}")
            st.info("Check the shapefile, GEOID join, and installed mapping dependencies.")


# ============================================================
# 10. MODEL RESULTS
# ============================================================

with models_tab:
    st.header("Machine-learning model evaluation")
    st.markdown(
        '<p class="section-subtitle">'
        "Review saved model evaluation outputs and artifacts. "
        "This page does not retrain models."
        "</p>",
        unsafe_allow_html=True,
    )

    result_files = find_result_files()
    model_files = find_model_files()

    if result_files:
        st.subheader("Saved evaluation CSV files")

        result_options = {
            str(path.relative_to(ROOT)): path
            for path in result_files
        }

        chosen_result_name = st.selectbox(
            "Choose a results file",
            list(result_options.keys()),
        )

        chosen_result_path = result_options[chosen_result_name]

        try:
            results_df = pd.read_csv(chosen_result_path)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download this results CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name=chosen_result_path.name,
                mime="text/csv",
                key="download_model_results_csv",
            )

            # Show a chart if the file contains familiar metric columns.
            numeric_cols = results_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_cols and len(results_df) > 1:
                st.subheader("Numeric result comparison")

                selected_numeric = st.multiselect(
                    "Numeric columns to plot",
                    numeric_cols,
                    default=numeric_cols[: min(4, len(numeric_cols))],
                )

                if selected_numeric:
                    chart_df = results_df[selected_numeric].copy()
                    chart_df.index = chart_df.index.astype(str)

                    st.bar_chart(chart_df)

        except Exception as exc:
            st.error(f"Could not read the selected results CSV: {exc}")

    else:
        st.info(
            "No evaluation CSV files were found in results/. "
            "Save model metrics to CSV to display them here."
        )

        st.code(
            'import pandas as pd\n\n'
            'pd.DataFrame([{\n'
            '    "model": "Random Forest",\n'
            '    "accuracy": 0.666,\n'
            '    "precision": 0.666,\n'
            '    "recall": 0.704,\n'
            '}]).to_csv("results/model_comparison.csv", index=False)'
        )

    st.divider()
    st.subheader("Saved model artifacts")

    if model_files:
        for path in model_files:
            st.write(f"- `{path.relative_to(ROOT)}`")
    else:
        st.info(
            "No model artifacts (.pkl, .joblib, .sav, .keras, or .h5) "
            "were found in models/. Your training scripts may not "
            "currently save trained models."
        )

    st.markdown(
        """
        **Important:** A saved model file is not enough to report
        predictive performance. The results page should display metrics
        measured on a held-out test set, along with the target definition,
        split strategy, and feature preparation used during evaluation.
        """
    )


# ============================================================
# 11. METHODOLOGY
# ============================================================

with methodology_tab:
    st.header("Methodology and limitations")

    st.subheader("Research objective")

    st.write(
        "Explore whether historical census-tract socioeconomic and "
        "housing indicators can help identify California neighborhoods "
        "that experienced rising housing costs between 2012 and 2016."
    )

    st.subheader("Historical target definition")

    st.latex(
        r"\Delta H = H_{2016} - H_{2012}"
    )

    st.latex(
        r"y = \begin{cases}"
        r"1, & \Delta H > 0 \\"
        r"0, & \Delta H \leq 0"
        r"\end{cases}"
    )

    st.write(
        "Here, H represents the selected housing-cost measure in the "
        "target CSV. The label identifies an increase in that measure; "
        "it does not independently establish that a neighborhood "
        "experienced gentrification."
    )

    st.subheader("Data sources used by this dashboard")

    st.markdown(
        """
        - `data/raw/housing_cost_target.csv`: historical target labels.
        - `data/processed/california_acs_2010_2016_clean.csv`:
          cleaned historical ACS indicators.
        - `data/raw/tract10/`: census-tract geographic boundaries.
        - `results/`: saved model evaluation CSVs, if available.
        - `models/`: saved model artifacts, if available.
        """
    )

    st.subheader("Modeling approach")

    st.markdown(
        """
        The research workflow has included:

        1. Logistic Regression.
        2. Random Forest.
        3. Random Forest with spatial features.
        4. Artificial Neural Network.
        5. Ensemble modeling.

        These methods should be compared using the same target
        definition and a consistent held-out test set. Spatial features
        and census geography should be handled carefully to avoid
        information leakage.
        """
    )

    st.subheader("Interpretation and limitations")

    st.markdown(
        """
        - Rising housing costs alone are not a complete definition
          of gentrification.
        - The historical target covers 2012–2016, while the features
          may span 2010–2016.
        - If features include information from 2016, they may leak
          information from the same period used to construct the target.
        - Census estimates may have sampling uncertainty.
        - Tract boundaries and identifiers must match across datasets.
        - A classification result is not proof of displacement,
          causation, or an individual household's experience.
        - A genuine early-warning evaluation should use features
          available before the prediction period.
        """
    )

    st.subheader("Current dataset summary")

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Valid matched tracts",
                "Median housing cost in 2012",
                "Median housing cost in 2016",
                "Median absolute change",
                "Tracts with an increase",
                "Tracts without an increase",
            ],
            "Value": [
                number(valid_tract_count),
                money(state_median_2012),
                money(state_median_2016),
                money(median_change),
                number(rising_count),
                number(non_rising_count),
            ],
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )
