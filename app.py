from pathlib import Path
import html
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

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
COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
COUNTY_CV_FILE = RESULTS_DIR / "county_cv.csv"
PREDICTIONS_FILE = RESULTS_DIR / "tract_predictions.csv"

st.set_page_config(
    page_title="California Gentrification Early Warning",
    page_icon=":material/holiday_village:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. DESIGN TOKENS
# ============================================================

PAPER = "#F4F5F7"
CARD = "#FFFFFF"
INK = "#0F1B2A"
MUTED = "#566375"
FAINT = "#8B95A3"
RULE = "#E1E5EA"
GRID = "#EDF0F3"
NAVY = "#12304F"
DEEP = "#0A1C30"

# Data colours (validated for colour-vision deficiency on white).
COOL = "#1F72A3"   # no increase / 2012
WARM = "#B8452C"   # increase / 2016
MID = "#ECEAE4"    # diverging midpoint
NO_DATA = "#CBD0CD"

FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
DISPLAY = "Manrope, Inter, system-ui, sans-serif"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {{
        --paper: {PAPER};
        --card: {CARD};
        --ink: {INK};
        --muted: {MUTED};
        --faint: {FAINT};
        --rule: {RULE};
        --navy: {NAVY};
        --deep: {DEEP};
        --cool: {COOL};
        --warm: {WARM};
        --radius: 16px;
        --shadow: 0 1px 2px rgba(15, 27, 42, .04), 0 10px 28px -14px rgba(15, 27, 42, .16);
    }}

    html, body, .stApp, .stApp p, .stApp li, .stApp label,
    .stApp input, .stApp button, .stApp textarea, .stMarkdown {{
        font-family: {FONT};
    }}

    .stApp {{
        color: var(--ink);
        background:
            radial-gradient(1200px 520px at 85% -120px, rgba(31, 114, 163, .10), transparent 60%),
            var(--paper);
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}

    .block-container {{
        max-width: 1320px;
        padding: 1.6rem 2rem 4rem 2rem;
    }}

    /* ---------- hero ---------- */
    .hero {{
        position: relative; overflow: hidden;
        display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2.5rem;
        align-items: center;
        padding: 2.1rem 2.4rem 2.2rem 2.4rem;
        border-radius: 22px;
        color: #fff;
        background:
            radial-gradient(520px 300px at 92% 110%, rgba(184, 69, 44, .38), transparent 70%),
            radial-gradient(600px 340px at 0% -20%, rgba(31, 114, 163, .45), transparent 70%),
            linear-gradient(135deg, var(--deep) 0%, var(--navy) 100%);
        box-shadow: 0 24px 48px -28px rgba(10, 28, 48, .7);
        margin-bottom: 1.6rem;
    }}
    .hero::before {{
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background-image: radial-gradient(rgba(255, 255, 255, .09) 1px, transparent 1px);
        background-size: 18px 18px;
        mask-image: linear-gradient(90deg, transparent 0%, #000 55%);
        -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 55%);
    }}
    .hero > * {{ position: relative; }}
    .wordmark {{
        display: flex; align-items: center; gap: .6rem;
        font-family: {DISPLAY}; font-weight: 700; font-size: .92rem;
        color: rgba(255, 255, 255, .92); letter-spacing: -.005em;
    }}
    .wordmark .logo {{
        display: grid; place-items: center; width: 32px; height: 32px;
        border-radius: 9px; background: rgba(255, 255, 255, .1);
        border: 1px solid rgba(255, 255, 255, .14);
    }}
    .eyebrow {{
        display: inline-flex; align-items: center; gap: .45rem;
        margin-left: .4rem; padding: .22rem .65rem; border-radius: 999px;
        font-family: {FONT}; font-size: .74rem; font-weight: 600;
        color: rgba(255, 255, 255, .78);
        background: rgba(255, 255, 255, .08); border: 1px solid rgba(255, 255, 255, .12);
    }}
    .eyebrow .dot {{ width: 6px; height: 6px; border-radius: 50%; background: #F08C6E; }}
    .stApp .finding {{
        font-family: {DISPLAY};
        font-size: clamp(1.75rem, 3.2vw, 2.7rem);
        line-height: 1.1; font-weight: 800; letter-spacing: -.03em;
        color: #fff; max-width: 24ch;
        margin: 1.25rem 0 .8rem 0;
        font-variant-numeric: tabular-nums;
    }}
    .stApp .finding em {{
        font-style: normal;
        background: linear-gradient(90deg, #FFB49C, #F08C6E);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }}
    .stApp .dek {{
        color: rgba(225, 233, 242, .82); font-size: 1rem; line-height: 1.6;
        max-width: 64ch; margin: 0;
    }}
    .ring {{ text-align: center; padding-right: .4rem; }}
    .ring svg {{ display: block; margin: 0 auto; }}
    .ring .big {{
        font-family: {DISPLAY}; font-weight: 800; font-size: 2rem;
        fill: #fff; letter-spacing: -.03em;
    }}
    .ring .small {{ font-size: .72rem; fill: rgba(225, 233, 242, .75); font-weight: 600; }}
    .ring-caption {{ margin-top: .55rem; font-size: .8rem; color: rgba(225, 233, 242, .75); }}
    @media (max-width: 860px) {{
        .hero {{ grid-template-columns: 1fr; padding: 1.6rem 1.4rem; gap: 1.2rem; }}
        .ring {{ display: none; }}
    }}

    /* ---------- tabs: segmented control ---------- */
    [data-testid="stTabs"] [role="tablist"] {{
        gap: .25rem; padding: .3rem; width: fit-content; max-width: 100%;
        background: var(--card); border: 1px solid var(--rule);
        border-radius: 14px; box-shadow: var(--shadow);
        overflow-x: auto;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        height: auto; padding: .5rem 1rem; margin: 0; border-radius: 10px;
        background: transparent !important; border: none;
        transition: background .15s ease, color .15s ease;
    }}
    [data-testid="stTabs"] [role="tab"]:hover {{ background: #EEF1F5 !important; }}
    [data-testid="stTabs"] [role="tab"] p {{
        font-weight: 600; font-size: .9rem; color: var(--muted); white-space: nowrap;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ background: var(--navy) !important; }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: #fff; }}
    [data-testid="stTabs"] .react-aria-SelectionIndicator,
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none !important; }}
    [data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 1.4rem; }}

    /* nested tabs (map detail) stay light */
    [class*="st-key-panel"] [data-testid="stTabs"] [role="tablist"] {{
        box-shadow: none; background: #F1F3F6; border-color: transparent;
    }}
    [class*="st-key-panel"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: var(--card) !important; box-shadow: 0 1px 3px rgba(15, 27, 42, .12);
    }}
    [class*="st-key-panel"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: var(--ink); }}
    [class*="st-key-panel"] [data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 1rem; }}

    /* ---------- KPI cards ---------- */
    .strip {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: .9rem; margin: 0 0 1.2rem 0;
    }}
    .strip .cell {{
        position: relative; overflow: hidden;
        background: var(--card); border: 1px solid var(--rule);
        border-radius: var(--radius); box-shadow: var(--shadow);
        padding: 1.05rem 1.2rem 1.1rem 1.2rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .strip .cell:hover {{ transform: translateY(-2px); }}
    .strip .cell::after {{
        content: ""; position: absolute; left: 1.2rem; right: 1.2rem; top: 0; height: 3px;
        border-radius: 0 0 3px 3px; background: var(--accent, var(--navy)); opacity: .9;
    }}
    .strip .k {{
        color: var(--muted); font-size: .76rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: .06em;
    }}
    .strip .v {{
        font-family: {DISPLAY};
        color: var(--ink); font-size: 1.85rem; font-weight: 800;
        letter-spacing: -.03em; margin-top: .4rem; line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }}
    .strip .n {{ color: var(--faint); font-size: .8rem; margin-top: .3rem; }}

    /* ---------- panels and section titles ---------- */
    [class*="st-key-panel"] {{
        background: var(--card);
        border: 1px solid var(--rule) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow);
        padding: 1.1rem 1.25rem 1.2rem 1.25rem;
    }}
    .st-key-panel_mapcanvas {{ padding: .5rem; }}
    .stApp .sec-title {{
        font-family: {DISPLAY};
        font-size: 1.08rem; font-weight: 700; color: var(--ink);
        letter-spacing: -.015em; margin: 0;
        display: flex; align-items: center; gap: .55rem; flex-wrap: wrap;
    }}
    .stApp .sec-note {{
        color: var(--muted); font-size: .88rem; line-height: 1.5;
        margin: .3rem 0 .5rem 0; max-width: 72ch;
    }}

    /* ---------- widgets ---------- */
    [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {{
        border-radius: 10px !important; border-color: var(--rule) !important;
        background: #FAFBFC !important;
    }}
    [data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within {{
        border-color: var(--navy) !important;
        box-shadow: 0 0 0 3px rgba(18, 48, 79, .12);
    }}
    [data-testid="stWidgetLabel"] p {{ font-weight: 600; font-size: .84rem; color: var(--ink); }}
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--rule); border-radius: 12px; overflow: hidden;
    }}
    [data-testid="stMetric"] {{
        background: #F7F9FB; border: 1px solid var(--rule);
        border-radius: 12px; padding: .8rem 1rem;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: .76rem; font-weight: 600; color: var(--muted);
        text-transform: uppercase; letter-spacing: .06em;
    }}
    [data-testid="stMetricValue"] {{
        font-family: {DISPLAY}; font-weight: 800; letter-spacing: -.02em;
    }}
    [data-testid="stExpander"] details {{
        background: var(--card); border: 1px solid var(--rule) !important;
        border-radius: var(--radius); box-shadow: var(--shadow);
    }}
    [data-testid="stExpander"] summary p {{ font-weight: 600; }}
    [data-testid="stCode"], [data-testid="stCodeBlock"] {{ border-radius: 12px; }}

    [data-testid="stDownloadButton"] button {{
        border-radius: 999px; border: 1px solid var(--navy);
        background: var(--navy); color: #fff; font-weight: 600;
        padding: .45rem 1.1rem;
        box-shadow: 0 6px 16px -8px rgba(18, 48, 79, .6);
        transition: transform .15s ease, background .15s ease;
    }}
    [data-testid="stDownloadButton"] button p {{ color: #fff; }}
    [data-testid="stDownloadButton"] button:hover {{
        background: var(--deep); border-color: var(--deep); color: #fff; transform: translateY(-1px);
    }}

    /* ---------- tags, legends, prose ---------- */
    .tag {{
        display: inline-flex; align-items: center; gap: .35rem;
        padding: .22rem .65rem; border-radius: 999px;
        font-family: {FONT}; font-size: .78rem; font-weight: 600; letter-spacing: 0;
    }}
    .tag::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
    .tag.up {{ background: #F9E6E0; color: #8E2F1B; }}
    .tag.flat {{ background: #DEEDF6; color: #134D72; }}
    .tag.none {{ background: #ECEEF1; color: var(--muted); }}

    .legend-row {{ display: flex; align-items: center; gap: .55rem; margin: .45rem 0; font-size: .86rem; color: var(--ink); }}
    .swatch {{ width: 12px; height: 12px; border-radius: 4px; flex: none; }}
    .ramp {{ height: 10px; border-radius: 999px; margin: .55rem 0 .3rem 0; }}
    .ramp-labels {{ display: flex; justify-content: space-between; font-size: .76rem; color: var(--muted); font-variant-numeric: tabular-nums; }}

    .prose-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 1rem;
    }}
    .prose {{
        background: var(--card); border: 1px solid var(--rule);
        border-radius: var(--radius); box-shadow: var(--shadow);
        padding: 1.3rem 1.5rem 1.1rem 1.5rem;
        font-size: .95rem; line-height: 1.65; color: var(--ink);
    }}
    .prose .step {{
        display: inline-grid; place-items: center; width: 26px; height: 26px;
        border-radius: 8px; background: #EAF0F6; color: var(--navy);
        font-family: {DISPLAY}; font-weight: 800; font-size: .8rem;
    }}
    .stApp .prose h4 {{
        font-family: {DISPLAY}; font-size: 1.05rem; font-weight: 700;
        letter-spacing: -.01em; margin: .7rem 0 .4rem 0; padding: 0;
    }}
    .stApp .prose p, .stApp .prose li {{ color: #2C3844; }}
    .prose ul {{ padding-left: 1.1rem; margin: 0; }}
    .prose li {{ margin: .25rem 0; }}
    .prose.wide {{ grid-column: 1 / -1; }}

    .stApp .footnote {{
        display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
        color: var(--faint); font-size: .8rem;
        margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid var(--rule);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

pio.templates["atlas"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[COOL, WARM],
        title=dict(font=dict(family=DISPLAY, size=14, color=INK), x=0, xanchor="left"),
        xaxis=dict(
            gridcolor=GRID, linecolor=RULE, zeroline=False, ticks="",
            title=dict(font=dict(size=12, color=MUTED)),
            tickfont=dict(color=MUTED),
        ),
        yaxis=dict(
            gridcolor=GRID, linecolor=RULE, zeroline=False, ticks="",
            title=dict(font=dict(size=12, color=MUTED)),
            tickfont=dict(color=MUTED),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(color=MUTED, size=12), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=CARD, bordercolor=RULE,
            font=dict(family=FONT, color=INK, size=12),
        ),
        margin=dict(l=8, r=8, t=36, b=8),
    )
)
pio.templates.default = "atlas"

CHART_CONFIG = {"displayModeBar": False}


def chart(fig):
    """Render a Plotly figure. Explicit layout values survive Streamlit's own chart theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK),
        hoverlabel=dict(bgcolor=CARD, bordercolor=RULE, font=dict(family=FONT, color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=RULE, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=RULE, zeroline=False)
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ============================================================
# 3. HELPERS
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
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def signed_money(value):
    if pd.isna(value):
        return "N/A"
    return ("+" if value > 0 else "") + money(value)


def number(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def stat_strip(items):
    """A row of KPI cards: [(label, value, note), ...] or (label, value, note, accent)."""
    cells = ""
    for item in items:
        k, v, n = item[:3]
        accent = item[3] if len(item) > 3 else NAVY
        cells += (
            f'<div class="cell" style="--accent:{accent}"><div class="k">{html.escape(k)}</div>'
            f'<div class="v">{html.escape(v)}</div>'
            f'<div class="n">{html.escape(n)}</div></div>'
        )
    st.markdown(f'<div class="strip">{cells}</div>', unsafe_allow_html=True)


def section(title, note=""):
    note_html = f'<p class="sec-note">{note}</p>' if note else ""
    st.markdown(
        f'<p class="sec-title">{html.escape(title)}</p>{note_html}',
        unsafe_allow_html=True,
    )


def panel(key):
    return st.container(border=True, key=f"panel_{key}")


def label_tag(value):
    if value is None or pd.isna(value):
        return '<span class="tag none">No matched data</span>'
    if int(value) == 1:
        return '<span class="tag up">Costs rose</span>'
    return '<span class="tag flat">No increase</span>'


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def mix(a, b, t):
    rgb = hex_to_rgb(a) + (hex_to_rgb(b) - hex_to_rgb(a)) * float(t)
    return "#{:02x}{:02x}{:02x}".format(*rgb.round().astype(int))


SEQ_COST = ("#EAF1F5", "#123F5E")
SEQ_PROB = ("#F7EBE6", "#7E2814")


# ============================================================
# 4. DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_target():
    """Load and validate the historical target dataset."""
    if not TARGET_FILE.exists():
        return None, f"Target dataset not found: {TARGET_FILE.relative_to(ROOT)}"

    df = pd.read_csv(TARGET_FILE, dtype={"tract_id": "string"}, low_memory=False)

    required = {
        "tract_id",
        "housing_cost_2012",
        "housing_cost_2016",
        "housing_cost_change",
        "gentrified",
    }
    missing = required.difference(df.columns)
    if missing:
        return None, "The target CSV is missing these columns: " + ", ".join(sorted(missing))

    df["tract_id"] = normalize_id(df["tract_id"])
    for col in required - {"tract_id"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required)).copy()
    df = df[
        df["housing_cost_2012"].between(1, 100000)
        & df["housing_cost_2016"].between(1, 100000)
    ].copy()

    # Recompute both derived columns so they are always consistent.
    df["housing_cost_change"] = df["housing_cost_2016"] - df["housing_cost_2012"]
    df["gentrified"] = (df["housing_cost_change"] > 0).astype(int)

    return df.drop_duplicates("tract_id", keep="first"), None


@st.cache_data(show_spinner=False)
def load_acs():
    """Load cleaned historical ACS features if available."""
    if not CLEAN_FILE.exists():
        return None
    df = pd.read_csv(CLEAN_FILE, dtype={"GEOID": "string"}, low_memory=False)
    if "GEOID" in df.columns:
        df["tract_id"] = normalize_id(df["GEOID"])
    elif "tract_id" in df.columns:
        df["tract_id"] = normalize_id(df["tract_id"])
    else:
        return None
    return df.drop_duplicates("tract_id", keep="first")


@st.cache_data(show_spinner=False)
def load_predictions():
    """Out-of-fold model probabilities written by src/train_ensemble.py."""
    if not PREDICTIONS_FILE.exists():
        return None
    df = pd.read_csv(PREDICTIONS_FILE, dtype={"tract_id": "string"})
    df["tract_id"] = normalize_id(df["tract_id"])
    return df.drop_duplicates("tract_id")


@st.cache_data(show_spinner=False)
def load_csv(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_map_data(target_signature):
    """Load tract boundaries and join historical targets (and predictions)."""
    try:
        import geopandas as gpd
    except ImportError:
        return None, "GeoPandas is not installed."

    if not SHAPEFILE.exists():
        return None, f"The census-tract shapefile was not found at {SHAPEFILE.relative_to(ROOT)}."

    target, error = load_target()
    if error or target is None:
        return None, error or "Could not load the target dataset."

    try:
        tracts = gpd.read_file(SHAPEFILE)
    except Exception as exc:
        return None, f"Could not read the shapefile: {exc}"

    geo_col = next(
        (c for c in ["GEOID10", "GEOID", "GEOID_10", "GEOID20"] if c in tracts.columns),
        None,
    )
    if geo_col is not None:
        tracts["tract_id"] = normalize_id(tracts[geo_col])
    elif all(c in tracts.columns for c in ["STATEFP10", "COUNTYFP10", "TRACTCE10"]):
        tracts["tract_id"] = (
            tracts["STATEFP10"].astype(str).str.zfill(2)
            + tracts["COUNTYFP10"].astype(str).str.zfill(3)
            + tracts["TRACTCE10"].astype(str).str.zfill(6)
        )
    elif all(c in tracts.columns for c in ["STATEFP", "COUNTYFP", "TRACTCE"]):
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

    tracts["county_fips"] = tracts["tract_id"].str[2:5]
    tracts = tracts[["tract_id", "county_fips", "geometry"]]

    map_gdf = tracts.merge(target, on="tract_id", how="left", validate="one_to_one")

    predictions = load_predictions()
    if predictions is not None:
        map_gdf = map_gdf.merge(
            predictions[["tract_id", "rf_oof_probability"]],
            on="tract_id",
            how="left",
        )

    if map_gdf.crs is not None:
        map_gdf = map_gdf.to_crs(epsg=4326)

    if map_gdf["gentrified"].notna().sum() == 0:
        return None, (
            "The map boundaries loaded, but no target rows matched. "
            "Check the GEOID format and geographic vintage."
        )

    return map_gdf, None


def find_model_files():
    if not MODELS_DIR.exists():
        return []
    extensions = {".pkl", ".joblib", ".sav", ".keras", ".h5"}
    return sorted(p for p in MODELS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


# ============================================================
# 5. LOAD DATA
# ============================================================

target_df, target_error = load_target()

if target_error or target_df is None:
    st.error(target_error or "Unable to load historical data.")
    st.markdown(
        "Run the data pipeline first so that `data/raw/housing_cost_target.csv` exists:\n\n"
        "```bash\nexport CENSUS_API_KEY=your_key\npython src/download_census.py\npython src/build_target.py\n```"
    )
    st.stop()

acs_df = load_acs()
predictions_df = load_predictions()

valid_tract_count = len(target_df)
rising_count = int(target_df["gentrified"].sum())
non_rising_count = valid_tract_count - rising_count
rising_pct = 100 * rising_count / valid_tract_count if valid_tract_count else 0

state_median_2012 = target_df["housing_cost_2012"].median()
state_median_2016 = target_df["housing_cost_2016"].median()
median_change = target_df["housing_cost_change"].median()

# ============================================================
# 6. MASTHEAD
# ============================================================

RING_R = 54
RING_C = 2 * np.pi * RING_R

st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="wordmark">
          <span class="logo">
            <svg width="18" height="18" viewBox="0 0 22 22" aria-hidden="true">
              <rect x="1" y="11" width="5" height="10" rx="1.2" fill="#7FB8DA"/>
              <rect x="8.5" y="6" width="5" height="15" rx="1.2" fill="#FFFFFF"/>
              <rect x="16" y="1" width="5" height="20" rx="1.2" fill="#F08C6E"/>
            </svg>
          </span>
          California Gentrification Early Warning
          <span class="eyebrow"><span class="dot"></span>ACS 2012 → 2016</span>
        </div>
        <p class="finding"><em>{rising_count:,}</em> of {valid_tract_count:,} California census tracts
        saw median housing costs rise between 2012 and 2016.</p>
        <p class="dek">Tract-level ACS 5-year estimates, compared with models that try to anticipate
        which tracts would rise using only data from 2010 and 2011. Dollar amounts are nominal
        (not adjusted for inflation).</p>
      </div>
      <div class="ring" aria-label="{rising_pct:.0f}% of tracts saw costs rise">
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle cx="75" cy="75" r="{RING_R}" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="12"/>
          <circle cx="75" cy="75" r="{RING_R}" fill="none" stroke="#F08C6E" stroke-width="12"
                  stroke-linecap="round" transform="rotate(-90 75 75)"
                  stroke-dasharray="{RING_C * rising_pct / 100:.1f} {RING_C:.1f}"/>
          <text x="75" y="80" text-anchor="middle" class="big">{rising_pct:.0f}%</text>
          <text x="75" y="100" text-anchor="middle" class="small">ROSE</text>
        </svg>
        <div class="ring-caption">Median change {html.escape(signed_money(median_change))}/mo</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, explorer_tab, map_tab, models_tab, methodology_tab = st.tabs(
    ["Overview", "Tract explorer", "Map", "Model results", "Methodology"]
)

# ============================================================
# 7. OVERVIEW
# ============================================================

with overview_tab:
    stat_strip(
        [
            ("Census tracts analysed", number(valid_tract_count), "Valid cost in both years"),
            ("Median monthly cost, 2012", money(state_median_2012), "ACS 2008–2012", COOL),
            ("Median monthly cost, 2016", money(state_median_2016), "ACS 2012–2016", WARM),
            ("Tracts where costs rose", pct(rising_pct), f"{number(rising_count)} tracts", WARM),
        ]
    )

    left, right = st.columns([1.15, 1], gap="medium")

    with left:
        with panel("scatter"):
            section(
                "Every tract, 2012 against 2016",
                "Points above the diagonal had higher monthly housing costs in 2016 than in 2012.",
            )
            hi = float(np.nanpercentile(target_df[["housing_cost_2012", "housing_cost_2016"]].to_numpy(), 99.5))
            fig = go.Figure()
            for label, name, color in [(0, "No increase", COOL), (1, "Costs rose", WARM)]:
                sub = target_df[target_df["gentrified"] == label]
                fig.add_trace(
                    go.Scattergl(
                        x=sub["housing_cost_2012"],
                        y=sub["housing_cost_2016"],
                        mode="markers",
                        name=f"{name} ({len(sub):,})",
                        marker=dict(size=4, color=color, opacity=0.45, line=dict(width=0)),
                        customdata=sub[["tract_id", "housing_cost_change"]],
                        hovertemplate=(
                            "Tract %{customdata[0]}<br>2012: $%{x:,.0f}<br>2016: $%{y:,.0f}"
                            "<br>Change: $%{customdata[1]:,.0f}<extra></extra>"
                        ),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[0, hi], y=[0, hi], mode="lines", hoverinfo="skip", showlegend=False,
                    line=dict(color=FAINT, width=1, dash="dot"),
                )
            )
            fig.update_layout(height=430)
            fig.update_xaxes(title="Monthly housing cost, 2012", tickprefix="$", range=[0, hi])
            fig.update_yaxes(title="Monthly housing cost, 2016", tickprefix="$", range=[0, hi])
            chart(fig)

    with right:
        with panel("hist"):
            section(
                "How much costs changed",
                f"Median change across tracts: {signed_money(median_change)} per month.",
            )
            change = target_df["housing_cost_change"].dropna()
            # Drop (not clip) the extreme 1% on each side; clipping would pile
            # them into the end bins and create false spikes.
            lo_c, hi_c = np.percentile(change, [1, 99])
            counts, edges = np.histogram(change[change.between(lo_c, hi_c)], bins=48)
            centers = (edges[:-1] + edges[1:]) / 2
            fig = go.Figure(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=np.diff(edges) * 0.9,
                    marker=dict(color=[WARM if c > 0 else COOL for c in centers], cornerradius=2),
                    customdata=np.column_stack([edges[:-1], edges[1:]]),
                    hovertemplate="$%{customdata[0]:,.0f} to $%{customdata[1]:,.0f}<br>%{y:,} tracts<extra></extra>",
                    showlegend=False,
                )
            )
            fig.add_vline(x=0, line=dict(color=INK, width=1))
            fig.add_annotation(
                x=0, y=1, yref="paper", text="No change", showarrow=False,
                xanchor="right", xshift=-6, font=dict(size=11, color=MUTED),
            )
            fig.update_layout(height=430, bargap=0)
            fig.update_xaxes(title="Change in monthly cost, 2012 to 2016 (1st–99th percentile)", tickprefix="$")
            fig.update_yaxes(title="Tracts")
            chart(fig)

    with panel("top"):
        section("Largest increases", "The ten tracts with the biggest rise in median monthly housing cost.")
        top = (
            target_df.sort_values("housing_cost_change", ascending=False)
            .head(10)[["tract_id", "housing_cost_2012", "housing_cost_2016", "housing_cost_change"]]
            .assign(pct_change=lambda d: 100 * d["housing_cost_change"] / d["housing_cost_2012"])
        )
        st.dataframe(
            top,
            hide_index=True,
            width="stretch",
            column_config={
                "tract_id": st.column_config.TextColumn("Census tract"),
                "housing_cost_2012": st.column_config.NumberColumn("2012", format="$%d"),
                "housing_cost_2016": st.column_config.NumberColumn("2016", format="$%d"),
                "housing_cost_change": st.column_config.NumberColumn("Change", format="$%d"),
                "pct_change": st.column_config.ProgressColumn(
                    "Change (%)", format="%.0f%%", min_value=0,
                    max_value=float(max(top["pct_change"].max(), 1)),
                ),
            },
        )

    st.download_button(
        "Download tract data (CSV)",
        data=target_df.to_csv(index=False).encode("utf-8"),
        file_name="california_historical_target.csv",
        mime="text/csv",
        key="download_historical_target_csv",
        icon=":material/download:",
    )


# ============================================================
# 8. TRACT EXPLORER
# ============================================================

with explorer_tab:
    tract_ids = sorted(target_df["tract_id"].dropna().unique().tolist())

    if not tract_ids:
        st.warning("No valid census tract IDs are available.")
    else:
        with panel("picker"):
            pick_col, info_col = st.columns([1, 2], vertical_alignment="bottom")
            with pick_col:
                selected_tract = st.selectbox(
                    "Census tract (11-digit GEOID)",
                    tract_ids,
                    key="explorer_tract",
                    help="Type to search. The first five digits are the state and county code.",
                )

            row = target_df[target_df["tract_id"] == selected_tract].iloc[0]

            with info_col:
                st.markdown(
                    f'<p class="sec-title">Tract {html.escape(selected_tract)} {label_tag(row["gentrified"])}</p>'
                    f'<p class="sec-note">County FIPS {html.escape(selected_tract[2:5])}, California</p>',
                    unsafe_allow_html=True,
                )

        pred_note = "Model output not available"
        pred_value = "N/A"
        if predictions_df is not None:
            match = predictions_df[predictions_df["tract_id"] == selected_tract]
            if not match.empty:
                pred_value = pct(100 * match.iloc[0]["rf_oof_probability"])
                pred_note = "Random-forest probability of a rise"

        stat_strip(
            [
                ("Monthly cost, 2012", money(row["housing_cost_2012"]), f"California median {money(state_median_2012)}", COOL),
                ("Monthly cost, 2016", money(row["housing_cost_2016"]), f"California median {money(state_median_2016)}", WARM),
                ("Change", signed_money(row["housing_cost_change"]),
                 pct(100 * row["housing_cost_change"] / row["housing_cost_2012"])),
                ("Model estimate", pred_value, pred_note),
            ]
        )

        with panel("dumbbell"):
            section("This tract against the state", "Each line runs from the 2012 value to the 2016 value.")
            rows_ = [
                ("California median", state_median_2012, state_median_2016),
                (f"Tract {selected_tract}", row["housing_cost_2012"], row["housing_cost_2016"]),
            ]
            fig = go.Figure()
            for name, a, b in rows_:
                fig.add_trace(
                    go.Scatter(
                        x=[a, b], y=[name, name], mode="lines",
                        line=dict(color=RULE, width=6), hoverinfo="skip", showlegend=False,
                    )
                )
            for year, idx, color in [("2012", 1, COOL), ("2016", 2, WARM)]:
                fig.add_trace(
                    go.Scatter(
                        x=[r[idx] for r in rows_],
                        y=[r[0] for r in rows_],
                        mode="markers+text",
                        name=year,
                        marker=dict(size=14, color=color, line=dict(color=CARD, width=2)),
                        text=[money(r[idx]) for r in rows_],
                        textposition="top center",
                        textfont=dict(size=12, color=INK),
                        hovertemplate=f"{year}: $%{{x:,.0f}}<extra></extra>",
                    )
                )
            fig.update_layout(height=240, margin=dict(l=8, r=24, t=40, b=8))
            fig.update_xaxes(tickprefix="$", title="Median monthly housing cost")
            fig.update_yaxes(showgrid=False, range=[-0.6, 1.7])
            chart(fig)

        with panel("indicators"):
            section("Census indicators for this tract", "All cleaned ACS variables available for the tract.")
            if acs_df is None:
                st.caption(
                    "The cleaned ACS feature file (data/processed/california_acs_2010_2016_clean.csv) "
                    "was not found, so only housing costs are shown."
                )
            else:
                tract_acs = acs_df[acs_df["tract_id"] == selected_tract]
                if tract_acs.empty:
                    st.caption("No cleaned ACS row was found for this tract.")
                else:
                    r = tract_acs.iloc[0]
                    cols = [
                        c for c in tract_acs.columns
                        if c not in {"tract_id", "GEOID"}
                        and any(c.endswith(f"_{y}") for y in range(2010, 2017))
                    ]
                    indicator_df = pd.DataFrame(
                        {
                            "Variable": [c.rsplit("_", 1)[0] for c in cols],
                            "Year": [c.rsplit("_", 1)[1] for c in cols],
                            "Value": [r[c] for c in cols],
                        }
                    )
                    query = st.text_input("Filter variables", placeholder="e.g. DP03_0062 or B25105")
                    if query:
                        indicator_df = indicator_df[
                            indicator_df["Variable"].str.contains(query, case=False, regex=False)
                        ]
                    st.dataframe(
                        indicator_df, hide_index=True, width="stretch", height=320,
                        column_config={"Value": st.column_config.NumberColumn(format="%.2f")},
                    )

        st.download_button(
            "Download this tract (CSV)",
            data=pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
            file_name=f"tract_{selected_tract}_history.csv",
            mime="text/csv",
            key="download_explorer_tract_csv",
        icon=":material/download:",
        )


# ============================================================
# 9. MAP
# ============================================================

with map_tab:
    map_gdf, map_error = load_map_data(
        f"{TARGET_FILE.stat().st_mtime if TARGET_FILE.exists() else 0}"
        f"-{PREDICTIONS_FILE.stat().st_mtime if PREDICTIONS_FILE.exists() else 0}"
    )

    if map_error or map_gdf is None:
        st.warning(map_error or "The map could not be loaded.")
        st.markdown(
            "Install the mapping packages and place the 2010 TIGER/Line tract shapefile here:\n\n"
            "```bash\npip install geopandas folium streamlit-folium\n"
            "data/raw/tract10/tl_2010_06_tract10.shp\n```"
        )
    else:
        try:
            import folium
            from streamlit_folium import st_folium

            map_gdf = map_gdf[map_gdf.geometry.notna() & ~map_gdf.geometry.is_empty].copy()
            map_gdf = map_gdf.reset_index(drop=True)

            matched = int(map_gdf["gentrified"].notna().sum())
            rising_map = int((map_gdf["gentrified"] == 1).sum())
            avg_pct = (
                100 * (map_gdf["housing_cost_2016"] / map_gdf["housing_cost_2012"] - 1)
            ).replace([np.inf, -np.inf], np.nan).dropna().mean()

            stat_strip(
                [
                    ("Tract polygons", number(len(map_gdf)), "2010 Census geography"),
                    ("Matched to cost data", number(matched), f"{number(len(map_gdf) - matched)} without data"),
                    ("Tracts where costs rose", number(rising_map), pct(100 * rising_map / matched if matched else 0), WARM),
                    ("Average change per tract", pct(avg_pct), "2012 to 2016, nominal"),
                ]
            )

            metric_options = {
                "Did costs rise?": "gentrified",
                "Change in monthly cost": "housing_cost_change",
                "Monthly cost, 2012": "housing_cost_2012",
                "Monthly cost, 2016": "housing_cost_2016",
            }
            if "rf_oof_probability" in map_gdf.columns:
                metric_options["Model probability of a rise"] = "rf_oof_probability"

            controls, canvas = st.columns([1, 3.2], gap="medium")
            tract_ids_map = sorted(map_gdf.loc[map_gdf["gentrified"].notna(), "tract_id"].astype(str).unique())

            with controls:
                with panel("mapcontrols"):
                    map_metric_label = st.selectbox("Colour tracts by", list(metric_options), key="map_color_metric")
                    metric_col = metric_options[map_metric_label]

                    initial = st.session_state.get("selected_map_tract")
                    if initial not in tract_ids_map:
                        initial = tract_ids_map[0] if tract_ids_map else None
                    if tract_ids_map:
                        st.session_state["selected_map_tract"] = st.selectbox(
                            "Find a tract",
                            tract_ids_map,
                            index=tract_ids_map.index(initial),
                            help="Or click a tract on the map.",
                        )

                    values = pd.to_numeric(map_gdf[metric_col], errors="coerce").dropna()

                    if metric_col == "gentrified":
                        legend = (
                            f'<div class="legend-row"><span class="swatch" style="background:{WARM}"></span>Costs rose</div>'
                            f'<div class="legend-row"><span class="swatch" style="background:{COOL}"></span>No increase</div>'
                        )
                        color_for = lambda v: WARM if v == 1 else COOL
                    elif metric_col == "housing_cost_change":
                        limit = float(values.abs().quantile(0.98)) or 1.0
                        legend = (
                            f'<div class="ramp" style="background:linear-gradient(90deg,{COOL},{MID},{WARM})"></div>'
                            f'<div class="ramp-labels"><span>{signed_money(-limit)}</span><span>$0</span>'
                            f'<span>{signed_money(limit)}</span></div>'
                        )

                        def color_for(v, limit=limit):
                            t = max(-1.0, min(1.0, v / limit))
                            return mix(MID, WARM, t) if t >= 0 else mix(MID, COOL, -t)
                    else:
                        lo_v, hi_v = (float(values.quantile(0.02)), float(values.quantile(0.98))) if len(values) else (0.0, 1.0)
                        if hi_v <= lo_v:
                            hi_v = lo_v + 1
                        ramp = SEQ_PROB if metric_col == "rf_oof_probability" else SEQ_COST
                        fmt = (lambda v: pct(100 * v)) if metric_col == "rf_oof_probability" else money
                        legend = (
                            f'<div class="ramp" style="background:linear-gradient(90deg,{ramp[0]},{ramp[1]})"></div>'
                            f'<div class="ramp-labels"><span>{fmt(lo_v)}</span><span>{fmt(hi_v)}</span></div>'
                        )

                        def color_for(v, lo_v=lo_v, hi_v=hi_v, ramp=ramp):
                            return mix(ramp[0], ramp[1], max(0.0, min(1.0, (v - lo_v) / (hi_v - lo_v))))

                    st.markdown(
                        f'<p class="sec-title" style="font-size:.9rem;margin-top:.6rem">{html.escape(map_metric_label)}</p>'
                        f"{legend}"
                        f'<div class="legend-row"><span class="swatch" style="background:{NO_DATA}"></span>No matched data</div>',
                        unsafe_allow_html=True,
                    )

            def style_function(feature):
                value = feature.get("properties", {}).get(metric_col)
                try:
                    value = float(value)
                    fill = color_for(value) if np.isfinite(value) else NO_DATA
                except (TypeError, ValueError):
                    fill = NO_DATA
                return {"fillColor": fill, "color": "#FFFFFF", "weight": 0.3, "fillOpacity": 0.86}

            # Send only the columns the map needs; this keeps the page fast.
            keep = ["tract_id", "housing_cost_2012", "housing_cost_2016", "housing_cost_change", "gentrified", "geometry"]
            if "rf_oof_probability" in map_gdf.columns:
                keep.insert(-1, "rf_oof_probability")
            display_gdf = map_gdf[keep].copy()
            try:
                display_gdf["geometry"] = display_gdf.geometry.simplify(0.0008, preserve_topology=True)
            except Exception:
                pass

            folium_map = folium.Map(
                location=[37.2, -119.5],
                zoom_start=6,
                # CARTO basemaps now require an API key; Esri's light-gray canvas does not.
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
                max_zoom=16,
                prefer_canvas=True,
                control_scale=True,
            )
            # No fit_bounds: the map is built while its tab is hidden (zero size),
            # which makes Leaflet jump to maximum zoom. The fixed view above covers California.

            folium.GeoJson(
                display_gdf.to_json(),
                name="California census tracts",
                style_function=style_function,
                highlight_function=lambda f: {"weight": 1.6, "color": INK, "fillOpacity": 0.95},
                tooltip=folium.GeoJsonTooltip(
                    fields=["tract_id", metric_col],
                    aliases=["Tract", map_metric_label],
                    localize=True,
                    sticky=False,
                    style=(
                        f"font-family:{FONT};font-size:12px;color:{INK};background:#fff;"
                        f"border:1px solid {RULE};border-radius:10px;box-shadow:0 8px 20px -10px rgba(15,27,42,.35);padding:8px 10px;"
                    ),
                ),
            ).add_to(folium_map)

            with canvas, panel("mapcanvas"):
                map_result = st_folium(
                    folium_map,
                    height=640,
                    use_container_width=True,
                    returned_objects=["last_clicked"],
                    key="california_tract_map_v3",
                )

            click = map_result.get("last_clicked") if isinstance(map_result, dict) else None
            if click and click.get("lat") is not None and click.get("lng") is not None:
                try:
                    from shapely.geometry import Point

                    hits = map_gdf[map_gdf.geometry.contains(Point(float(click["lng"]), float(click["lat"])))]
                    if not hits.empty:
                        st.session_state["selected_map_tract"] = str(hits.iloc[0]["tract_id"])
                except Exception:
                    pass

            selected_tract = st.session_state.get("selected_map_tract")
            selected_rows = map_gdf[map_gdf["tract_id"].astype(str) == str(selected_tract)] if selected_tract else map_gdf.iloc[0:0]

            if not selected_rows.empty:
                sel = selected_rows.iloc[0]

                with panel("mapdetail"):
                    st.markdown(
                        f'<p class="sec-title">Tract {html.escape(str(selected_tract))} {label_tag(sel.get("gentrified"))}</p>'
                        f'<p class="sec-note">County FIPS {html.escape(str(sel.get("county_fips", "")))}, California</p>',
                        unsafe_allow_html=True,
                    )

                    detail_tabs = st.tabs(["Housing cost", "Model estimate", "Nearby tracts"])

                    with detail_tabs[0]:
                        a, b, c = st.columns(3)
                        a.metric("2012", money(sel["housing_cost_2012"]))
                        b.metric("2016", money(sel["housing_cost_2016"]))
                        c.metric(
                            "Change",
                            signed_money(sel["housing_cost_change"]),
                            delta=pct(100 * sel["housing_cost_change"] / sel["housing_cost_2012"])
                            if pd.notna(sel["housing_cost_2012"]) else None,
                            delta_color="off",
                        )

                    with detail_tabs[1]:
                        prob = sel.get("rf_oof_probability", np.nan)
                        if prob is None or pd.isna(prob):
                            st.caption(
                                "No model estimate for this tract. Run `python src/train_ensemble.py` "
                                "to write results/tract_predictions.csv."
                            )
                        else:
                            st.metric("Random-forest probability that costs rose", pct(100 * prob))
                            st.caption(
                                "Out-of-fold estimate: the model that produced this number was trained "
                                "without this tract, using only 2010–2011 features."
                            )

                    with detail_tabs[2]:
                        try:
                            point = sel.geometry.representative_point()
                            distances = map_gdf.geometry.representative_point().distance(point)
                            nearby = map_gdf.loc[distances[distances > 0].nsmallest(6).index].copy()
                            st.dataframe(
                                pd.DataFrame(
                                    {
                                        "Tract": nearby["tract_id"],
                                        "Change": nearby["housing_cost_change"],
                                        "Outcome": nearby["gentrified"].map({1.0: "Costs rose", 0.0: "No increase"}).fillna("No data"),
                                    }
                                ),
                                hide_index=True,
                                width="stretch",
                                column_config={"Change": st.column_config.NumberColumn(format="$%d")},
                            )
                        except Exception:
                            st.caption("Nearby tracts could not be calculated for this geometry.")

        except ImportError:
            st.error("The map needs folium and streamlit-folium. Install them in the environment that runs Streamlit.")
            st.code("pip install geopandas folium streamlit-folium")
        except Exception as exc:
            st.error(f"The map could not be drawn: {exc}")


# ============================================================
# 10. MODEL RESULTS
# ============================================================

with models_tab:
    comparison = load_csv(str(COMPARISON_FILE))
    county_cv = load_csv(str(COUNTY_CV_FILE))

    if comparison is None:
        with panel("noresults"):
            section(
                "No model results yet",
                "Train the models to see their scores here. Results are read from results/model_comparison.csv.",
            )
            st.code("python src/train_ann.py\npython src/train_ensemble.py", language="bash")
    else:
        has_auc = "ROC AUC" in comparison.columns
        best = comparison.dropna(subset=["ROC AUC"]).sort_values("ROC AUC").iloc[-1] if has_auc else None
        baseline = comparison[comparison["Model"].str.contains("baseline", case=False)]
        base_acc = float(baseline["Accuracy"].iloc[0]) if not baseline.empty else np.nan

        items = []
        if best is not None:
            items.append(("Best ROC AUC", f"{best['ROC AUC']:.3f}", str(best["Model"])))
            items.append(("Its accuracy", pct(100 * best["Accuracy"]), f"Always guessing 'rose': {pct(100 * base_acc)}"))
            if "Balanced accuracy" in comparison.columns:
                items.append(("Its balanced accuracy", pct(100 * best["Balanced accuracy"]), "0.5 = no better than chance"))
        if county_cv is not None and len(county_cv) == 2:
            grouped = county_cv[county_cv["Validation"].str.contains("County")]
            if not grouped.empty:
                items.append(
                    ("Unseen-county ROC AUC", f"{grouped['ROC AUC mean'].iloc[0]:.3f}",
                     f"± {grouped['ROC AUC std'].iloc[0]:.3f}, random forest")
                )
        if items:
            stat_strip(items[:4])

        if has_auc:
            with panel("auc"):
                section(
                    "How well each model ranks tracts",
                    "ROC AUC on the held-out 20% test set. 0.5 is a coin flip; 1.0 is perfect.",
                )
                plot_df = comparison.sort_values("ROC AUC")
                is_base = plot_df["Model"].str.contains("baseline", case=False)
                fig = go.Figure(
                    go.Bar(
                        x=plot_df["ROC AUC"],
                        y=plot_df["Model"],
                        orientation="h",
                        marker=dict(color=[NO_DATA if b else COOL for b in is_base], cornerradius=4),
                        text=[f"{v:.3f}" for v in plot_df["ROC AUC"]],
                        textposition="outside",
                        textfont=dict(color=INK, size=12),
                        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                        showlegend=False,
                    )
                )
                fig.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"))
                fig.update_layout(height=60 + 46 * len(plot_df), bargap=0.35, margin=dict(l=8, r=40, t=10, b=8))
                fig.update_xaxes(range=[0, 1.05], title="ROC AUC")
                fig.update_yaxes(showgrid=False)
                chart(fig)

        with panel("table"):
            section("All metrics", "Accuracy alone is misleading here because most tracts rose; compare every model with the baseline row.")
            fmt_cols = {
                c: st.column_config.NumberColumn(c, format="%.3f")
                for c in comparison.columns if c != "Model"
            }
            st.dataframe(comparison, hide_index=True, width="stretch", column_config=fmt_cols)
            if county_cv is not None:
                st.caption(
                    "Cross-validation check: random folds put neighbouring tracts in both training and test data. "
                    "County-grouped folds hold out whole counties, which is closer to predicting a new place."
                )
                st.dataframe(
                    county_cv, hide_index=True, width="stretch",
                    column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in ["ROC AUC mean", "ROC AUC std"]},
                )

    model_files = find_model_files()
    with st.expander(f"Saved model files ({len(model_files)})"):
        if model_files:
            st.markdown("\n".join(f"- `{p.relative_to(ROOT)}`" for p in model_files))
        else:
            st.caption("No model files were found in models/.")


# ============================================================
# 11. METHODOLOGY
# ============================================================

with methodology_tab:
    st.markdown(
        r"""
<div class="prose-grid">
<div class="prose">
<span class="step">1</span>
<h4>Question</h4>
<p>Can 2010–2011 census-tract characteristics anticipate which California tracts saw
housing costs rise between 2012 and 2016?</p>
</div>
<div class="prose">
<span class="step">2</span>
<h4>Outcome</h4>
<p>For each tract, H is the ACS median monthly housing cost (table B25105). A tract is
labelled 1 if H<sub>2016</sub> − H<sub>2012</sub> &gt; 0 and 0 otherwise. This measures a
rise in housing cost, which is one symptom of gentrification and not a definition of it.</p>
</div>
<div class="prose">
<span class="step">3</span>
<h4>Features</h4>
<ul>
<li>All DP03 economic-profile variables for 2010 and 2011 (ACS 5-year).</li>
<li>Spatial features: the neighbour average and local Moran's I of the 2010→2011 change in
housing cost and in income distribution (Queen contiguity).</li>
<li>Census "no estimate" codes such as −666666666 are treated as missing; missing values are
filled with training-set medians.</li>
</ul>
</div>
<div class="prose">
<span class="step">4</span>
<h4>Models and evaluation</h4>
<ul>
<li>Logistic regression, random forest, a feed-forward neural network, and two ensembles
(majority vote and averaged probability).</li>
<li>One stratified 80/20 split shared by all models. Hyper-parameters are chosen on the
training data only (cross-validation or a validation split); the test set is used once.</li>
<li>Every score is reported next to a majority-class baseline, and the random forest is also
checked with folds that hold out whole counties.</li>
</ul>
</div>
<div class="prose wide">
<span class="step">!</span>
<h4>Limitations</h4>
<ul>
<li>Dollar values are nominal. Inflation between the 2012 and 2016 releases pushes most tracts
into the "rose" class even when real costs were flat.</li>
<li>ACS 5-year windows overlap (2010 covers 2006–10, 2011 covers 2007–11, 2012 covers 2008–12),
so year-to-year changes between adjacent releases are small and noisy.</li>
<li>Tract estimates carry sampling error; small changes may not be statistically meaningful.</li>
<li>A prediction is not evidence of displacement, causation or any household's experience.</li>
</ul>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="footnote">Source: U.S. Census Bureau, American Community Survey 5-year estimates; '
    "2010 TIGER/Line census tracts. Historical analysis, not a forecast.</p>",
    unsafe_allow_html=True,
)
