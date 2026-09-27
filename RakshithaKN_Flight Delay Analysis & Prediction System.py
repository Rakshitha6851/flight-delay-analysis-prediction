"""
Flight Delay Analysis & Prediction System
==========================================
A complete Streamlit dashboard for EDA, business intelligence,
and ML-based flight delay prediction using the 2024 BTS dataset.
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# scikit-learn is optional — the dashboard works without it;
# only the ML Prediction tab requires it.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
        roc_auc_score, roc_curve,
    )
    SKLEARN_AVAILABLE = True
except Exception:  # ImportError or DLL / binary compatibility issues
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Analysis & Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        margin-bottom: 8px;
    }
    .kpi-card.red   { border-left-color: #ef4444; }
    .kpi-card.green { border-left-color: #22c55e; }
    .kpi-card.amber { border-left-color: #f59e0b; }
    .kpi-label { font-size: 0.78rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e293b; line-height: 1.2; }
    .kpi-sub   { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }
    .section-title { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin: 1.2rem 0 0.5rem 0; }
    .insight-box {
        background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 6px;
        padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #1e3a5f;
    }
    .risk-box {
        background: #fff7ed; border-left: 4px solid #f59e0b; border-radius: 6px;
        padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #78350f;
    }
    .action-box {
        background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px;
        padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #14532d;
    }
    div[data-testid="stMetric"] { background: white; border-radius:8px; padding:12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.87rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DELAY_THRESHOLD = 15          # FAA standard: delayed if dep_delay > 15 min
CARRIER_NAMES = {
    "9E": "Endeavor Air", "AA": "American Airlines", "AS": "Alaska Airlines",
    "B6": "JetBlue", "DL": "Delta Air Lines", "F9": "Frontier Airlines",
    "G4": "Allegiant Air", "HA": "Hawaiian Airlines", "MQ": "Envoy Air",
    "NK": "Spirit Airlines", "OH": "PSA Airlines", "OO": "SkyWest Airlines",
    "QX": "Horizon Air", "UA": "United Airlines", "WN": "Southwest Airlines",
    "YV": "Mesa Airlines", "YX": "Republic Airways", "ZW": "Air Wisconsin",
}

DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(filepath: str) -> pd.DataFrame:
    """Load, inspect, and return the raw dataframe."""
    df = pd.read_csv(filepath, low_memory=False)
    return df


@st.cache_data(show_spinner="Cleaning data…")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Standardise column names to snake_case
    2. Drop exact duplicates
    3. Fix date / numeric types
    4. Fill delay-column NaNs with 0 (per dataset documentation)
    5. Drop rows where both dep_delay AND arr_delay are NaN (cancelled flights
       have no delay data; keep them but mark separately)
    6. Derive analytical features
    """
    # ── 1. Normalise column names ──────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── 2. Drop duplicates ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    dupes_removed = before - len(df)

    # ── 3. Type corrections ────────────────────────────────────────────────
    # Flight date
    if "fl_date" in df.columns:
        df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    # Ensure numeric columns are numeric
    numeric_cols = [
        "dep_delay", "arr_delay", "taxi_out", "taxi_in",
        "crs_elapsed_time", "actual_elapsed_time", "air_time",
        "distance", "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
        "crs_dep_time", "dep_time", "crs_arr_time", "arr_time",
        "wheels_off", "wheels_on",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cancelled / diverted → int (0/1)
    for col in ["cancelled", "diverted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── 4. Fill delay NaNs with 0 ──────────────────────────────────────────
    delay_fill_cols = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    for col in delay_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── 5. Derived features ────────────────────────────────────────────────
    # Binary delay target (FAA standard: > 15 min departure delay)
    if "dep_delay" in df.columns:
        df["is_delayed"] = (df["dep_delay"] > DELAY_THRESHOLD).astype(int)

    # Departure hour from scheduled departure time (hhmm format)
    if "crs_dep_time" in df.columns:
        df["dep_hour"] = (df["crs_dep_time"] // 100).astype("Int64")
        df["dep_hour"] = df["dep_hour"].where(df["dep_hour"].between(0, 23), other=pd.NA)

    # Season
    if "month" in df.columns:
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall",
        }
        df["season"] = df["month"].map(season_map)

    # Route label
    if "origin" in df.columns and "dest" in df.columns:
        df["route"] = df["origin"].astype(str) + " → " + df["dest"].astype(str)

    # Carrier full name
    if "op_unique_carrier" in df.columns:
        df["carrier_name"] = df["op_unique_carrier"].map(CARRIER_NAMES).fillna(df["op_unique_carrier"])

    # Day name / Month name
    if "day_of_week" in df.columns:
        df["day_name"] = df["day_of_week"].map(DAY_NAMES)
    if "month" in df.columns:
        df["month_name"] = df["month"].map(MONTH_NAMES)

    # Store metadata on the df so it can be surfaced in the UI
    df.attrs["dupes_removed"] = dupes_removed
    return df


# ─────────────────────────────────────────────
# ML PIPELINE (cached on the filtered df hash)
# ─────────────────────────────────────────────
ML_SAMPLE_SIZE = 150_000   # max rows used for ML training (keeps training fast)

@st.cache_data(show_spinner="Training ML model…")
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier to predict is_delayed.
    Uses a reproducible stratified sample (≤ ML_SAMPLE_SIZE rows) so training
    completes quickly on the full 7M-row dataset without affecting dashboard EDA.
    Returns model, encoders, feature list, and evaluation metrics.
    Returns all-None tuple if sklearn is unavailable.
    """
    if not SKLEARN_AVAILABLE:
        return None, None, None, None, None
    required = ["is_delayed", "month", "day_of_week", "distance", "op_unique_carrier", "origin"]
    # Only keep rows where target and key features are non-null
    cols_needed = [c for c in required if c in df.columns]
    model_df = df[cols_needed].dropna()

    # Need at least both classes present
    if model_df["is_delayed"].nunique() < 2 or len(model_df) < 200:
        return None, None, None, None, None

    # Add dep_hour if available (still from full df, before sampling)
    extra_feats = []
    if "dep_hour" in df.columns:
        model_df = model_df.copy()
        model_df["dep_hour"] = df.loc[model_df.index, "dep_hour"]
        model_df = model_df.dropna(subset=["dep_hour"])
        extra_feats = ["dep_hour"]

    # ── Reproducible sample to keep training time bounded ─────────────────
    # Only the ML pipeline is sampled; dashboard/EDA always use the full df.
    if len(model_df) > ML_SAMPLE_SIZE:
        model_df = model_df.sample(n=ML_SAMPLE_SIZE, random_state=42)

    # Encode categoricals
    le_carrier = LabelEncoder()
    le_origin = LabelEncoder()
    model_df = model_df.copy()
    model_df["carrier_enc"] = le_carrier.fit_transform(model_df["op_unique_carrier"].astype(str))
    model_df["origin_enc"] = le_origin.fit_transform(model_df["origin"].astype(str))

    feature_cols = ["month", "day_of_week", "distance", "carrier_enc", "origin_enc"] + extra_feats
    X = model_df[feature_cols]
    y = model_df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "cm":        confusion_matrix(y_test, y_pred),
        "y_test":    y_test.values,
        "y_prob":    y_prob,
        "report":    classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "feature_importance": dict(zip(feature_cols, clf.feature_importances_)),
    }

    encoders = {"carrier": le_carrier, "origin": le_origin}
    return clf, encoders, feature_cols, metrics, model_df


# ─────────────────────────────────────────────
# HELPER: KPI CARD
# ─────────────────────────────────────────────
def kpi_card(label, value, sub="", color="blue"):
    color_class = {"blue": "", "red": " red", "green": " green", "amber": " amber"}.get(color, "")
    st.markdown(f"""
    <div class="kpi-card{color_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def risk(text):
    st.markdown(f'<div class="risk-box">⚠️ {text}</div>', unsafe_allow_html=True)

def action(text):
    st.markdown(f'<div class="action-box">✅ {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
CHART_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Set2

def bar_chart(df_agg, x, y, title, color=None, orientation="v", text=None):
    fig = px.bar(
        df_agg, x=x, y=y, title=title,
        color=color, orientation=orientation,
        text=text, template=CHART_TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=360)
    fig.update_traces(texttemplate="%{text:.1f}%" if text else None, textposition="outside")
    return fig

def line_chart(df_agg, x, y, title, color=None):
    fig = px.line(
        df_agg, x=x, y=y, title=title,
        color=color, markers=True, template=CHART_TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=350)
    return fig

def pie_chart(names, values, title):
    fig = px.pie(
        names=names, values=values, title=title,
        template=CHART_TEMPLATE, color_discrete_sequence=COLOR_SEQ, hole=0.35,
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=340)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/1200px-Amazon_logo.svg.png",
        width=50,
    ) if False else None  # placeholder logo disabled

    st.sidebar.title("✈️ Flight Delay System")
    st.sidebar.markdown("---")

    # ── Dataset selector ──────────────────────────────────────────────────
    st.sidebar.subheader("📂 Dataset")
    data_options = {}
    if os.path.exists("flight_data_2024.csv"):
        data_options["Full Dataset (~7M rows)"] = "flight_data_2024.csv"
    if os.path.exists("flight_data_2024_sample.csv"):
        data_options["Sample Dataset (10K rows)"] = "flight_data_2024_sample.csv"

    if not data_options:
        st.sidebar.error(
            "No dataset found.\n\n"
            "Please place **flight_data_2024_sample.csv** or "
            "**flight_data_2024.csv** in the same folder as app.py."
        )
        st.stop()

    chosen_label = st.sidebar.selectbox("Select dataset", list(data_options.keys()))
    filepath = data_options[chosen_label]

    raw_df = load_data(filepath)
    df = clean_data(raw_df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")

    # Airline filter
    if "op_unique_carrier" in df.columns:
        carriers = sorted(df["op_unique_carrier"].dropna().unique().tolist())
        sel_carriers = st.sidebar.multiselect(
            "Airline (carrier code)", carriers, default=carriers[:min(len(carriers), 5)]
        )
        if sel_carriers:
            df = df[df["op_unique_carrier"].isin(sel_carriers)]

    # Month filter
    if "month" in df.columns:
        months = sorted(df["month"].dropna().unique().tolist())
        sel_months = st.sidebar.multiselect(
            "Month", months,
            format_func=lambda m: MONTH_NAMES.get(int(m), str(m)),
            default=months,
        )
        if sel_months:
            df = df[df["month"].isin(sel_months)]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Rows loaded:** {len(df):,}  \n"
        f"**Columns:** {df.shape[1]}  \n"
        f"**Dupes removed:** {df.attrs.get('dupes_removed', 0):,}"
    )
    return df, raw_df


# ─────────────────────────────────────────────
# TAB 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────
def tab_executive(df: pd.DataFrame):
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "A high-level summary of 2024 domestic flight performance — designed for business decision-makers."
    )

    total_flights = len(df)
    cancelled_col = "cancelled" if "cancelled" in df.columns else None
    delayed_col   = "is_delayed" if "is_delayed" in df.columns else None

    cancelled_n   = int(df[cancelled_col].sum()) if cancelled_col else 0
    delayed_n     = int(df[delayed_col].sum())  if delayed_col else 0
    avg_dep_delay = df["dep_delay"].mean() if "dep_delay" in df.columns else 0
    delay_rate    = delayed_n / total_flights * 100 if total_flights else 0
    cancel_rate   = cancelled_n / total_flights * 100 if total_flights else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Flights", f"{total_flights:,}", "in filtered selection", "blue")
    with c2: kpi_card("Delay Rate", f"{delay_rate:.1f}%", f"dep delay > {DELAY_THRESHOLD} min", "red" if delay_rate > 20 else "amber")
    with c3: kpi_card("Avg Dep Delay", f"{avg_dep_delay:.1f} min", "all flights incl. early", "amber")
    with c4: kpi_card("Cancellations", f"{cancelled_n:,}", f"{cancel_rate:.2f}% of flights", "red")
    with c5:
        avg_delayed_only = df.loc[df["dep_delay"] > 0, "dep_delay"].mean() if "dep_delay" in df.columns else 0
        kpi_card("Avg Delay (delayed)", f"{avg_delayed_only:.1f} min", "flights with dep_delay > 0", "red")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    monthly = None  # defined here so it's always in scope below

    # Delay rate by month
    with col_a:
        if "month" in df.columns and delayed_col:
            monthly = (
                df.groupby("month")
                .agg(total=("is_delayed", "count"), delayed=("is_delayed", "sum"))
                .reset_index()
            )
            monthly["delay_rate"] = monthly["delayed"] / monthly["total"] * 100
            monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
            fig = px.bar(
                monthly, x="month_name", y="delay_rate",
                title="Monthly Delay Rate (%)",
                template=CHART_TEMPLATE, color="delay_rate",
                color_continuous_scale="Reds",
                labels={"delay_rate": "Delay Rate (%)", "month_name": "Month"},
            )
            fig.update_layout(margin=dict(t=40, b=0), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Delay breakdown by cause
    with col_b:
        cause_cols = {
            "carrier_delay": "Carrier", "weather_delay": "Weather",
            "nas_delay": "NAS", "security_delay": "Security",
            "late_aircraft_delay": "Late Aircraft",
        }
        present = {v: df[k].sum() for k, v in cause_cols.items() if k in df.columns and df[k].sum() > 0}
        if present:
            fig = pie_chart(list(present.keys()), list(present.values()), "Delay Cause Breakdown (total minutes)")
            st.plotly_chart(fig, use_container_width=True)

    # Business narrative
    st.markdown("### 📌 Key Findings")
    col_i, col_r, col_a2 = st.columns(3)
    with col_i:
        st.markdown("**Insights**")
        insight(f"Overall delay rate is **{delay_rate:.1f}%** — roughly 1 in {max(1, round(100/delay_rate)) if delay_rate > 0 else 'N/A'} flights departs late.")
        if monthly is not None:
            worst_month = monthly.loc[monthly["delay_rate"].idxmax(), "month_name"]
            insight(f"**{worst_month}** sees the highest departure delays of the year.")
        if present:
            top_cause = max(present, key=present.get)
            insight(f"**{top_cause}** delays account for the largest share of total delay minutes.")
    with col_r:
        st.markdown("**Risks**")
        risk(f"A {delay_rate:.1f}% delay rate means millions of passengers experience disruption annually.")
        risk("Late-aircraft and weather delays are associated with a large share of total delay minutes — flights arriving late tend to depart late on the next leg.")
        risk(f"Cancellation rate of {cancel_rate:.2f}% indicates supply-demand gaps on high-demand days.")
    with col_a2:
        st.markdown("**Recommended Actions**")
        action("Prioritise buffer time for routes with consistently high NAS / late-aircraft delays.")
        action("Strengthen crew/aircraft positioning strategies during peak summer & holiday periods.")
        action("Deploy delay prediction model for proactive passenger communication.")


# ─────────────────────────────────────────────
# TAB 2 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def tab_eda(df: pd.DataFrame):
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # ── Delay distribution ────────────────────────────────────────────────
    if "dep_delay" in df.columns:
        st.markdown("#### Departure Delay Distribution")
        clip_val = 180
        delay_data = df["dep_delay"].clip(-60, clip_val).dropna()
        fig = px.histogram(
            delay_data, nbins=80, title=f"Departure Delay Distribution (clipped to −60 … {clip_val} min)",
            template=CHART_TEMPLATE, color_discrete_sequence=["#3b82f6"],
            labels={"value": "Dep Delay (min)", "count": "Flights"},
        )
        fig.add_vline(x=DELAY_THRESHOLD, line_dash="dash", line_color="red",
                      annotation_text=f"Delay threshold ({DELAY_THRESHOLD} min)")
        fig.update_layout(height=320, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        insight(f"Most flights depart within ±{DELAY_THRESHOLD} minutes of schedule. The distribution has a long right tail — a small proportion of flights incur very large delays.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Day of week ───────────────────────────────────────────────────────
    with col1:
        if "day_of_week" in df.columns and "is_delayed" in df.columns:
            dow = (
                df.groupby("day_of_week")
                .agg(total=("is_delayed", "count"), delayed=("is_delayed", "sum"))
                .reset_index()
            )
            dow["delay_pct"] = dow["delayed"] / dow["total"] * 100
            dow["day"] = dow["day_of_week"].map(DAY_NAMES)
            fig = px.bar(
                dow, x="day", y="delay_pct",
                title="Delay Rate by Day of Week (%)",
                template=CHART_TEMPLATE,
                color="delay_pct", color_continuous_scale="Blues",
                labels={"delay_pct": "Delay Rate (%)", "day": "Day"},
            )
            fig.update_layout(height=320, margin=dict(t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            worst_day = dow.loc[dow["delay_pct"].idxmax(), "day"]
            insight(f"**{worst_day}** has the highest delay rate — likely due to network congestion accumulation.")

    # ── Departure hour ────────────────────────────────────────────────────
    with col2:
        if "dep_hour" in df.columns and "is_delayed" in df.columns:
            hourly = (
                df.dropna(subset=["dep_hour"])
                .groupby("dep_hour")
                .agg(total=("is_delayed", "count"), delayed=("is_delayed", "sum"))
                .reset_index()
            )
            hourly["delay_pct"] = hourly["delayed"] / hourly["total"] * 100
            fig = px.line(
                hourly, x="dep_hour", y="delay_pct", markers=True,
                title="Delay Rate by Departure Hour (%)",
                template=CHART_TEMPLATE,
                labels={"delay_pct": "Delay Rate (%)", "dep_hour": "Hour (local)"},
            )
            fig.update_layout(height=320, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            insight("Early-morning flights (5–7 AM) have the lowest delay rates. Delay rates rise sharply through the afternoon and peak in the evening as network congestion accumulates.")

    st.markdown("---")
    col3, col4 = st.columns(2)

    # ── Season ────────────────────────────────────────────────────────────
    with col3:
        if "season" in df.columns and "is_delayed" in df.columns:
            seas = (
                df.groupby("season")
                .agg(total=("is_delayed", "count"), delayed=("is_delayed", "sum"))
                .reset_index()
            )
            seas["delay_pct"] = seas["delayed"] / seas["total"] * 100
            fig = px.bar(
                seas, x="season", y="delay_pct",
                title="Delay Rate by Season (%)",
                template=CHART_TEMPLATE, color="season",
                color_discrete_sequence=COLOR_SEQ,
                labels={"delay_pct": "Delay Rate (%)", "season": "Season"},
            )
            fig.update_layout(height=320, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # ── Cancelled vs on-time ──────────────────────────────────────────────
    with col4:
        if "cancelled" in df.columns:
            cancel_by_month = df.groupby("month")["cancelled"].mean().reset_index()
            cancel_by_month["cancel_pct"] = cancel_by_month["cancelled"] * 100
            cancel_by_month["month_name"] = cancel_by_month["month"].map(MONTH_NAMES)
            fig = px.bar(
                cancel_by_month, x="month_name", y="cancel_pct",
                title="Cancellation Rate by Month (%)",
                template=CHART_TEMPLATE, color="cancel_pct",
                color_continuous_scale="Oranges",
                labels={"cancel_pct": "Cancel Rate (%)", "month_name": "Month"},
            )
            fig.update_layout(height=320, margin=dict(t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("Winter months (Jan, Feb, Dec) see elevated cancellation rates due to weather events.")

    # ── Delay cause over months ───────────────────────────────────────────
    st.markdown("#### Delay Cause Trends by Month")
    cause_cols_map = {
        "carrier_delay": "Carrier", "weather_delay": "Weather",
        "nas_delay": "NAS", "security_delay": "Security",
        "late_aircraft_delay": "Late Aircraft",
    }
    present_causes = [k for k in cause_cols_map if k in df.columns]
    if present_causes and "month" in df.columns:
        monthly_causes = df.groupby("month")[present_causes].mean().reset_index()
        monthly_causes["month_name"] = monthly_causes["month"].map(MONTH_NAMES)
        melted = monthly_causes.melt(
            id_vars=["month", "month_name"],
            value_vars=present_causes,
            var_name="cause", value_name="avg_minutes",
        )
        melted["cause"] = melted["cause"].map(cause_cols_map)
        fig = px.line(
            melted, x="month_name", y="avg_minutes", color="cause",
            markers=True, title="Average Delay Minutes by Cause & Month",
            template=CHART_TEMPLATE,
            labels={"avg_minutes": "Avg Minutes", "month_name": "Month"},
        )
        fig.update_layout(height=360, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3 — AIRLINE ANALYSIS
# ─────────────────────────────────────────────
def tab_airlines(df: pd.DataFrame):
    st.markdown('<div class="section-title">Airline Performance Analysis</div>', unsafe_allow_html=True)

    if "op_unique_carrier" not in df.columns:
        st.info("Carrier column not found.")
        return

    _agg_dict = {
        "total_flights":    ("is_delayed", "count"),
        "delayed_flights":  ("is_delayed", "sum"),
        "avg_dep_delay":    ("dep_delay", "mean"),
    }
    if "arr_delay" in df.columns:
        _agg_dict["avg_arr_delay"] = ("arr_delay", "mean")
    if "cancelled" in df.columns:
        _agg_dict["cancelled_flights"] = ("cancelled", "sum")

    agg = (
        df.groupby(["op_unique_carrier", "carrier_name"])
        .agg(**_agg_dict)
        .reset_index()
    )
    agg["delay_rate"] = agg["delayed_flights"] / agg["total_flights"] * 100
    agg["cancel_rate"] = (
        agg["cancelled_flights"] / agg["total_flights"] * 100
        if "cancelled_flights" in agg.columns else 0
    )
    agg = agg.sort_values("delay_rate", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            agg, x="delay_rate", y="carrier_name", orientation="h",
            title="Delay Rate by Airline (%)",
            template=CHART_TEMPLATE, color="delay_rate",
            color_continuous_scale="RdYlGn_r",
            labels={"delay_rate": "Delay Rate (%)", "carrier_name": "Airline"},
        )
        fig.update_layout(height=420, margin=dict(t=40, b=0), coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            agg.sort_values("avg_dep_delay", ascending=False),
            x="avg_dep_delay", y="carrier_name", orientation="h",
            title="Average Departure Delay by Airline (min)",
            template=CHART_TEMPLATE, color="avg_dep_delay",
            color_continuous_scale="Blues",
            labels={"avg_dep_delay": "Avg Dep Delay (min)", "carrier_name": "Airline"},
        )
        fig.update_layout(height=420, margin=dict(t=40, b=0), coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: volume vs delay rate
    st.markdown("#### Airline Performance Matrix")
    fig = px.scatter(
        agg, x="total_flights", y="delay_rate",
        size="total_flights", color="carrier_name",
        hover_name="carrier_name", text="op_unique_carrier",
        title="Flights Volume vs Delay Rate (bubble size = flight volume)",
        template=CHART_TEMPLATE,
        labels={"total_flights": "Total Flights", "delay_rate": "Delay Rate (%)"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=400, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### Full Airline Scorecard")
    display_cols = ["carrier_name", "total_flights", "delay_rate", "avg_dep_delay", "cancel_rate"]
    display_cols = [c for c in display_cols if c in agg.columns]
    st.dataframe(
        agg[display_cols].rename(columns={
            "carrier_name": "Airline", "total_flights": "Total Flights",
            "delay_rate": "Delay Rate (%)", "avg_dep_delay": "Avg Dep Delay (min)",
            "cancel_rate": "Cancel Rate (%)",
        }).round(2).reset_index(drop=True),
        use_container_width=True,
    )

    worst = agg.iloc[0]
    best  = agg.iloc[-1]
    risk(f"**{worst['carrier_name']}** has the highest observed delay rate ({worst['delay_rate']:.1f}%) in the selected data.")
    action(f"**{best['carrier_name']}** is the on-time performance leader — benchmark its operations for best-practice adoption.")


# ─────────────────────────────────────────────
# TAB 4 — AIRPORT & ROUTE ANALYSIS
# ─────────────────────────────────────────────
def tab_airports(df: pd.DataFrame):
    st.markdown('<div class="section-title">Airport & Route Analysis</div>', unsafe_allow_html=True)
    top_n = 20

    col1, col2 = st.columns(2)

    # ── Top origin airports ───────────────────────────────────────────────
    with col1:
        if "origin" in df.columns and "is_delayed" in df.columns:
            orig = (
                df.groupby("origin")
                .agg(total=("is_delayed", "count"), delayed=("is_delayed", "sum"))
                .reset_index()
            )
            orig["delay_rate"] = orig["delayed"] / orig["total"] * 100
            top_orig = orig.nlargest(top_n, "total").sort_values("delay_rate", ascending=False)
            fig = px.bar(
                top_orig, x="delay_rate", y="origin", orientation="h",
                title=f"Delay Rate — Top {top_n} Origin Airports (%)",
                template=CHART_TEMPLATE, color="delay_rate",
                color_continuous_scale="RdYlGn_r",
                labels={"delay_rate": "Delay Rate (%)", "origin": "Origin Airport"},
            )
            fig.update_layout(height=500, margin=dict(t=40, b=0), coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Top destination airports ──────────────────────────────────────────
    with col2:
        if "dest" in df.columns and "arr_delay" in df.columns:
            dest_agg = (
                df.groupby("dest")
                .agg(total=("arr_delay", "count"), avg_arr=("arr_delay", "mean"))
                .reset_index()
            )
            top_dest = dest_agg.nlargest(top_n, "total").sort_values("avg_arr", ascending=False)
            fig = px.bar(
                top_dest, x="avg_arr", y="dest", orientation="h",
                title=f"Avg Arrival Delay — Top {top_n} Destination Airports (min)",
                template=CHART_TEMPLATE, color="avg_arr",
                color_continuous_scale="Blues",
                labels={"avg_arr": "Avg Arr Delay (min)", "dest": "Dest Airport"},
            )
            fig.update_layout(height=500, margin=dict(t=40, b=0), coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Busiest routes ────────────────────────────────────────────────────
    if "route" in df.columns and "is_delayed" in df.columns:
        st.markdown("#### Top Routes by Traffic and Delay")
        routes = (
            df.groupby("route")
            .agg(
                total=("is_delayed", "count"),
                delayed=("is_delayed", "sum"),
                avg_dep_delay=("dep_delay", "mean"),
            )
            .reset_index()
        )
        routes["delay_rate"] = routes["delayed"] / routes["total"] * 100
        top_routes = routes.nlargest(25, "total")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                top_routes.sort_values("total", ascending=False).head(15),
                x="total", y="route", orientation="h",
                title="Top 15 Busiest Routes",
                template=CHART_TEMPLATE, color="delay_rate",
                color_continuous_scale="RdYlGn_r",
                labels={"total": "Total Flights", "route": "Route"},
            )
            fig.update_layout(height=440, margin=dict(t=40, b=0), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            high_risk = routes[routes["total"] >= routes["total"].quantile(0.7)].nlargest(15, "delay_rate")
            fig = px.bar(
                high_risk, x="delay_rate", y="route", orientation="h",
                title="High-Risk Routes (high traffic, high delay)",
                template=CHART_TEMPLATE, color="delay_rate",
                color_continuous_scale="Reds",
                labels={"delay_rate": "Delay Rate (%)", "route": "Route"},
            )
            fig.update_layout(height=440, margin=dict(t=40, b=0), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        worst_route = routes.nlargest(1, "delay_rate").iloc[0]
        busiest_route = routes.nlargest(1, "total").iloc[0]
        risk(f"Route **{worst_route['route']}** has the highest delay rate ({worst_route['delay_rate']:.1f}%) among frequently-flown routes.")
        insight(f"Busiest route is **{busiest_route['route']}** with {busiest_route['total']:,} flights in the period.")


# ─────────────────────────────────────────────
# TAB 5 — DELAY DRIVERS
# ─────────────────────────────────────────────
def tab_drivers(df: pd.DataFrame):
    st.markdown('<div class="section-title">Delay Drivers & Root Cause Analysis</div>', unsafe_allow_html=True)

    cause_map = {
        "carrier_delay": "Carrier", "weather_delay": "Weather",
        "nas_delay": "NAS (Air Traffic)", "security_delay": "Security",
        "late_aircraft_delay": "Late Aircraft",
    }
    present = {v: df[k].sum() for k, v in cause_map.items() if k in df.columns}

    if present:
        col1, col2 = st.columns(2)
        with col1:
            fig = pie_chart(list(present.keys()), list(present.values()), "Total Delay Minutes by Cause")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cause_df = pd.DataFrame({"Cause": list(present.keys()), "Total Minutes": list(present.values())})
            cause_df = cause_df.sort_values("Total Minutes", ascending=True)
            fig = px.bar(
                cause_df, x="Total Minutes", y="Cause", orientation="h",
                title="Total Delay Minutes by Cause",
                template=CHART_TEMPLATE, color="Total Minutes",
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=340, margin=dict(t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Cause breakdown by airline
        if "carrier_name" in df.columns:
            st.markdown("#### Delay Cause Breakdown by Airline")
            cause_by_carrier = df.groupby("carrier_name")[[k for k in cause_map if k in df.columns]].mean().reset_index()
            cause_by_carrier = cause_by_carrier.rename(columns={k: v for k, v in cause_map.items()})
            melt = cause_by_carrier.melt(id_vars="carrier_name", var_name="Cause", value_name="Avg Minutes")
            fig = px.bar(
                melt, x="carrier_name", y="Avg Minutes", color="Cause",
                barmode="stack", title="Average Delay by Cause per Airline (stacked)",
                template=CHART_TEMPLATE,
                labels={"carrier_name": "Airline"},
            )
            fig.update_layout(height=400, margin=dict(t=40, b=0), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Insights
        top_cause = max(present, key=present.get)
        insight(f"**{top_cause}** delays make up the largest share of total delayed minutes.")
        action("Carrier delays are within airline control — review turnaround time, crew scheduling, and maintenance reliability.")
        action("Late-aircraft delays are associated with a large share of total delay minutes — improving early-morning on-time performance may reduce downstream delays.")
        risk("Weather delays are exogenous — however, proactive cancellation decisions can prevent greater cascading disruption.")

    # Distance vs delay scatter
    if "distance" in df.columns and "dep_delay" in df.columns:
        st.markdown("#### Distance vs Departure Delay")
        sample = df[["distance", "dep_delay"]].dropna().sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="distance", y="dep_delay", opacity=0.3,
            title="Distance vs Departure Delay (random sample)",
            template=CHART_TEMPLATE,
            labels={"distance": "Distance (miles)", "dep_delay": "Dep Delay (min)"},
            trendline="ols",
        )
        fig.update_layout(height=360, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        insight("Distance has minimal direct correlation with departure delay — delays are driven by operational factors, not route length.")


# ─────────────────────────────────────────────
# TAB 6 — ML PREDICTION
# ─────────────────────────────────────────────
def tab_ml(df: pd.DataFrame):
    st.markdown('<div class="section-title">Machine Learning — Delay Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        "A **Random Forest Classifier** trained to predict whether a flight will be delayed "
        f"(departure delay > {DELAY_THRESHOLD} min). Features: month, day of week, distance, "
        "airline, origin airport, and departure hour."
    )

    if not SKLEARN_AVAILABLE:
        st.error(
            "**scikit-learn is not available** in this environment.\n\n"
            "Install it with: `pip install scikit-learn scipy`\n\n"
            "All other dashboard tabs (EDA, Airlines, Airports, Delay Drivers) work without scikit-learn."
        )
        return

    clf, encoders, feature_cols, metrics, model_df = train_model(df)

    if clf is None:
        st.warning("Insufficient data to train a model with the current filters. Expand the airline/month selection.")
        return

    # ── Evaluation metrics ────────────────────────────────────────────────
    st.markdown("### Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Accuracy",  f"{metrics['accuracy']*100:.1f}%",  "", "blue")
    with c2: kpi_card("Precision", f"{metrics['precision']*100:.1f}%", "", "green")
    with c3: kpi_card("Recall",    f"{metrics['recall']*100:.1f}%",    "", "amber")
    with c4: kpi_card("F1-Score",  f"{metrics['f1']*100:.1f}%",        "", "blue")
    with c5: kpi_card("ROC-AUC",   f"{metrics['roc_auc']:.3f}",        "", "green")

    col1, col2 = st.columns(2)

    # Confusion matrix
    with col1:
        cm = metrics["cm"]
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Delayed", "Delayed"], y=["Not Delayed", "Delayed"],
            title="Confusion Matrix",
        )
        fig.update_layout(height=340, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ROC curve
    with col2:
        fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_prob"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={metrics['roc_auc']:.3f})", line=dict(color="#3b82f6", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(
            title="ROC Curve", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", height=340,
            margin=dict(t=40, b=0), template=CHART_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("### Feature Importance")
    fi = pd.DataFrame({
        "Feature": list(metrics["feature_importance"].keys()),
        "Importance": list(metrics["feature_importance"].values()),
    }).sort_values("Importance", ascending=True)
    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Random Forest Feature Importance",
        template=CHART_TEMPLATE, color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=320, margin=dict(t=40, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Live prediction ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔮 Predict a Single Flight")
    st.markdown("Enter flight details to get an instant delay probability.")

    carriers_available = sorted(model_df["op_unique_carrier"].unique().tolist())
    origins_available  = sorted(model_df["origin"].unique().tolist())

    with st.form("predict_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_month  = st.selectbox("Month", list(MONTH_NAMES.keys()), format_func=lambda m: MONTH_NAMES[m])
            p_dow    = st.selectbox("Day of Week", list(DAY_NAMES.keys()), format_func=lambda d: DAY_NAMES[d])
        with pc2:
            p_carrier = st.selectbox("Airline (carrier code)", carriers_available)
            p_origin  = st.selectbox("Origin Airport", origins_available)
        with pc3:
            p_distance = st.slider("Distance (miles)", 50, 5500, 800)
            p_hour     = st.slider("Scheduled Departure Hour", 0, 23, 8) if "dep_hour" in feature_cols else None

        submitted = st.form_submit_button("Predict Delay Probability", use_container_width=True)

    if submitted:
        try:
            carrier_enc_val = encoders["carrier"].transform([p_carrier])[0]
            origin_enc_val  = encoders["origin"].transform([p_origin])[0]
            row = [p_month, p_dow, p_distance, carrier_enc_val, origin_enc_val]
            if p_hour is not None and "dep_hour" in feature_cols:
                row.append(p_hour)
            X_pred = pd.DataFrame([row], columns=feature_cols)
            prob = clf.predict_proba(X_pred)[0][1]
            pred = clf.predict(X_pred)[0]

            if pred == 1:
                st.error(f"⚠️ **Likely DELAYED** — estimated probability: **{prob*100:.1f}%**")
            else:
                st.success(f"✅ **Likely ON TIME** — delay probability: **{prob*100:.1f}%**")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                title={"text": "Delay Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if prob > 0.5 else "#22c55e"},
                    "steps": [
                        {"range": [0, 33], "color": "#dcfce7"},
                        {"range": [33, 66], "color": "#fef9c3"},
                        {"range": [66, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                },
                number={"suffix": "%", "font": {"size": 36}},
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Classification report table
    with st.expander("📋 Full Classification Report"):
        report_df = pd.DataFrame(metrics["report"]).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    df, raw_df = render_sidebar()

    st.title("✈️ Flight Delay Analysis & Prediction System")
    st.markdown(
        "**2024 U.S. Domestic Flights** · Source: BTS TranStats On-Time Performance Database  \n"
        "Use the sidebar to select the dataset and apply airline / month filters."
    )

    tabs = st.tabs([
        "🏠 Executive Overview",
        "📊 Exploratory Analysis",
        "🛫 Airline Analysis",
        "🗺️ Airport & Routes",
        "🔍 Delay Drivers",
        "🤖 ML Prediction",
    ])

    with tabs[0]: tab_executive(df)
    with tabs[1]: tab_eda(df)
    with tabs[2]: tab_airlines(df)
    with tabs[3]: tab_airports(df)
    with tabs[4]: tab_drivers(df)
    with tabs[5]: tab_ml(df)

    st.markdown("---")
    st.caption("Flight Delay Analysis & Prediction System · Data: BTS TranStats 2024 · Built with Streamlit + scikit-learn")


if __name__ == "__main__":
    main()
