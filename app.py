"""
app.py
======
Wildfire "Major Incident" Risk Predictor — Streamlit Frontend

3 sections:
    1. Overview        -> dataset summary, map, key stats
    2. Model Comparison -> Accuracy / Precision / Recall / F1 + Confusion Matrices
    3. Predict          -> naya incident daal kar risk predict karna

Ye file sirf backend.py ke functions use karta hai — training logic
backend.py me hi rehta hai (single source of truth).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import backend

# ---------------------------------------------------------------------------
# Page config (sab se pehla Streamlit call hona zaroori hai)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wildfire Risk Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — ember / charred-wood palette (warm primary + cool smoke accent)
# ---------------------------------------------------------------------------
BG = "#15110D"
BG_ELEV = "#211A14"
BG_ELEV_2 = "#2B2118"
EMBER = "#FF7A33"
AMBER = "#FFB627"
SMOKE = "#6FA89B"
DANGER = "#E63946"
ASH_VIOLET = "#9C8AA5"
TEXT = "#F5EFE6"
TEXT_MUTED = "#B8AC9C"
BORDER = "#3A2E22"

MODEL_COLORS = {
    "SVM": EMBER,
    "Random Forest": AMBER,
    "ANN": SMOKE,
    "CNN": ASH_VIOLET,
}

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, .stApp {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {TEXT} !important;
        letter-spacing: -0.01em;
    }}

    [data-testid="stSidebar"] {{
        background-color: {BG_ELEV} !important;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        color: {EMBER} !important;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

    .stButton > button {{
        background: linear-gradient(135deg, {EMBER}, {AMBER});
        color: #15110D;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        padding: 0.6rem 1.6rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(255, 122, 51, 0.35);
    }}

    [data-testid="stExpander"] {{
        background-color: {BG_ELEV};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}

    .fc-hero {{
        padding: 1.6rem 0 0.6rem 0;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 1.4rem;
    }}
    .fc-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        color: {EMBER};
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        text-transform: uppercase;
    }}
    .fc-card {{
        background: {BG_ELEV};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .fc-badge-major {{
        display:inline-block; padding: 0.55rem 1.2rem; border-radius: 999px;
        background: rgba(230,57,70,0.15); color: {DANGER};
        border: 1px solid {DANGER}; font-family:'Space Grotesk',sans-serif; font-weight:600;
        font-size: 1.05rem;
    }}
    .fc-badge-safe {{
        display:inline-block; padding: 0.55rem 1.2rem; border-radius: 999px;
        background: rgba(111,168,155,0.15); color: {SMOKE};
        border: 1px solid {SMOKE}; font-family:'Space Grotesk',sans-serif; font-weight:600;
        font-size: 1.05rem;
    }}
    .fc-caption {{ color: {TEXT_MUTED}; font-size: 0.88rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def plotly_theme(fig, height=380):
    fig.update_layout(
        paper_bgcolor=BG_ELEV,
        plot_bgcolor=BG_ELEV,
        font=dict(family="Inter", color=TEXT),
        height=height,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ---------------------------------------------------------------------------
# Load data + train models (cached — sirf pehli baar chalta hai)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔥 Dataset load ho raha hai aur models train ho rahe hain...")
def load_pipeline():
    return backend.run_pipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(
        "⚠️ Dataset nahi mili.\n\n"
        f"{e}\n\n"
        "'California_Fire_Incidents.csv' ko is repo ke root folder me "
        "backend.py / app.py ke sath upload karein."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Pipeline load karte waqt error aaya: {e}")
    st.stop()

models = pipeline["models"]
results = pipeline["results"]
df = pipeline["df"]
top_counties = pipeline["top_counties"]
feature_columns = pipeline["feature_columns"]
scaler = pipeline["scaler"]
available_model_names = list(models.keys())

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔥 Wildfire Risk")
    st.markdown(
        "<span class='fc-caption'>California Fire Incidents — ML risk engine</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Model Comparison", "🔮 Predict Risk"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"<span class='fc-caption'>Models loaded: {', '.join(available_model_names)}</span>",
        unsafe_allow_html=True,
    )
    if not pipeline["tensorflow_available"]:
        st.markdown(
            "<span class='fc-caption'>ℹ️ CNN skip ho gaya — TensorFlow is "
            "environment me available nahi. SVM, Random Forest, aur ANN "
            "normally kaam kar rahe hain.</span>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class='fc-hero'>
        <div class='fc-eyebrow'>ML CLASSIFICATION SYSTEM</div>
        <h1 style='margin:0.2rem 0 0.2rem 0;'>Wildfire Major-Incident Risk Predictor</h1>
        <p class='fc-caption' style='font-size:1rem;'>
            California Fire Incidents dataset par train kiye gaye SVM, Random Forest,
            ANN aur CNN models — kisi fire incident ke Major Incident ban'ne ka risk predict karte hain.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Incidents", f"{len(df):,}")
    c2.metric("Total Acres Burned", f"{df['AcresBurned'].sum():,.0f}")
    major_pct = (df["MajorIncident"].sum() / len(df)) * 100
    c3.metric("Major Incident Rate", f"{major_pct:.1f}%")
    c4.metric("Counties Covered", f"{df['Counties'].nunique()}")

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("#### 📍 Incident Locations")
        map_df = df[["Latitude", "Longitude"]].dropna()
        st.map(map_df, color=EMBER, size=4000)

    with col_right:
        st.markdown("#### 🗺️ Top Counties by Incident Count")
        top10 = df["Counties"].str.split(",").str[0].str.strip().value_counts().head(10)
        fig = px.bar(
            x=top10.values,
            y=top10.index,
            orientation="h",
            labels={"x": "Incidents", "y": ""},
            color_discrete_sequence=[EMBER],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plotly_theme(fig, height=400), use_container_width=True)

    st.markdown("#### 📋 Sample Records")
    preview_cols = [
        "Name", "Counties", "Started", "AcresBurned",
        "PercentContained", "MajorIncident", "Status",
    ]
    st.dataframe(df[preview_cols].head(15), use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE 2: MODEL COMPARISON
# ===========================================================================
elif page == "📊 Model Comparison":
    st.markdown("#### ⚖️ Performance Comparison")

    metrics_df = pd.DataFrame(
        {
            name: {
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1"],
            }
            for name, m in results.items()
        }
    ).T

    best_model = metrics_df["F1-Score"].idxmax()

    fig = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                text=[f"{v:.2f}" for v in metrics_df[metric]],
                textposition="outside",
            )
        )
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1], title="Score"))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(plotly_theme(fig, height=420), use_container_width=True)

    st.markdown(
        f"<div class='fc-card'>🏆 <b>Best model (by F1-Score): "
        f"<span style='color:{EMBER}'>{best_model}</span></b> — F1 = "
        f"{metrics_df.loc[best_model, 'F1-Score']:.3f}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧮 Metrics Table")
    st.dataframe(
        metrics_df.style.format("{:.4f}").background_gradient(
            cmap="Oranges", vmin=0, vmax=1
        ),
        use_container_width=True,
    )

    st.markdown("#### 🔢 Confusion Matrices")
    cm_cols = st.columns(len(results))
    labels = ["Not Major", "Major"]
    for col, (name, m) in zip(cm_cols, results.items()):
        with col:
            cm = np.array(m["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=labels,
                y=labels,
                color_continuous_scale=[[0, BG_ELEV_2], [1, EMBER]],
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig_cm.update_layout(coloraxis_showscale=False, title=name)
            st.plotly_chart(plotly_theme(fig_cm, height=320), use_container_width=True)

    st.markdown(
        "<p class='fc-caption'>Accuracy = sahi predictions ka percentage · "
        "Precision = jin ko 'Major' predict kiya un mein se kitne sach me Major thay · "
        "Recall = actual Major incidents mein se kitne pakray gaye · "
        "F1-Score = Precision aur Recall ka balance.</p>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE 3: PREDICT
# ===========================================================================
elif page == "🔮 Predict Risk":
    st.markdown(
        "<p class='fc-caption'>Naye / hypothetical fire incident ki details "
        "daalein aur model se predict karwayein ke ye Major Incident ban sakta hai ya nahi.</p>",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown("##### 📌 Incident Details")
        f1, f2 = st.columns(2)
        county = f1.selectbox("County", top_counties + ["Other"])
        month_name = f2.selectbox("Month", MONTH_NAMES, index=6)
        month = MONTH_NAMES.index(month_name) + 1

        f3, f4 = st.columns(2)
        latitude = f3.number_input("Latitude", value=37.5, format="%.4f")
        longitude = f4.number_input("Longitude", value=-119.5, format="%.4f")

        acres = st.number_input("Acres Burned (so far)", min_value=0.0, value=300.0, step=10.0)
        contained = st.slider("Percent Contained (%)", 0, 100, 20)

        with st.expander("⚙️ Resources Deployed & Impact (optional)"):
            r1, r2, r3 = st.columns(3)
            crews = r1.number_input("Crews", min_value=0, value=5)
            engines = r2.number_input("Engines", min_value=0, value=3)
            dozers = r3.number_input("Dozers", min_value=0, value=0)
            r4, r5, r6 = st.columns(3)
            helicopters = r4.number_input("Helicopters", min_value=0, value=1)
            air_tankers = r5.number_input("Air Tankers", min_value=0, value=0)
            water_tenders = r6.number_input("Water Tenders", min_value=0, value=1)
            personnel = st.number_input("Personnel Involved", min_value=0, value=80)

            i1, i2, i3 = st.columns(3)
            struct_threat = i1.number_input("Structures Threatened", min_value=0, value=0)
            struct_destroyed = i2.number_input("Structures Destroyed", min_value=0, value=0)
            struct_damaged = i3.number_input("Structures Damaged", min_value=0, value=0)
            i4, i5 = st.columns(2)
            injuries = i4.number_input("Injuries", min_value=0, value=0)
            fatalities = i5.number_input("Fatalities", min_value=0, value=0)

        model_choice = st.selectbox("Model", available_model_names)
        predict_clicked = st.button("🔥 Predict Risk", use_container_width=True)

    with col_result:
        st.markdown("##### 🎯 Result")
        if predict_clicked:
            raw_input = {
                "Latitude": latitude,
                "Longitude": longitude,
                "AcresBurned": acres,
                "PercentContained": contained,
                "CrewsInvolved": crews,
                "Engines": engines,
                "Dozers": dozers,
                "Helicopters": helicopters,
                "AirTankers": air_tankers,
                "WaterTenders": water_tenders,
                "PersonnelInvolved": personnel,
                "StructuresThreatened": struct_threat,
                "StructuresDestroyed": struct_destroyed,
                "StructuresDamaged": struct_damaged,
                "Injuries": injuries,
                "Fatalities": fatalities,
                "Month": month,
                "County": county,
            }
            label, prob = backend.predict_incident(
                model_choice, models, scaler, raw_input, top_counties, feature_columns
            )

            if label == 1:
                st.markdown(
                    "<span class='fc-badge-major'>🔥 MAJOR INCIDENT — High Risk</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='fc-badge-safe'>✅ Not Major — Lower Risk</span>",
                    unsafe_allow_html=True,
                )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": TEXT}},
                    title={"text": "Major Incident Probability", "font": {"color": TEXT_MUTED, "size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
                        "bar": {"color": EMBER},
                        "bgcolor": BG_ELEV_2,
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(111,168,155,0.25)"},
                            {"range": [40, 70], "color": "rgba(255,182,39,0.25)"},
                            {"range": [70, 100], "color": "rgba(230,57,70,0.25)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                paper_bgcolor=BG_ELEV,
                font=dict(family="Inter", color=TEXT),
                height=300,
                margin=dict(l=20, r=20, t=50, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(
                f"<span class='fc-caption'>Model used: <b>{model_choice}</b></span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='fc-card'><span class='fc-caption'>Form fill karke "
                "'Predict Risk' button dabayein — result yahan dikhega.</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<p class='fc-caption' style='margin-top:1.5rem;'>ℹ️ Ye model CalFire ke "
        "historical incident records par train hua hai (real-time weather/ignition "
        "sensors par nahi) — isay ek demo / decision-support tool ki tarah samjhein, "
        "official fire-risk assessment ka replacement nahi.</p>",
        unsafe_allow_html=True,
    )
