"""
Medical Insurance Cost Predictor
--------------------------------
Streamlit front-end for an XGBRegressor saved as `xgboost.pkl`.

Model inputs (in this exact order):
    age, sex, bmi, children, smoker, region

The model was trained on integer-encoded categories. The mappings below follow
the standard alphabetical LabelEncoder order. If you trained with a different
encoding, just edit the three dictionaries.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "xgboost.pkl"

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

SEX_MAP = {"Female": 0, "Male": 1}
SMOKER_MAP = {"No": 0, "Yes": 1}
REGION_MAP = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
        border-radius: 22px;
        padding: 2.2rem 2.4rem;
        color: #fff;
        box-shadow: 0 12px 32px rgba(79, 70, 229, 0.30);
        margin-bottom: 1.6rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color: #fff;}
    .hero p {margin: .5rem 0 0; font-size: 1.05rem; opacity: .92; color: #fff;}

    .result-card {
        background: linear-gradient(135deg, rgba(79,70,229,.14), rgba(6,182,212,.14));
        border: 1px solid rgba(124,58,237,.35);
        border-radius: 20px;
        padding: 1.8rem 1.6rem;
        text-align: center;
    }
    .result-label {font-size: .9rem; letter-spacing: .12em; text-transform: uppercase; opacity: .7;}
    .result-value {
        font-size: 3.4rem; font-weight: 800; line-height: 1.15; margin: .3rem 0;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .result-sub {font-size: .95rem; opacity: .75;}

    .metric-card {
        background: rgba(127,127,127,.08);
        border: 1px solid rgba(127,127,127,.22);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .metric-title {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}
    .metric-value {font-size: 1.5rem; font-weight: 700; margin-top: .15rem;}

    .bmi-track {
        position: relative; height: 12px; border-radius: 99px; margin: .6rem 0 .4rem;
        background: linear-gradient(90deg, #38bdf8 0%, #22c55e 30%, #facc15 55%, #f97316 75%, #ef4444 100%);
    }
    .bmi-dot {
        position: absolute; top: -5px; width: 22px; height: 22px; border-radius: 50%;
        background: #fff; border: 4px solid #4f46e5; transform: translateX(-50%);
        box-shadow: 0 2px 8px rgba(0,0,0,.3);
    }
    .bmi-scale {display: flex; justify-content: space-between; font-size: .72rem; opacity: .6;}

    .section-title {font-size: 1.25rem; font-weight: 700; margin: 1.8rem 0 .6rem;}
    .note {font-size: .82rem; opacity: .6; margin-top: 2rem; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def make_frame(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """Build a one-row DataFrame with the dtypes the model was trained on."""
    return pd.DataFrame(
        {
            "age": [int(age)],
            "sex": [int(sex)],
            "bmi": [float(bmi)],
            "children": [int(children)],
            "smoker": [int(smoker)],
            "region": [int(region)],
        }
    )[FEATURES]


def predict(model, **kwargs) -> float:
    return float(model.predict(make_frame(**kwargs))[0])


def bmi_category(bmi: float):
    if bmi < 18.5:
        return "Underweight", "#38bdf8"
    if bmi < 25:
        return "Healthy", "#22c55e"
    if bmi < 30:
        return "Overweight", "#eab308"
    return "Obese", "#ef4444"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Insurance Cost Predictor</h1>
        <p>Enter a few personal details and our XGBoost model will estimate the
        yearly medical insurance charges instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model file `xgboost.pkl` was not found. Place it in the same folder as "
        "`app.py` (and commit it to your repository)."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model: {exc}")
    st.info(
        "This usually means the `xgboost` / `scikit-learn` versions in "
        "`requirements.txt` differ from the ones used for training."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧾 Your details")
    age = st.slider("Age", 18, 100, 35)
    bmi = st.slider("BMI", 10.0, 55.0, 27.0, 0.1)
    children = st.slider("Children covered", 0, 10, 0)
    sex_label = st.radio("Sex", list(SEX_MAP), horizontal=True)
    smoker_label = st.radio("Smoker", list(SMOKER_MAP), horizontal=True)
    region_label = st.selectbox("Region", list(REGION_MAP))
    st.caption("Results update automatically as you change the inputs.")

profile = dict(
    age=age,
    sex=SEX_MAP[sex_label],
    bmi=bmi,
    children=children,
    smoker=SMOKER_MAP[smoker_label],
    region=REGION_MAP[region_label],
)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
estimate = max(predict(model, **profile), 0.0)
monthly = estimate / 12

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated yearly charges</div>
            <div class="result-value">${estimate:,.0f}</div>
            <div class="result-sub">≈ ${monthly:,.0f} per month</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    cat, color = bmi_category(bmi)
    pos = min(max((bmi - 10) / (55 - 10), 0), 1) * 100
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Body Mass Index</div>
            <div class="metric-value">{bmi:.1f}
                <span style="color:{color}; font-size:1rem;">· {cat}</span></div>
            <div class="bmi-track"><div class="bmi-dot" style="left:{pos}%"></div></div>
            <div class="bmi-scale"><span>10</span><span>18.5</span><span>25</span><span>30</span><span>55</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Quick summary tiles
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
tiles = [
    ("Age", f"{age} yrs"),
    ("Sex", sex_label),
    ("Children", str(children)),
    ("Smoker", smoker_label),
    ("Region", region_label),
]
for col, (title, value) in zip(st.columns(len(tiles)), tiles):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# What-if analysis
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">🔍 What-if analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Smoker vs Non-smoker", "Cost by age", "Cost by BMI"])

with tab1:
    non_smoker = max(predict(model, **{**profile, "smoker": 0}), 0.0)
    smoker = max(predict(model, **{**profile, "smoker": 1}), 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Non-smoker", f"${non_smoker:,.0f}")
    c2.metric("Smoker", f"${smoker:,.0f}")
    c3.metric("Difference", f"${smoker - non_smoker:,.0f}")
    st.bar_chart(
        pd.DataFrame({"Yearly charges ($)": [non_smoker, smoker]}, index=["Non-smoker", "Smoker"]),
        color="#7c3aed",
    )

with tab2:
    ages = list(range(18, 101, 2))
    costs = [max(predict(model, **{**profile, "age": a}), 0.0) for a in ages]
    st.line_chart(pd.DataFrame({"Yearly charges ($)": costs}, index=ages), color="#4f46e5")
    st.caption("Predicted charges as age changes, with your other details held constant.")

with tab3:
    bmis = [round(10 + i * 0.5, 1) for i in range(0, 91)]
    costs = [max(predict(model, **{**profile, "bmi": b}), 0.0) for b in bmis]
    st.line_chart(pd.DataFrame({"Yearly charges ($)": costs}, index=bmis), color="#06b6d4")
    st.caption("Predicted charges as BMI changes, with your other details held constant.")

# ----------------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------------
with st.expander("📊 What drives the model's predictions?"):
    try:
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(importances, horizontal=True, color="#7c3aed")
    except Exception:  # noqa: BLE001
        st.write("Feature importances are not available for this model.")

st.markdown(
    '<div class="note">This tool provides an estimate from a machine-learning model for '
    "educational purposes only. It is not an official insurance quote.</div>",
    unsafe_allow_html=True,
)
