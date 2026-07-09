import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🩺",
    layout="centered"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0E1613 0%, #131F1B 100%);
    color: #E7EDE9;
}

#MainMenu, footer {visibility: hidden;}

/* Hero */
.eyebrow {
    color: #6FBF9E;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Source Serif 4', serif;
    font-size: 2.5rem;
    font-weight: 600;
    color: #EFF5F1;
    margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
}
.hero-sub {
    color: #93A39B;
    font-size: 1rem;
    margin-bottom: 1.8rem;
    line-height: 1.5;
}

/* Guide box */
.guide-box {
    background: #16211D;
    border: 1px solid #223028;
    border-left: 4px solid #6FBF9E;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 26px;
    font-size: 0.92rem;
    color: #B9C6BF;
    line-height: 1.65;
}
.guide-box b { color: #EFF5F1; }

/* Input card */
.input-card {
    background: #16211D;
    border: 1px solid #223028;
    border-radius: 14px;
    padding: 26px 26px 6px 26px;
    margin-bottom: 22px;
}
.section-label {
    color: #6FBF9E;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

label, .stSelectbox label, .stNumberInput label, .stRadio label {
    color: #C7D3CC !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.stSelectbox > div > div, .stNumberInput input {
    background-color: #101A16 !important;
    border: 1px solid #29392F !important;
    border-radius: 8px !important;
    color: #E7EDE9 !important;
}

/* Model selector highlight */
.model-pill {
    display: inline-block;
    background: #1B2921;
    border: 1px solid #2C4136;
    color: #6FBF9E;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 6px;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #6FBF9E 0%, #4C9878 100%);
    color: #0E1613;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(111, 191, 158, 0.3);
    color: #0E1613;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #192720 0%, #16211D 100%);
    border: 1px solid #2C4136;
    border-left: 5px solid #6FBF9E;
    border-radius: 14px;
    padding: 28px 30px;
    margin-top: 22px;
    text-align: center;
}
.result-label {
    color: #93A39B;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.result-amount {
    font-family: 'Source Serif 4', serif;
    font-size: 2.6rem;
    color: #EFF5F1;
    font-weight: 600;
}
.result-model {
    color: #6FBF9E;
    font-size: 0.88rem;
    margin-top: 8px;
}

/* Footer */
.dev-footer {
    text-align: center;
    color: #4A5A50;
    font-size: 0.82rem;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #1D2A23;
}
.dev-footer span {
    color: #6FBF9E;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS + SCALER
# ============================================================
models = {
    "Random Forest": joblib.load("rf_model.pkl"),
    "Linear Regression": joblib.load("linear_model.pkl"),
    "Decision Tree": joblib.load("dtree_model.pkl"),
}
scaler = joblib.load("insurance_scaler.pkl")

# Approximate R2 scores from training, shown to help users pick a model
model_scores = {
    "Random Forest": "R² ≈ 0.86 — most accurate on this dataset",
    "Linear Regression": "R² ≈ 0.78 — fast, simple, easy to explain",
    "Decision Tree": "R² ≈ 0.75 — simple rule-based logic",
}

# ============================================================
# HERO
# ============================================================
st.markdown('<div class="eyebrow">Health Insurance Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Estimate a person\'s annual medical insurance charges using machine learning trained on real insurance records.</div>',
    unsafe_allow_html=True
)

# ============================================================
# USAGE GUIDE
# ============================================================
st.markdown("""
<div class="guide-box">
<b>How to use this tool</b><br>
1. Choose which model you'd like predictions from (Random Forest is the most accurate).<br>
2. Fill in the person's age, sex, BMI, number of children, smoking status, and region.<br>
3. Click <b>Predict Cost</b> to see the estimated annual insurance charge in US dollars.
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL SELECTOR
# ============================================================
st.markdown('<div class="section-label">Choose a Model</div>', unsafe_allow_html=True)
selected_model_name = st.selectbox(
    "Model",
    list(models.keys()),
    label_visibility="collapsed"
)
st.caption(model_scores[selected_model_name])

# ============================================================
# INPUT FORM
# ============================================================
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Customer Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("Predict Cost")

# ============================================================
# PREDICTION
# ============================================================
if predict:
    sex_num = 1 if sex == "Male" else 0
    smoker_num = 1 if smoker == "Yes" else 0

    # One-hot encode region (must match training column order)
    region_sw = 1 if region == "Southwest" else 0
    region_se = 1 if region == "Southeast" else 0
    region_nw = 1 if region == "Northwest" else 0
    region_ne = 1 if region == "Northeast" else 0

    input_data = np.array([[age, sex_num, bmi, children, smoker_num,
                             region_ne, region_nw, region_se, region_sw]])
    input_scaled = scaler.transform(input_data)

    model = models[selected_model_name]
    prediction = model.predict(input_scaled)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Annual Cost</div>
        <div class="result-amount">${prediction:,.2f}</div>
        <div class="result-model">Predicted using {selected_model_name}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="dev-footer">Built by <span>Aurang Zeb</span> — Machine Learning Project</div>',
    unsafe_allow_html=True
)
