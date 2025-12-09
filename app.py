# app.py  –  Diabetes Risk Prediction Dashboard (XGBoost + Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL + PREPROCESSOR
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_diabetes_model.pkl")      # best_xgb
    preprocessor = joblib.load("preprocessor.pkl")        # ColumnTransformer
    return model, preprocessor

model, preprocessor = load_artifacts()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧪 Diabetes Risk App")
st.sidebar.markdown(
    """
    This app uses a **trained XGBoost model** on ~100k patients  
    to estimate **diabetes risk probability**.

    **Risk Legend**  
    - 🟢 0–20% : Low risk  
    - 🟡 20–50% : Moderate risk  
    - 🔴 50%+ : High risk  

    > ⚠ This is a data science demo, *not* medical advice.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Abdul Qadir** with ❤️ & ML")

# =========================
# MAIN HEADER
# =========================
st.title("🩺 Diabetes Risk Prediction Dashboard")
st.caption("Gradient EDA + XGBoost • Healthcare Domain")

st.markdown("---")

# =========================
# INPUT FORM
# =========================
st.subheader("Enter Patient Details")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age (years)", min_value=0, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        smoking_history = st.selectbox(
            "Smoking History",
            ["never", "No Info", "current", "former", "ever", "not current"],
            index=0,
        )

    with col2:
        hypertension = st.selectbox("Hypertension", ["No (0)", "Yes (1)"])
        heart_disease = st.selectbox("Heart Disease", ["No (0)", "Yes (1)"])
        bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)

    with col3:
        hba1c = st.slider("HbA1c Level (%)", min_value=3.5, max_value=12.0, value=6.0, step=0.1)
        glucose = st.slider(
            "Blood Glucose Level (mg/dL)",
            min_value=70,
            max_value=300,
            value=140,
            step=1,
        )

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

# =========================
# PREDICTION
# =========================
if submitted:
    input_dict = {
        "age": age,
        "hypertension": 1 if "Yes" in hypertension else 0,
        "heart_disease": 1 if "Yes" in heart_disease else 0,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "gender": gender,
        "smoking_history": smoking_history,
    }

    input_df = pd.DataFrame([input_dict])

    X_proc = preprocessor.transform(input_df)

    prob = float(model.predict_proba(X_proc)[0, 1])
    pred = int(model.predict(X_proc)[0])

    if prob < 0.20:
        risk_label = "Low"
        risk_color = "green"
    elif prob < 0.50:
        risk_label = "Moderate"
        risk_color = "gold"
    else:
        risk_label = "High"
        risk_color = "red"

    # =========================
    # RESULT CARDS
    # =========================
    st.subheader("Prediction Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Diabetes Probability", f"{prob*100:.1f} %")

    with c2:
        st.metric("Risk Category", f"{risk_label}")

    with c3:
        st.metric("Model Decision", "Diabetic (1)" if pred == 1 else "Non-Diabetic (0)")

    # =========================
    # NICE GAUGE PLOT
    # =========================
    st.markdown("## 📊 Risk Gauge")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 20], "color": "#e0f7e9"},
                    {"range": [20, 50], "color": "#fff7d1"},
                    {"range": [50, 100], "color": "#ffe0e0"},
                ],
            },
            title={"text": "Estimated Diabetes Risk"},
        )
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    # =========================
    # FEATURE SNAPSHOT
    # =========================
    st.markdown("## 📌 Input Feature Snapshot")

    feat_df = pd.DataFrame(
        {
            "Feature": ["Age", "BMI", "HbA1c", "Glucose"],
            "Value": [age, bmi, hba1c, glucose],
        }
    )

    fig_bar = go.Figure(
        data=[
            go.Bar(
                x=feat_df["Feature"],
                y=feat_df["Value"],
                text=[f"{v:.1f}" for v in feat_df["Value"]],
                textposition="auto",
            )
        ]
    )
    fig_bar.update_layout(yaxis_title="Value", template="simple_white")

    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # RAW INPUT TABLE
    # =========================
    with st.expander("🔎 See Raw Input Data"):
        st.write(input_df)

else:
    st.info("👆 Fill the form above and click **Predict Diabetes Risk** to see results.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "⚠ Disclaimer: This app is for educational and demo purposes only and does not replace professional medical advice."
)



