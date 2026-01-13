import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Title
st.title("❤️ Heart Disease Prediction System")
st.markdown("### AI-Powered Risk Assessment Tool (Educational Use Only)")

# Sidebar for inputs
with st.sidebar:
    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 52)
        sex = st.selectbox("Gender", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 125)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 212)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 168)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

    slope = st.selectbox("ST Slope", [1, 2, 3])
    ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 2)
    thal = st.selectbox("Thalassemia", [3, 6, 7])

    # Convert categorical inputs
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    features = [
        age, sex_val, cp, trestbps, chol, fbs_val, restecg,
        thalach, exang_val, oldpeak, slope, ca, thal
    ]

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

# Main prediction logic
if predict_btn:
    try:
        model = joblib.load("model.joblib")

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1] * 100

        st.header("📊 Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Diagnosis",
                      "Heart Disease Detected" if prediction == 1 else "No Heart Disease")

        with c2:
            st.metric("Probability", f"{probability:.1f}%")

        with c3:
            if probability < 30:
                st.metric("Risk Level", "LOW 🟢")
            elif probability < 70:
                st.metric("Risk Level", "MEDIUM 🟡")
            else:
                st.metric("Risk Level", "HIGH 🔴")

        st.subheader("📋 Medical Recommendation")

        if probability < 30:
            st.success("Low risk detected. Maintain a healthy lifestyle and regular checkups.")
        elif probability < 70:
            st.warning("Moderate risk detected. Medical consultation is recommended.")
        else:
            st.error("High risk detected. Immediate medical evaluation is advised.")

        st.subheader("🔍 Key Risk Factors")
        risks = []
        if age > 55: risks.append("Advanced age")
        if trestbps > 140: risks.append("High blood pressure")
        if chol > 240: risks.append("High cholesterol")
        if exang_val == 1: risks.append("Exercise-induced angina")
        if oldpeak > 2: risks.append("High ST depression")
        if ca > 1: risks.append("Multiple blocked vessels")

        if risks:
            for r in risks:
                st.write(f"• {r}")
        else:
            st.info("No major risk factors detected.")

        # Patient summary table
        try:
            if probability < 30:
                risk_level = "LOW"
            elif probability < 70:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            summary = pd.DataFrame({
                "Age": [age],
                "Gender": [sex],
                "Chest Pain Type": [cp],
                "Resting BP": [trestbps],
                "Cholesterol": [chol],
                "Fasting BS": ["Yes" if fbs_val==1 else "No"],
                "Resting ECG": [restecg],
                "Max HR": [thalach],
                "Exercise Angina": ["Yes" if exang_val==1 else "No"],
                "ST Depression": [oldpeak],
                "ST Slope": [slope],
                "Major Vessels": [ca],
                "Thalassemia": [thal],
                "Prediction": ["Heart Disease" if prediction==1 else "No Heart Disease"],
                "Probability (%)": [f"{probability:.1f}"],
                "Risk Level": [risk_level]
            })

            st.subheader("🧾 Patient Summary")
            st.table(summary)
        except Exception:
            # If something goes wrong building the table, silently skip it
            pass

    except Exception as e:
        st.error(f"Error loading model: {e}")

else:
    st.info("👈 Enter patient data from the sidebar and click **Predict Risk**")

    st.markdown("---")
    st.markdown("""
    **Model Information**
    - Algorithm: Logistic Regression  
    - Accuracy: 91.67%  
    - Dataset: Cleveland Heart Disease (Kaggle / UCI)  
    - Features: 13 clinical attributes  

    **Disclaimer:**  
    This system is developed for academic purposes only and must not be used
    for real medical diagnosis.
    """)

st.markdown("---")
st.caption(
    "Heart Disease Prediction System | AI Mini Project | Dire Dawa University | "
    "Logistic Regression | Accuracy: 91.67%"
)
