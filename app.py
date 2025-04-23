import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
with open("credit_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

st.title("Credit Risk Prediction App")
st.write("This app predicts whether a customer is a good or bad credit risk.")

# Input form
with st.form("credit_form"):
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 18, 100, 30)
    job = st.selectbox("Job", [0, 1, 2, 3])  # Assuming 4 types of jobs
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
    checking = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
    duration = st.slider("Duration (months)", 4, 72, 24)
    purpose = st.selectbox("Purpose", ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'repairs', 'vacation/others', 'unknown'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Sex": sex,
        "Age": age,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Duration": duration,
        "Purpose": purpose
    }])

    # Predict
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][prediction]

    # Output
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Good Credit Risk (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Bad Credit Risk (Confidence: {prob:.2f})")
