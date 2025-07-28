import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Predict the likelihood of diabetes based on basic health parameters.")

# Load the trained pipeline
MODEL_PATH = Path("models/best_pipeline.pkl")
pipe = joblib.load(MODEL_PATH)

st.sidebar.header("Input Patient Data")

def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    Glucose = st.sidebar.number_input('Glucose', 0, 300, 100)
    BloodPressure = st.sidebar.number_input('Blood Pressure', 0, 200, 70)
    SkinThickness = st.sidebar.number_input('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.number_input('Insulin', 0, 1000, 80)
    BMI = st.sidebar.number_input('BMI', 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    Age = st.sidebar.number_input('Age', 20, 100, 30)

    input_df = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    return input_df

input_df = user_input_features()

# Feature engineering (same as in FeatureEngineer from training)
input_df['BMI_category'] = pd.cut(
    input_df['BMI'],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=['underweight', 'Normal', 'overweight', 'obese'],
    right=False
)

input_df['Age_group'] = pd.cut(
    input_df['Age'],
    bins=[20, 30, 40, 50, 60, 100],
    labels=['20-30', '30-40', '40-50', '50-60', '60+'],
    right=False
)

# Prediction
if st.button("Predict"):
    prediction = pipe.predict(input_df)[0]
    if hasattr(pipe.named_steps['classifier'], "predict_proba"):
        proba = pipe.predict_proba(input_df)[0][1]
    else:
        proba = None

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Likely Diabetic{' - Probability: {:.2f}'.format(proba) if proba else ''}")
    else:
        st.success(f"✅ Unlikely Diabetic{' - Probability: {:.2f}'.format(proba) if proba else ''}")
