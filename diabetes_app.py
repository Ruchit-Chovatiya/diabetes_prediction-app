import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import bmi_category

st.title("Diabetes Prediction App")
st.write("""
Predict the likelihood of diabetes from basic health parameters.
""")

# Load pipeline
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

    df = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })
    return df

input_df = user_input_features()

# Feature engineering to match training
input_df['BMI_category'] = input_df['BMI'].apply(bmi_category)
bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']
input_df['Age_group'] = pd.cut(input_df['Age'], bins=bins, labels=labels, right=False)

# Predict
prediction = pipe.predict(input_df)
if hasattr(pipe.named_steps['classifier'], "predict_proba"):
    probability = pipe.predict_proba(input_df)[0][1]
else:
    probability = None

st.subheader("Prediction")
if prediction[0] == 1:
    if probability is not None:
        st.error(f"Likely diabetic. Probability: {probability:.2f}")
    else:
        st.error(f"Likely diabetic.")
else:
    if probability is not None:
        st.success(f"Unlikely diabetic. Probability: {probability:.2f}")
    else:
        st.success(f"Unlikely diabetic.")
