# 🩺 Diabetes Prediction App

A simple and interactive Streamlit web application to predict the likelihood of diabetes in patients based on key health metrics using a trained machine learning pipeline.

---

## 🚀 Features

- Takes user input for key health parameters
- Displays prediction with probability
- Handles feature engineering like BMI category and Age group
- Uses a trained pipeline (preprocessing + model) with `joblib`

---

## 🧪 Technologies Used

- Python 🐍
- Streamlit 🌐
- scikit-learn 🤖
- Pandas & NumPy 📊
- Joblib 💾

---

## 📊 Input Features

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Additionally, it creates:
- `BMI_category`: underweight, normal, overweight, obese
- `Age_group`: 20–30, 30–40, ...

---


---

## ▶️ How to Run the App Locally

### 1. Clone the repo

```bash
git clone https://github.com/Ruchit-Chovatiya/diabetes-prediction-app.git
cd diabetes-prediction-app
