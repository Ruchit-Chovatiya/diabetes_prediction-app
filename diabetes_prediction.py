"""
Diabetes Prediction using Machine Learning
Author: Chovatiya Ruchit
"""

# 1. Import Libraries
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load Dataset
df = pd.read_csv("diabetes.csv")

# 3. Handle Invalid Zero Values
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# Impute missing values with column mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[cols_with_invalid_zeros] = imputer.fit_transform(df[cols_with_invalid_zeros])

# 4. Feature Engineering

# BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMI_category'] = df['BMI'].apply(bmi_category)

# Age Group Binning
bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# One-hot encoding categorical features
df_encoded = pd.get_dummies(df, columns=['BMI_category', 'Age_group'], drop_first=True, dtype=int)

# 5. Feature Scaling
numeric_cols = df_encoded.drop(columns=['Outcome']).select_dtypes(include='number').columns
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# 6. Feature-Target Split
X = df_encoded.drop(columns=['Outcome'])
y = df_encoded['Outcome']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Model Training & Evaluation

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

# K-Nearest Neighbors
kn_model = KNeighborsClassifier()
kn_model.fit(X_train, y_train)
y_pred_kn = kn_model.predict(X_test)
acc_kn = accuracy_score(y_test, y_pred_kn)
report_kn = classification_report(y_test, y_pred_kn)

# 9. Hyperparameter Tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Parameters & Final Model
print("Best Parameters Found:")
print(grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Final prediction using best model
best_lr = grid_search.best_estimator_
final_pred = best_lr.predict(X_test)
final_accuracy = accuracy_score(y_test, final_pred)

print(f"\nFinal Accuracy on Test Set: {final_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, final_pred))

import joblib

# Save model
joblib.dump(best_lr, 'best_logistic_model.pkl')

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
