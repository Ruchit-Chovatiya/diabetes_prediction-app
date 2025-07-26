import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Read data from CSV
df = pd.read_csv("diabetes.csv")

cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[cols_with_invalid_zeros] = imputer.fit_transform(df[cols_with_invalid_zeros])

# Feature engineering
def bmi_category(bmi):
    if bmi<18.5:
        return 'underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'overweight'
    else:
        return 'obese'
    
df['BMI_category'] = df['BMI'].apply(bmi_category)

bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']

df['Age_group'] = pd.cut(df['Age'], bins = bins, labels = labels, right = False)
# print(df.head())

df_encoded = pd.get_dummies(df, columns=['BMI_category', 'Age_group'], drop_first=True, dtype = int)
# print(df_encoded.head())

# Scaling
numeric_cols = df_encoded.drop(columns = ['Outcome']).select_dtypes(include='number').columns

scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

X = df_encoded.drop(columns = ['Outcome'], axis = 1)
y = df_encoded['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

# Models

# 1. LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
classification_report_lr = classification_report(y_test, y_pred_lr)

# print(accuracy_lr)
# print(classification_report_lr)

# 2. DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_report_dt = classification_report(y_test, y_pred_dt)

# print(accuracy_dt)
# print(classification_report_dt)

# 3. RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

# print(accuracy_rf)
# print(classification_report_rf)

# 4. KNeighborsClassifier

kn_model = KNeighborsClassifier()
kn_model.fit(X_train, y_train)
y_pred_kn = kn_model.predict(X_test)

accuracy_kn = accuracy_score(y_test, y_pred_kn)
classification_report_kn = classification_report(y_test, y_pred_kn)

# print(accuracy_kn)
# print(classification_report_kn)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameter: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_}")

best_lr = grid_search.best_estimator_
y_pred = best_lr.predict(X_test)