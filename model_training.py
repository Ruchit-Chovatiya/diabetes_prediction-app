import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# 1) Load data
df = pd.read_csv("diabetes.csv")

# Replace invalid 0s with NaN
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# Inline BMI category (no external function)
df['BMI_category'] = pd.cut(
    df['BMI'],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=['underweight', 'Normal', 'overweight', 'obese'],
    right=False
)

# Age group
bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 2) Preprocessing
numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features = ['BMI_category', 'Age_group']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(drop='first', dtype=int))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train multiple models
models = {
    "LogisticRegression": LogisticRegression(solver='liblinear'),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}

best_model_name = None
best_score = 0.0
best_pipe = None

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_pipe = pipe

print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")

# 5) Save the best pipeline
joblib.dump(best_pipe, "best_pipeline.pkl")
print("✅ Saved pipeline to best_pipeline.pkl")