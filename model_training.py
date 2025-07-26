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
from utils import bmi_category  # ✅ Import the function from utils

# 1) Load data
df = pd.read_csv("diabetes.csv")

# Replace 0s with NaNs in columns where 0 is invalid
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# Add BMI category using imported function
df['BMI_category'] = df['BMI'].apply(bmi_category)

# Create Age group column
bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Define features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Define preprocessing for numeric and categorical features
numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features = ['BMI_category', 'Age_group']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', dtype=int))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train multiple models
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

# Save best pipeline
Path("models").mkdir(exist_ok=True, parents=True)
joblib.dump(best_pipe, "models/best_pipeline.pkl")
print("✅ Saved best model to models/best_pipeline.pkl")
