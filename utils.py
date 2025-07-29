# utils.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['BMI_category'] = pd.cut(
            X['BMI'], bins=[0, 18.5, 25, 30, np.inf],
            labels=['underweight', 'Normal', 'overweight', 'obese'],
            right=False
        )
        X['Age_group'] = pd.cut(
            X['Age'], bins=[20, 30, 40, 50, 60, 100],
            labels=['20-30', '30-40', '40-50', '50-60', '60+'],
            right=False
        )
        return X
