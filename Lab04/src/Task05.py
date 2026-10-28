import os
import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)

# ==========================================
# 1. Custom Transformer for Clipping Outliers (from Task 4)
# ==========================================
class ClipOutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X, y=None):
        self.lower_bounds = np.percentile(X, 1, axis=0)
        self.upper_bounds = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds, self.upper_bounds)

# ==========================================
# 2. Dataset Ingestion & Split
# ==========================================
print("Fetching UCI Adult dataset...")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(repo_root, 'data')
os.makedirs(data_dir, exist_ok=True)

adult_data = fetch_openml(
    name='adult',
    version=2,
    as_frame=True,
    data_home=data_dir,
    parser='auto'
)

X = adult_data.data
y = adult_data.target

# Stratified split BEFORE any cleaning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Drop duplicates from train only
train_df = pd.concat([X_train, y_train], axis=1)
duplicated_mask = train_df.duplicated()
X_train_clean = X_train[~duplicated_mask].reset_index(drop=True)
y_train_clean = y_train[~duplicated_mask].reset_index(drop=True)

# Encode target to binary (1 if '>50K', else 0) for reliable F1 scoring
y_train_bin = (y_train_clean == '>50K').astype(int)
y_test_bin = (y_test == '>50K').astype(int)

print(f"Clean train -> {len(X_train_clean)}, Test (untouched) -> {len(X_test)}")

# ==========================================
# 3. Baseline 1: WITHOUT Derived Features
# ==========================================
print("\n--- Baseline 1: WITHOUT Derived Features ---")
numeric_features_1 = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', ClipOutliers())
])

# FIX: Added OneHotEncoder to convert strings to numbers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_1 = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features_1),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X_train_proc_1 = preprocessor_1.fit_transform(X_train_clean)
X_test_proc_1 = preprocessor_1.transform(X_test)

lr_1 = LogisticRegression(max_iter=1000, random_state=42)
lr_1.fit(X_train_proc_1, y_train_bin)
y_pred_1 = lr_1.predict(X_test_proc_1)

acc_1 = accuracy_score(y_test_bin, y_pred_1)
f1_1 = f1_score(y_test_bin, y_pred_1)

print(f"Accuracy: {acc_1:.4f}")
print(f"F1 Score:  {f1_1:.4f}")

# ==========================================
# 4. Baseline 2: WITH Derived Features
# ==========================================
print("\n--- Baseline 2: WITH Derived Features ---")
# Create copies to avoid SettingWithCopyWarning
X_train_eng = X_train_clean.copy()
X_test_eng = X_test.copy()

# Engineer features
X_train_eng['capital_net'] = X_train_eng['capital-gain'] - X_train_eng['capital-loss']
X_train_eng['hours_per_age'] = X_train_eng['hours-per-week'] / (X_train_eng['age'] - 16)

X_test_eng['capital_net'] = X_test_eng['capital-gain'] - X_test_eng['capital-loss']
X_test_eng['hours_per_age'] = X_test_eng['hours-per-week'] / (X_test_eng['age'] - 16)

# Update numeric features list to include the new derived features
numeric_features_2 = numeric_features_1 + ['capital_net', 'hours_per_age']

preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features_2),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X_train_proc_2 = preprocessor_2.fit_transform(X_train_eng)
X_test_proc_2 = preprocessor_2.transform(X_test_eng)

lr_2 = LogisticRegression(max_iter=1000, random_state=42)
lr_2.fit(X_train_proc_2, y_train_bin)
y_pred_2 = lr_2.predict(X_test_proc_2)

acc_2 = accuracy_score(y_test_bin, y_pred_2)
f1_2 = f1_score(y_test_bin, y_pred_2)

print(f"Accuracy: {acc_2:.4f}")
print(f"F1 Score:  {f1_2:.4f}")

# ==========================================
# 5. Save Results
# ==========================================
results_dir = os.path.join(repo_root, 'lab04', 'results')
os.makedirs(results_dir, exist_ok=True)

results_text = (
    "=== FEATURE ENGINEERING RESULTS ===\n\n"
    "Baseline 1 (Without Derived Features):\n"
    f"  - Accuracy: {acc_1:.4f}\n"
    f"  - F1 Score: {f1_1:.4f}\n\n"
    "Baseline 2 (With Derived Features: capital_net, hours_per_age):\n"
    f"  - Accuracy: {acc_2:.4f}\n"
    f"  - F1 Score: {f1_2:.4f}\n"
)

print("\n" + results_text)

with open(os.path.join(results_dir, 'feature_engineering_results.txt'), 'w') as f:
    f.write(results_text)

print(" Task 5 Complete: Results saved to results/feature_engineering_results.txt")