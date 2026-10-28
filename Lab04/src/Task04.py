import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin



random.seed(42)
np.random.seed(42)


class ClipOutliers(BaseEstimator, TransformerMixin):
    """Clips numeric features to the 1st and 99th percentiles fitted on the training data."""
    def __init__(self):
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X, y=None):
        # ColumnTransformer passes NumPy arrays, not DataFrames
        # np.percentile calculates the 1st and 99th percentiles for each column (axis=0)
        self.lower_bounds = np.percentile(X, 1, axis=0)
        self.upper_bounds = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        # np.clip efficiently bounds all values across all columns at once
        return np.clip(X, self.lower_bounds, self.upper_bounds)


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


print("\n--- Task 4: Leakage-Free Cleaning Pipeline ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)
print(f"Initial split -> Train: {len(X_train)}, Test: {len(X_test)}")



# Combine to find exact row duplicates across features AND target
train_df = pd.concat([X_train, y_train], axis=1)
initial_train_len = len(train_df)

# Find duplicated rows (returns a boolean mask)
duplicated_mask = train_df.duplicated()
duplicates_dropped = duplicated_mask.sum()

# Apply the mask to the original X and y
X_train_clean = X_train[~duplicated_mask].reset_index(drop=True)
y_train_clean = y_train[~duplicated_mask].reset_index(drop=True)

print(f"Dropped {duplicates_dropped} exact duplicate rows from training set.")
print(f"Clean train -> {len(X_train_clean)}, Test (untouched) -> {len(X_test)}")


numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Numeric pipeline: Median imputation -> Clip to 1st/99th percentile
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', ClipOutliers())
])

# Categorical pipeline: Mode (most_frequent) imputation
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


print("\nFitting preprocessor on training data only...")
preprocessor.fit(X_train_clean)

print("Transforming training data...")
X_train_processed = preprocessor.transform(X_train_clean)

print("Transforming test data (using train-derived parameters, NO LEAKAGE)...")
X_test_processed = preprocessor.transform(X_test)

# Convert back to DataFrame for easier inspection
X_train_df = pd.DataFrame(X_train_processed, columns=numeric_features + categorical_features)
X_test_df = pd.DataFrame(X_test_processed, columns=numeric_features + categorical_features)

print("\nSample of processed training data (first 2 rows):")
print(X_train_df.head(2))


results_dir = os.path.join(repo_root, 'lab04', 'results')
os.makedirs(results_dir, exist_ok=True)
preprocessor_path = os.path.join(results_dir, 'preprocessor.joblib')

joblib.dump(preprocessor, preprocessor_path)
print(f"\n Task 4 Complete: Fitted preprocessor saved to {preprocessor_path}")