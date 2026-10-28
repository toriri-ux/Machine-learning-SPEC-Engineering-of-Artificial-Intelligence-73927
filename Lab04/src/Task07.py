import os
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class ClipOutliers(BaseEstimator, TransformerMixin):
    """Clips numeric features to the 1st and 99th percentiles fitted on the training data."""
    def __init__(self):
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X, y=None):
        self.lower_bounds = np.percentile(X, 1, axis=0)
        self.upper_bounds = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds, self.upper_bounds)


print("Loading data and applying saved preprocessor...")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(repo_root, 'data')

adult_data = fetch_openml(name='adult', version=2, as_frame=True, data_home=data_dir, parser='auto')
X = adult_data.data
y = adult_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# Drop duplicates from train only (same logic as Task 4)
train_df = pd.concat([X_train, y_train], axis=1)
duplicated_mask = train_df.duplicated()
X_train_clean = X_train[~duplicated_mask].reset_index(drop=True)

# Load saved preprocessor from Task 4
preprocessor_path = os.path.join(repo_root, 'lab04', 'results', 'preprocessor.joblib')
preprocessor = joblib.load(preprocessor_path)

# Apply the preprocessor
X_train_proc = preprocessor.transform(X_train_clean)
X_test_proc = preprocessor.transform(X_test)



def compute_sha256(data_array):
    # Ensure the array is contiguous in memory for consistent hashing
    return hashlib.sha256(np.ascontiguousarray(data_array).tobytes()).hexdigest()

hash_train = compute_sha256(X_train_proc)
hash_test = compute_sha256(X_test_proc)

print(f"X_train_processed SHA256: {hash_train}")
print(f"X_test_processed SHA256:  {hash_test}")


yaml_content = f"""dataset: UCI Adult
source: OpenML (name='adult', version=2)
random_seed: 42
split_ratio:
  train: 0.70
  test: 0.30
preprocessing:
  numeric_imputation: median
  categorical_imputation: most_frequent
  outlier_clipping: 1st and 99th percentile
  duplicate_handling: dropped from train only
  categorical_encoding: one-hot encoded
data_hashes:
  X_train_processed: {hash_train}
  X_test_processed: {hash_test}
"""

results_dir = os.path.join(repo_root, 'lab04', 'results')
os.makedirs(results_dir, exist_ok=True)
config_path = os.path.join(results_dir, 'config.yaml')
with open(config_path, 'w') as f:
    f.write(yaml_content)

print(f"Config saved to {config_path}")