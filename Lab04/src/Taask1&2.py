import os
import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import joblib

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)

# ==========================================
# 1. Dataset Ingestion (Tasks 1 & 2)
# ==========================================
print("Fetching UCI Adult dataset from OpenML...")

# Global rule: Data folder should be at repo root
# We go up two directories from lab04/src/ to reach the repo root, then into 'data'
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(repo_root, 'data')
os.makedirs(data_dir, exist_ok=True)

# Fetch the dataset (version=2 is the cleaned version of the Adult dataset)
adult_data = fetch_openml(
    name='adult',
    version=2,
    as_frame=True,
    data_home=data_dir,
    parser='auto' # Suppresses pandas parser warning
)

X_raw = adult_data.data
y_raw = adult_data.target

# Combine into a single raw dataframe for initial logging (Task 2)
df_raw = X_raw.copy()
df_raw['income'] = y_raw


print("\n--- Ingestion Step: Raw Data Logging ---")
print(f"Total row count: {df_raw.shape[0]} (Expected: 48842)")
print(f"Total column count: {df_raw.shape[1]} (Expected: 15)")
print(f"\nFeature names (14 inputs + 1 target):")
print(list(df_raw.columns))

print("\nFirst 3 rows of raw data:")
print(df_raw.head(3))

print("\nValue counts for target (income):")
print(df_raw['income'].value_counts())

print("\n Tasks 1 & 2 Complete: Data ingested without modification and logged.")