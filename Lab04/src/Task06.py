import os
import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


random.seed(42)
np.random.seed(42)


print("Fetching UCI Adult dataset for label-quality check...")
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


df = pd.concat([adult_data.data, adult_data.target], axis=1)
class adult_data:
    ...

target_col = adult_data.target.name 
feature_cols = [c for c in df.columns if c != target_col]

print(f"Total rows in dataset: {len(df)}")


print("\n--- Task 6: Label-Quality Check ---")
print("Running duplicate-conflict audit...")

# FIX: Hash the 14 feature columns to avoid pandas MultiIndex Cartesian product explosion
feature_hashes = pd.util.hash_pandas_object(df[feature_cols], index=False)

# Create a temporary dataframe to group by the hash
df_temp = df.copy()
df_temp['feature_hash'] = feature_hashes

# Count unique labels per feature hash
unique_labels_per_hash = df_temp.groupby('feature_hash')[target_col].nunique()

# A conflict exists if a hash has > 1 unique label
conflict_count = (unique_labels_per_hash > 1).sum()

print(f" Duplicate-Conflict Audit Complete:")
print(f"   -> Found {conflict_count} instance(s) where identical features have conflicting income labels.")


print("\nDrawing 50 random rows for manual data dictionary check...")
np.random.seed(42)
sample_50 = df.sample(50, random_state=42).copy()

# Helper: Flag obviously impossible values based on Task 3 ranges / data dictionary
sample_50['flag_impossible'] = (
        (sample_50['age'] < 17) | (sample_50['age'] > 90) |
        (sample_50['hours-per-week'] < 1) | (sample_50['hours-per-week'] > 99) |
        (sample_50['capital-gain'] < 0) | (sample_50['capital-gain'] > 99999) |
        (sample_50['capital-loss'] < 0) | (sample_50['capital-loss'] > 4356)
)

impossible_count = sample_50['flag_impossible'].sum()

print(f"Sampled 50 rows. Flagged {impossible_count} row(s) with impossible values based on data dictionary ranges.")

# ==========================================
# 4. Save Results
# ==========================================
results_dir = os.path.join(repo_root, 'lab04', 'results')
os.makedirs(results_dir, exist_ok=True)

# Save the 50 rows to a CSV so you can manually inspect them for your report
sample_path = os.path.join(results_dir, 'label_quality_sample_50.csv')
sample_50.to_csv(sample_path, index=False)

report_text = (
    "=== LABEL QUALITY CHECK REPORT ===\n\n"
    f"1. Duplicate-Conflict Audit:\n"
    f"   - Found {conflict_count} instance(s) of identical features with conflicting income labels.\n\n"
    f"2. Manual Re-check Sample:\n"
    f"   - Drew 50 random rows (saved to results/label_quality_sample_50.csv).\n"
    f"   - Automated flagging found {impossible_count} row(s) with values outside standard data dictionary ranges (e.g., age < 17 or > 90, hours < 1 or > 99).\n"
    f"   - These rows should be manually reviewed in the CSV for contextual impossibilities.\n"
)

print("\n" + report_text)

with open(os.path.join(results_dir, 'label_quality_report.txt'), 'w') as f:
    f.write(report_text)

print(f"Task 6 Complete: Reports saved to {results_dir}")