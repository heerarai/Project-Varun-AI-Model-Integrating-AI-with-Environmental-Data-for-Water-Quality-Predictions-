import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
input_path = "data/processed/merged_data.csv"
output_path = "data/processed/labeled_data.csv"

# -----------------------------
# Load merged dataset
# -----------------------------
df = pd.read_csv(input_path)

# -----------------------------
# Helper functions for scoring
# -----------------------------
def score_ph(ph):
    if pd.isna(ph):
        return np.nan
    if 6.5 <= ph <= 8.5:
        return 2   # good
    elif 6.0 <= ph <= 9.0:
        return 1   # fair
    else:
        return 0   # poor

def score_do(do):
    if pd.isna(do):
        return np.nan
    if do >= 7:
        return 2
    elif do >= 5:
        return 1
    else:
        return 0

def score_turbidity(turb):
    if pd.isna(turb):
        return np.nan
    if turb <= 1:
        return 2
    elif turb <= 5:
        return 1
    else:
        return 0

def score_nitrate(n):
    if pd.isna(n):
        return np.nan
    if n <= 1:
        return 2
    elif n <= 5:
        return 1
    else:
        return 0

# -----------------------------
# Apply scores
# -----------------------------
df["ph_score"] = df["ph"].apply(score_ph)
df["do_score"] = df["dissolved_oxygen"].apply(score_do)
df["turb_score"] = df["turbidity"].apply(score_turbidity)
df["nitrate_score"] = df["nitrate"].apply(score_nitrate)

# -----------------------------
# Aggregate WQI score
# -----------------------------
score_cols = ["ph_score", "do_score", "turb_score", "nitrate_score"]
df["wqi_score"] = df[score_cols].mean(axis=1)

# -----------------------------
# Assign category
# -----------------------------
def wqi_category(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 1.5:
        return "Good"
    elif score >= 0.8:
        return "Fair"
    else:
        return "Poor"

df["wqi_category"] = df["wqi_score"].apply(wqi_category)

# -----------------------------
# Save labeled dataset
# -----------------------------
df.to_csv(output_path, index=False)

print("✅ WQI categories created")
print("Final shape:", df.shape)
print(df[["ph", "dissolved_oxygen", "turbidity", "nitrate", "wqi_score", "wqi_category"]].head())
print("\nCategory counts:")
print(df["wqi_category"].value_counts())
