import pandas as pd

# -----------------------------
# Load raw water-quality data
# -----------------------------
input_path = "data/raw/Water_Quality_Sampling_Data.csv"
output_path = "data/processed/water_clean.csv"

df = pd.read_csv(input_path)

# -----------------------------
# Keep only parameters we care about
# -----------------------------
target_parameters = [
    "PH",
    "WATER TEMPERATURE",
    "DISSOLVED OXYGEN",
    "TURBIDITY",
    "NITRATE/NITRITE AS N",
    "PHOSPHORUS AS P"
]

df = df[df["PARAMETER"].isin(target_parameters)]

# -----------------------------
# Convert date
# -----------------------------
df["SAMPLE_DATE"] = pd.to_datetime(df["SAMPLE_DATE"]).dt.date

# -----------------------------
# Keep essential columns only
# -----------------------------
df = df[
    [
        "SAMPLE_ID",
        "SAMPLE_DATE",
        "PARAMETER",
        "RESULT"
    ]
]

# -----------------------------
# Convert RESULT to numeric
# -----------------------------
df["RESULT"] = pd.to_numeric(df["RESULT"], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=["RESULT"])

# -----------------------------
# Pivot LONG → WIDE
# -----------------------------
water_wide = df.pivot_table(
    index=["SAMPLE_ID", "SAMPLE_DATE"],
    columns="PARAMETER",
    values="RESULT",
    aggfunc="mean"
).reset_index()

# -----------------------------
# Rename columns to clean names
# -----------------------------
water_wide = water_wide.rename(
    columns={
        "PH": "ph",
        "WATER TEMPERATURE": "water_temp",
        "DISSOLVED OXYGEN": "dissolved_oxygen",
        "TURBIDITY": "turbidity",
        "NITRATE/NITRITE AS N": "nitrate",
        "PHOSPHORUS AS P": "phosphorus",
    }
)

# -----------------------------
# Drop rows with too many missing values
# -----------------------------
water_wide = water_wide.dropna(thresh=4)

# -----------------------------
# Save cleaned water data
# -----------------------------
water_wide.to_csv(output_path, index=False)

print("✅ Water data cleaned and saved to:", output_path)
print("Final shape:", water_wide.shape)
print(water_wide.head())
