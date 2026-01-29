import pandas as pd

# -----------------------------
# Paths
# -----------------------------
water_path = "data/processed/water_clean.csv"
weather_path = "data/processed/weather_fixed.csv"
output_path = "data/processed/merged_data.csv"

# -----------------------------
# Load datasets
# -----------------------------
water = pd.read_csv(water_path)
weather = pd.read_csv(weather_path)

# -----------------------------
# Ensure date formats match
# -----------------------------
water["SAMPLE_DATE"] = pd.to_datetime(water["SAMPLE_DATE"]).dt.date
weather["date"] = pd.to_datetime(weather["date"]).dt.date

# -----------------------------
# Merge on date
# -----------------------------
merged = water.merge(
    weather,
    left_on="SAMPLE_DATE",
    right_on="date",
    how="inner"
)

# Drop duplicate date column
merged = merged.drop(columns=["date"])

# -----------------------------
# Save merged dataset
# -----------------------------
merged.to_csv(output_path, index=False)

print("✅ Merged dataset saved to:", output_path)
print("Final shape:", merged.shape)
print(merged.head())
