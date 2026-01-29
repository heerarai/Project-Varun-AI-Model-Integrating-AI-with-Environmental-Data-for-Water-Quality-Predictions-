import pandas as pd

# -----------------------------
# Load labeled data
# -----------------------------
data_path = "data/processed/labeled_data.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Feature importance from baseline model
# (hard-coded from training output)
# -----------------------------
feature_importance = {
    "dissolved_oxygen": 0.500377,
    "nitrate": 0.149537,
    "ph": 0.142672,
    "turbidity": 0.065193,
    "water_temp": 0.062149,
    "phosphorus": 0.040312,
    "daylight_duration": 0.039760
}

# -----------------------------
# Generate explanation text
# -----------------------------
def generate_explanation(row):
    drivers = []

    if row["dissolved_oxygen"] < 5:
        drivers.append("low dissolved oxygen")
    elif row["dissolved_oxygen"] >= 7:
        drivers.append("high dissolved oxygen")

    if row["nitrate"] > 5:
        drivers.append("elevated nitrate levels")
    elif row["nitrate"] <= 1:
        drivers.append("low nitrate levels")

    if row["ph"] < 6.5 or row["ph"] > 8.5:
        drivers.append("pH outside optimal range")
    else:
        drivers.append("pH within optimal range")

    if pd.notna(row["turbidity"]) and row["turbidity"] > 5:
        drivers.append("high turbidity")

    if pd.notna(row["daylight_duration"]):
        drivers.append("seasonal daylight effects")

    if not drivers:
        drivers.append("balanced water chemistry")

    explanation = "Water quality is primarily influenced by " + ", ".join(drivers) + "."
    return explanation

# -----------------------------
# Apply explanations
# -----------------------------
df["explanation"] = df.apply(generate_explanation, axis=1)

# -----------------------------
# Save updated dataset
# -----------------------------
output_path = "data/processed/explained_data.csv"
df.to_csv(output_path, index=False)

print("✅ Explanations generated")
print(df[["wqi_category", "explanation"]].head())
