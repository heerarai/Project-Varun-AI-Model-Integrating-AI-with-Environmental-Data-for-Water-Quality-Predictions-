import pandas as pd

# -----------------------------
# Paths
# -----------------------------
input_path = "data/processed/explained_data.csv"
output_path = "data/processed/flan_dataset.csv"

# -----------------------------
# Load explained data
# -----------------------------
df = pd.read_csv(input_path)

# -----------------------------
# Build FLAN-style prompts
# -----------------------------
def build_prompt(row):
    prompt = (
        "Given the following water quality and environmental measurements, "
        "predict the water quality category and explain the reasoning.\n\n"
        f"pH: {row['ph']}\n"
        f"Dissolved Oxygen: {row['dissolved_oxygen']}\n"
        f"Turbidity: {row['turbidity']}\n"
        f"Nitrate: {row['nitrate']}\n"
        f"Phosphorus: {row['phosphorus']}\n"
        f"Water Temperature: {row['water_temp']}\n"
        f"Daylight Duration (seconds): {row['daylight_duration']}\n"
    )
    return prompt

def build_response(row):
    response = (
        f"Water Quality Category: {row['wqi_category']}. "
        f"{row['explanation']}"
    )
    return response

# -----------------------------
# Create FLAN dataset
# -----------------------------
flan_df = pd.DataFrame()
flan_df["input_text"] = df.apply(build_prompt, axis=1)
flan_df["target_text"] = df.apply(build_response, axis=1)

# -----------------------------
# Save dataset
# -----------------------------
flan_df.to_csv(output_path, index=False)

print("✅ FLAN-T5 dataset prepared")
print("Shape:", flan_df.shape)
print("\nSample input:")
print(flan_df.iloc[0]["input_text"])
print("\nSample target:")
print(flan_df.iloc[0]["target_text"])
