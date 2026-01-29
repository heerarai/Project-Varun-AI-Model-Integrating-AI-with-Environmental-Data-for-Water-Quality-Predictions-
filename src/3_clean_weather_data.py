import pandas as pd

# -----------------------------
# Paths
# -----------------------------
input_path = "data/raw/open-meteo-30.27N97.75W157m.csv"
output_path = "data/processed/weather_clean.csv"

# -----------------------------
# Read raw file WITHOUT assuming headers
# -----------------------------
raw = pd.read_csv(input_path, header=None)

# -----------------------------
# Find the row that contains the real header
# (the row where the first column == 'time')
# -----------------------------
header_row_index = raw[raw[0] == "time"].index[0]

# -----------------------------
# Re-read CSV starting from the header row
# -----------------------------
weather = pd.read_csv(
    input_path,
    skiprows=header_row_index
)

# -----------------------------
# Convert time column to date
# -----------------------------
weather["date"] = pd.to_datetime(weather["time"]).dt.date

# -----------------------------
# Select useful weather features
# (keep only what exists)
# -----------------------------
columns_to_keep = ["date"]

for col in [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max"
]:
    if col in weather.columns:
        columns_to_keep.append(col)

weather = weather[columns_to_keep]

# -----------------------------
# Rename columns for clarity
# -----------------------------
weather = weather.rename(
    columns={
        "temperature_2m_mean": "air_temp_mean",
        "precipitation_sum": "precipitation_sum",
        "wind_speed_10m_max": "wind_speed_max"
    }
)

# -----------------------------
# Drop rows with missing dates
# -----------------------------
weather = weather.dropna(subset=["date"])

# -----------------------------
# Save cleaned weather data
# -----------------------------
weather.to_csv(output_path, index=False)

print("✅ Weather data cleaned and saved to:", output_path)
print("Final shape:", weather.shape)
print(weather.head())
