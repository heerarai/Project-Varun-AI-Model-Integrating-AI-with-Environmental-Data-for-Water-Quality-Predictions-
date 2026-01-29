import pandas as pd

input_path = "data/raw/open-meteo-30.27N97.75W157m.csv"
output_path = "data/processed/weather_fixed.csv"

# ---------------------------------
# Read raw file with no headers
# ---------------------------------
raw = pd.read_csv(input_path, header=None)

# ---------------------------------
# Find the row that contains 'time'
# (this is the REAL header row)
# ---------------------------------
header_row_idx = None
for i in range(len(raw)):
    row_as_str = raw.iloc[i].astype(str).str.lower().tolist()
    if any("time" == cell.strip() for cell in row_as_str):
        header_row_idx = i
        break

if header_row_idx is None:
    raise ValueError("❌ Could not find header row containing 'time'")

# ---------------------------------
# Use that row as header
# ---------------------------------
header = raw.iloc[header_row_idx].tolist()
data = raw.iloc[header_row_idx + 1:].copy()
data.columns = header

# ---------------------------------
# Drop completely empty rows
# ---------------------------------
data = data.dropna(how="all")

# ---------------------------------
# Normalize column names (strip units)
# ---------------------------------
data.columns = [
    col.split("(")[0].strip().lower().replace(" ", "_")
    for col in data.columns
]

# ---------------------------------
# Keep useful weather columns
# ---------------------------------
keep_cols = ["time"]

for c in [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "daylight_duration",
    "et0_fao_evapotranspiration"
]:
    if c in data.columns:
        keep_cols.append(c)

data = data[keep_cols]

# ---------------------------------
# Convert date
# ---------------------------------
data["date"] = pd.to_datetime(data["time"]).dt.date
data = data.drop(columns=["time"])

# ---------------------------------
# Save fixed weather file
# ---------------------------------
data.to_csv(output_path, index=False)

print("✅ Weather file fixed successfully")
print("Columns:", list(data.columns))
print("Shape:", data.shape)
print(data.head())
