import pandas as pd

# Load raw datasets
water_path = "data/raw/Water_Quality_Sampling_Data.csv"
weather_path = "data/raw/open-meteo-30.27N97.75W157m.csv"

water = pd.read_csv(water_path)
weather = pd.read_csv(weather_path)

print("===== WATER QUALITY DATA =====")
print("Shape:", water.shape)
print("Columns:")
print(water.columns)

print("Missing values (%):")
print((water.isnull().mean() * 100).round(2))

print("Sample rows:")
print(water.head())

print("\n===== WEATHER DATA =====")
print("Shape:", weather.shape)
print("Columns:")
print(weather.columns)

print("Missing values (%):")
print((weather.isnull().mean() * 100).round(2))

print("Sample rows:")
print(weather.head())
