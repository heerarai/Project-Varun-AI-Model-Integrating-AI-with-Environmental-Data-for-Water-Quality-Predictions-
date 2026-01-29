import pandas as pd

OLD_PATH = "data/raw/open-meteo-30.27N97.75W157m_OLD.csv"
NEW_PATH = "data/raw/open-meteo-30.27N97.75W157m_NEW.csv"
OUT_PATH = "data/raw/open-meteo-30.27N97.75W157m.csv"

def load_and_fix(path):
    raw = pd.read_csv(path, header=None)

    header_row = None
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower().str.strip().tolist()
        if any(cell == "time" for cell in row):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Could not find header row in {path}")

    header = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    data = data.dropna(how="all")

    data.columns = [
        c.split("(")[0].strip().lower().replace(" ", "_")
        for c in data.columns
    ]

    keep = ["time"]
    for c in [
        "temperature_2m_max",
        "precipitation_sum",
        "daylight_duration",
        "et0_fao_evapotranspiration",
    ]:
        if c in data.columns:
            keep.append(c)

    data = data[keep]
    data["date"] = pd.to_datetime(data["time"], errors="coerce").dt.date
    data = data.drop(columns=["time"])
    data = data.dropna(subset=["date"])

    return data

old_df = load_and_fix(OLD_PATH)
new_df = load_and_fix(NEW_PATH)

combined = pd.concat([old_df, new_df], ignore_index=True)
combined["_filled"] = combined.notna().sum(axis=1)
combined = combined.sort_values(["date", "_filled"], ascending=[True, False])
combined = combined.drop_duplicates(subset=["date"], keep="first")
combined = combined.drop(columns=["_filled"])
combined = combined.sort_values("date").reset_index(drop=True)

combined.to_csv(OUT_PATH, index=False)

print("✅ Combined weather file created")
print("Rows:", len(combined))
print("Date range:", combined["date"].min(), "to", combined["date"].max())
print("Columns:", list(combined.columns))
