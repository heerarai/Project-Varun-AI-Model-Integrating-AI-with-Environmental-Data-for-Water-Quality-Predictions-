import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load labeled data
# -----------------------------
data_path = "data/processed/labeled_data.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Select features
# -----------------------------
feature_cols = [
    "ph",
    "dissolved_oxygen",
    "turbidity",
    "nitrate",
    "phosphorus",
    "water_temp",
    "daylight_duration"
]

X = df[feature_cols]
y = df["wqi_category"]

# -----------------------------
# Handle missing values
# -----------------------------
# Use median imputation (simple + defensible)
X = X.fillna(X.median())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("✅ Baseline Random Forest Results")
print("Accuracy:", round(acc, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Feature importance
# -----------------------------
importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)
