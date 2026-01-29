import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

MODEL_DIR = "models/flan_t5_varuna"

# -----------------------------
# WQI thresholds (for display)
# -----------------------------
THRESHOLDS = {
    "pH": "6.5–8.5 (optimal)",
    "Dissolved Oxygen": "≥ 7 mg/L (healthy)",
    "Turbidity": "≤ 5 NTU (clear–moderate)",
    "Nitrate": "≤ 5 mg/L (safe range)",
}

def quick_reasoning(ph, do, turb, nitrate):
    reasons = []

    if do < 5:
        reasons.append("Dissolved oxygen is low (<5 mg/L), which reduces water quality.")
    elif do < 7:
        reasons.append("Dissolved oxygen is moderate (5–7 mg/L).")
    else:
        reasons.append("Dissolved oxygen is high (≥7 mg/L), which supports healthier water.")

    if 6.5 <= ph <= 8.5:
        reasons.append("pH is within the optimal 6.5–8.5 range.")
    else:
        reasons.append("pH is outside the optimal 6.5–8.5 range.")

    if turb > 5:
        reasons.append("Turbidity is high (>5 NTU), indicating cloudy water.")
    elif turb > 1:
        reasons.append("Turbidity is moderate (1–5 NTU).")
    else:
        reasons.append("Turbidity is low (≤1 NTU), indicating clear water.")

    if nitrate > 5:
        reasons.append("Nitrate is elevated (>5 mg/L), suggesting nutrient pollution.")
    elif nitrate > 1:
        reasons.append("Nitrate is moderate (1–5 mg/L).")
    else:
        reasons.append("Nitrate is low (≤1 mg/L).")

    return reasons

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def extract_category(text):
    match = re.search(r"(Good|Fair|Poor)", text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Project Varuna Demo", page_icon="💧")

st.title("💧 Project Varuna — Water Quality AI Demo")
st.write("Enter measured water values to receive a predicted water quality category and explanation.")

with st.expander("📊 Reference ranges used in this project"):
    for k, v in THRESHOLDS.items():
        st.write(f"**{k}:** {v}")

st.subheader("Input Measurements")

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.5, 0.01)
    do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 7.0, 0.01)
    turb = st.number_input("Turbidity (NTU)", 0.0, 100.0, 2.0, 0.1)

with col2:
    nitrate = st.number_input("Nitrate (mg/L)", 0.0, 50.0, 1.0, 0.1)
    phos = st.number_input("Phosphorus (mg/L)", 0.0, 5.0, 0.05, 0.01)
    temp = st.number_input("Water Temperature (°C)", -5.0, 50.0, 25.0, 0.1)
    daylight = st.number_input("Daylight Duration (seconds)", 30000.0, 50000.0, 39000.0, 10.0)

tokenizer, model, device = load_model()

prompt = (
    "Given the following water quality and environmental measurements, "
    "predict the water quality category and explain the reasoning.\n\n"
    f"pH: {ph}\n"
    f"Dissolved Oxygen: {do}\n"
    f"Turbidity: {turb}\n"
    f"Nitrate: {nitrate}\n"
    f"Phosphorus: {phos}\n"
    f"Water Temperature: {temp}\n"
    f"Daylight Duration (seconds): {daylight}\n"
)

if st.button("🔮 Predict Water Quality"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=160, num_beams=4)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    category = extract_category(text)

    st.markdown("---")
    st.subheader("Prediction")

    if category == "Good":
        st.success("✅ Water Quality: GOOD")
    elif category == "Fair":
        st.warning("⚠️ Water Quality: FAIR")
    elif category == "Poor":
        st.error("⛔ Water Quality: POOR")
    else:
        st.info("ℹ️ Water Quality: UNKNOWN")

    st.markdown("### Model Explanation")
    st.markdown(f"**{text}**")

    st.markdown("### Why this result makes sense")
    for r in quick_reasoning(ph, do, turb, nitrate):
        st.write("• " + r)

    st.caption("This is a research demonstration system, not a certified water safety test.")
