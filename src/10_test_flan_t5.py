import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------------
# Load trained model
# -----------------------------
MODEL_DIR = "models/flan_t5_varuna"

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Example input (you can edit these values)
# -----------------------------
prompt = (
    "Given the following water quality and environmental measurements, "
    "predict the water quality category and explain the reasoning.\n\n"
    "pH: 8.3\n"
    "Dissolved Oxygen: 4.9\n"
    "Turbidity: 3.2\n"
    "Nitrate: 2.8\n"
    "Phosphorus: 0.08\n"
    "Water Temperature: 27.0\n"
    "Daylight Duration (seconds): 41000\n"


)

# -----------------------------
# Tokenize input
# -----------------------------
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True
).to(device)

# -----------------------------
# Generate output
# -----------------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

# -----------------------------
# Decode and print result
# -----------------------------
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🔮 FLAN-T5 Prediction:\n")
print(prediction)
