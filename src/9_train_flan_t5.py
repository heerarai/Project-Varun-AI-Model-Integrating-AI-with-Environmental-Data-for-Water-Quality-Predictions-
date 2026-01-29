import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "data/processed/flan_dataset.csv"
OUTPUT_DIR = "models/flan_t5_varuna"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df)

# -----------------------------
# Load tokenizer & model
# -----------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# -----------------------------
# Tokenization
# -----------------------------
def preprocess(batch):
    inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    targets = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET_LENGTH
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names
)

# -----------------------------
# Train / validation split
# -----------------------------
split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
val_data = split["test"]

# -----------------------------
# Training arguments (SAFE for laptop)
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=10,
    logging_strategy="steps",
    fp16=torch.cuda.is_available(),
    report_to="none"
)


# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ FLAN-T5 fine-tuning complete")
print("Model saved to:", OUTPUT_DIR)
