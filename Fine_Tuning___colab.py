# Upgrade pip first (optional but recommended)
!pip install --upgrade pip

# Install Hugging Face Transformers, Datasets, and Accelerate (for Trainer)
!pip install --upgrade transformers datasets accelerate

# If you want PyTorch explicitly (Colab usually has a compatible version)
!pip install torch --upgrade




# Fine_Tuning_Colab_GPU.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# ---------- CONFIG ----------
MODEL_NAME = "facebook/bart-large-mnli"
TRAIN_CSV = "/content/train.csv"  # update path
VALID_CSV = "/content/valid.csv"  # update path
OUTPUT_DIR = "./intent_model"
NUM_LABELS = 5
BATCH_SIZE = 8  # increase for GPU
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50

# ---------- CHECK DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- LOAD TOKENIZER & MODEL ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)
model.to(device)

# ---------- LOAD DATA ----------
if not os.path.exists(TRAIN_CSV) or not os.path.exists(VALID_CSV):
    raise FileNotFoundError(f"Cannot find CSV files:\n{TRAIN_CSV}\n{VALID_CSV}")

dataset = load_dataset("csv", data_files={
    "train": TRAIN_CSV,
    "validation": VALID_CSV
})

# ---------- TOKENIZE ----------
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------- TRAINING ARGUMENTS ----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # enable mixed precision for GPU
    report_to="none"
)

# ---------- TRAINER ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer
)

# ---------- TRAIN ----------
trainer.train()

# ---------- SAVE MODEL ----------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete. Model saved to {OUTPUT_DIR}")
