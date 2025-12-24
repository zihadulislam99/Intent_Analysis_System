from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "xlm-roberta-base"  # or your fine-tuned HF repo
SAVE_PATH = "./Intent_Analysis_System/intent_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)
print("Model downloaded and saved locally.")
