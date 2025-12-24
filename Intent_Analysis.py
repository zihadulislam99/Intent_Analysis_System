from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./Intent_Analysis_System/intent_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()  # IMPORTANT for inference

def predict_intent(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    intent_map = {0: "Hate / Violence", 1: "Anti-state propaganda", 2: "Panic spreading", 3: "Misinformation", 4: "Normal"}
    # intent_map = {0: "Hate / Violence", 1: "Anti-state propaganda", 2: "Panic spreading", 3: "Misinformation", 4: "Normal"}
    return [intent_map[p] for p in torch.argmax(probs, dim=-1).tolist()]

texts = [ "I will kill you", "The government must be destroyed", "Banks will collapse tomorrow", "Drinking bleach cures disease", "I love learning AI", "সারা দেশে ব্যাংক বন্ধ হয়ে যাবে", "Ti amo", "I love you"]

# print(predict_intent("সারা দেশে ব্যাংক বন্ধ হয়ে যাবে")[0])
# print(predict_intent(["Ti amo", "I love you"]))
for text, intent in zip(texts, predict_intent(texts)):
    print(f"Text: {text}  ||  Intent: {intent}")
