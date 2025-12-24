[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![NLP](https://img.shields.io/badge/NLP-Intent%20Analysis-purple.svg)](#)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CONTRIBUTING.md)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

# Multilingual Intent Analysis System

A **multilingual text intent classification system** built with **Python, PyTorch, and Hugging Face Transformers**.  
The system predicts user intent across **five categories** — *Hate/Violence, Anti-state Propaganda, Panic Spreading, Misinformation, Neutral* — and supports **20+ languages**, including **English and Bengali (বাংলা)**.  
It can run fully **offline** using locally stored model files, suitable for secure or low-connectivity environments.

---

## **Features**

* **Multilingual Intent Classification:** Supports over 20 global languages.
* **5-Class Classification:** Hate/Violence → Neutral.
* **Offline Inference:** No internet required during prediction.
* **Batch & Single Text Support:** Analyze one or multiple texts at once.
* **Transformer-Based Model:** High accuracy using modern NLP architectures.
* **Lightweight Inference Mode:** Uses `model.eval()` and `torch.no_grad()`.
* **Easy Integration:** Can be plugged into APIs, web apps, or data pipelines.

---

## **Task Details**

| Property              | Description                                               |
| --------------------- | --------------------------------------------------------- |
| **Task**              | Text Classification (Intent Analysis)                    |
| **Number of Classes** | 5                                                         |
| **Labels**            | Hate/Violence, Anti-state Propaganda, Panic Spreading, Misinformation, Neutral |
| **Framework**         | PyTorch                                                   |
| **Model Type**        | Transformer (Hugging Face)                                |
| **Inference Mode**    | Offline / Local                                           |

---

## **Supported Languages**

English, 中文 (Chinese), Español (Spanish), हिन्दी (Hindi), العربية (Arabic), বাংলা (Bengali), Português (Portuguese), Русский (Russian), 日本語 (Japanese), Deutsch (German), Bahasa Melayu (Malay), తెలుగు (Telugu), Tiếng Việt (Vietnamese), 한국어 (Korean), Français (French), Türkçe (Turkish), Italiano (Italian), Polski (Polish), Українська (Ukrainian), Tagalog, Nederlands (Dutch), Schweizerdeutsch (Swiss German), Kiswahili (Swahili)

---

## **Technology Stack**

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **NLP Library:** Hugging Face Transformers
* **Tokenizer:** AutoTokenizer
* **Model Loader:** AutoModelForSequenceClassification

---

## **Project Structure**

```

Intent_Analysis_System/
│
├── intent_model/            # Local trained model & tokenizer files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab files
│
├── inference.py             # Intent prediction script
└── README.md                # Project documentation

````

---

## **Setup Instructions**

### 1. Install Dependencies
```bash
pip install torch transformers datasets
````

> ⚠️ Internet is **not required** during inference if the model is already stored locally.

---

### 2. Model Preparation

Ensure your trained model and tokenizer are available locally at:

```
./intent_model
```

The system loads the model using:

```python
AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
```

---

### 3. Run Intent Prediction

Example inference script:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./intent_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

def predict_intent(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    intent_map = {
        0: "Hate/Violence",
        1: "Anti-state Propaganda",
        2: "Panic Spreading",
        3: "Misinformation",
        4: "Neutral"
    }
    return [intent_map[i] for i in torch.argmax(probs, dim=-1).tolist()]
```

---

## **Usage Example**

```python
print(predict_intent("দেশের বিরুদ্ধে অপপ্রচার চালানো হয়েছে।")[0])
# Output: Anti-state Propaganda

texts = ["I am scared", "This is fake news"]
print(predict_intent(texts))
# Output: ['Panic Spreading', 'Misinformation']
```

---

## **Tips for Better Results**

* Keep text length under **512 tokens**.
* Use clear, complete sentences for higher accuracy.
* Batch processing improves performance for large datasets.
* Ideal for news, social media, and customer feedback content.

---

## **Applications**

* News and social media monitoring
* Misinformation detection
* Customer feedback analysis
* Multilingual NLP pipelines
* AI-based decision and risk analysis
* Offline or secure environments

---

## **License**

This project is **MIT licensed** — free to use for **personal, educational, and research purposes**.

---

## **Author**

**Zihadul Islam**
GitHub: [https://github.com/zihadulislam99](https://github.com/zihadulislam99)
