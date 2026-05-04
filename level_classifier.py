from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------- LOAD YOUR MODEL --------
MODEL_PATH = "models/jlpt_bert"   # change to your path

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


LABELS = ["N1", "N2", "N3", "N4", "N5"]


# -------- PREDICT LEVEL --------
def predict_level(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    return LABELS[pred]