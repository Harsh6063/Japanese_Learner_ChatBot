import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "models/jlpt_bert"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

labels = ["N5", "N4", "N3", "N2", "N1"]

def predict_level(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs).item()

    return labels[pred], probs[0][pred].item()