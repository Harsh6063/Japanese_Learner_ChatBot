import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cpu"   # keep CPU to avoid CUDA OOM

# -------- MODELS --------
JA_EN = "Helsinki-NLP/opus-mt-ja-en"
EN_JA = "Helsinki-NLP/opus-mt-en-jap"

# Load models
tok_ja_en = AutoTokenizer.from_pretrained(JA_EN)
mod_ja_en = AutoModelForSeq2SeqLM.from_pretrained(JA_EN).to(device)

tok_en_ja = AutoTokenizer.from_pretrained(EN_JA)
mod_en_ja = AutoModelForSeq2SeqLM.from_pretrained(EN_JA).to(device)


# -------- DETECT LANGUAGE --------
def is_japanese(text):
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9faf]', text))


# -------- JA → EN --------
def ja_to_en(text):
    inputs = tok_ja_en(text, return_tensors="pt", truncation=True).to(device)

    outputs = mod_ja_en.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=4,
        early_stopping=True
    )

    return tok_ja_en.decode(outputs[0], skip_special_tokens=True)


# -------- EN → JA --------
def en_to_ja(text):
    inputs = tok_en_ja(text, return_tensors="pt", truncation=True).to(device)

    outputs = mod_en_ja.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=4,
        early_stopping=True
    )

    return tok_en_ja.decode(outputs[0], skip_special_tokens=True)


# -------- AUTO --------
def translate_auto(text):
    if is_japanese(text):
        return ja_to_en(text)
    else:
        return en_to_ja(text)