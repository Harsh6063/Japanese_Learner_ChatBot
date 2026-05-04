import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ----------- LOAD DATA -----------

df = pd.read_csv("data/processed/Sentences.csv")

# keep only required columns
df = df[["text", "level"]]

# encode labels (N5 → 0, ..., N1 → 4)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["level"])

# save mapping
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_map)

# convert to HF dataset
dataset = Dataset.from_pandas(df)

# ----------- TOKENIZER -----------

MODEL_NAME = "cl-tohoku/bert-base-japanese"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize)

# train-test split
dataset = dataset.train_test_split(test_size=0.1)

# ----------- MODEL -----------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)

# ----------- METRICS -----------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# ----------- TRAINING -----------

training_args = TrainingArguments(
    output_dir="models/jlpt_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# ----------- TRAIN -----------

trainer.train()

# ----------- SAVE -----------

trainer.save_model("models/jlpt_bert")
tokenizer.save_pretrained("models/jlpt_bert")

print("✅ Training complete!")