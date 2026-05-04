import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = []

def clean(text):
    return text.replace("\n", " ").strip()

# -------- WORDS --------
with open("data/processed/words.jsonl", encoding="utf-8") as f:
    for line in f:
        w = json.loads(line)

        docs.append(Document(
            page_content=clean(f"""
Word: {w['word']}
Reading: {w['reading']}
Meaning: {w['meaning']}
Level: {w['level']}

This Japanese word {w['word']} means {w['meaning']}.
"""),
            metadata={"type": "word"}
        ))

# -------- GRAMMAR --------
with open("data/processed/grammar_final.jsonl", encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)

        docs.append(Document(
            page_content=clean(f"""
Grammar: {g['pattern']}
Explanation: {g.get('explanation','')}
Example: {g['example']}
Translation: {g.get('translation','')}
Level: {g['level']}
"""),
            metadata={"type": "grammar"}
        ))

# -------- KANJI --------
for file in [
    "data/processed/n5_complete_syllabus.json",
    "data/processed/n4_complete_syllabus.json",
    "data/processed/n3_complete_syllabus.json",
    "data/processed/n2_complete_syllabus.json", 
    "data/processed/n1_complete_syllabus.json"
]:
    with open(file, encoding="utf-8") as f:
        raw = json.load(f)
        key = list(raw.keys())[0]
        data = raw[key]

        for k in data.get("kanji_50", []):
            docs.append(Document(
                page_content=clean(f"""
Kanji: {k['kanji']}
Meaning: {k['meaning']}
Reading: {k['readings']}

The kanji {k['kanji']} means {k['meaning']}.
"""),
                metadata={"type": "kanji"}
            ))

# -------- EMBEDDINGS --------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
db.save_local("vector_db")

print("✅ Vector DB ready")