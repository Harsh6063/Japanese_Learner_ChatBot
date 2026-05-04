import json
import re

FILES = {
    "N5": "data/processed/n5_complete_syllabus.json",
    "N4": "data/processed/n4_complete_syllabus.json",
    "N3": "data/processed/n3_complete_syllabus.json",
    "N2": "data/processed/n2_complete_syllabus.json",
    "N1": "data/processed/n1_complete_syllabus.json"
}

JLPT_DATA = {}

for level, path in FILES.items():
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
            key = f"{level}_Complete_Syllabus"
            JLPT_DATA[level] = raw.get(key, {})
    except:
        JLPT_DATA[level] = {}


def extract_level(text):
    match = re.search(r"n[1-5]", text.lower())
    return match.group().upper() if match else None


def detect_topic(text):
    t = text.lower()
    if "grammar" in t:
        return "grammar"
    if "kanji" in t:
        return "kanji"
    if "plan" in t or "study" in t or "start" in t:
        return "plan"
    return "full"


def syllabus_chatbot(text):
    level = extract_level(text)
    if not level:
        return "Please specify JLPT level."

    topic = detect_topic(text)
    data = JLPT_DATA.get(level, {})

    grammar = data.get("grammar", [])
    kanji = data.get("kanji_50", [])
    plan = data.get("study_plan", [])

    if topic == "grammar":
        formatted = "🧩 Grammar Points:\n\n"

        for g in grammar[:10]:
            formatted += (
                f"🔹 {g['pattern']}\n"
                f"Meaning: {g.get('explanation','')}\n"
                f"Example: {g.get('example','')}\n"
                f"Translation: {g.get('translation','')}\n\n"
        )

        return formatted.strip()


    if topic == "kanji":
        formatted = "🈶 Kanji List:\n\n"

        for k in kanji[:20]:
            formatted += (
                f"🔹 {k['kanji']}\n → "
                f"Meaning: {k['meaning']}\n → "
                f"Reading: {k['readings']}\n\n"
        )

        return formatted.strip()

    if topic == "plan":
        return "\n".join(plan[:5])

    req = data.get("requirements", {})
    desc = data.get("description", "")

    return f"""
📘 {level} Syllabus

🧠 Overview:
{desc}

📊 Requirements:
- Kanji: {req.get("kanji", "unknown")}
- Vocabulary: {req.get("vocabulary", "unknown")}
- Listening: {req.get("listening_skills", "N/A")}
- Reading: {req.get("reading_skills", "N/A")}

📅 Study Plan:
{" ".join(plan[:3]) if plan else "Practice daily with reading, listening, and grammar."}

💡 You can also ask:
- {level} grammar
- {level} kanji
- {level} plan
"""