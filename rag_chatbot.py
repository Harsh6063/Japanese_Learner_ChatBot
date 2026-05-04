import os
import re
from dotenv import load_dotenv
from functools import lru_cache

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from syllabus import syllabus_chatbot
from translator import translate_auto
from level_classifier import predict_level


# ---------------- USER PROFILE ----------------
USER_PROFILE = {
    "level": None,
    "goal": None
}


# ---------------- HISTORY ----------------
def format_history(messages, limit=4):
    if not messages:
        return ""
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages[-limit:]])


# ---------------- NORMALIZE ----------------
def normalize_query(q):
    q = q.lower()
    fixes = {
        "wat": "what",
        "meanig": "meaning",
        "kanjii": "kanji",
        "grammer": "grammar"
    }
    for k, v in fixes.items():
        q = q.replace(k, v)
    return q


# ---------------- FORMATTERS ----------------
def format_kanji(text):
    # split using bullet OR pattern
    parts = re.split(r'•|\n', text)

    clean_parts = []

    for p in parts:
        p = p.strip()
        if "→" in p and re.search(r'[\u4e00-\u9faf]', p):
            clean_parts.append(p)

    # remove duplicates
    clean_parts = list(dict.fromkeys(clean_parts))

    # format nicely
    formatted = (
    "🈶 Important Kanji for this level\n"
    "━━━━━━━━━━━━━━━━━━\n\n"
)

    for item in clean_parts:
        formatted += f"• {item}\n"

    return formatted.strip()


def format_grammar(text):
    parts = re.split(r'\n|•', text)

    clean_parts = []

    for p in parts:
        p = p.strip()
        if "→" in p:
            clean_parts.append(p)

    clean_parts = list(dict.fromkeys(clean_parts))

    formatted = "🧩 Important Grammar Points:\n\n"

    for item in clean_parts:
        formatted += f"• {item}\n"

    return formatted.strip()


# ---------------- GREETING ----------------
def is_greeting(q):
    return any(w in q for w in ["hi", "hello", "hey", "help"])


# ---------------- LEVEL ----------------
def extract_level(text):
    match = re.search(r"n[1-5]", text.lower())
    return match.group().upper() if match else None


# ---------------- GOAL ----------------
def extract_goal(text):
    goals = {
        "conversation": "Speaking",
        "anime": "Anime",
        "exam": "JLPT Exam",
        "travel": "Travel"
    }
    for k, v in goals.items():
        if k in text.lower():
            return v
    return None


# ---------------- TRANSLATION ----------------
def is_translation_request(q):
    return any(k in q for k in ["translate", "meaning","in english", "in japanese"])


def clean_text(q):
    return re.sub(r"(translate|meaning|in english|in japanese)", "", q).strip()


# ---------------- LOAD SYSTEM ----------------
@lru_cache(maxsize=1)
def load_system():
    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search("", k=200)

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 6

    faiss = db.as_retriever(search_kwargs={"k": 6})

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return client, faiss, bm25, reranker


# ---------------- SEARCH ----------------
def hybrid_search(query, faiss, bm25):
    d1 = faiss.invoke(query)
    d2 = bm25.invoke(query)

    seen = set()
    results = []

    for d in d1 + d2:
        if d.page_content not in seen:
            results.append(d)
            seen.add(d.page_content)

    return results[:8]


def rerank(query, docs, reranker, top_k=3):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = list(zip(docs, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]


# ---------------- MAIN ----------------
def chatbot(query, history=None):

    try:
        q = normalize_query(query)

        # GREETING
        if is_greeting(q):
            return """👋 Hello!  こんにちは (Konnichiwa) I am your AI Japanese Tutor NIHONGO

🎯 My purpose:
I help you learn Japanese step-by-step — grammar, kanji, vocabulary, and translation.

📘 What is JLPT?
The Japanese-Language Proficiency Test (JLPT) has 5 levels:

- N5 → Beginner
- N4 → Basic
- N3 → Intermediate
- N2 → Upper Intermediate
- N1 → Advanced

👉 Tell me:
- Your level (N5–N1)
- Your goal (conversation / anime / exam / travel)

Example:
N5 + conversation

Let’s start learning 🚀
"""

        auto_level = None

# detect level only if user didn't specify
        if not extract_level(q):
            try:
                auto_level = predict_level(query)
            except:
                auto_level = None

        # PROFILE
        level = extract_level(q)
        goal = extract_goal(q)

        if level and ("+" in query or goal):
            USER_PROFILE["level"] = level
            USER_PROFILE["goal"] = goal

            return f"""✅ Profile set!

Level: {level}
Goal: {goal}
"""

        # SYLLABUS
        if level:
            return syllabus_chatbot(query)

        # TRANSLATION
        if is_translation_request(q) and not level:
            return f"🌐 {translate_auto(clean_text(query))}"

        client, faiss, bm25, reranker = load_system()

        # SEARCH
        docs = hybrid_search(q, faiss, bm25)

        if docs:
            docs = rerank(q, docs, reranker)

        if not docs:
            return "⚠️ No relevant info found."

        context = "\n\n".join([d.page_content for d in docs])

        history_text = format_history(history)

        prompt = f"""
You are a Japanese tutor.
Give a SHORT explanation (max 4 lines).

Format:
• Meaning
• Usage
• 1 example

User Level (manual): {USER_PROFILE.get('level')}
Detected Level (BERT): {auto_level}

Adapt explanation:
- N5 → very simple
- N3 → moderate
- N1 → detailed

Goal: {USER_PROFILE.get('goal')}

Conversation:
{history_text}

Context:
{context}

User:
{query}

Answer clearly.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content
        if auto_level:
            result = f"📊 Detected Level: {auto_level}\n\n" + result

        # -------- FORMAT OUTPUT --------
        if "kanji" in q and "→" in result:
            return format_kanji(result)

        if "grammar" in q and "→" in result:
            return format_grammar(result)

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"