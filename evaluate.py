from rag_chatbot import chatbot
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from collections import defaultdict


# ---------------- LOAD MODEL ----------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------- TEST DATA ----------------
tests = [
    {
        "q": "translate はじめまして",
        "expected": "nice to meet you",
        "type": "translation"
    },
    {
        "q": "translate ありがとう",
        "expected": "thank you",
        "type": "translation"
    },
    {
        "q": "What does 本 mean",
        "expected": "book",
        "type": "kanji"
    },
    {
        "q": "Explain 〜ている",
        "expected": "ongoing action present continuous",
        "type": "grammar"
    },
    {
        "q": "N5 grammar",
        "expected": "〜は 〜です basic grammar",
        "type": "syllabus"
    },
]


# ---------------- SEMANTIC SIMILARITY ----------------
def semantic_score(expected, output):
    # limit long outputs (important fix)
    output = output[:300]

    emb1 = model.encode([expected])
    emb2 = model.encode([output])

    score = cosine_similarity(emb1, emb2)[0][0]
    return float(score)


# ---------------- TASK-AWARE SCORING ----------------
def score_response(test, output):
    ttype = test["type"]
    expected = test["expected"]

    if ttype == "translation":
        return semantic_score(expected, output)

    elif ttype == "kanji":
        return semantic_score(expected, output)

    elif ttype == "grammar":
        # check if correct grammar appears
        if "〜ている" in output or "ongoing" in output:
            return 0.9
        return semantic_score(expected, output)

    elif ttype == "syllabus":
        if "Grammar Points" in output or "grammar" in output.lower():
            return 0.9
        return 0.5

    return semantic_score(expected, output)


# ---------------- MAIN EVALUATION ----------------
def evaluate():

    scores = []
    latencies = []
    category_scores = defaultdict(list)

    print("\n🚀 Running Evaluation...\n")

    for i, t in enumerate(tests):

        start = time.time()
        output = chatbot(t["q"])
        latency = time.time() - start

        # ignore first run latency (model load)
        if i != 0:
            latencies.append(latency)

        score = score_response(t, output)

        scores.append(score)
        category_scores[t["type"]].append(score)

        print("\n============================")
        print("Query:", t["q"])
        print("Expected:", t["expected"])
        print("Output:", output[:200])
        print("Score:", round(score, 3))
        print("Latency:", round(latency, 2), "s")

        if score > 0.75:
            print("🔥 Strong match")
        elif score > 0.5:
            print("👍 Partial match")
        else:
            print("⚠️ Weak match")

    # ---------------- FINAL METRICS ----------------
    avg_score = np.mean(scores)
    avg_latency = np.mean(latencies) if latencies else 0

    print("\n============================")
    print(f"📊 Overall Score: {avg_score:.3f}")
    print(f"⚡ Avg Latency: {avg_latency:.2f}s")

    print("\n📊 Category-wise Performance:")
    for k, v in category_scores.items():
        print(f"{k}: {round(np.mean(v), 3)}")

    if avg_score > 0.75:
        print("\n🔥 Overall: Strong system")
    elif avg_score > 0.5:
        print("\n👍 Overall: Decent system")
    else:
        print("\n⚠️ Needs improvement")


# ---------------- RUN ----------------
if __name__ == "__main__":
    evaluate()