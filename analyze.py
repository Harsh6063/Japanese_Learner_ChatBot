from transformers import pipeline

# ----------- MODEL -----------

classifier = pipeline(
    "text-classification",
    model="models/jlpt_bert",
    tokenizer="models/jlpt_bert"
)

label_map = {
    "LABEL_0": "N1",
    "LABEL_1": "N2",
    "LABEL_2": "N3",
    "LABEL_3": "N4",
    "LABEL_4": "N5"
}


# ----------- LEVEL -----------

def get_level(text):
    result = classifier(text)[0]
    return label_map.get(result["label"], result["label"])


# ----------- WORD DETECTION -----------

from tokenizer import tokenize

def link_words(text, word_index):
    tokens = tokenize(text)
    words = []

    for t in tokens:
        surface = t["surface"]
        base = t["lemma"] if t["lemma"] != "*" else surface

        # check both base and surface
        if base in word_index:
            data = word_index[base]
        elif surface in word_index:
            data = word_index[surface]
        else:
            continue

        words.append({
            "word": surface,
            "reading": data.get("reading", ""),
            "meaning": data.get("meaning", ""),
            "level": data.get("level", "")
        })

    return words


# ----------- GRAMMAR DETECTION -----------

def detect_grammar(text, grammar_list):
    found = []
    seen = set()

    for g in grammar_list:
        match = g.get("match", "").replace("〜", "")

        # skip very small patterns
        if len(match) <= 1:
            continue

        if match in text and match not in seen:
            found.append({
                "pattern": g.get("pattern"),
                "meaning": g.get("meaning", g.get("type", "")),
                "example": g.get("example", "")
            })
            seen.add(match)

    return found


# ----------- KANJI EXTRACTION -----------

def extract_kanji(text, kanji_db):
    result = []

    for ch in text:
        if ch in kanji_db:
            info = kanji_db[ch]

            result.append({
                "kanji": ch,
                "meaning": ", ".join(info.get("meanings", [])),
                "on_readings": info.get("readings_on", []),
                "kun_readings": info.get("readings_kun", []),
                "level": f"N{info.get('jlpt_new', '')}"
            })

    return result


def detect_particles(tokens):
    particles = []

    for t in tokens:
        if t["pos"] == "助詞":  # particle
            particles.append({
                "pattern": t["surface"],
                "meaning": "particle"
            })

    return particles


# ----------- MAIN ANALYZE -----------

from tokenizer import tokenize

def analyze_full(text, word_index, grammar_list, kanji_db):

    tokens = tokenize(text)

    level = get_level(text)

    words = link_words(text, word_index)
    grammar = detect_grammar(text, grammar_list)
    particles = detect_particles(tokens)
    grammar.extend(particles)

    kanji = extract_kanji(text, kanji_db)

    return {
        "text": text,
        "level": level,
        "words": words,
        "grammar": grammar,
        "kanji": kanji
    }