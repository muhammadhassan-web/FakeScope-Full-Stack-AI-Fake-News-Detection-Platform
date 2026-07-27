"""
preprocess.py — Text Preprocessing Pipeline for Fake News Detection
Handles cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import string

# ---------------------------------------------------------------------------
# Stopwords — bundled fallback (no network needed for NLTK download)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn","said","say","says","also","would","could","one","like",
    "get","got","make","made","know","think","time","year","new","may","even",
    "well","way","back","see","go","come","people","take","use","good","give",
    "look","want","seem","help","show","put","keep","last","let","large",
    "end","need","long","hand","place","big","right","high","something",
    "tell","every","found","still","us","set","mr","mrs","ms","dr",
}

# Simple suffix-based stemmer (faster than full Porter, good enough for this task)
def simple_stem(word: str) -> str:
    suffixes = ["ing","tion","tions","ness","ment","ments","ful","ous","ious",
                "er","ers","est","ed","ly","ies","es","s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[:-len(suffix)]
    return word


def clean_text(text: str, use_stemming: bool = True) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove numbers and punctuation
    4. Tokenize (whitespace split)
    5. Remove stopwords
    6. Stem tokens
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove special chars / punctuation (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenize
    tokens = text.split()

    # 7. Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    # 8. Stem
    if use_stemming:
        tokens = [simple_stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df, text_col: str = "text", title_col: str = "title",
                          use_stemming: bool = True):
    """
    Preprocess a DataFrame: combine title + text, clean, return processed copy.
    """
    df = df.copy()

    # Combine title and text for richer features
    if title_col in df.columns and text_col in df.columns:
        df["combined"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    elif text_col in df.columns:
        df["combined"] = df[text_col].fillna("")
    elif title_col in df.columns:
        df["combined"] = df[title_col].fillna("")
    else:
        raise ValueError(f"Neither '{text_col}' nor '{title_col}' found in DataFrame.")

    print("  Cleaning text…", end=" ", flush=True)
    df["cleaned"] = df["combined"].apply(lambda x: clean_text(x, use_stemming))
    print("done.")

    # Drop rows with empty cleaned text
    before = len(df)
    df = df[df["cleaned"].str.len() > 0].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed} empty rows after cleaning.")

    return df


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "BREAKING!! Government secretly PUTTING chemicals in tap water — whistleblower REVEALS!!!",
        "Scientists at MIT discovered a new antibiotic compound using machine learning techniques.",
    ]
    print("=== Preprocessing Smoke Test ===")
    for s in samples:
        print(f"\nRaw  : {s}")
        print(f"Clean: {clean_text(s)}")
