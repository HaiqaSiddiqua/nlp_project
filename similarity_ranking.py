import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util


# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("cleaned_travel_data.csv")

# Tokenize text (used for Word2Vec training)
df["tokens"] = df["cleaned_text"].apply(lambda x: x.split())


# -----------------------------------------------------
# 2. TF-IDF MODEL (Precompute vectors)
# -----------------------------------------------------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])


# -----------------------------------------------------
# 3. SBERT MODEL (Precompute embeddings)
# -----------------------------------------------------
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_embeddings = sbert_model.encode(
    df["cleaned_text"].tolist(),
    convert_to_tensor=True
)


# -----------------------------------------------------
# 4. WORD2VEC MODEL + Sentence Vector Function
# -----------------------------------------------------
w2v_model = Word2Vec(
    df["tokens"].tolist(), 
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=20
)


def sentence_vector(tokens, model, dim=100):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


# Precompute Word2Vec embeddings for all rows
df["w2v_vector"] = df["tokens"].apply(lambda x: sentence_vector(x, w2v_model))


# -----------------------------------------------------
# 5. SPACY FOR QUERY PROCESSING
# -----------------------------------------------------
nlp = spacy.load("en_core_web_sm")


def preprocess_query(query):
    doc = nlp(query.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]


# -----------------------------------------------------
# 6. MAIN FUNCTION TO GET TOP RESULTS
# -----------------------------------------------------
def get_results(query, top_n=5):

    # ---------------- TF-IDF ----------------
    q_tfidf = tfidf.transform([query])
    tfidf_scores = cosine_similarity(q_tfidf, tfidf_matrix).flatten()

    tfidf_rank = np.argsort(tfidf_scores)[::-1][:top_n]
    tfidf_results = [
        {"key": df.iloc[i]["_key"], "score": float(tfidf_scores[i])}
        for i in tfidf_rank
    ]

    # ---------------- SBERT ----------------
    q_sbert = sbert_model.encode([query], convert_to_tensor=True)
    sbert_scores = util.cos_sim(q_sbert, sbert_embeddings).cpu().numpy().flatten()

    sbert_rank = np.argsort(sbert_scores)[::-1][:top_n]
    sbert_results = [
        {"key": df.iloc[i]["_key"], "score": float(sbert_scores[i])}
        for i in sbert_rank
    ]

    # ---------------- Word2Vec ----------------
    q_tokens = preprocess_query(query)
    q_vec = sentence_vector(q_tokens, w2v_model).reshape(1, -1)

    all_vecs = np.vstack(df["w2v_vector"].values)
    w2v_scores = cosine_similarity(q_vec, all_vecs)[0]

    w2v_rank = np.argsort(w2v_scores)[::-1][:top_n]
    w2v_results = [
        {"key": df.iloc[i]["_key"], "score": float(w2v_scores[i])}
        for i in w2v_rank
    ]

    # ------------ Return Everything Cleanly -----------
    return {
        "TFIDF": tfidf_results,
        "SBERT": sbert_results,
        "Word2Vec": w2v_results,
    }
