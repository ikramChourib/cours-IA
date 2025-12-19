import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re

# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(page_title="Analyse intelligente des avis", layout="wide")
st.title("🧠 Analyse intelligente des avis clients (NLP + Dashboard)")

# -----------------------------s&
# NLP: Sentiment pipeline
# -----------------------------
# Modèle simple multi-langues (donne un score 1..5 étoiles)
from textblob_fr import PatternTagger, PatternAnalyzer
from textblob import Blobber

tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

def analyze_sentiment(texts):
    rows = []
    for t in texts:
        pol = tb(t).sentiment[0]  # [-1..1]
        if pol > 0.1:
            s = "Positif"
        elif pol < -0.1:
            s = "Négatif"
        else:
            s = "Neutre"
        rows.append({"review": t, "sentiment": s, "score": float(pol), "label_raw": None})
    return pd.DataFrame(rows)


def top_keywords(texts, k=15):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words=None  # simple: pas de stopwords FR pour garder le code léger
    )
    X = vectorizer.fit_transform(texts)
    # moyenne TF-IDF par terme
    mean_tfidf = X.mean(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    df_kw = pd.DataFrame({"keyword": vocab, "tfidf": mean_tfidf}).sort_values("tfidf", ascending=False)
    return df_kw.head(k)

# -----------------------------
# Input UI
# -----------------------------
st.sidebar.header("📥 Données")
mode = st.sidebar.radio("Mode d'entrée", ["Uploader un CSV", "Coller des avis"], index=0)

reviews = []

if mode == "Uploader un CSV":
    uploaded = st.sidebar.file_uploader("CSV avec une colonne 'review'", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        if "review" not in df_in.columns:
            st.error("Le CSV doit contenir une colonne nommée 'review'.")
            st.stop()
        reviews = df_in["review"].dropna().astype(str).tolist()
        st.sidebar.success(f"{len(reviews)} avis chargés ✅")
else:
    raw = st.sidebar.text_area("Colle tes avis (1 avis par ligne)", height=220)
    if raw.strip():
        reviews = [line.strip() for line in raw.splitlines() if line.strip()]
        st.sidebar.success(f"{len(reviews)} avis détectés ✅")

# -----------------------------
# Run analysis
# -----------------------------
if not reviews:
    st.info("➡️ Uploade un CSV (colonne `review`) ou colle des avis pour démarrer.")
    st.stop()

with st.spinner("Analyse en cours (sentiment + mots-clés)..."):
    df_sent = analyze_sentiment(reviews)
    df_kw = top_keywords(df_sent["review"].tolist(), k=20)

# -----------------------------
# Dashboard
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Nombre d'avis", len(df_sent))

with col2:
    pos_rate = (df_sent["sentiment"] == "Positif").mean() * 100
    st.metric("Taux positif", f"{pos_rate:.1f}%")

with col3:
    neg_rate = (df_sent["sentiment"] == "Négatif").mean() * 100
    st.metric("Taux négatif", f"{neg_rate:.1f}%")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("📊 Répartition des sentiments")
    counts = df_sent["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    st.bar_chart(counts.set_index("sentiment"))

with right:
    st.subheader("🔎 Top mots-clés (TF-IDF)")
    st.dataframe(df_kw, use_container_width=True, hide_index=True)

st.divider()

st.subheader("🧾 Détails des avis")
filter_sent = st.multiselect(
    "Filtrer par sentiment",
    options=["Positif", "Neutre", "Négatif"],
    default=["Positif", "Neutre", "Négatif"]
)
df_view = df_sent[df_sent["sentiment"].isin(filter_sent)].copy()

st.dataframe(
    df_view[["sentiment", "label_raw", "score", "review"]],
    use_container_width=True,
    hide_index=True
)

# Export
st.download_button(
    "⬇️ Télécharger les résultats (CSV)",
    data=df_sent.to_csv(index=False).encode("utf-8"),
    file_name="resultats_avis.csv",
    mime="text/csv"
)
