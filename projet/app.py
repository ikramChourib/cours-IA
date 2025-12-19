import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import re

# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(page_title="Analyse intelligente des avis", layout="wide")
st.title("🧠 Analyse intelligente des avis clients (NLP + Dashboard)")

# -----------------------------
# NLP: Sentiment pipeline
# -----------------------------
# Modèle simple multi-langues (donne un score 1..5 étoiles)
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

sentiment_model = load_sentiment_model()

def stars_to_sentiment(label: str) -> str:
    # label: "1 star" .. "5 stars"
    m = re.search(r"(\d)", label)
    stars = int(m.group(1)) if m else 3
    if stars <= 2:
        return "Négatif"
    elif stars == 3:
        return "Neutre"
    return "Positif"

def analyze_sentiment(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = sentiment_model(batch)
        for t, p in zip(batch, preds):
            results.append({
                "review": t,
                "label_raw": p["label"],
                "score": float(p["score"]),
                "sentiment": stars_to_sentiment(p["label"])
            })
    return pd.DataFrame(results)

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
