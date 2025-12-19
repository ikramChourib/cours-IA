# app_reviews_dashboard.py
# -*- coding: utf-8 -*-
"""
Dashboard Streamlit : Avis -> nettoyage -> sentiment -> thèmes -> visualisations

Optimisations clés (très importantes pour ton CSV Amazon WDC) :
- Lecture CSV avec usecols (on ignore prices/sourceURLs/etc. très lourds)
- Mapping Amazon robuste (text = title + body)
- Nettoyage rapide (regex HTML au lieu de BeautifulSoup)
- Cache Streamlit : preprocess / sentiment / topics
- BERT topics uniquement sur échantillon (sinon très lent)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import json
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import emoji

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords

# Google Play collector (non-official)
try:
    from google_play_scraper import reviews as gp_reviews, Sort as gp_Sort
except Exception:
    gp_reviews = None
    gp_Sort = None


# -----------------------------
# Helpers: NLTK stopwords
# -----------------------------
@st.cache_resource(show_spinner=False)
def ensure_nltk():
    try:
        _ = stopwords.words("french")
    except LookupError:
        nltk.download("stopwords")


@st.cache_resource(show_spinner=False)
def get_stopwords(lang: str) -> set:
    ensure_nltk()
    if lang.lower().startswith("fr"):
        sw = set(stopwords.words("french"))
    elif lang.lower().startswith("en"):
        sw = set(stopwords.words("english"))
    else:
        sw = set(stopwords.words("french")).union(set(stopwords.words("english")))
    extra = {
        "très", "trop", "pas", "plus", "moins", "encore", "vraiment",
        "application", "appli", "hotel", "hôtel", "restaurant", "séjour"
    }
    return sw.union(extra)


# -----------------------------
# Data collection (upload)
# -----------------------------
def load_uploaded_file(uploaded_file, source_name: str) -> pd.DataFrame:
    """Charge un fichier uploadé via Streamlit (CSV/Parquet/JSON).
    OPTI: lecture CSV avec usecols pour éviter les colonnes énormes (prices, sourceURLs, etc.).
    """
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        # On garde seulement les colonnes utiles au dashboard "reviews"
        keep = {
            "asins", "asin", "name", "brand", "categories",
            "reviews.date", "reviews.rating", "reviews.title", "reviews.text", "reviews.username",
            "date", "rating", "text", "title", "username",  # au cas où autre schéma
        }
        df = pd.read_csv(
            uploaded_file,
            usecols=lambda c: c in keep,
            low_memory=False,
            dtype={
                "asins": "string",
                "asin": "string",
                "name": "string",
                "brand": "string",
                "categories": "string",
                "reviews.title": "string",
                "reviews.text": "string",
                "reviews.username": "string",
                "text": "string",
                "title": "string",
                "username": "string",
            },
        )
    elif name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    elif name.endswith(".json"):
        try:
            obj = json.load(uploaded_file)
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                key = next((k for k in ["data", "reviews", "items"] if k in obj), None)
                df = pd.DataFrame(obj[key]) if key else pd.DataFrame([obj])
            else:
                df = pd.DataFrame()
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_json(uploaded_file)
    else:
        raise ValueError("Format non supporté. Utilise CSV / Parquet / JSON.")

    df["source"] = source_name
    return df


def collect_google_play(app_id: str, lang: str = "fr", country: str = "fr", n: int = 500) -> pd.DataFrame:
    if gp_reviews is None:
        raise RuntimeError("google-play-scraper n'est pas installé. pip install google-play-scraper")

    all_rows = []
    batch_size = 200
    remaining = n
    next_token = None

    while remaining > 0:
        k = min(batch_size, remaining)
        result, next_token = gp_reviews(
            app_id,
            lang=lang,
            country=country,
            sort=gp_Sort.NEWEST,
            count=k,
            continuation_token=next_token
        )
        if not result:
            break

        for r in result:
            all_rows.append({
                "text": r.get("content", ""),
                "rating": r.get("score", None),
                "date": r.get("at", None),
                "author": r.get("userName", None),
                "source": "GooglePlay",
                "item_id": app_id,
                "name": app_id
            })

        remaining -= len(result)
        if next_token is None:
            break
        time.sleep(0.2)

    return pd.DataFrame(all_rows)


# -----------------------------
# Text cleaning / preprocessing (FAST)
# -----------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACES_RE = re.compile(r"\s+")
NON_LETTER_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ\s'-]+")
HTML_RE = re.compile(r"<[^>]+>")  # fast HTML removal


def strip_html_fast(text: str) -> str:
    return HTML_RE.sub(" ", str(text))


def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = strip_html_fast(text)
    t = t.replace("\n", " ").replace("\r", " ")
    t = URL_RE.sub(" ", t)
    t = emoji.replace_emoji(t, replace=" ")
    t = t.lower()
    t = NON_LETTER_RE.sub(" ", t)
    t = SPACES_RE.sub(" ", t).strip()
    return t


def ensure_single_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Si df contient plusieurs colonnes portant le même nom `col`, on les fusionne en 1 seule."""
    if col not in df.columns:
        return df

    mask = (df.columns == col)
    if mask.sum() <= 1:
        return df

    merged = df.loc[:, mask].fillna("").astype(str).agg(" ".join, axis=1)
    df = df.loc[:, ~mask].copy()
    df[col] = merged
    return df


def preprocess_df(df: pd.DataFrame, text_col_candidates=("text", "review", "content", "comment", "body")) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy().reset_index(drop=True)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    df = ensure_single_column(df, "text")
    df = ensure_single_column(df, "text_raw")

    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError("Impossible de trouver une colonne texte.")
        text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())

    df.rename(columns={text_col: "text_raw"}, inplace=True)
    df = ensure_single_column(df, "text_raw")

    df["text_raw"] = df["text_raw"].fillna("").astype(str)

    # fast map + to_numpy (anti reindex)
    df["text_clean"] = df["text_raw"].map(clean_text).to_numpy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df[df["text_clean"].astype(str).str.len() > 0].reset_index(drop=True)
    return df


# -----------------------------
# Schema mapping: Amazon WDC (reviews.*)
# -----------------------------
def map_amazon_wdc_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mappe le dataset Amazon WDC (reviews.text, reviews.rating, reviews.date...)
    vers un schéma standard: text, rating, date, author, item_id, name/brand/categories
    OPTI: construit text = title + body avant rename.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Construire un champ texte complet: title + body
    title = df["reviews.title"] if "reviews.title" in df.columns else df.get("title", None)
    body = df["reviews.text"] if "reviews.text" in df.columns else df.get("text", None)

    if title is not None or body is not None:
        t1 = title.fillna("").astype(str) if title is not None else ""
        t2 = body.fillna("").astype(str) if body is not None else ""
        df["text"] = (t1 + ". " + t2).str.replace(r"\s+", " ", regex=True).str.strip(". ").str.strip()

    # Renames standard
    rename_map = {}
    if "reviews.rating" in df.columns:
        rename_map["reviews.rating"] = "rating"
    if "reviews.date" in df.columns:
        rename_map["reviews.date"] = "date"
    if "reviews.username" in df.columns:
        rename_map["reviews.username"] = "author"
    if "username" in df.columns and "author" not in df.columns:
        rename_map["username"] = "author"

    df.rename(columns=rename_map, inplace=True)

    # item_id
    if "item_id" not in df.columns:
        if "asins" in df.columns:
            df["item_id"] = df["asins"].astype(str)
        elif "asin" in df.columns:
            df["item_id"] = df["asin"].astype(str)
        else:
            df["item_id"] = None

    # S'il n'y a pas de text (fallback)
    if "text" not in df.columns:
        if "reviews.text" in df.columns:
            df["text"] = df["reviews.text"].fillna("").astype(str)
        else:
            df["text"] = ""

    df = ensure_single_column(df, "text")
    return df


# -----------------------------
# Sentiment analysis (multilingual) - faster settings
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_name: str):
    from transformers import pipeline as hf_pipeline
    import torch
    torch.set_num_threads(1)
    return hf_pipeline("sentiment-analysis", model=model_name)


def map_stars_to_sentiment(label: str) -> Tuple[str, float]:
    m = re.search(r"([1-5])", str(label))
    if not m:
        return ("unknown", np.nan)
    stars = int(m.group(1))
    if stars <= 2:
        return ("negative", stars)
    elif stars == 3:
        return ("neutral", stars)
    else:
        return ("positive", stars)


def run_sentiment(df: pd.DataFrame, model_name: str, batch_size: int = 64, max_chars: int = 512) -> pd.DataFrame:
    if df.empty:
        return df
    pipe = load_sentiment_model(model_name)

    texts = df["text_raw"].fillna("").astype(str).str.slice(0, max_chars).tolist()
    preds = []
    for i in range(0, len(texts), batch_size):
        out = pipe(texts[i:i + batch_size], truncation=True)
        preds.extend(out)

    df = df.copy()
    df["sent_label_raw"] = [p.get("label") for p in preds]
    df["sent_score_raw"] = [p.get("score") for p in preds]

    mapped = [map_stars_to_sentiment(lab) for lab in df["sent_label_raw"].tolist()]
    df["sentiment"] = [m[0] for m in mapped]
    df["sent_stars_pred"] = [m[1] for m in mapped]
    return df


# -----------------------------
# Topic modeling: LDA (fast)
# -----------------------------
def run_lda_topics(
    df: pd.DataFrame,
    lang: str = "fr",
    n_topics: int = 8,
    max_features: int = 8000,
    min_df: int = 5,
    max_df: float = 0.5,
    n_top_words: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame()

    sw = get_stopwords(lang)

    vectorizer = CountVectorizer(
        stop_words=list(sw),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["text_clean"].astype(str).tolist())

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )
    doc_topic = lda.fit_transform(X)
    topic_id = doc_topic.argmax(axis=1)

    vocab = np.array(vectorizer.get_feature_names_out())
    topics_rows = []
    for k in range(n_topics):
        top_idx = np.argsort(lda.components_[k])[::-1][:n_top_words]
        topics_rows.append({"topic_id": k, "keywords": ", ".join(vocab[top_idx].tolist())})

    topics_df = pd.DataFrame(topics_rows)

    out = df.copy()
    out["topic_model"] = "LDA"
    out["topic_id"] = topic_id
    out["topic_conf"] = doc_topic.max(axis=1)
    return out, topics_df


# -----------------------------
# Topic modeling: BERT embeddings + clustering (slow) -> sample
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def ctfidf_keywords(texts: List[str], labels: np.ndarray, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    df_tmp = pd.DataFrame({"text": texts, "label": labels})
    clusters = sorted([c for c in df_tmp["label"].unique() if c != -1])
    if not clusters:
        return {}

    docs_per_cluster = (
        df_tmp[df_tmp["label"].isin(clusters)]
        .groupby("label")["text"]
        .apply(lambda x: " ".join(x))
        .to_dict()
    )

    tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
    cluster_ids = list(docs_per_cluster.keys())
    cluster_docs = [docs_per_cluster[c] for c in cluster_ids]
    X = tfidf.fit_transform(cluster_docs)
    X = normalize(X, norm="l1", axis=1)

    terms = np.array(tfidf.get_feature_names_out())
    out = {}
    for idx, c in enumerate(cluster_ids):
        row = X[idx].toarray().ravel()
        top = np.argsort(row)[::-1][:top_n]
        out[c] = [(terms[i], float(row[i])) for i in top if row[i] > 0]
    return out


def run_bert_clustering(
    df: pd.DataFrame,
    lang: str = "fr",
    embed_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    min_cluster_size: int = 15,
    top_words: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import umap
    import hdbscan

    if df.empty:
        return df, pd.DataFrame()

    embedder = load_embedding_model(embed_model_name)
    texts = df["text_clean"].astype(str).tolist()

    embeddings = embedder.encode(
        texts,
        show_progress_bar=False,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        metric="cosine",
        random_state=42
    )
    emb_2 = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(emb_2)

    kw = ctfidf_keywords(texts, labels, top_n=top_words)

    topics_rows = []
    for c, words in sorted(kw.items(), key=lambda x: x[0]):
        topics_rows.append({
            "topic_id": int(c),
            "keywords": ", ".join([w for w, _ in words])
        })
    topics_df = pd.DataFrame(topics_rows)

    out = df.copy()
    out["topic_model"] = "BERT_CLUSTER"
    out["topic_id"] = labels
    out["topic_conf"] = np.nan
    return out, topics_df


# -----------------------------
# Schema unification
# -----------------------------
def unify_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy().reset_index(drop=True)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    for c in ["source", "item_id", "author", "rating", "date", "name", "brand", "categories"]:
        if c not in df.columns:
            df[c] = None

    df = ensure_single_column(df, "text")
    df = ensure_single_column(df, "text_raw")

    if "text_raw" not in df.columns:
        if "text" in df.columns:
            df["text_raw"] = df["text"].astype(str)
        else:
            df["text_raw"] = ""

    if "text_clean" not in df.columns:
        df["text_clean"] = df["text_raw"].fillna("").astype(str).map(clean_text).to_numpy()

    return df


# -----------------------------
# Cached pipeline steps
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = unify_schema(df_raw)
    return preprocess_df(df_raw)


@st.cache_data(show_spinner=False)
def cached_sentiment(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return run_sentiment(df, model_name=model_name)


@st.cache_data(show_spinner=False)
def cached_lda(df: pd.DataFrame, lang: str, n_topics: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_lda_topics(df, lang=lang, n_topics=n_topics)


@st.cache_data(show_spinner=False)
def cached_bert_topics(df: pd.DataFrame, lang: str, embed_model: str, min_cluster_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_bert_clustering(df, lang=lang, embed_model_name=embed_model, min_cluster_size=min_cluster_size)


# -----------------------------
# Dashboard UI (main)
# -----------------------------
def main():
    st.set_page_config(page_title="Avis -> Sentiments & Thèmes", layout="wide")
    st.title("Pipeline Avis : collecte → nettoyage → sentiment → thèmes → dashboard (optimisé)")

    st.markdown(
        """
Ce dashboard fonctionne **sans scraping** :  
- **Amazon** & **TripAdvisor** : charge un **open dataset local** (CSV/Parquet/JSON).  
- **Google Play** : collecte via `google-play-scraper` (lib non-officielle).

**Optimisations activées** : lecture CSV avec `usecols`, nettoyage rapide, cache Streamlit,
BERT topics limité à un échantillon.
        """
    )

    st.sidebar.header("1) Collecte des données")

    use_amazon = st.sidebar.checkbox("Charger Amazon (dataset local)", value=True)
    amazon_file = st.sidebar.file_uploader(
        "Dataset Amazon (CSV/Parquet/JSON)",
        type=["csv", "parquet", "json"]
    )

    use_trip = st.sidebar.checkbox("Charger TripAdvisor (dataset local)", value=False)
    trip_file = st.sidebar.file_uploader(
        "Dataset TripAdvisor (CSV/Parquet/JSON)",
        type=["csv", "parquet", "json"]
    )

    use_gp = st.sidebar.checkbox("Collecter Google Play", value=False)
    gp_app_id = st.sidebar.text_input("Google Play app_id (ex: com.spotify.music)", value="")
    gp_lang = st.sidebar.text_input("Langue (ex: fr)", value="fr")
    gp_country = st.sidebar.text_input("Pays (ex: fr)", value="fr")
    gp_n = st.sidebar.slider("Nombre d'avis Google Play", 50, 2000, 300, 50)

    st.sidebar.header("2) NLP")
    lang = st.sidebar.selectbox("Langue principale (stopwords)", ["fr", "en", "mix"], index=0)

    do_sent = st.sidebar.checkbox("Détecter le sentiment", value=True)
    sentiment_model = st.sidebar.text_input(
        "Modèle sentiment (HuggingFace)",
        value="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    topic_method = st.sidebar.selectbox("Extraction de thèmes", ["Aucun", "LDA", "BERT clustering"], index=1)

    st.sidebar.subheader("Paramètres LDA")
    lda_topics = st.sidebar.slider("Nombre de topics", 3, 20, 8, 1)

    st.sidebar.subheader("Paramètres BERT clustering")
    embed_model = st.sidebar.text_input("Embedding model", value="paraphrase-multilingual-MiniLM-L12-v2")
    min_cluster_size = st.sidebar.slider("min_cluster_size (HDBSCAN)", 5, 60, 15, 1)
    bert_max_n = st.sidebar.slider("Max avis pour BERT (échantillon)", 300, 6000, 2000, 100)

    run_btn = st.sidebar.button("🚀 Exécuter le pipeline")

    if not run_btn:
        st.info("Upload tes fichiers (Amazon/TripAdvisor) ou active Google Play, puis clique sur **Exécuter le pipeline**.")
        return

    dfs = []
    errors = []

    if use_amazon:
        try:
            df_amz = load_uploaded_file(amazon_file, "Amazon")
            if not df_amz.empty:
                df_amz = map_amazon_wdc_schema(df_amz)  # mapping avant concat
                dfs.append(df_amz)
        except Exception as e:
            errors.append(f"Amazon: {e}")

    if use_trip:
        try:
            df_trip = load_uploaded_file(trip_file, "TripAdvisor")
            if not df_trip.empty:
                # si tripadvisor a des colonnes différentes, on laisse preprocess choisir la meilleure colonne texte
                dfs.append(df_trip)
        except Exception as e:
            errors.append(f"TripAdvisor: {e}")

    if use_gp and gp_app_id.strip():
        try:
            dfs.append(collect_google_play(
                gp_app_id.strip(),
                lang=gp_lang.strip(),
                country=gp_country.strip(),
                n=int(gp_n)
            ))
        except Exception as e:
            errors.append(f"Google Play: {e}")

    if errors:
        st.warning("Certaines sources n'ont pas pu être chargées :\n- " + "\n- ".join(errors))

    if not dfs:
        st.error("Aucune donnée chargée. Upload un dataset Amazon/TripAdvisor ou active Google Play.")
        return

    df_raw = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    # Harmonisation + preprocess (cached)
    with st.spinner("Nettoyage & préparation..."):
        df = cached_preprocess(df_raw)

    # dédoublonnage léger (optionnel mais utile)
    # adapte la liste si author/date manquent
    subset_cols = [c for c in ["item_id", "author", "date", "rating", "text_raw"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)

    # Sentiment (cached)
    if do_sent:
        with st.spinner("Analyse de sentiment..."):
            try:
                df = cached_sentiment(df, model_name=sentiment_model)
            except Exception as e:
                st.exception(e)
                st.warning("Sentiment désactivé suite à une erreur.")
                do_sent = False

    # Topics
    topics_table = pd.DataFrame()
    if topic_method == "LDA":
        with st.spinner("Extraction des thèmes (LDA)..."):
            df, topics_table = cached_lda(df, lang=lang, n_topics=int(lda_topics))

    elif topic_method == "BERT clustering":
        with st.spinner("Extraction des thèmes (BERT clustering) sur échantillon..."):
            try:
                df_for_topics = df if len(df) <= int(bert_max_n) else df.sample(int(bert_max_n), random_state=42)
                df_topics, topics_table = cached_bert_topics(
                    df_for_topics,
                    lang=lang,
                    embed_model=embed_model,
                    min_cluster_size=int(min_cluster_size)
                )

                # merge labels back into full df (only for sampled indices)
                df = df.copy()
                df["topic_model"] = None
                df["topic_id"] = np.nan
                df.loc[df_topics.index, "topic_model"] = df_topics["topic_model"]
                df.loc[df_topics.index, "topic_id"] = df_topics["topic_id"].astype(float)

            except Exception as e:
                st.exception(e)
                st.warning("BERT clustering désactivé suite à une erreur.")
                topic_method = "Aucun"

    # -----------------------------
    # Dashboard
    # -----------------------------
    st.success(f"Données prêtes : {len(df):,} avis")

    st.subheader("Filtres")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sources = sorted(df["source"].dropna().unique().tolist())
        sel_sources = st.multiselect("Source", options=sources, default=sources)

    with c2:
        if "sentiment" in df.columns:
            sents = ["negative", "neutral", "positive", "unknown"]
            sel_sent = st.multiselect("Sentiment", options=sents, default=sents)
        else:
            sel_sent = None
            st.caption("Sentiment non calculé.")

    with c3:
        max_len = int(df["text_clean"].astype(str).str.len().max()) if len(df) else 0
        sel_min_len = st.slider("Longueur min (caractères)", 0, max(50, max_len), min(20, max_len))

    with c4:
        query = st.text_input("Recherche (contient)", value="")

    fdf = df[df["source"].isin(sel_sources)].copy()
    fdf = fdf[fdf["text_clean"].astype(str).str.len() >= sel_min_len]

    if sel_sent is not None and "sentiment" in fdf.columns:
        fdf = fdf[fdf["sentiment"].isin(sel_sent)]

    if query.strip():
        q = query.strip().lower()
        fdf = fdf[fdf["text_clean"].astype(str).str.contains(re.escape(q), na=False)]

    st.write(f"Après filtres : **{len(fdf):,}** avis")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Avis", f"{len(fdf):,}")
    with k2:
        if "rating" in fdf.columns and fdf["rating"].notna().any():
            st.metric("Rating moyen", f"{fdf['rating'].mean():.2f}")
        else:
            st.metric("Rating moyen", "—")
    with k3:
        if "sentiment" in fdf.columns:
            pos_rate = (fdf["sentiment"] == "positive").mean() * 100
            st.metric("% Positif", f"{pos_rate:.1f}%")
        else:
            st.metric("% Positif", "—")
    with k4:
        if "date" in fdf.columns and fdf["date"].notna().any():
            st.metric("Période", f"{fdf['date'].min().date()} → {fdf['date'].max().date()}")
        else:
            st.metric("Période", "—")

    st.subheader("Visualisations")
    left, right = st.columns(2)
    with left:
        if "sentiment" in fdf.columns:
            fig = px.histogram(fdf, x="sentiment", title="Répartition des sentiments")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Active l'option 'Détecter le sentiment' pour cette vue.")
    with right:
        if "rating" in fdf.columns and fdf["rating"].notna().any():
            fig = px.histogram(fdf.dropna(subset=["rating"]), x="rating", nbins=10, title="Distribution des ratings")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune colonne rating exploitable.")

    if "date" in fdf.columns and fdf["date"].notna().any():
        st.subheader("Évolution temporelle")
        tmp = fdf.dropna(subset=["date"]).copy()
        tmp["day"] = tmp["date"].dt.date
        agg = tmp.groupby(["day", "source"]).size().reset_index(name="count")
        fig = px.line(agg, x="day", y="count", color="source", title="Nombre d'avis par jour")
        st.plotly_chart(fig, use_container_width=True)

    if topic_method != "Aucun" and "topic_id" in fdf.columns and not topics_table.empty:
        st.subheader("Thèmes")
        t1, t2 = st.columns([1, 2])

        with t1:
            topics_sorted = topics_table.sort_values("topic_id")
            st.dataframe(topics_sorted, use_container_width=True, height=250)
            topic_ids = sorted([int(x) for x in fdf["topic_id"].dropna().unique().tolist()
                                if not pd.isna(x) and int(x) >= 0])
            if topic_ids:
                sel_topic = st.selectbox("Explorer un topic", options=topic_ids, index=0)
            else:
                sel_topic = None
                st.info("Aucun topic disponible (ou seulement -1).")

        with t2:
            if sel_topic is not None:
                subset = fdf[fdf["topic_id"] == sel_topic].copy()
                st.write(f"**{len(subset):,}** avis dans le topic **{sel_topic}**")
                cols = [c for c in ["source", "name", "brand", "rating", "sentiment", "text_raw"] if c in subset.columns]
                st.dataframe(
                    subset[cols].rename(columns={"text_raw": "review"}).head(50),
                    use_container_width=True,
                    height=350
                )

        tc = fdf.dropna(subset=["topic_id"]).groupby("topic_id").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(tc, x="topic_id", y="count", title="Taille des topics (sur avis taggés)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Données")
    show_cols = [c for c in ["source", "item_id", "name", "brand", "author", "date", "rating", "sentiment", "topic_model", "topic_id", "text_raw"] if c in fdf.columns]
    st.dataframe(
        fdf[show_cols].rename(columns={"text_raw": "review"}),
        use_container_width=True,
        height=420
    )

    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger CSV (après pipeline)",
        data=csv,
        file_name="reviews_processed.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
