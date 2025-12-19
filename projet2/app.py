import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Dashboard Tendances Réseaux Sociaux", layout="wide")
st.title("📈 Dashboard d’analyse des tendances sur les réseaux sociaux")

st.sidebar.header("📥 Données")
uploaded = st.sidebar.file_uploader("Uploader un CSV (colonnes: date, platform, user, text, likes, shares, comments)", type=["csv"])

if uploaded is None:
    st.info("➡️ Uploade un CSV pour démarrer (ex: social_posts.csv).")
    st.stop()

df = pd.read_csv(uploaded)

# --- checks simples ---
needed = {"date", "platform", "user", "text"}
missing = needed - set(df.columns)
if missing:
    st.error(f"Colonnes manquantes : {missing}. Il faut au minimum : date, platform, user, text.")
    st.stop()

# parse date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "text"])
df["text"] = df["text"].astype(str)

# engagement (si dispo)
for col in ["likes", "shares", "comments"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["engagement"] = df["likes"] + 2*df["shares"] + df["comments"]

# filtres
platforms = sorted(df["platform"].astype(str).unique().tolist())
selected_platforms = st.sidebar.multiselect("Plateformes", platforms, default=platforms)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Période", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    d1, d2 = pd.to_datetime(min_date), pd.to_datetime(max_date)

df = df[df["platform"].astype(str).isin(selected_platforms)]
df = df[(df["date"] >= d1) & (df["date"] <= d2)]

if df.empty:
    st.warning("Aucune donnée après filtrage.")
    st.stop()

# ---------- Hashtags ----------
def extract_hashtags(text):
    return [h.lower() for h in re.findall(r"#\w+", text)]

df["hashtags"] = df["text"].apply(extract_hashtags)
all_hashtags = df["hashtags"].explode().dropna()

top_hashtags = all_hashtags.value_counts().head(15).reset_index()
top_hashtags.columns = ["hashtag", "count"]

# ---------- Mots-clés (TF-IDF) ----------
def top_keywords(texts, k=15):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    mean = X.mean(axis=0).A1
    vocab = vec.get_feature_names_out()
    out = pd.DataFrame({"keyword": vocab, "score": mean}).sort_values("score", ascending=False)
    return out.head(k)

kw = top_keywords(df["text"].tolist(), k=20)

# ---------- Série temporelle ----------
df["day"] = df["date"].dt.to_period("D").dt.to_timestamp()
daily_posts = df.groupby("day").size().reset_index(name="posts")
daily_eng = df.groupby("day")["engagement"].sum().reset_index(name="engagement_total")

# ---------- Top posts ----------
top_posts = df.sort_values("engagement", ascending=False).head(20)[
    ["date", "platform", "user", "engagement", "likes", "shares", "comments", "text"]
]

# ---------- Dashboard ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Nombre de posts", len(df))
with c2:
    st.metric("Engagement total", int(df["engagement"].sum()))
with c3:
    st.metric("Plateformes", len(selected_platforms))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("🔥 Top hashtags")
    if len(top_hashtags) == 0:
        st.info("Aucun hashtag détecté (#...).")
    else:
        st.bar_chart(top_hashtags.set_index("hashtag"))

with right:
    st.subheader("🔎 Top mots-clés (TF-IDF)")
    st.dataframe(kw, use_container_width=True, hide_index=True)

st.divider()

c4, c5 = st.columns(2)

with c4:
    st.subheader("📆 Volume de posts par jour")
    st.line_chart(daily_posts.set_index("day"))

with c5:
    st.subheader("💥 Engagement total par jour")
    st.line_chart(daily_eng.set_index("day"))

st.divider()

st.subheader("🏆 Top posts (par engagement)")
st.dataframe(top_posts, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Télécharger les posts filtrés (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="posts_filtres.csv",
    mime="text/csv"
)
