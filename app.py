import os, re, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                              silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from difflib import get_close_matches

st.set_page_config(
    page_title="Netflix Content Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
:root {
    --netflix-red:#E50914; --netflix-dark:#141414; --netflix-card:#1f1f1f;
    --netflix-card2:#2a2a2a; --netflix-text:#e5e5e5; --netflix-muted:#808080;
    --netflix-gold:#f5c518; --netflix-green:#46d369;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background-color:var(--netflix-dark)!important;color:var(--netflix-text)!important;}
.main{background-color:var(--netflix-dark)!important;}
.block-container{padding-top:2rem;padding-bottom:2rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d0d0d 0%,#1a1a1a 100%)!important;border-right:1px solid #333;}
[data-testid="stSidebar"] *{color:var(--netflix-text)!important;}
.netflix-logo{font-size:2.2rem;font-weight:900;color:var(--netflix-red);letter-spacing:-1px;text-align:center;padding:0.5rem 0 1.5rem 0;text-shadow:0 0 30px rgba(229,9,20,0.4);}
.hero-banner{background:linear-gradient(135deg,#0d0d0d 0%,#1a0000 40%,#2d0000 100%);border:1px solid #3a0000;border-radius:16px;padding:3rem 3rem 2.5rem 3rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse at top right,rgba(229,9,20,0.15) 0%,transparent 60%);}
.hero-title{font-size:3rem;font-weight:900;background:linear-gradient(90deg,#fff 0%,#e50914 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin-bottom:0.5rem;}
.hero-sub{font-size:1.1rem;color:#aaa;font-weight:300;}
.metric-card{background:linear-gradient(135deg,var(--netflix-card) 0%,var(--netflix-card2) 100%);border:1px solid #333;border-radius:12px;padding:1.5rem 1.2rem;text-align:center;transition:transform 0.2s,border-color 0.2s,box-shadow 0.2s;}
.metric-card:hover{transform:translateY(-4px);border-color:var(--netflix-red);box-shadow:0 8px 32px rgba(229,9,20,0.2);}
.metric-value{font-size:2.4rem;font-weight:900;color:var(--netflix-red);line-height:1;}
.metric-label{font-size:0.85rem;color:var(--netflix-muted);margin-top:0.3rem;text-transform:uppercase;letter-spacing:1px;}
.metric-icon{font-size:1.8rem;margin-bottom:0.5rem;}
.section-header{display:flex;align-items:center;gap:0.75rem;font-size:1.6rem;font-weight:700;border-left:4px solid var(--netflix-red);padding-left:1rem;margin:2rem 0 1.2rem 0;}
.rec-card{background:linear-gradient(135deg,#1e1e1e,#252525);border:1px solid #333;border-radius:12px;padding:1.2rem;margin-bottom:0.8rem;transition:all 0.2s;position:relative;overflow:hidden;}
.rec-card:hover{border-color:var(--netflix-red);box-shadow:0 4px 24px rgba(229,9,20,0.15);transform:translateX(4px);}
.rec-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--netflix-red);opacity:0;transition:opacity 0.2s;}
.rec-card:hover::before{opacity:1;}
.rec-title{font-size:1rem;font-weight:700;color:#fff;}
.rec-meta{font-size:0.78rem;color:var(--netflix-muted);margin-top:0.2rem;}
.badge{display:inline-block;padding:0.15rem 0.55rem;border-radius:20px;font-size:0.72rem;font-weight:600;margin-right:0.3rem;margin-top:0.4rem;}
.badge-excellent{background:rgba(70,211,105,0.2);color:var(--netflix-green);border:1px solid rgba(70,211,105,0.4);}
.badge-good{background:rgba(245,197,24,0.2);color:var(--netflix-gold);border:1px solid rgba(245,197,24,0.4);}
.badge-fair{background:rgba(229,9,20,0.2);color:#ff6b6b;border:1px solid rgba(229,9,20,0.4);}
.badge-type{background:rgba(255,255,255,0.1);color:#ccc;border:1px solid rgba(255,255,255,0.15);}
.score-bar-bg{background:#333;border-radius:4px;height:6px;margin-top:0.5rem;}
.score-bar-fill{height:6px;border-radius:4px;background:linear-gradient(90deg,#e50914,#ff6b6b);}
.xai-box{background:linear-gradient(135deg,#0f0f28,#1a1a38);border:1px solid #2a2a5a;border-radius:12px;padding:1.4rem;margin-top:0.5rem;}
.xai-title{font-size:1rem;font-weight:700;color:#9ba8f0;margin-bottom:1rem;}
.xai-row{display:flex;align-items:center;gap:0.75rem;margin-bottom:0.7rem;}
.xai-label{font-size:0.78rem;color:#aaa;width:145px;flex-shrink:0;}
.xai-bar-bg{flex:1;background:#2a2a3a;border-radius:4px;height:9px;}
.xai-bar-fill{height:9px;border-radius:4px;}
.xai-val{font-size:0.78rem;color:#fff;width:50px;text-align:right;flex-shrink:0;font-weight:700;}
.shared-tag{display:inline-block;background:rgba(123,140,222,0.12);color:#9ba8f0;border:1px solid rgba(123,140,222,0.3);border-radius:20px;padding:0.15rem 0.6rem;font-size:0.72rem;margin:0.2rem;}
.desc-box{background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:0.9rem 1rem;font-size:0.8rem;color:#bbb;line-height:1.6;margin-top:0.4rem;}
.cluster-badge{display:inline-block;background:rgba(70,211,105,0.12);color:#46d369;border:1px solid rgba(70,211,105,0.3);border-radius:20px;padding:0.2rem 0.8rem;font-size:0.78rem;font-weight:700;}
.weight-info{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:0.8rem 1rem;font-size:0.8rem;color:#aaa;margin-bottom:1rem;}
[data-baseweb="tab-list"]{background:#1a1a1a!important;border-bottom:2px solid #e50914!important;gap:0;}
[data-baseweb="tab"]{color:#808080!important;padding:0.6rem 1.5rem!important;font-weight:600!important;}
[aria-selected="true"][data-baseweb="tab"]{color:#fff!important;border-bottom:2px solid #e50914!important;}
[data-testid="stSelectbox"]>div,[data-testid="stSlider"]{color:#fff!important;}
input,textarea,select{background:#1f1f1f!important;color:#fff!important;border:1px solid #444!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#1a1a1a;}
::-webkit-scrollbar-thumb{background:#e50914;border-radius:3px;}
hr{border-color:#333!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "netflix_titles_cleaned.csv")
    df = pd.read_csv(path).reset_index(drop=True)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    return df

def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()

def names_to_tokens(s):
    s = "" if pd.isna(s) else str(s)
    return " ".join(p.strip().lower().replace(" ", "_") for p in s.split(",") if p.strip())

def normalize_title(t):
    t = "" if pd.isna(t) else str(t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t.lower().strip())).strip()

def genre_set(s):
    if pd.isna(s): return set()
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}

def cast_set(s):
    if pd.isna(s): return set()
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}

def jaccard(a, b):
    if not a and not b: return 0.0
    return len(a & b) / len(a | b)

def year_closeness(y1, y2, scale=10.0):
    if pd.isna(y1) or pd.isna(y2): return 0.0
    return float(np.exp(-abs(float(y1) - float(y2)) / scale))

def minmax_scale(values):
    v = np.asarray(values, dtype=float)
    mn, mx = np.nanmin(v), np.nanmax(v)
    if np.isclose(mn, mx): return np.zeros_like(v)
    return (v - mn) / (mx - mn)

def match_quality(score):
    if score >= 0.70: return "Excellent"
    if score >= 0.45: return "Good"
    if score >= 0.25: return "Fair"
    return "Weak"

def repeat_tokens(text, weight):
    tokens = text.split()
    return " ".join(t for t in tokens for _ in range(weight))

def build_profile(row):
    parts = [
        repeat_tokens(clean_text(row.get("listed_in", "")), 3),
        repeat_tokens(names_to_tokens(row.get("cast", "")), 2),
        repeat_tokens(names_to_tokens(row.get("director", "")), 2),
        repeat_tokens(clean_text(row.get("description", "")), 4),
        repeat_tokens(clean_text(row.get("type", "")), 1),
        repeat_tokens(clean_text(row.get("rating", "")), 1),
    ]
    return " ".join(p for p in parts if p)

@st.cache_resource
def build_model(_hash):
    df = load_data()
    df["content_profile"] = df.apply(build_profile, axis=1)
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.80,
                            ngram_range=(1, 1), max_features=20_000, sublinear_tf=True)
    X = tfidf.fit_transform(df["content_profile"])
    df["title_norm"] = df["title"].apply(normalize_title)
    idx_by_title = defaultdict(list)
    for row_idx, tn in df["title_norm"].items():
        idx_by_title[tn].append(row_idx)
    for key in idx_by_title:
        idx_by_title[key].sort(
            key=lambda i: df.loc[i, "release_year"] if not pd.isna(df.loc[i, "release_year"]) else 0,
            reverse=True)
    return df, X, tfidf, idx_by_title

@st.cache_resource
def build_clusters(n_clusters=8):
    df, X, _, _ = build_model(0)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_red = normalize(svd.fit_transform(X))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_red)
    sil = silhouette_score(X_red, labels, sample_size=2000, random_state=42)
    dbi = davies_bouldin_score(X_red, labels)
    chi = calinski_harabasz_score(X_red, labels)
    return labels, X_red, sil, dbi, chi, km

# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def _score_candidates(q_idx, df, X):
    q_year   = df.loc[q_idx, "release_year"]
    q_genres = genre_set(df.loc[q_idx, "listed_in"])
    text_sims = cosine_similarity(X[q_idx:q_idx+1], X).ravel()
    cand = df[["title","type","release_year","rating","listed_in","description","cast","director"]].copy()
    cand["text_sim"]      = text_sims
    cand["genre_overlap"] = cand["listed_in"].apply(lambda x: jaccard(q_genres, genre_set(x)))
    cand["year_close"]    = cand["release_year"].apply(lambda y: year_closeness(q_year, y))
    return cand

def get_recommendations(title, top_n, df, X, idx_by_title, w_text, w_genre, w_year):
    tnorm = normalize_title(title)
    if tnorm not in idx_by_title:
        close = get_close_matches(tnorm, list(idx_by_title.keys()), n=5, cutoff=0.5)
        return None, close
    q_idx = idx_by_title[tnorm][0]
    cand = _score_candidates(q_idx, df, X)
    cand = cand.loc[cand.index != q_idx].copy()
    cand["text_sim_n"]  = minmax_scale(cand["text_sim"].to_numpy())
    cand["genre_n"]     = minmax_scale(cand["genre_overlap"].to_numpy())
    cand["year_n"]      = minmax_scale(cand["year_close"].to_numpy())
    cand["final_score"] = w_text*cand["text_sim_n"] + w_genre*cand["genre_n"] + w_year*cand["year_n"]
    cand["match_quality"] = cand["final_score"].apply(match_quality)
    result = cand.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=False)
    result = result.rename(columns={"index": "orig_idx"})
    query_row = df.loc[q_idx]
    return result, query_row

def get_batch_recommendations(seed_titles, top_n, df, X, idx_by_title, w_text, w_genre, w_year):
    resolved, failed = {}, []
    for title in seed_titles:
        tnorm = normalize_title(title)
        if tnorm in idx_by_title:
            resolved[title] = idx_by_title[tnorm][0]
        else:
            close = get_close_matches(tnorm, list(idx_by_title.keys()), n=1, cutoff=0.5)
            if close:
                resolved[title] = idx_by_title[close[0]][0]
            else:
                failed.append(title)
    if not resolved:
        return None, failed
    seed_indices = set(resolved.values())
    agg_text  = np.zeros(len(df))
    agg_genre = np.zeros(len(df))
    agg_year  = np.zeros(len(df))
    for q_idx in seed_indices:
        cand = _score_candidates(q_idx, df, X)
        agg_text  += cand["text_sim"].to_numpy()
        agg_genre += cand["genre_overlap"].to_numpy()
        agg_year  += cand["year_close"].to_numpy()
    n = len(seed_indices)
    agg_text /= n; agg_genre /= n; agg_year /= n
    rdf = df[["title","type","release_year","rating","listed_in","description","cast","director"]].copy()
    rdf["text_sim"]      = agg_text
    rdf["genre_overlap"] = agg_genre
    rdf["year_close"]    = agg_year
    rdf = rdf.loc[~rdf.index.isin(seed_indices)].copy()
    rdf["text_sim_n"]  = minmax_scale(rdf["text_sim"].to_numpy())
    rdf["genre_n"]     = minmax_scale(rdf["genre_overlap"].to_numpy())
    rdf["year_n"]      = minmax_scale(rdf["year_close"].to_numpy())
    rdf["final_score"] = w_text*rdf["text_sim_n"] + w_genre*rdf["genre_n"] + w_year*rdf["year_n"]
    rdf["match_quality"] = rdf["final_score"].apply(match_quality)
    rdf = rdf.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=False)
    rdf = rdf.rename(columns={"index": "orig_idx"})
    return rdf, failed

def explain_recommendation(query_row, rec_row, cluster_labels=None, df=None):
    """Full XAI explanation including clusters, descriptions, shared features."""
    q_genres = genre_set(query_row.get("listed_in",""))
    r_genres = genre_set(rec_row.get("listed_in",""))
    shared_genres = sorted(q_genres & r_genres)

    q_cast = cast_set(query_row.get("cast",""))
    r_cast = cast_set(rec_row.get("cast",""))
    shared_cast = sorted(q_cast & r_cast)

    q_dir = str(query_row.get("director","") or "").strip().lower()
    r_dir = str(rec_row.get("director","") or "").strip().lower()
    same_director = bool(q_dir and q_dir == r_dir and q_dir not in ("","nan"))

    year_diff = None
    qy = query_row.get("release_year")
    ry = rec_row.get("release_year")
    if not pd.isna(qy) and not pd.isna(ry):
        year_diff = abs(int(qy) - int(ry))

    # Cluster info
    same_cluster = None
    cluster_id   = None
    if cluster_labels is not None and df is not None:
        # find query title index
        qt_norm = normalize_title(str(query_row.get("title","")))
        qt_matches = df[df["title"].apply(normalize_title) == qt_norm].index
        rt_idx = int(rec_row.get("orig_idx", -1)) if "orig_idx" in rec_row else -1
        if len(qt_matches) > 0 and rt_idx >= 0:
            q_cluster = cluster_labels[qt_matches[0]]
            r_cluster = cluster_labels[rt_idx]
            same_cluster = (q_cluster == r_cluster)
            cluster_id   = int(q_cluster)

    # descriptions
    q_desc = str(query_row.get("description","") or "")
    r_desc = str(rec_row.get("description","") or "")

    return {
        "shared_genres":  shared_genres,
        "shared_cast":    shared_cast,
        "same_director":  same_director,
        "director":       q_dir.replace("_"," ").title() if same_director else None,
        "year_diff":      year_diff,
        "same_cluster":   same_cluster,
        "cluster_id":     cluster_id,
        "q_desc":         q_desc[:250] + ("…" if len(q_desc)>250 else ""),
        "r_desc":         r_desc[:250] + ("…" if len(r_desc)>250 else ""),
        "text_sim_n":     float(rec_row.get("text_sim_n", 0)),
        "genre_n":        float(rec_row.get("genre_n", 0)),
        "year_n":         float(rec_row.get("year_n", 0)),
        "final_score":    float(rec_row.get("final_score", 0)),
        "text_sim_raw":   float(rec_row.get("text_sim", 0)),
        "genre_raw":      float(rec_row.get("genre_overlap", 0)),
        "year_raw":       float(rec_row.get("year_close", 0)),
    }

# ─────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
    font=dict(color="#e5e5e5", family="Inter"),
    title_font=dict(color="#ffffff", size=15),
    legend=dict(bgcolor="#1f1f1f", bordercolor="#333", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=30),
)
NF_COLORS = ["#E50914","#FF6B6B","#FF8C00","#FFD700","#46D369",
             "#00B4D8","#7B4FFF","#FF69B4","#20B2AA","#FA8072"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="netflix-logo">NETFLIX</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate",
                    ["🏠 Overview","📊 EDA","🔵 Clustering","🎬 Recommend"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="color:#555;font-size:0.75rem;text-align:center;">🎬 Content-Based Recommender<br>Netflix Dataset · 8,804 titles</p>',
                unsafe_allow_html=True)

with st.spinner("Loading Netflix catalog..."):
    df = load_data()

# ═══════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Netflix Content<br>Recommendation System</div>
        <div class="hero-sub">Content-Based AI · TF-IDF · Hybrid Scoring · KMeans Clustering</div>
    </div>""", unsafe_allow_html=True)

    n_movies    = int((df["type"]=="Movie").sum())
    n_shows     = int((df["type"]=="TV Show").sum())
    n_countries = df["primary_country"].replace("unknown", np.nan).dropna().nunique()

    c1,c2,c3,c4 = st.columns(4)
    for col,icon,val,lbl in [
        (c1,"🎬",f"{len(df):,}","Total Titles"),
        (c2,"🎥",f"{n_movies:,}","Movies"),
        (c3,"📺",f"{n_shows:,}","TV Shows"),
        (c4,"🌍",f"{n_countries}","Countries"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl,cr = st.columns([1.1,1])
    with cl:
        st.markdown('<div class="section-header">📅 Content Added Over Time</div>', unsafe_allow_html=True)
        td = df.groupby(["year_added","type"]).size().reset_index(name="count").dropna(subset=["year_added"])
        fig = px.area(td, x="year_added", y="count", color="type",
                      color_discrete_map={"Movie":"#E50914","TV Show":"#FF6B6B"},
                      labels={"year_added":"Year","count":"Titles Added","type":"Type"})
        fig.update_layout(**PLOT_LAYOUT); fig.update_traces(line_width=2)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.markdown('<div class="section-header">🎭 Content Split</div>', unsafe_allow_html=True)
        tc = df["type"].value_counts()
        fig2 = go.Figure(go.Pie(labels=tc.index, values=tc.values, hole=0.6,
                                marker=dict(colors=["#E50914","#FF6B6B"],line=dict(color="#141414",width=2)),
                                textfont=dict(color="#fff")))
        fig2.add_annotation(text=f"{len(df):,}<br><span style='font-size:10px'>Titles</span>",
                            x=0.5,y=0.5,showarrow=False,font=dict(size=18,color="#fff"))
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">⚙️ Recommendation Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:1.5rem;font-size:0.9rem;line-height:2;">
    <b style="color:#E50914;">Step 1 — Content Profile</b>&nbsp; Build weighted text (genres×3, desc×4, cast×2, director×2)<br>
    <b style="color:#E50914;">Step 2 — TF-IDF</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Vectorize into a 20k-feature sparse matrix<br>
    <b style="color:#E50914;">Step 3 — Cosine Similarity</b>&nbsp; Measure angle-based similarity across all title pairs<br>
    <b style="color:#E50914;">Step 4 — Hybrid Score</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w_text×TextSim + w_genre×GenreOverlap + w_year×YearClose<br>
    <b style="color:#E50914;">Step 5 — Clustering</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; KMeans (k=8) on SVD-reduced TF-IDF for content grouping<br>
    <b style="color:#E50914;">Step 6 — XAI</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Full explanation: scores, shared genres/cast/director, cluster, descriptions
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="hero-title" style="font-size:2rem;margin-bottom:1.5rem;">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(["🎭 Genres","🌍 Countries","📆 Trends","⏱️ Duration"])

    with tab1:
        st.markdown('<div class="section-header">Top Genres by Count</div>', unsafe_allow_html=True)
        gs = df["listed_in"].dropna().str.split(",").explode().str.strip()
        tg = gs.value_counts().head(20).reset_index(); tg.columns=["Genre","Count"]
        fig = px.bar(tg, x="Count", y="Genre", orientation="h",
                     color="Count", color_continuous_scale=["#600000","#E50914","#FF6B6B"],
                     labels={"Genre":"","Count":"Number of Titles"})
        fig.update_layout(**PLOT_LAYOUT, height=520, yaxis={"categoryorder":"total ascending"})
        fig.update_coloraxes(showscale=False); st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Genre Distribution by Type</div>', unsafe_allow_html=True)
        gt = (df[["type","listed_in"]].dropna().assign(Genre=df["listed_in"].str.split(",")).explode("Genre"))
        gt["Genre"] = gt["Genre"].str.strip()
        gtc = gt.groupby(["Genre","type"]).size().reset_index(name="Count")
        top15 = gs.value_counts().head(15).index
        fig2 = px.bar(gtc[gtc["Genre"].isin(top15)], x="Count", y="Genre", color="type",
                      orientation="h", barmode="group",
                      color_discrete_map={"Movie":"#E50914","TV Show":"#FF6B6B"})
        fig2.update_layout(**PLOT_LAYOUT, height=480, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Top 20 Countries</div>', unsafe_allow_html=True)
        cd = df["primary_country"].replace("unknown",np.nan).dropna().value_counts().head(20).reset_index()
        cd.columns=["Country","Count"]
        fig = px.bar(cd, x="Country", y="Count",
                     color="Count", color_continuous_scale=["#600000","#E50914"],
                     labels={"Count":"Titles","Country":""})
        fig.update_layout(**PLOT_LAYOUT, height=400); fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Global Content Map</div>', unsafe_allow_html=True)
        md = df["primary_country"].replace("unknown",np.nan).dropna().value_counts().reset_index()
        md.columns=["Country","Count"]
        fig2 = px.choropleth(md, locations="Country", locationmode="country names", color="Count",
                             color_continuous_scale=["#141414","#600000","#E50914"], labels={"Count":"Titles"})
        fig2.update_layout(**PLOT_LAYOUT, height=400, geo=dict(bgcolor="#141414",showframe=False))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Releases Over the Years</div>', unsafe_allow_html=True)
        yr = df.groupby(["release_year","type"]).size().reset_index(name="count")
        yr = yr[yr["release_year"]>=1990]
        fig = px.line(yr, x="release_year", y="count", color="type",
                      color_discrete_map={"Movie":"#E50914","TV Show":"#FF6B6B"},
                      markers=True, labels={"release_year":"Year","count":"Releases","type":""})
        fig.update_layout(**PLOT_LAYOUT); fig.update_traces(line_width=2, marker_size=4)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Content Rating Distribution</div>', unsafe_allow_html=True)
        rd = df["rating"].dropna().value_counts().reset_index(); rd.columns=["Rating","Count"]
        fig2 = px.pie(rd, names="Rating", values="Count", hole=0.45, color_discrete_sequence=NF_COLORS)
        fig2.update_layout(**PLOT_LAYOUT); fig2.update_traces(textfont_color="#fff")
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-header">Movie Duration Distribution</div>', unsafe_allow_html=True)
        mv = df[(df["type"]=="Movie") & df["duration_value"].notna()]
        fig = px.histogram(mv, x="duration_value", nbins=60,
                           color_discrete_sequence=["#E50914"],
                           labels={"duration_value":"Duration (minutes)","count":"Count"})
        fig.update_layout(**PLOT_LAYOUT); st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">TV Show Seasons Distribution</div>', unsafe_allow_html=True)
        sw = df[(df["type"]=="TV Show") & df["duration_value"].notna()]
        sc = sw["duration_value"].value_counts().reset_index(); sc.columns=["Seasons","Count"]
        sc = sc[sc["Seasons"]<=15].sort_values("Seasons")
        fig2 = px.bar(sc, x="Seasons", y="Count",
                      color="Count", color_continuous_scale=["#600000","#E50914"],
                      labels={"Seasons":"Number of Seasons","Count":"Shows"})
        fig2.update_layout(**PLOT_LAYOUT); fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.markdown('<div class="hero-title" style="font-size:2rem;margin-bottom:1.5rem;">🔵 Content Clustering</div>', unsafe_allow_html=True)
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 4, 16, 8)

    with st.spinner("Running KMeans clustering…"):
        labels, X_red, sil, dbi, chi, km = build_clusters(n_clusters)

    df_cls = df.copy(); df_cls["cluster"] = labels

    c1,c2,c3 = st.columns(3)
    for col,icon,val,lbl,tip in [
        (c1,"📐",f"{sil:.4f}","Silhouette Score","Closer to 1 = better"),
        (c2,"📉",f"{dbi:.4f}","Davies-Bouldin Index","Closer to 0 = better"),
        (c3,"📈",f"{chi:,.0f}","Calinski-Harabasz","Higher = better"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div><div style="font-size:0.72rem;color:#555;margin-top:0.3rem;">{tip}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🗺️ Cluster Scatter (2D PCA)</div>', unsafe_allow_html=True)
    from sklearn.decomposition import PCA
    coords = PCA(n_components=2, random_state=42).fit_transform(X_red)
    sdf = pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Cluster":labels.astype(str),
                         "Title":df["title"],"Type":df["type"],"Genre":df["listed_in"].fillna("")})
    fig = px.scatter(sdf.sample(min(3000,len(sdf)),random_state=42),
                     x="PC1",y="PC2",color="Cluster",
                     hover_data={"Title":True,"Type":True,"PC1":False,"PC2":False},
                     color_discrete_sequence=NF_COLORS, opacity=0.7)
    fig.update_traces(marker_size=4); fig.update_layout(**PLOT_LAYOUT, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📋 Cluster Profiles</div>', unsafe_allow_html=True)
    for cid in sorted(df_cls["cluster"].unique()):
        sub = df_cls[df_cls["cluster"]==cid]
        tg = (sub["listed_in"].dropna().str.split(",").explode().str.strip()
              .value_counts().head(3).index.tolist())
        n_mov = int((sub["type"]=="Movie").sum())
        n_sho = int((sub["type"]=="TV Show").sum())
        yr_med = int(sub["release_year"].median()) if sub["release_year"].notna().any() else "N/A"
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">🔵 Cluster {cid} &nbsp;—&nbsp; {len(sub)} titles</div>
            <div class="rec-meta">
                <span class="badge badge-type">🎥 {n_mov} Movies</span>
                <span class="badge badge-type">📺 {n_sho} TV Shows</span>
                <span class="badge badge-type">📅 Median Year: {yr_med}</span>
            </div>
            <div class="rec-meta" style="margin-top:0.4rem;">Top genres: {', '.join(tg) if tg else 'N/A'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Cluster Size Comparison</div>', unsafe_allow_html=True)
    cs = df_cls.groupby(["cluster","type"]).size().reset_index(name="count")
    cs["cluster"] = "Cluster "+cs["cluster"].astype(str)
    fig2 = px.bar(cs, x="cluster", y="count", color="type", barmode="stack",
                  color_discrete_map={"Movie":"#E50914","TV Show":"#FF6B6B"},
                  labels={"cluster":"","count":"Titles","type":""})
    fig2.update_layout(**PLOT_LAYOUT); st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════
# RECOMMEND
# ═══════════════════════════════════════════════
elif page == "🎬 Recommend":
    st.markdown('<div class="hero-title" style="font-size:2rem;margin-bottom:1.5rem;">🎬 Find Your Next Watch</div>',
                unsafe_allow_html=True)

    with st.spinner("Preparing model…"):
        df_m, X_m, tfidf_m, idx_m = build_model(0)

    # Pre-compute clusters for XAI
    with st.spinner("Loading clusters for XAI…"):
        cl_labels, _, _, _, _, _ = build_clusters(8)

    all_titles = sorted(df_m["title"].dropna().unique().tolist())

    # ── Weight sliders (sidebar) ─────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚖️ Scoring Weights")
        st.markdown('<div style="font-size:0.75rem;color:#666;">Adjust how much each factor contributes. Weights are auto-normalised.</div>',
                    unsafe_allow_html=True)
        w_text_raw  = st.slider("🧠 Text Similarity",  0, 10, 6)
        w_genre_raw = st.slider("🎭 Genre Overlap",    0, 10, 3)
        w_year_raw  = st.slider("📅 Year Closeness",   0, 10, 1)
        total_w = w_text_raw + w_genre_raw + w_year_raw
        if total_w == 0: total_w = 1
        w_text  = w_text_raw  / total_w
        w_genre = w_genre_raw / total_w
        w_year  = w_year_raw  / total_w
        st.markdown(f'<div class="weight-info">Normalised: Text <b>{w_text:.2f}</b> · Genre <b>{w_genre:.2f}</b> · Year <b>{w_year:.2f}</b></div>',
                    unsafe_allow_html=True)

    # ── Mode toggle ──────────────────────────────
    mode = st.radio("Recommendation Mode",
                    ["🎯 Single Title","🧺 Batch (Multi-Seed)"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # SINGLE MODE
    # ════════════════════════════════════════════
    if mode == "🎯 Single Title":
        ci, cc = st.columns([2,1])
        with ci:
            selected = st.selectbox("🔍 Search for a title", options=all_titles,
                                    index=all_titles.index("Breaking Bad") if "Breaking Bad" in all_titles else 0)
            filter_type = st.radio("Filter results by", ["All","Movie","TV Show"], horizontal=True)
        with cc:
            top_n = st.slider("Number of recommendations", 5, 20, 10)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🎬 Get Recommendations", use_container_width=True)

        if run_btn or selected:
            with st.spinner("Computing recommendations…"):
                result, query = get_recommendations(selected, top_n+5, df_m, X_m, idx_m, w_text, w_genre, w_year)

            if result is None:
                st.error(f"Title not found. Did you mean: {', '.join(query) if query else 'N/A'}")
            else:
                q = query
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a0000,#2d0000);border:1px solid #3a0000;
                     border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;">
                    <div style="font-size:1.4rem;font-weight:900;color:#fff;">📌 {q['title']}</div>
                    <div style="color:#aaa;font-size:0.85rem;margin-top:0.3rem;">
                        <span class="badge badge-type">{q['type']}</span>
                        <span class="badge badge-type">⭐ {q.get('rating','N/A')}</span>
                        <span class="badge badge-type">📅 {int(q['release_year']) if not pd.isna(q['release_year']) else 'N/A'}</span>
                    </div>
                    <div style="color:#ccc;font-size:0.85rem;margin-top:0.6rem;">🎭 {q['listed_in']}</div>
                    <div style="color:#999;font-size:0.82rem;margin-top:0.5rem;">{str(q.get('description',''))[:280]}…</div>
                </div>""", unsafe_allow_html=True)

                if filter_type != "All":
                    result = result[result["type"]==filter_type]
                result = result.head(top_n)

                st.markdown(f'<div class="section-header">🎯 Top {len(result)} Recommendations</div>', unsafe_allow_html=True)

                cl, cr = st.columns([1.2,1])
                with cl:
                    for _, row in result.iterrows():
                        mq = row["match_quality"]
                        badge_cls = {"Excellent":"badge-excellent","Good":"badge-good"}.get(mq,"badge-fair")
                        score_pct = int(row["final_score"]*100)
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-title">{row['title']}</div>
                            <div class="rec-meta">
                                <span class="badge badge-type">{row['type']}</span>
                                <span class="badge badge-type">📅 {int(row['release_year']) if not pd.isna(row['release_year']) else 'N/A'}</span>
                                <span class="badge badge-type">⭐ {row.get('rating','N/A')}</span>
                                <span class="badge {badge_cls}">{mq}</span>
                            </div>
                            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_pct}%;"></div></div>
                            <div class="rec-meta" style="text-align:right;font-size:0.72rem;color:#E50914;">{score_pct}% match</div>
                        </div>""", unsafe_allow_html=True)

                with cr:
                    st.markdown('<div class="section-header" style="font-size:1.1rem;">📈 Score Breakdown</div>',
                                unsafe_allow_html=True)
                    fig = px.bar(result.head(10), x="final_score", y="title", orientation="h",
                                 color="final_score",
                                 color_continuous_scale=["#600000","#E50914","#FF6B6B"],
                                 labels={"final_score":"Match Score","title":""})
                    fig.update_layout(**PLOT_LAYOUT, height=380, yaxis={"categoryorder":"total ascending"})
                    fig.update_coloraxes(showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                    if not result.empty:
                        top = result.iloc[0]
                        fig2 = go.Figure(go.Scatterpolar(
                            r=[top["text_sim_n"], top["genre_n"], top["year_n"], top["final_score"]],
                            theta=["Text Sim","Genre Overlap","Year Closeness","Final Score"],
                            fill="toself", line_color="#E50914",
                            fillcolor="rgba(229,9,20,0.15)", name=top["title"][:25],
                        ))
                        fig2.update_layout(**PLOT_LAYOUT, height=270,
                                           polar=dict(bgcolor="#1a1a1a",
                                                      radialaxis=dict(visible=True,range=[0,1],gridcolor="#333"),
                                                      angularaxis=dict(gridcolor="#333")))
                        st.plotly_chart(fig2, use_container_width=True)

                # ══════════════════════════════
                # XAI PANEL – TOP-1
                # ══════════════════════════════
                if not result.empty:
                    top_rec = result.iloc[0]
                    xai = explain_recommendation(query, top_rec, cl_labels, df_m)

                    st.markdown("---")
                    st.markdown(f"""
                    <div style="font-size:1.15rem;font-weight:700;color:#9ba8f0;margin-bottom:0.3rem;">
                        🔍 Why <em>"{top_rec['title']}"</em> was recommended for <em>"{q['title']}"</em>
                    </div>""", unsafe_allow_html=True)

                    xa, xb, xc = st.columns([1.1, 1, 1])

                    # — Column A: Score components —
                    with xa:
                        score_parts = []
                        for _lbl, _val, _raw, _clr in [
                            ("🧠 Text Similarity",  xai["text_sim_n"], xai["text_sim_raw"], "#E50914"),
                            ("🎭 Genre Overlap",    xai["genre_n"],    xai["genre_raw"],    "#FF8C00"),
                            ("📅 Year Closeness",   xai["year_n"],     xai["year_raw"],     "#46D369"),
                            ("⭐ Final Score",       xai["final_score"],xai["final_score"],  "#7B4FFF"),
                        ]:
                            _pct = int(_val * 100)
                            score_parts.append(
                                '<div class="xai-row">'
                                '<div class="xai-label">' + _lbl + '</div>'
                                '<div class="xai-bar-bg"><div class="xai-bar-fill" style="width:' + str(_pct) + '%;background:' + _clr + ';"></div></div>'
                                '<div class="xai-val">' + f"{_raw:.3f}" + '</div>'
                                '</div>'
                            )
                        if xai["same_cluster"] is not None:
                            _sc = ("✅ Cluster " + str(xai["cluster_id"])) if xai["same_cluster"] else "❌ Different clusters"
                            score_parts.append('<div style="margin-top:0.7rem;"><span class="cluster-badge">' + _sc + '</span></div>')
                        st.markdown(
                            '<div class="xai-box"><div class="xai-title">📊 Score Components</div>'
                            + "".join(score_parts)
                            + '</div>',
                            unsafe_allow_html=True
                        )

                    # — Column B: Shared features —
                    with xb:
                        g_tags = "".join(f'<span class="shared-tag">🎭 {g.title()}</span>' for g in xai["shared_genres"]) \
                                 or '<span style="color:#555;font-size:0.78rem;">No shared genres</span>'
                        c_tags = "".join(f'<span class="shared-tag">🎬 {c.replace("_"," ").title()}</span>'
                                         for c in xai["shared_cast"][:6]) \
                                 or '<span style="color:#555;font-size:0.78rem;">No shared cast</span>'
                        dir_html = ""
                        if xai["same_director"]:
                            dir_html = f'<div style="margin-top:0.7rem;"><span style="color:#aaa;font-size:0.78rem;">🎥 Same Director:</span><br><span class="shared-tag">{xai["director"]}</span></div>'
                        year_html = ""
                        if xai["year_diff"] is not None:
                            year_html = f'<div style="margin-top:0.7rem;font-size:0.8rem;color:#aaa;">📅 Year gap: <b style="color:#fff;">{xai["year_diff"]} yr{"s" if xai["year_diff"]!=1 else ""}</b></div>'
                        st.markdown(f"""
                        <div class="xai-box">
                            <div class="xai-title">🔗 Shared Features</div>
                            <div style="color:#aaa;font-size:0.76rem;margin-bottom:0.3rem;">Genres in common:</div>
                            <div>{g_tags}</div>
                            <div style="color:#aaa;font-size:0.76rem;margin-top:0.7rem;margin-bottom:0.3rem;">Cast in common:</div>
                            <div>{c_tags}</div>
                            {dir_html}
                            {year_html}
                        </div>""", unsafe_allow_html=True)

                    # — Column C: Descriptions —
                    with xc:
                        st.markdown(f"""
                        <div class="xai-box">
                            <div class="xai-title">📖 Descriptions</div>
                            <div style="color:#9ba8f0;font-size:0.76rem;margin-bottom:0.3rem;">📌 {q['title']}</div>
                            <div class="desc-box">{xai['q_desc']}</div>
                            <div style="color:#9ba8f0;font-size:0.76rem;margin-top:0.8rem;margin-bottom:0.3rem;">🎯 {top_rec['title']}</div>
                            <div class="desc-box">{xai['r_desc']}</div>
                        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # BATCH MODE
    # ════════════════════════════════════════════
    else:
        st.markdown("""
        <div style="background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:1rem;margin-bottom:1rem;font-size:0.88rem;color:#aaa;">
        💡 <b style="color:#fff;">Batch Mode</b> — Select 2–5 seed titles. Scores are averaged across all seeds.
        </div>""", unsafe_allow_html=True)

        ci, cc = st.columns([2,1])
        with ci:
            defaults = [t for t in ["Breaking Bad","Narcos"] if t in all_titles][:2]
            seeds = st.multiselect("🔍 Select 2–5 seed titles", options=all_titles,
                                   default=defaults if defaults else all_titles[:2], max_selections=5)
            filter_type_b = st.radio("Filter results by", ["All","Movie","TV Show"], horizontal=True)
        with cc:
            top_n_b = st.slider("Number of recommendations", 5, 20, 10)
            st.markdown("<br>", unsafe_allow_html=True)
            run_batch = st.button("🧺 Get Batch Recommendations", use_container_width=True)

        if run_batch and len(seeds)>=2:
            with st.spinner("Computing batch recommendations…"):
                batch_result, failed = get_batch_recommendations(
                    seeds, top_n_b+5, df_m, X_m, idx_m, w_text, w_genre, w_year)

            if failed:
                st.warning(f"Could not resolve: {', '.join(failed)}")
            if batch_result is None:
                st.error("None of the selected titles could be resolved.")
            else:
                pills = " ".join(f'<span class="badge badge-type">📌 {s}</span>' for s in seeds)
                st.markdown(f'<div style="margin-bottom:1rem;">Seeded from: {pills}</div>', unsafe_allow_html=True)

                if filter_type_b != "All":
                    batch_result = batch_result[batch_result["type"]==filter_type_b]
                batch_result = batch_result.head(top_n_b)

                st.markdown(f'<div class="section-header">🎯 Top {len(batch_result)} Batch Recommendations</div>',
                            unsafe_allow_html=True)

                cl, cr = st.columns([1.2,1])
                with cl:
                    for _, row in batch_result.iterrows():
                        mq = row["match_quality"]
                        badge_cls = {"Excellent":"badge-excellent","Good":"badge-good"}.get(mq,"badge-fair")
                        score_pct = int(row["final_score"]*100)
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-title">{row['title']}</div>
                            <div class="rec-meta">
                                <span class="badge badge-type">{row['type']}</span>
                                <span class="badge badge-type">📅 {int(row['release_year']) if not pd.isna(row['release_year']) else 'N/A'}</span>
                                <span class="badge badge-type">⭐ {row.get('rating','N/A')}</span>
                                <span class="badge {badge_cls}">{mq}</span>
                            </div>
                            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_pct}%;"></div></div>
                            <div class="rec-meta" style="text-align:right;font-size:0.72rem;color:#E50914;">{score_pct}% match</div>
                        </div>""", unsafe_allow_html=True)

                with cr:
                    st.markdown('<div class="section-header" style="font-size:1.1rem;">📈 Score Breakdown</div>',
                                unsafe_allow_html=True)
                    fig_b = px.bar(batch_result.head(10), x="final_score", y="title", orientation="h",
                                   color="final_score",
                                   color_continuous_scale=["#600000","#E50914","#FF6B6B"],
                                   labels={"final_score":"Batch Match Score","title":""})
                    fig_b.update_layout(**PLOT_LAYOUT, height=380, yaxis={"categoryorder":"total ascending"})
                    fig_b.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_b, use_container_width=True)

                    if len(batch_result)>=2:
                        top5 = batch_result.head(5)
                        comp_df = pd.DataFrame({
                            "Title":         top5["title"].str[:28].tolist(),
                            "Text Sim":      (top5["text_sim_n"]*w_text).tolist(),
                            "Genre Overlap": (top5["genre_n"]*w_genre).tolist(),
                            "Year Closeness":(top5["year_n"]*w_year).tolist(),
                        })
                        comp_long = comp_df.melt(id_vars="Title", var_name="Component", value_name="Score")
                        fig_comp = px.bar(comp_long, x="Score", y="Title", color="Component",
                                          orientation="h", barmode="stack",
                                          color_discrete_map={"Text Sim":"#E50914","Genre Overlap":"#FF8C00","Year Closeness":"#46D369"},
                                          labels={"Score":"Weighted Score","Title":""})
                        fig_comp.update_layout(**PLOT_LAYOUT, height=280,
                                               title="Score Components (Top 5)",
                                               yaxis={"categoryorder":"total ascending"})
                        st.plotly_chart(fig_comp, use_container_width=True)

        elif run_batch and len(seeds)<2:
            st.warning("Please select at least 2 seed titles for batch mode.")
