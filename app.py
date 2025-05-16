import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import fasttext
import re
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Optional: OpenAI/Cohere/LLama2 support
from bertopic.backend import OpenAIBackend, CohereBackend
from bertopic.representation import OpenAIRepresentation

# ========== MODEL LOADING AND CONFIG ==========
@st.cache_resource
def load_models(embed_backend, openai_api_key=None, cohere_api_key=None):
    if embed_backend == "SentenceTransformers":
        embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    elif embed_backend == "OpenAI":
        embed_model = OpenAIBackend("text-embedding-ada-002", openai_api_key=openai_api_key)
    elif embed_backend == "Cohere":
        embed_model = CohereBackend("embed-english-v3.0", cohere_api_key=cohere_api_key)
    # Add Llama2 or other backends here as needed
    else:
        embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    ft_model = fasttext.load_model("lid.176.ftz")
    return embed_model, ft_model

def detect_language(text, ft_model):
    try:
        lang = ft_model.predict(text)[0][0].split("__")[-1]
    except:
        lang = "unknown"
    return lang

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# ========== SIDEBAR CONFIG ==========
st.sidebar.header("Embedding & Model Configuration")
embed_backend = st.sidebar.selectbox(
    "Embedding Backend",
    ["SentenceTransformers", "OpenAI", "Cohere"],
    help="Choose embedding provider. For OpenAI/Cohere, set your API key."
)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password") if embed_backend == "OpenAI" else None
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password") if embed_backend == "Cohere" else None

min_cluster_size = st.sidebar.slider("HDBSCAN Min Cluster Size", 2, 100, 10)
min_samples = st.sidebar.slider("HDBSCAN Min Samples", 1, 20, 2)
min_df = st.sidebar.slider("CountVectorizer Min Document Frequency", 1, 20, 5)
use_bigrams = st.sidebar.checkbox("Use Bigrams", True)

# Advanced options
st.sidebar.header("Advanced Options")
zero_shot_labels = st.sidebar.text_area("Zero-Shot Topic Labels (comma-separated)", "")
seed_words = st.sidebar.text_area("Seed Words (topic: word1,word2; ...)", "")
incremental_mode = st.sidebar.checkbox("Enable Incremental/Online Learning")
visuals = st.sidebar.multiselect(
    "Advanced Visualizations",
    ["UMAP Embedding", "Topic Timeline", "Topic Hierarchy", "Word Cloud", "Geospatial Heatmap", "Topic Drift", "LLM Topic Labels"]
)

# ========== MAIN APP ==========
st.title("BERTopic Enterprise Search & Geospatial Analyzer")

uploaded_file = st.file_uploader("Upload CSV with 'query' and optional 'location'/'timestamp' columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'query' not in df.columns:
        st.error("CSV must contain a 'query' column")
        st.stop()

    # Load models
    with st.spinner("Loading models..."):
        embed_model, ft_model = load_models(embed_backend, openai_api_key, cohere_api_key)

    # Preprocess
    with st.spinner("Preprocessing and embedding queries..."):
        df['lang'] = df['query'].apply(lambda x: detect_language(str(x), ft_model))
        df['cleaned'] = df['query'].astype(str).apply(clean_text)
        embeddings = embed_model.encode(df['cleaned'].tolist(), show_progress_bar=True) if hasattr(embed_model, "encode") else embed_model.embed(df['cleaned'].tolist())

    # BERTopic config
    from hdbscan import HDBSCAN
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', prediction_data=True)
    vectorizer = CountVectorizer(ngram_range=(1, 2) if use_bigrams else (1, 1), min_df=min_df, stop_words="english")

    topic_model = BERTopic(
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        embedding_model=embed_model,
        calculate_probabilities=False,
        low_memory=True,
    )

    # Fit/Transform
    with st.spinner("Running BERTopic..."):
        docs = df['cleaned'].tolist()
        # Zero-shot/seeded logic
        if zero_shot_labels.strip():
            labels = [x.strip() for x in zero_shot_labels.split(",") if x.strip()]
            topics, probs = topic_model.fit_transform(docs, y=[None]*len(docs), topics=labels)
        elif seed_words.strip():
            seed_dict = {}
            for t in seed_words.split(";"):
                if ":" in t:
                    topic, words = t.split(":")
                    seed_dict[topic.strip()] = [w.strip() for w in words.split(",")]
            topics, probs = topic_model.fit_transform(docs, seed_topics=seed_dict)
        else:
            topics, probs = topic_model.fit_transform(docs, embeddings)
        df['topic'] = topics

    st.success("Topic modeling complete!")

    # Incremental/online learning
    if incremental_mode:
        st.info("Incremental/Online Learning: Upload a new batch for merging or partial_fit.")
        new_file = st.file_uploader("Upload new batch CSV for incremental update", type=["csv"], key="inc")
        if new_file:
            new_df = pd.read_csv(new_file)
            new_df['cleaned'] = new_df['query'].astype(str).apply(clean_text)
            new_embeddings = embed_model.encode(new_df['cleaned'].tolist(), show_progress_bar=True) if hasattr(embed_model, "encode") else embed_model.embed(new_df['cleaned'].tolist())
            new_topics, _ = topic_model.fit_transform(new_df['cleaned'].tolist(), new_embeddings)
            topic_model = topic_model.merge_models([topic_model, BERTopic().fit(new_df['cleaned'].tolist(), new_embeddings)])
            st.success("Merged with new batch!")
            # Optionally update df and topics

    # ========== VISUALIZATIONS ==========
    st.write("### Top Topics")
    st.dataframe(topic_model.get_topic_info())

    if "UMAP Embedding" in visuals:
        st.write("### UMAP Embedding")
        fig_umap = topic_model.visualize_documents(docs, topics=topics)
        st.plotly_chart(fig_umap, use_container_width=True)

    if "Topic Timeline" in visuals and 'timestamp' in df.columns:
        st.write("### Topics Over Time")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        topics_over_time = topic_model.topics_over_time(docs, topics, df['timestamp'])
        fig_time = topic_model.visualize_topics_over_time(topics_over_time)
        st.plotly_chart(fig_time, use_container_width=True)

    if "Topic Hierarchy" in visuals:
        st.write("### Topic Hierarchy")
        fig_hier = topic_model.visualize_hierarchy()
        st.plotly_chart(fig_hier, use_container_width=True)

    if "Word Cloud" in visuals:
        st.write("### Word Cloud")
        topic_id = st.selectbox("Select topic for word cloud", sorted(set(topics)))
        words = dict(topic_model.get_topic(topic_id))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    if "Geospatial Heatmap" in visuals and 'location' in df.columns:
        st.write("### Geospatial Topic Distribution")
        fig = px.scatter_geo(
            df.dropna(subset=['location']),
            locations="location",  # ISO alpha codes or full country names
            locationmode='country names',
            color="topic",
            hover_name="query",
            title="Topics by Location"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Topic Drift" in visuals and 'timestamp' in df.columns:
        st.write("### Topic Drift")
        drift = []
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for topic in set(topics):
            topic_docs = df[df['topic'] == topic].sort_values("timestamp")
            if len(topic_docs) > 10:
                split = len(topic_docs) // 2
                first_half = topic_docs.iloc[:split]['cleaned'].tolist()
                second_half = topic_docs.iloc[split:]['cleaned'].tolist()
                words1 = dict(topic_model.get_topic(topic, docs=first_half))
                words2 = dict(topic_model.get_topic(topic, docs=second_half))
                drift.append({
                    "topic": topic,
                    "start_top_words": list(words1.keys())[:5],
                    "end_top_words": list(words2.keys())[:5]
                })
        st.dataframe(pd.DataFrame(drift))

    if "LLM Topic Labels" in visuals and embed_backend == "OpenAI" and openai_api_key:
        st.write("### LLM-Powered Topic Labels")
        rep = OpenAIRepresentation(openai_api_key=openai_api_key)
        topic_model.set_topic_representation(rep)
        st.write(topic_model.get_topic_info())

    # ========== DOWNLOAD ==========
    st.write("### Download Topic Assignments")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="topic_assignments.csv")
else:
    st.info("Upload a CSV file to get started.")

st.markdown("""
---
**Features:**  
- Multilingual embeddings & language detection  
- Zero-shot and seeded topic modeling  
- Incremental/online learning  
- UMAP, timeline, hierarchy, drift, word cloud, geospatial heatmap  
- LLM topic labeling (OpenAI)  
- Enterprise-scale ready  
""")
