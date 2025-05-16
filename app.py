import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Optional: OpenAI/Cohere/LLama2 support
from bertopic.backend import OpenAIBackend, CohereBackend

# FastText is optional - use if available
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

# ========== MODEL LOADING AND CONFIG ==========
@st.cache_resource
def load_models(embed_backend, openai_api_key=None, cohere_api_key=None):
    if embed_backend == "SentenceTransformers":
        embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    elif embed_backend == "OpenAI":
        if not openai_api_key:
            st.error("OpenAI API key is required when using OpenAI embeddings")
            st.stop()
        embed_model = OpenAIBackend("text-embedding-3-small", openai_api_key=openai_api_key)
    elif embed_backend == "Cohere":
        if not cohere_api_key:
            st.error("Cohere API key is required when using Cohere embeddings")
            st.stop()
        embed_model = CohereBackend("embed-english-v3.0", cohere_api_key=cohere_api_key)
    # Add Llama2 or other backends here as needed
    else:
        embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Try to load FastText language model if available
    ft_model = None
    if FASTTEXT_AVAILABLE:
        model_path = "lid.176.ftz"
        # Check if model exists
        if os.path.exists(model_path):
            ft_model = fasttext.load_model(model_path)
        else:
            st.warning("FastText language detection model not found. Language detection will be disabled.")
    
    return embed_model, ft_model

def detect_language(text, ft_model):
    if ft_model is None:
        return "unknown"
    try:
        # Clean the text for fasttext prediction
        cleaned_text = re.sub(r'\s+', ' ', str(text).strip())
        if not cleaned_text:
            return "unknown"
        prediction = ft_model.predict(cleaned_text)
        lang = prediction[0][0].split("__")[-1]
    except Exception as e:
        st.warning(f"Language detection error: {e}")
        lang = "unknown"
    return lang

def clean_text(text):
    # Convert to string first in case we get non-string input
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

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
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'query' not in df.columns:
            st.error("CSV must contain a 'query' column")
            st.stop()

        # Check for empty data
        if df.empty:
            st.error("The uploaded CSV file is empty")
            st.stop()
            
        # Fill any NaN values in query column
        df['query'] = df['query'].fillna("").astype(str)
        if df['query'].str.strip().eq('').any():
            st.warning("Your data contains empty queries which might affect topic modeling quality")

        # Load models
        with st.spinner("Loading models..."):
            embed_model, ft_model = load_models(embed_backend, openai_api_key, cohere_api_key)

        # Preprocess
        with st.spinner("Preprocessing and embedding queries..."):
            if ft_model is not None:
                df['lang'] = df['query'].apply(lambda x: detect_language(str(x), ft_model))
            else:
                df['lang'] = "unknown"
                
            df['cleaned'] = df['query'].apply(clean_text)
            
            # Check for empty cleaned text
            empty_count = df['cleaned'].str.strip().eq('').sum()
            if empty_count > 0:
                st.warning(f"{empty_count} queries resulted in empty text after cleaning")
                df = df[df['cleaned'].str.strip() != '']
                if df.empty:
                    st.error("All queries resulted in empty text after cleaning")
                    st.stop()
            
            # Handle embeddings based on model type
            if hasattr(embed_model, "encode"):
                embeddings = embed_model.encode(df['cleaned'].tolist(), show_progress_bar=True)
            else:
                embeddings = embed_model.embed(df['cleaned'].tolist())

        # BERTopic config
        try:
            from hdbscan import HDBSCAN
            hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                                    min_samples=min_samples, 
                                    metric='euclidean', 
                                    prediction_data=True)
            
            vectorizer = CountVectorizer(ngram_range=(1, 2) if use_bigrams else (1, 1), 
                                        min_df=min_df, 
                                        stop_words="english")

            topic_model = BERTopic(
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                embedding_model=embed_model,
                calculate_probabilities=True,
                verbose=True,
            )
        except Exception as e:
            st.error(f"Error initializing BERTopic model: {str(e)}")
            st.stop()

        # Fit/Transform
        with st.spinner("Running BERTopic..."):
            docs = df['cleaned'].tolist()
            
            try:
                # Zero-shot/seeded logic
                if zero_shot_labels.strip():
                    labels = [x.strip() for x in zero_shot_labels.split(",") if x.strip()]
                    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, topics=labels)
                elif seed_words.strip():
                    seed_dict = {}
                    for t in seed_words.split(";"):
                        if ":" in t:
                            topic, words = t.split(":")
                            seed_dict[topic.strip()] = [w.strip() for w in words.split(",")]
                    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, seed_topics=seed_dict)
                else:
                    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
                
                df['topic'] = topics
            except Exception as e:
                st.error(f"Error during topic modeling: {str(e)}")
                st.stop()

        st.success("Topic modeling complete!")

        # Incremental/online learning
        if incremental_mode:
            st.info("Incremental/Online Learning: Upload a new batch for merging or partial_fit.")
            new_file = st.file_uploader("Upload new batch CSV for incremental update", type=["csv"], key="inc")
            if new_file:
                try:
                    new_df = pd.read_csv(new_file)
                    if 'query' not in new_df.columns:
                        st.error("Incremental CSV must contain a 'query' column")
                    else:
                        new_df['cleaned'] = new_df['query'].astype(str).apply(clean_text)
                        if hasattr(embed_model, "encode"):
                            new_embeddings = embed_model.encode(new_df['cleaned'].tolist(), show_progress_bar=True)
                        else:
                            new_embeddings = embed_model.embed(new_df['cleaned'].tolist())
                            
                        # Create a temporary model for the new data
                        temp_model = BERTopic(
                            hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples),
                            vectorizer_model=CountVectorizer(ngram_range=(1, 2) if use_bigrams else (1, 1), min_df=min_df)
                        )
                        temp_model.fit(new_df['cleaned'].tolist(), new_embeddings)
                        
                        # Merge models
                        topic_model = topic_model.merge_models([topic_model, temp_model])
                        st.success("Merged with new batch!")
                except Exception as e:
                    st.error(f"Error during incremental learning: {str(e)}")

        # ========== VISUALIZATIONS ==========
        st.write("### Top Topics")
        topic_info = topic_model.get_topic_info()
        st.dataframe(topic_info)

        if "UMAP Embedding" in visuals:
            st.write("### UMAP Embedding")
            try:
                fig_umap = topic_model.visualize_documents(docs, topics=topics)
                st.plotly_chart(fig_umap, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating UMAP visualization: {str(e)}")

        if "Topic Timeline" in visuals and 'timestamp' in df.columns:
            st.write("### Topics Over Time")
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                if df['timestamp'].isna().all():
                    st.error("Could not parse timestamp column. Ensure it's in a valid datetime format.")
                else:
                    df = df.dropna(subset=['timestamp'])
                    topics_over_time = topic_model.topics_over_time(docs, topics, df['timestamp'])
                    fig_time = topic_model.visualize_topics_over_time(topics_over_time)
                    st.plotly_chart(fig_time, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating timeline visualization: {str(e)}")

        if "Topic Hierarchy" in visuals:
            st.write("### Topic Hierarchy")
            try:
                fig_hier = topic_model.visualize_hierarchy()
                st.plotly_chart(fig_hier, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating hierarchy visualization: {str(e)}")

        if "Word Cloud" in visuals:
            st.write("### Word Cloud")
            try:
                topic_options = sorted(set(topics))
                if topic_options:
                    topic_id = st.selectbox("Select topic for word cloud", topic_options)
                    words = dict(topic_model.get_topic(topic_id))
                    if words:
                        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
                        fig, ax = plt.subplots(figsize=(8,4))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.warning("No keywords found for this topic.")
                else:
                    st.warning("No topics available for word cloud visualization.")
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")

        if "Geospatial Heatmap" in visuals and 'location' in df.columns:
            st.write("### Geospatial Topic Distribution")
            try:
                geo_df = df.dropna(subset=['location'])
                if not geo_df.empty:
                    fig = px.scatter_geo(
                        geo_df,
                        locations="location",
                        locationmode='country names',
                        color="topic",
                        hover_name="query",
                        title="Topics by Location"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid location data found for geospatial visualization.")
            except Exception as e:
                st.error(f"Error generating geospatial visualization: {str(e)}")

        if "Topic Drift" in visuals and 'timestamp' in df.columns:
            st.write("### Topic Drift")
            try:
                drift = []
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                if not df.empty:
                    for topic in set(topics):
                        topic_docs = df[df['topic'] == topic].sort_values("timestamp")
                        if len(topic_docs) > 10:
                            split = len(topic_docs) // 2
                            first_half = topic_docs.iloc[:split]['cleaned'].tolist()
                            second_half = topic_docs.iloc[split:]['cleaned'].tolist()
                            words1 = dict(topic_model.get_topic(topic, docs=first_half))
                            words2 = dict(topic_model.get_topic(topic, docs=second_half))
                            if words1 and words2:
                                drift.append({
                                    "topic": topic,
                                    "start_top_words": list(words1.keys())[:5],
                                    "end_top_words": list(words2.keys())[:5]
                                })
                    
                    if drift:
                        st.dataframe(pd.DataFrame(drift))
                    else:
                        st.warning("Insufficient data for topic drift analysis.")
                else:
                    st.warning("No valid timestamp data for drift analysis.")
            except Exception as e:
                st.error(f"Error analyzing topic drift: {str(e)}")

        if "LLM Topic Labels" in visuals and embed_backend == "OpenAI" and openai_api_key:
            st.write("### LLM-Powered Topic Labels")
            try:
                from bertopic.representation import OpenAIRepresentation
                rep = OpenAIRepresentation(openai_api_key=openai_api_key)
                topic_model.update_topics(docs, topics, representation_model=rep)
                st.write(topic_model.get_topic_info())
            except Exception as e:
                st.error(f"Error generating LLM topic labels: {str(e)}")

        # ========== DOWNLOAD ==========
        st.write("### Download Topic Assignments")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="topic_assignments.csv")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        
else:
    st.info("Upload a CSV file to get started.")
