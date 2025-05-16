import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.backend import OpenAIBackend, CohereBackend
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic.representation import OpenAIRepresentation

def load_sample_data():
    data = pd.DataFrame({
        "query": [
            "weather in NYC", "buy iPhone", "football scores", "rain forecast",
            "cheap flights", "NBA playoffs", "snow in Boston", "discount shoes",
            "Yankees game", "best pizza", "weather in LA", "buy Samsung", "tennis results"
        ] * 10,
        "region": ["NY", "CA", "TX", "NY", "CA", "TX", "NY", "CA", "TX", "NY", "CA", "TX", "NY"] * 10,
    })
    return data

def get_embedding_model(backend_name):
    if backend_name == "OpenAI":
        return OpenAIBackend("text-embedding-ada-002")
    elif backend_name == "Cohere":
        return CohereBackend("embed-english-v3.0")
    elif backend_name == "Llama2 (local)":
        from bertopic.backend import TransformerBackend
        return TransformerBackend("TheBloke/Llama-2-7B-Chat-GGML")
    else:
        return None  # Default MiniLM

def create_topic_model(
    data,
    min_cluster_size=10,
    min_samples=1,
    use_bigram=True,
    min_df=5,
    embedding_backend="all-MiniLM-L6-v2 (default)",
    zero_shot_labels=None,
    seed_words=None,
):
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        ngram_range=(1, 2) if use_bigram else (1, 1),
        min_df=min_df,
        stop_words="english",
    )
    embedding_model = get_embedding_model(embedding_backend)
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        embedding_model=embedding_model,
        calculate_probabilities=False,
        low_memory=True,
    )
    docs = data["query"].tolist()
    if zero_shot_labels:
        topics, _ = topic_model.fit_transform(
            docs, y=[None]*len(docs), topics=zero_shot_labels
        )
    elif seed_words:
        seed_dict = {}
        for t in seed_words.split(";"):
            if ":" in t:
                topic, words = t.split(":")
                seed_dict[topic.strip()] = [w.strip() for w in words.split(",")]
        topics, _ = topic_model.fit_transform(docs, seed_topics=seed_dict)
    else:
        topics, _ = topic_model.fit_transform(docs)
    return topic_model, topics, None

def visualize_topics(topic_model, data, topics):
    fig = topic_model.visualize_barchart(top_n_topics=10)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Sample topic assignments:")
    st.dataframe(
        pd.DataFrame({"query": data["query"], "topic": topics}).head(20)
    )

def visualize_umap(topic_model, docs, topics):
    fig = topic_model.visualize_documents(docs, topics=topics)
    st.plotly_chart(fig, use_container_width=True)

def visualize_topic_timeline(topic_model, docs, topics, timestamps):
    df = pd.DataFrame({"topic": topics, "timestamp": timestamps})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    timeline = df.groupby([pd.Grouper(key="timestamp", freq="D"), "topic"]).size().unstack(fill_value=0)
    fig = px.line(timeline, title="Topic Frequency Over Time")
    st.plotly_chart(fig, use_container_width=True)

def visualize_wordcloud(topic_model, topic_id):
    words = dict(topic_model.get_topic(topic_id))
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def visualize_hierarchy(topic_model):
    fig = topic_model.visualize_hierarchy()
    st.plotly_chart(fig, use_container_width=True)

def visualize_geospatial_heatmap(data, topics, region_col="region"):
    df = data.copy()
    df["topic"] = topics
    topic_counts = df.groupby([region_col, "topic"]).size().reset_index(name="count")
    fig = px.density_heatmap(topic_counts, x=region_col, y="topic", z="count", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

def visualize_topic_drift(topic_model, docs, topics, timestamps):
    df = pd.DataFrame({"doc": docs, "topic": topics, "timestamp": pd.to_datetime(timestamps)})
    drift = []
    for topic in set(topics):
        topic_docs = df[df["topic"] == topic].sort_values("timestamp")
        if len(topic_docs) > 10:
            split = len(topic_docs) // 2
            first_half = topic_docs.iloc[:split]["doc"].tolist()
            second_half = topic_docs.iloc[split:]["doc"].tolist()
            words1 = dict(topic_model.get_topic(topic, docs=first_half))
            words2 = dict(topic_model.get_topic(topic, docs=second_half))
            drift.append({
                "topic": topic,
                "start_top_words": list(words1.keys())[:5],
                "end_top_words": list(words2.keys())[:5]
            })
    st.write("Topic Drift (Top Words Start vs End):")
    st.dataframe(pd.DataFrame(drift))

def llm_topic_labels(topic_model, openai_api_key):
    rep = OpenAIRepresentation(openai_api_key=openai_api_key)
    topic_model.set_topic_representation(rep)
    st.success("LLM-powered topic labels updated!")
    st.write(topic_model.get_topic_info())

def incremental_update(model, new_data):
    new_model, new_topics, _ = create_topic_model(new_data)
    merged_model = model.merge_models([model, new_model])
    return merged_model, new_topics
