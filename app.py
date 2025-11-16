import streamlit as st
from similarity_ranking import get_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NLP-Driven Place Matching", layout="wide")
st.title("🌍 NLP-Driven Place Matching")
st.write("Enter a query and get top recommendations using different similarity models.")

# ------------------ Input ------------------
query = st.text_input("🔎 Enter your query:")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None

# ------------------ Search ------------------
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query!")
    else:
        st.session_state.results = get_results(query)

# ------------------ Display Recommendations ------------------
if st.session_state.results:
    results = st.session_state.results

    def display_results(model_name, data):
        st.subheader(f"{model_name} Top 5 Recommendations")
        if isinstance(data, list):
            df = pd.DataFrame([{"Place": item["key"], "Score": round(item["score"], 3)} for item in data])
            st.table(df)
        else:
            st.write(data)

    display_results("TF-IDF", results.get("TFIDF", []))
    display_results("SBERT", results.get("SBERT", []))
    display_results("Word2Vec", results.get("Word2Vec", []))

    # ------------------ Evaluation Button ------------------
    if st.button("Check Model Evaluation"):
        st.subheader("📊 Model Evaluation Metrics")

        def evaluate_models_relevance(df, top_n=5):
            models = ['TF-IDF', 'SBERT', 'Word2Vec']
            results = []

            for model in models:
                precisions, mrrs, ndcgs = [], [], []
                for q in df['Query'].unique():
                    df_query = df[df['Query'] == q].sort_values('Rank')
                    rels = df_query[f'{model} Relevant'].values[:top_n]

                    precisions.append(np.sum(rels > 0) / top_n)
                    ranks = np.where(rels > 0)[0]
                    mrrs.append(1 / (ranks[0] + 1) if len(ranks) > 0 else 0)
                    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rels))
                    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted(rels, reverse=True)))
                    ndcgs.append(dcg / idcg if idcg != 0 else 0)

                results.append({
                    'Model': model,
                    f'Precision@{top_n}': round(np.mean(precisions), 3),
                    f'MRR@{top_n}': round(np.mean(mrrs), 3),
                    f'NDCG@{top_n}': round(np.mean(ndcgs), 3)
                })
            return pd.DataFrame(results)

        try:
            eval_df = pd.read_csv("Preprocessed Data - testing.csv")
            evaluation_results = evaluate_models_relevance(eval_df, top_n=5)
            st.table(evaluation_results)
        except FileNotFoundError:
            st.error("Evaluation CSV file not found! Make sure 'Preprocessed Data - testing.csv' exists.")
        # After displaying evaluation_results dataframe
        st.subheader("📊 Combined Model Performance Comparison")

        metrics = ["Precision@5", "MRR@5", "NDCG@5"]
        models = evaluation_results["Model"].tolist()

        # Prepare data matrix
        scores = evaluation_results[metrics].values   # Shape = (3 models, 3 metrics)

        # X positions for metrics
        x = np.arange(len(metrics))
        width = 0.25   # width of each bar

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot bars for each model
        for i, model in enumerate(models):
            ax.bar(
                x + (i - 1) * width,
                scores[i],
                width,
                label=model
            )

        # Labels & settings
        ax.set_ylabel("Scores")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(title="Models")

        # Add value labels on bars
        for i in range(len(models)):
            for j in range(len(metrics)):
                ax.text(
                    x[j] + (i - 1) * width,
                    scores[i][j],
                    f"{scores[i][j]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        st.pyplot(fig)

        # st.subheader("📊 Model Performance Comparison")

        # metrics = ["Precision@5", "MRR@5", "NDCG@5"]
        # models = evaluation_results.index.tolist()

        # for metric in metrics:
        #     st.write(f"### {metric}")

        #     fig, ax = plt.subplots(figsize=(6, 4))
            
        #     bars = ax.bar(models, evaluation_results[metric])
        #     ax.set_xlabel("Models")
        #     ax.set_ylabel(metric)
        #     ax.set_title(f"{metric} Comparison Across Models")

        #     # Add values on top of bars
        #     for bar in bars:
        #         height = bar.get_height()
        #         ax.text(
        #             bar.get_x() + bar.get_width() / 2,
        #             height,
        #             f"{height:.3f}",
        #             ha="center",
        #             va="bottom",
        #             fontsize=5
        #         )

        #     st.pyplot(fig)