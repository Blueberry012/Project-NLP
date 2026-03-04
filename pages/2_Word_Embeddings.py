# =====================================================
# Imports
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import gensim.downloader
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity

from evaluation import (
    evaluation_level1,
    evaluation_level2,
    evaluation_level3,
    evaluation_level4,
    evaluation_level5,
    evaluation_level6,
    evaluation_level7,
    evaluation_level8,
)

# =====================================================
# Config
# =====================================================

st.set_page_config(layout="wide")
st.title("🎯 Word Embedding Recommendation System")

# =====================================================
# Session State
# =====================================================

if "run" not in st.session_state:
    st.session_state.run = False

if "query_id" not in st.session_state:
    st.session_state.query_id = None


# =====================================================
# Embedding Interface
# =====================================================

class gensim_interface:

    def __init__(self, embeddingName):

        fileName = embeddingName + ".vecs"

        if os.path.isfile(fileName):
            print("loading embeddings...")
            self.loadPreparedEmbedding(embeddingName)
        else:
            print("embedding preparation...")
            self.embeddingPreparation(embeddingName)

        self.vectors = self.embeddingVectors.vectors

    def embeddingPreparation(self, embeddingName):
        self.embeddingVectors = gensim.downloader.load(embeddingName)

        with open(embeddingName + ".vecs", "wb") as fd:
            pickle.dump(self.embeddingVectors, fd)

    def loadPreparedEmbedding(self, embeddingName):
        with open(embeddingName + ".vecs", "rb") as fd:
            self.embeddingVectors = pickle.load(fd)

    def isVec(self, word):
        return word in self.embeddingVectors

    def getVec(self, word):
        return self.embeddingVectors[word]

    def nbDims(self):
        return self.embeddingVectors.vector_size


# =====================================================
# Load Data
# =====================================================

@st.cache_data
def load_data():

    tripadvisor = pd.read_csv("data/tripadvisor_final.csv")

    part1 = pd.read_csv("data/X_train_part1.csv")
    part2 = pd.read_csv("data/X_train_part2.csv")

    X_train02 = pd.concat([part1, part2], ignore_index=True)

    X_test02 = pd.read_csv("data/X_test.csv")

    return tripadvisor, X_train02, X_test02


tripadvisor, X_train02, X_test02 = load_data()
tripadvisor_meta = tripadvisor.set_index("id")


# =====================================================
# Build Model
# =====================================================

@st.cache_data
def build_embedding_model(X_train02, X_test02):

    emb = gensim_interface("glove-wiki-gigaword-100")

    X_train02 = X_train02.reset_index(drop=True)
    X_test02 = X_test02.reset_index(drop=True)

    # Document embedding
    def get_document_embedding(text, emb):

        words = text.split()
        vectors = []

        for word in words:
            if emb.isVec(word):
                vectors.append(emb.getVec(word))

        if len(vectors) == 0:
            return np.zeros(emb.nbDims())

        return np.mean(vectors, axis=0)

    # Train embeddings
    train_embeddings = []
    for text in X_train02["cleaned_review"]:
        train_embeddings.append(
            get_document_embedding(text, emb)
        )

    train_embeddings = np.array(train_embeddings)

    # Test embeddings
    test_embeddings = []
    for text in X_test02["cleaned_review"]:
        test_embeddings.append(
            get_document_embedding(text, emb)
        )

    test_embeddings = np.array(test_embeddings)

    similarity_matrix = cosine_similarity(
        test_embeddings,
        train_embeddings
    )

    return X_train02, X_test02, train_embeddings, test_embeddings, similarity_matrix, emb


X_train02, X_test02, train_embeddings, test_embeddings, similarity_matrix_m02, emb_model = \
    build_embedding_model(X_train02, X_test02)


# =====================================================
# Explainability
# =====================================================

def explain_similarity_words(query_text, similar_text,
                             emb,
                             top_k_words=8,
                             threshold=0.5):

    query_words = list(set(query_text.split()))
    similar_words = list(set(similar_text.split()))

    matched_pairs = []

    for w1 in query_words:

        if not emb.isVec(w1):
            continue

        v1 = emb.getVec(w1)

        for w2 in similar_words:

            if not emb.isVec(w2):
                continue

            v2 = emb.getVec(w2)

            sim = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2)
            )

            if sim > threshold:
                matched_pairs.append((w1, w2, sim))

    matched_pairs = sorted(
        matched_pairs,
        key=lambda x: x[2],
        reverse=True
    )

    return matched_pairs[:top_k_words]


# =====================================================
# Recommendation Engine
# =====================================================

def recommend_similar_place_embedding(
        idplace_query,
        X_test,
        X_train,
        sim_matrix,
        test_embeddings,
        train_embeddings,
        emb,
        top_k=5,
        top_n_words=5):

    X_test_reset = X_test.reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)

    if idplace_query not in X_test_reset["idplace"].values:
        st.warning("ID not in TEST dataset")
        return []

    query_idx = X_test_reset.index[
        X_test_reset["idplace"] == int(idplace_query)
    ][0]

    sims = sim_matrix[query_idx].copy()

    top_indices = np.argsort(sims)[::-1][:top_k]

    query_text = X_test_reset.iloc[query_idx]["cleaned_review"]

    st.subheader("🔎 Recommendations")

    rec_ids = []

    cols = st.columns(top_k)

    for rank, (col, idx) in enumerate(zip(cols, top_indices), start=1):

        with col:

            rec_id = int(X_train_reset.iloc[idx]["idplace"])
            rec_ids.append(rec_id)

            score = sims[idx]

            similar_text = X_train_reset.iloc[idx]["cleaned_review"]

            st.markdown(f"""
            **Reco {rank}**
            🏨 ID : {rec_id}
            ⭐ Similarity : {score:.3f}
            """)

            st.write("🧠 Semantic matching words")

            matched_words = explain_similarity_words(
                query_text,
                similar_text,
                emb,
                top_k_words=top_n_words
            )

            if len(matched_words) == 0:
                st.caption("No strong semantic match found")
            else:
                for w1, w2, sim in matched_words:
                    st.caption(f"{w1} ↔ {w2} ({sim:.3f})")

    return rec_ids


# =====================================================
# Sidebar
# =====================================================

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

test_places = X_test02.merge(
    tripadvisor,
    left_on="idplace",
    right_on="id",
    how="left"
)

test_places["display_name"] = (
    test_places["idplace"].astype(str)
    + " - "
    + test_places["nom"].astype(str)
)

test_places = test_places.drop_duplicates("idplace")

default_place = test_places[
    test_places["idplace"] == 1725986
]["display_name"].values

default_index = 0

if len(default_place) > 0:
    default_index = test_places["display_name"].tolist().index(
        default_place[0]
    )

selected_place = st.sidebar.selectbox(
    "🏨 Choose place",
    test_places["display_name"].tolist(),
    index=default_index
)

query_id = int(selected_place.split(" - ")[0])

if st.sidebar.button("🚀 Search"):
    st.session_state.run = True
    st.session_state.query_id = query_id


# =====================================================
# Execution
# =====================================================

if st.session_state.get("run"):

    rec_ids = recommend_similar_place_embedding(
        st.session_state.query_id,
        X_test02,
        X_train02,
        similarity_matrix_m02,
        test_embeddings,
        train_embeddings,
        emb_model,
        top_k
    )

    # Map
    st.header("🗺 Map")

    center = [48.8566, 2.3522]

    m = folium.Map(location=center, zoom_start=13)

    if st.session_state.query_id in tripadvisor_meta.index:

        row = tripadvisor_meta.loc[st.session_state.query_id]

        popup_text = f"""
        🏨 {row['nom']}<br>
        ⭐ Rating : {row['rating']}<br>
        📍 {row['adresse'] if 'adresse' in row else ''}
        """

        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="red")
        ).add_to(m)

    for rec_id in rec_ids:

        if rec_id not in tripadvisor_meta.index:
            continue

        if rec_id == st.session_state.query_id:
            continue

        row = tripadvisor_meta.loc[rec_id]

        popup_text = f"""
        🏨 {row['nom']}<br>
        ⭐ Rating : {row['rating']}<br>
        📍 {row['adresse'] if 'adresse' in row else ''}
        """

        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="blue", icon="star", prefix="fa")
        ).add_to(m)

    st_folium(m, width=1000, height=600)


    # Table
    st.header("📋 Places Details")

    ordered_ids = [query_id] + rec_ids

    tripadvisor_test = tripadvisor[
        tripadvisor["id"].isin(ordered_ids)
    ].copy()

    rank_map = {query_id: 0}

    for i, rid in enumerate(rec_ids, start=1):
        rank_map[rid] = i

    tripadvisor_test["Top"] = tripadvisor_test["id"].map(rank_map)

    tripadvisor_test = tripadvisor_test.sort_values("Top")

    cols = list(tripadvisor_test.columns)

    if "Top" in cols:
        cols.remove("Top")
        cols = ["Top"] + cols

    st.dataframe(tripadvisor_test[cols], use_container_width=True)


    # Metrics
    st.subheader("📊 Evaluation Metrics")

    st.write(f"Precision Level 1 : {evaluation_level1(X_test02,X_train02,similarity_matrix_m02,tripadvisor,top_k=5):.4f}")
    st.write(f"Precision Level 2 : {evaluation_level2(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta,top_k=5):.4f}")
    st.write(f"Precision Level 3 : {evaluation_level3(X_test02,X_train02,similarity_matrix_m02,tripadvisor):.4f}")
    st.write(f"Precision Level 4 : {evaluation_level4(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta):.4f}")
    st.write(f"Precision Level 5 : {evaluation_level5(X_test02,X_train02,similarity_matrix_m02,tripadvisor,top_k=5):.4f}")
    st.write(f"Precision Level 6 : {evaluation_level6(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta,top_k=5):.4f}")
    st.write(f"Precision Level 7 : {evaluation_level7(X_test02,X_train02,similarity_matrix_m02,tripadvisor):.2f}")
    st.write(f"Precision Level 8 : {evaluation_level8(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta):.2f}")


# Footer
st.markdown("---")
st.caption("Recommendation System")
