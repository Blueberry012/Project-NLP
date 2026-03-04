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
            with open(fileName, "rb") as fd:
                self.embeddingVectors = pickle.load(fd)
        else:
            self.embeddingVectors = gensim.downloader.load(embeddingName)

            with open(fileName, "wb") as fd:
                pickle.dump(self.embeddingVectors, fd)

        self.vectors = self.embeddingVectors.vectors

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
# Load Precomputed Embeddings
# =====================================================

train_embeddings = np.load("data/train_embeddings.npy")
test_embeddings = np.load("data/test_embeddings.npy")
similarity_matrix_m02 = np.load("data/similarity_matrix.npy")

emb_model = gensim_interface("glove-wiki-gigaword-100")


# =====================================================
# Explainability (Simple → Just w1)
# =====================================================

def explain_similarity_words(query_text, top_k_words=5):

    query_words = list(set(query_text.split()))[:top_k_words]

    return [(w, "", 0) for w in query_words if emb_model.isVec(w)]


# =====================================================
# Recommendation Engine
# =====================================================

def recommend_similar_place_embedding(
        idplace_query,
        X_test,
        X_train,
        sim_matrix,
        top_k=5):

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

            st.markdown(f"""
            **Reco {rank}**
            🏨 ID : {rec_id}
            ⭐ Similarity : {score:.3f}
            """)

            st.write("🧠 Keywords")

            matched_words = explain_similarity_words(query_text)

            if len(matched_words) == 0:
                st.caption("No keyword found")
            else:
                for w1, _, _ in matched_words:
                    st.caption(f"• {w1}")

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


# ⭐ DEFAULT PLACE
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
        top_k
    )

    # Map
    st.header("🗺 Map")

    center = [48.8566, 2.3522]

    m = folium.Map(location=center, zoom_start=13)

    if st.session_state.query_id in tripadvisor_meta.index:

        row = tripadvisor_meta.loc[st.session_state.query_id]

        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=row["nom"],
            icon=folium.Icon(color="red")
        ).add_to(m)

    for rec_id in rec_ids:

        if rec_id not in tripadvisor_meta.index:
            continue

        if rec_id == st.session_state.query_id:
            continue

        row = tripadvisor_meta.loc[rec_id]

        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=row["nom"],
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

    st.dataframe(tripadvisor_test, use_container_width=True)


    # Metrics
    st.subheader("📊 Evaluation Metrics")

    st.write(f"Level1 : {evaluation_level1(X_test02,X_train02,similarity_matrix_m02,tripadvisor,top_k=5):.4f}")
    st.write(f"Level2 : {evaluation_level2(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta,top_k=5):.4f}")
    st.write(f"Level3 : {evaluation_level3(X_test02,X_train02,similarity_matrix_m02,tripadvisor):.4f}")
    st.write(f"Level4 : {evaluation_level4(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta):.4f}")
    st.write(f"Level5 : {evaluation_level5(X_test02,X_train02,similarity_matrix_m02,tripadvisor,top_k=5):.4f}")
    st.write(f"Level6 : {evaluation_level6(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta,top_k=5):.4f}")
    st.write(f"Level7 : {evaluation_level7(X_test02,X_train02,similarity_matrix_m02,tripadvisor):.2f}")
    st.write(f"Level8 : {evaluation_level8(X_test02,X_train02,similarity_matrix_m02,tripadvisor_meta):.2f}")


st.markdown("---")
st.caption("Recommendation System")
