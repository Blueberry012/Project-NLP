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

    """
    #List of possible embeddings:

    ['fasttext-wiki-news-subwords-300',
     'conceptnet-numberbatch-17-06-300',
     'word2vec-ruscorpora-300',
     'word2vec-google-news-300',
     'glove-wiki-gigaword-50',
     'glove-wiki-gigaword-100',
     'glove-wiki-gigaword-200',
     'glove-wiki-gigaword-300',
     'glove-twitter-25',
     'glove-twitter-50',
     'glove-twitter-100',
     'glove-twitter-200',
     '__testing_word2vec-matrix-synopsis']
    """

    #load and serialize the embeddings in order to accelerate next loads
    #input: name of the embedding wanted (example : 'glove-twitter-25')
    #output: the vector embeddings
    def embeddingPreparation(self,embeddingName):
        self.embeddingVectors = gensim.downloader.load(embeddingName)
        with open(embeddingName + ".vecs","wb") as fd:
            pickle.dump(self.embeddingVectors,fd)
        #return self.embeddingVectors

    #load already serialized embeddings (call of embeddingPreparation() needed before)
    #input: name of the embedding wanted (example : 'glove-twitter-25')
    #output: the vector embeddings
    def loadPreparedEmbedding(self,embeddingName):
        with open(embeddingName + ".vecs","rb") as fd:
            self.embeddingVectors = pickle.load(fd)
        #return self.embeddingVectors

    #input: word or id of the word
    #output: True if word in the embedding list, False otherwise
    def isVec(self,word):
        if word in self.embeddingVectors:
            return True
        else:
            return False

    #input: word(string) or word id(int)
    #output: vector embedding
    def getVec(self,word):
        return self.embeddingVectors[word]

    #input: word(string)
    #output: word id(int)
    def getId(self,word):
        return self.embeddingVectors.key_to_index[word]

    #input: word id(int)
    #output: word(string)
    def getWord(self,id):
        return self.embeddingVectors.index_to_key[id]

    #output : list of all words covered by the embedding
    def getVocabList(self):
        return self.embeddingVectors.index_to_key

    #output : return vocabulary dictionary associating word and idWords (key=word, value=idWord)
    def getVocabDic(self):
        return self.embeddingVectors.key_to_index

    #output : number of words covered by the embedding
    def getLenVocab(self):
        return len(self.embeddingVectors.index_to_key)

    #output :  number of dimensions of the embedding
    def nbDims(self):
        return self.embeddingVectors.vector_size

    #output : list of all embeddings names availables on gensim
    def getAvailableEmbeddings(self):
        return list(gensim.downloader.info()['models'].keys())

    #input : word , number of most similar words wanted
    #output : list of n most similar words
    def getMostSimilar(self,word,n):
        ms = self.embeddingVectors.most_similar(word,topn=n)
        neighbours = [elem[0] for elem in ms]
        return neighbours

    ########################################################
    ########################################################
    ########################################################

    def __init__(self,embeddingName):
        #test if file exists
        fileName = embeddingName + ".vecs"
        if os.path.isfile(fileName):
            print("loading embeddings...")
            self.loadPreparedEmbedding(embeddingName)
        else:
            print("embedding preparation...")
            self.embeddingPreparation(embeddingName)
        self.vectors = self.embeddingVectors.vectors


# =====================================================
# Load Data
# =====================================================

@st.cache_data
def load_data():

    tripadvisor = pd.read_csv("data/tripadvisor_final.csv")
    X_train02 = pd.read_csv("data/X_train.csv")
    X_test02 = pd.read_csv("data/X_test.csv")

    return tripadvisor, X_train02, X_test02


tripadvisor, X_train02, X_test02 = load_data()
tripadvisor_meta = tripadvisor.set_index("id")

# =====================================================
# Build Model
# =====================================================

@st.cache_data
def build_embedding_model(X_train02, X_test02):
    emb = gensim_interface('glove-wiki-gigaword-100')

    X_train02 = X_train02.reset_index(drop=True)
    X_test02 = X_test02.reset_index(drop=True)

    # ===================================
    # Fonction embedding document
    # ===================================
    def get_document_embedding(text, emb):
        words = text.split()
        vectors = []
        for word in words:
            if emb.isVec(word):
                vectors.append(emb.getVec(word))

        if len(vectors) == 0:
            return np.zeros(emb.nbDims())

        return np.mean(vectors, axis=0)

    # ===================================
    # Construire embeddings TRAIN
    # ===================================
    train_embeddings = []
    for text in X_train02['cleaned_review']:
        train_embeddings.append(
            get_document_embedding(text, emb)
        )

    train_embeddings = np.array(train_embeddings)

    # ===================================
    # Construire embeddings TEST
    # ===================================
    test_embeddings = []
    for text in X_test02['cleaned_review']:
        test_embeddings.append(
            get_document_embedding(text, emb)
        )

    test_embeddings = np.array(test_embeddings)

    # ===================================
    # Similarité cosine Test vs Train
    # ===================================
    similarity_matrix_m02 = cosine_similarity(
        test_embeddings,
        train_embeddings
    )

    return X_train02, X_test02, train_embeddings, test_embeddings, similarity_matrix_m02


X_train02, X_test02, train_embeddings, test_embeddings, similarity_matrix_m02 = build_embedding_model(X_train02, X_test02)

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
        top_k=5,
        top_n_dims=5
):

    # ===============================
    # Reset index propre
    # ===============================
    X_test_reset = X_test.reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)

    # ===============================
    # Check existence query
    # ===============================
    if idplace_query not in X_test_reset["idplace"].values:
        st.warning("ID not in TEST dataset")
        return []

    # ===============================
    # Position query
    # ===============================
    query_idx = X_test_reset.index[
        X_test_reset["idplace"] == int(idplace_query)
    ][0]

    sims = sim_matrix[query_idx].copy()

    # ===============================
    # Top-K recommendations
    # ===============================
    top_indices = np.argsort(sims)[::-1][:top_k]

    query_vec = test_embeddings[query_idx]

    rec_ids = []

    st.subheader("🔎 Recommendations")

    cols = st.columns(top_k)

    # ===============================
    # Display recommendations
    # ===============================
    for rank, (col, idx) in enumerate(zip(cols, top_indices), start=1):

        with col:

            rec_id = int(X_train_reset.iloc[idx]["idplace"])
            rec_ids.append(rec_id)

            score = sims[idx]
            similar_vec = train_embeddings[idx]

            # ===============================
            # Explainability (dimension contribution)
            # ===============================
            contribution = query_vec * similar_vec
            top_dims = contribution.argsort()[::-1][:top_n_dims]

            st.markdown(f"""
            **Reco {rank}**

            🏨 ID : {rec_id}

            ⭐ Similarity : {score:.3f}
            """)

            st.write("🔑 Top Contributing Dimensions")

            for dim in top_dims:
                if contribution[dim] > 0:
                    st.caption(
                        f"Dim {dim} → {contribution[dim]:.6f}"
                    )

    return rec_ids

# =====================================================
# Sidebar Selection
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
    st.session_state.query_id = query_id   # ⭐ TRÈS IMPORTANT

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
        top_k
    )

    # =====================================================
    # Map Section
    # =====================================================

    st.header("🗺 Map")

    center = [48.8566, 2.3522]

    m = folium.Map(
        location=center,
        zoom_start=13
    )

    # =====================================================
    # Query marker
    # =====================================================

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
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(m)

    # =====================================================
    # Recommendation markers
    # =====================================================

    for rec_id in rec_ids:

        # éviter doublons éventuels
        if rec_id not in tripadvisor_meta.index:
            continue

        # éviter d'afficher le query en reco
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
            icon=folium.Icon(
                color="blue",
                icon="star",
                prefix="fa"
            )
        ).add_to(m)

    # =====================================================
    # Display map
    # =====================================================

    st_folium(m, width=1000, height=600)

    # ==========================
    # Details Table
    # ==========================

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

    # ==========================
    # Evaluation Metrics
    # ==========================

    st.subheader("📊 Evaluation Metrics")

    level1 = evaluation_level1(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor,
        top_k=5
    )

    level2 = evaluation_level2(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor_meta,
        top_k=5
    )

    level3 = evaluation_level3(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor
    )

    level4 = evaluation_level4(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor_meta
    )

    level5 = evaluation_level5(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor,
        top_k=5
    )

    level6 = evaluation_level6(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor_meta,
        top_k=5
    )

    level7 = evaluation_level7(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor
    )

    level8 = evaluation_level8(
        X_test02,
        X_train02,
        similarity_matrix_m02,
        tripadvisor_meta
    )
    

    st.write(f"Precision Level 1 (Normalized Top-5 Type Score) : {level1:.4f}")
    st.write(f"Precision Level 2 (Normalized Top-5 Metadata Score) : {level2:.4f}")
    st.write(f"Precision Level 3 (Top-1 Type Accuracy) : {level3:.4f}")
    st.write(f"Precision Level 4 (Top-1 Metadata Accuracy) : {level4:.4f}")
    st.write(f"Precision Level 5 (Top-5 Type Recall) : {level5:.4f}")
    st.write(f"Precision Level 6 (Top-5 Metadata Recall) : {level6:.4f}")
    st.write(f"Precision Level 7 (Mean Rank of First Relevant Type Match) : {level7:.2f}")
    st.write(f"Precision Level 8 (Mean Rank of First Relevant Metadata Match) : {level8:.2f}")

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("Recommendation System")