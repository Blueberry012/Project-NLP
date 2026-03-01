# =====================================================
# Imports
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from sklearn.feature_extraction.text import TfidfVectorizer
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
# Streamlit Config
# =====================================================

st.set_page_config(layout="wide")
st.title("🎯 TF-IDF NLP Pretreatment Recommendation System")

# =====================================================
# Session State
# =====================================================

if "run" not in st.session_state:
    st.session_state.run = False

if "query_id" not in st.session_state:
    st.session_state.query_id = None

# =====================================================
# Load Data
# =====================================================

@st.cache_data
def load_data():

    tripadvisor = pd.read_csv("data/tripadvisor_final.csv")
    part1 = pd.read_csv("data/X_train_part1.csv")
    part2 = pd.read_csv("data/X_train_part2.csv")
    X_train04 = pd.concat([part1, part2], ignore_index=True)
    #X_train04 = pd.read_csv("data/X_train.csv")
    X_test04 = pd.read_csv("data/X_test.csv")

    return tripadvisor, X_train04, X_test04


tripadvisor, X_train04, X_test04 = load_data()

tripadvisor_meta = tripadvisor.set_index("id")

# =====================================================
# TFIDF Model
# =====================================================

@st.cache_data
def build_model(X_train, X_test, tripadvisor):

    # ===================================
    # Mapping id → typeR
    # ===================================
    id_to_typeR = tripadvisor.set_index('id')['typeR'].to_dict()

    X_train['typeR'] = X_train['idplace'].map(id_to_typeR)
    X_test['typeR'] = X_test['idplace'].map(id_to_typeR)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # ===================================
    # TF-IDF Representation
    # ===================================
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),   # add bigrams
        min_df=2
    )

    tfidf_train = vectorizer.fit_transform(X_train['cleaned_review'])
    tfidf_test = vectorizer.transform(X_test['cleaned_review'])

    # ===================================
    # Cosine Similarity
    # ===================================
    similarity_matrix = cosine_similarity(
        tfidf_test,
        tfidf_train
    )

    return (
        vectorizer,
        tfidf_train,
        tfidf_test,
        similarity_matrix,
    )

vectorizer, tfidf_train, tfidf_test, similarity_matrix_m04 = build_model(
    X_train04,
    X_test04,
    tripadvisor
)

# =====================================================
# Recommendation Engine
# =====================================================

def recommend_similar_place_tfidf_nlp_pretreatment(idplace_query, top_k=5, top_n_words=5):

    if idplace_query not in X_test04['idplace'].values:
        st.warning("ID not found")
        return []

    X_test_reset = X_test04.reset_index(drop=True)

    query_pos = X_test_reset.index[
        X_test_reset['idplace'] == idplace_query
    ][0]

    sims = similarity_matrix_m04[query_pos]

    top_indices_sim = np.argsort(sims)[::-1][:top_k]

    query_vector = tfidf_test[query_pos].toarray()[0]
    feature_names = np.array(vectorizer.get_feature_names_out())

    rec_ids = []

    st.subheader("🔎 Recommendations")

    cols = st.columns(top_k)

    for i, idx in enumerate(top_indices_sim):

        with cols[i]:

            similar_id = int(X_train04.iloc[idx]['idplace'])
            rec_ids.append(similar_id)

            score = sims[idx]

            st.markdown(f"""
            **Reco {i+1}**

            ID : {similar_id}

            ⭐ Similarity : {score:.3f}
            """)

            similar_vector = tfidf_train[idx].toarray()[0]

            contribution = query_vector * similar_vector
            top_word_indices = contribution.argsort()[::-1][:top_n_words]

            st.write("🔑 Keywords")

            for word_idx in top_word_indices:
                if contribution[word_idx] > 0:
                    word = feature_names[word_idx]
                    sc = contribution[word_idx]
                    st.caption(f"{word} ({sc:.4f})")

    return rec_ids


# =====================================================
# Sidebar Selection
# =====================================================

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# Merge test + places info
test_places = X_test04.merge(
    tripadvisor,
    left_on="idplace",
    right_on="id",
    how="left"
)

test_places["display_name"] = (
    test_places["idplace"].astype(str) +
    " - " +
    test_places["nom"].astype(str)
)

test_places = test_places.drop_duplicates("idplace")

default_place = test_places[
    test_places["idplace"] == 1725986
]["display_name"].values

default_index = 0

if len(default_place) > 0:
    default_value = default_place[0]
    default_index = test_places["display_name"].tolist().index(default_value)

selected_place = st.sidebar.selectbox(
    "🏨 Choose a place (TEST dataset only)",
    test_places["display_name"].tolist(),
    index=default_index
)

query_id = int(selected_place.split(" - ")[0])

st.session_state.query_id = query_id

if st.sidebar.button("🚀 Search"):
    st.session_state.run = True

# =====================================================
# Execution
# =====================================================

if st.session_state.run:

    rec_ids = recommend_similar_place_tfidf_nlp_pretreatment(
        st.session_state.query_id,
        top_k=top_k
    )

    # =====================================================
    # Map Section
    # =====================================================

    st.header("🗺 Paris Map")

    center = [48.8566, 2.3522]

    m = folium.Map(
        location=center,
        zoom_start=13
    )

    # Query marker
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

    # =====================================================
    # Recommendation markers
    # =====================================================

    for rec_id in rec_ids:

        if rec_id not in tripadvisor_meta.index:
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

    st_folium(m, width=1000, height=600)

    # =====================================================
    # Details Section
    # =====================================================

    st.header("📋 Places Details")

    ordered_ids = [st.session_state.query_id] + rec_ids

    tripadvisor_test = tripadvisor[
        tripadvisor['id'].isin(ordered_ids)
    ].copy()

    rank_map = {st.session_state.query_id: 0}

    for i, rec_id in enumerate(rec_ids, start=1):
        rank_map[rec_id] = i

    tripadvisor_test["Top"] = tripadvisor_test["id"].map(rank_map)

    tripadvisor_test = tripadvisor_test.sort_values("Top")

    cols = list(tripadvisor_test.columns)

    if "Top" in cols:
        cols.remove("Top")
        cols = ["Top"] + cols

    tripadvisor_test = tripadvisor_test[cols]

    st.dataframe(
        tripadvisor_test,
        use_container_width=True
    )

    # =====================================================
    # Evaluation Metrics
    # =====================================================

    st.subheader("📊 Evaluation Metrics")

    level1 = evaluation_level1(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor,
        top_k=5
    )

    level2 = evaluation_level2(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor_meta,
        top_k=5
    )

    level3 = evaluation_level3(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor
    )

    level4 = evaluation_level4(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor_meta
    )

    level5 = evaluation_level5(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor,
        top_k=5
    )

    level6 = evaluation_level6(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor_meta,
        top_k=5
    )

    level7 = evaluation_level7(
        X_test04,
        X_train04,
        similarity_matrix_m04,
        tripadvisor
    )

    level8 = evaluation_level8(
        X_test04,
        X_train04,
        similarity_matrix_m04,
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