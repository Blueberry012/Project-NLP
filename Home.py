import streamlit as st

# =====================================================
# Page Config
# =====================================================

st.set_page_config(layout="wide")

# Title
st.title("🎯 TripAdvisor Recommendation System")
st.subheader("Information Retrieval as a Recommender System")
st.image("image/tripadvisor.png", width=750, clamp=True)

st.markdown("---")

# =====================================================
# Project Description
# =====================================================

st.header("📌 Project Overview")

st.markdown("""
This project focuses on **Information Retrieval techniques applied to recommendation systems**.

### 🎯 Main Goal
Based on past reviews of a place, recommend the most similar experiences.

### 💡 Main Hypothesis
Similar experiences (restaurants, hotels, attractions) are described using **similar words and expressions**.

If this hypothesis is valid, then we can recommend similar places relying **only on textual reviews**, without using metadata.
""")

# =====================================================
# Input / Output Model
# =====================================================

st.header("🔄 Input / Output Model")

st.markdown("""
### ✅ Input
- One review or a set of reviews describing a place.

### ✅ Output
- Top-K most similar places based on textual similarity.

The model relies mainly on NLP representations of reviews.
""")

# =====================================================
# Models Used
# =====================================================

st.header("🧠 Models & Baselines")

st.markdown("""
### 📊 Baseline Model
- **BM25 Ranking Model**
- Available here:
https://pypi.org/project/rank-bm25/

BM25 is used as a classical Information Retrieval baseline.

---

### 🚀 Advanced Models Designed
In this project, we designed models stronger than BM25:

✅ TF-IDF Vector Space Model  
✅ Word Embedding Representation (GloVe)  
✅ Hybrid Recommendation Approaches (optional extension)

These models aim to capture **semantic similarity**, not only lexical similarity.
""")

# =====================================================
# Evaluation Strategy
# =====================================================

st.header("📈 Evaluation Methodology")

st.markdown("""
Even if the system only uses reviews, we still need evaluation.

The recommendation quality is validated using metadata consistency.

### Evaluation Idea
For a given query experience:

If the system is good →  
Recommended places should belong to **the same category**:

- Restaurant → Restaurant
- Hotel → Hotel
- Attraction → Attraction

Metrics used:

- Precision@K
- Recall@K
- Rank-based metrics
""")

# =====================================================
# Features of the System
# =====================================================

st.header("✨ System Features")

st.markdown("""
✔ Interactive recommendation interface  
✔ Map visualization of recommendations  
✔ Explainability via keyword contributions  
✔ Multiple NLP models comparison  
✔ Evaluation dashboards  
""")

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("TripAdvisor NLP Recommendation Project")
