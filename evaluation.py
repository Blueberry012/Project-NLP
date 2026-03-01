import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


# ==========================================================
# TEXT PREPROCESSING
# ==========================================================

def preprocess_text(text):
    """
    Preprocess text:
    - lowercase
    - tokenize sentences
    - remove stopwords
    - keep alphabetic words only
    - lemmatize (verb mode)
    """
    stops = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(text)
    processed_sentences = []

    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w.isalpha() and w not in stops]
        words = [lemmatizer.lemmatize(w, pos='v') for w in words]
        processed_sentences.append(" ".join(words))

    return " ".join(processed_sentences)


# ==========================================================
# METADATA UTILITIES
# ==========================================================

metadata_columns = [
    'typeR',
    'priceRange',
    'activiteType',
    'activiteSubType',
    'activiteCategory',
    'restaurantType',
    'restaurantCategory',
    'restaurantCuisine',
    'restaurantDietaryRestrictions',
    'hotelType',
    'hotelpriceRange'
]


def to_set(value):
    if pd.isna(value):
        return set()
    return set(v.strip().lower() for v in str(value).split(','))


def metadata_precision_score(query_id, recommended_id, tripadvisor_meta):
    if query_id not in tripadvisor_meta.index or recommended_id not in tripadvisor_meta.index:
        return 0

    q = tripadvisor_meta.loc[query_id]
    r = tripadvisor_meta.loc[recommended_id]

    match_count = 0
    total_count = 0

    for col in metadata_columns:
        if pd.isna(q[col]) and pd.isna(r[col]):
            continue

        total_count += 1

        q_set = to_set(q[col])
        r_set = to_set(r[col])

        if len(q_set & r_set) > 0:
            match_count += 1

    if total_count == 0:
        return 0

    return match_count / total_count


# ==========================================================
# EVALUATION LEVEL 1
# ==========================================================

def evaluation_level1(X_test, X_train, similarity_matrix, tripadvisor, top_k=5):
    id_to_typeR = tripadvisor.set_index('id')['typeR'].to_dict()

    train_categories = X_train['idplace'].map(id_to_typeR).fillna("UNKNOWN").values
    test_categories = X_test['idplace'].map(id_to_typeR).fillna("UNKNOWN").values

    scores = []

    for i in range(len(test_categories)):
        sims = similarity_matrix[i]
        top_k_indices = sims.argsort()[::-1][:top_k]

        retrieved_types = train_categories[top_k_indices]
        correct = np.sum(retrieved_types == test_categories[i])
        scores.append(correct / top_k)

    return np.mean(scores) if scores else 0


# ==========================================================
# EVALUATION LEVEL 2
# ==========================================================

def evaluation_level2(X_test, X_train, similarity_matrix, tripadvisor_meta, top_k=5):
    scores = []

    for i in range(len(X_test)):
        query_id = X_test.iloc[i]['idplace']
        sims = similarity_matrix[i]
        top_k_indices = sims.argsort()[::-1][:top_k]

        metadata_scores = []

        for idx in top_k_indices:
            recommended_id = X_train.iloc[idx]['idplace']
            score = metadata_precision_score(query_id, recommended_id, tripadvisor_meta)
            metadata_scores.append(score)

        if metadata_scores:
            scores.append(np.mean(metadata_scores))

    return np.mean(scores) if scores else 0


# ==========================================================
# EVALUATION LEVEL 3
# ==========================================================

def evaluation_level3(X_test, X_train, similarity_matrix, tripadvisor):
    id_to_typeR = tripadvisor.set_index('id')['typeR'].to_dict()

    train_categories = X_train['idplace'].map(id_to_typeR).values
    test_categories = X_test['idplace'].map(id_to_typeR).values

    correct = 0

    for i in range(len(test_categories)):
        sims = similarity_matrix[i]
        most_similar_idx = sims.argmax()

        if test_categories[i] == train_categories[most_similar_idx]:
            correct += 1

    return correct / len(test_categories) if len(test_categories) > 0 else 0


# ==========================================================
# EVALUATION LEVEL 4
# ==========================================================

def evaluation_level4(X_test, X_train, similarity_matrix, tripadvisor_meta):
    scores = []

    for i in range(len(X_test)):
        query_id = X_test.iloc[i]['idplace']
        sims = similarity_matrix[i]

        best_idx = sims.argmax()
        recommended_id = X_train.iloc[best_idx]['idplace']

        score = metadata_precision_score(query_id, recommended_id, tripadvisor_meta)
        scores.append(score)

    return np.mean(scores) if scores else 0


# ==========================================================
# EVALUATION LEVEL 5
# ==========================================================

def evaluation_level5(X_test, X_train, similarity_matrix, tripadvisor, top_k=5):
    id_to_typeR = tripadvisor.set_index('id')['typeR'].to_dict()

    train_categories = X_train['idplace'].map(id_to_typeR).values
    test_categories = X_test['idplace'].map(id_to_typeR).values

    correct = 0

    for i in range(len(test_categories)):
        sims = similarity_matrix[i]
        top_k_indices = sims.argsort()[::-1][:top_k]

        if test_categories[i] in train_categories[top_k_indices]:
            correct += 1

    return correct / len(test_categories) if len(test_categories) > 0 else 0


# ==========================================================
# EVALUATION LEVEL 6
# ==========================================================

def evaluation_level6(X_test, X_train, similarity_matrix, tripadvisor_meta, top_k=5):
    correct = 0

    for i in range(len(X_test)):
        query_id = X_test.iloc[i]['idplace']
        sims = similarity_matrix[i]
        top_k_indices = sims.argsort()[::-1][:top_k]

        match_found = False

        for idx in top_k_indices:
            recommended_id = X_train.iloc[idx]['idplace']
            score = metadata_precision_score(query_id, recommended_id, tripadvisor_meta)

            if score == 1:
                match_found = True
                break

        if match_found:
            correct += 1

    return correct / len(X_test) if len(X_test) > 0 else 0


# ==========================================================
# EVALUATION LEVEL 7
# ==========================================================

def evaluation_level7(X_test, X_train, similarity_matrix, tripadvisor):
    id_to_typeR = tripadvisor.set_index('id')['typeR'].to_dict()

    train_categories = X_train['idplace'].map(id_to_typeR).fillna("UNKNOWN").values
    test_categories = X_test['idplace'].map(id_to_typeR).fillna("UNKNOWN").values

    ranks = []

    for i in range(len(X_test)):
        sims = similarity_matrix[i]
        sorted_indices = sims.argsort()[::-1]

        query_type = test_categories[i]
        found_rank = None

        for rank, idx in enumerate(sorted_indices, start=1):
            if train_categories[idx] == query_type:
                found_rank = rank
                break

        ranks.append(found_rank if found_rank is not None else len(sorted_indices))

    return np.mean(ranks) if ranks else 0


# ==========================================================
# EVALUATION LEVEL 8
# ==========================================================

def evaluation_level8(X_test, X_train, similarity_matrix, tripadvisor_meta):
    ranks = []

    for i in range(len(X_test)):
        query_id = X_test.iloc[i]['idplace']
        sims = similarity_matrix[i]
        sorted_indices = sims.argsort()[::-1]

        found_rank = None

        for rank, idx in enumerate(sorted_indices, start=1):
            recommended_id = X_train.iloc[idx]['idplace']
            score = metadata_precision_score(query_id, recommended_id, tripadvisor_meta)

            if score == 1:
                found_rank = rank
                break

        ranks.append(found_rank if found_rank is not None else len(sorted_indices))

    return np.mean(ranks) if ranks else 0