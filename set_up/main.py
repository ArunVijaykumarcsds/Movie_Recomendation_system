import os
import ast
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def convert(obj):
    """Convert JSON-like string of [{'name': ...}, ...] to list of names."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i.get("name", "").strip())
    return L


def convert_top3_cast(obj):
    """Take only top 3 cast members."""
    L = []
    ct = 0
    for i in ast.literal_eval(obj):
        if ct != 3:
            L.append(i.get("name", "").strip())
            ct += 1
        else:
            break
    return L


def fetch_director(obj):
    """Extract director name from crew list."""
    L = []
    for i in ast.literal_eval(obj):
        if i.get("job") == "Director":
            name = i.get("name", "").strip()
            if name:
                L.append(name)
            break
    return L


def stem(text):
    """
    Simple text cleaner / 'stemmer' replacement that:
    - lowercases
    - removes common English stopwords
    (No NLTK required)
    """
    tokens = []
    for word in text.split():
        w = word.lower()
        if w not in ENGLISH_STOP_WORDS:
            tokens.append(w)
    return " ".join(tokens)


# -------------------------------------------------------------------
# Data load + preprocessing
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_and_process_data():
    """
    Load movies & credits CSVs, build the tag corpus and similarity matrix.
    Expects:
      - credits.csv
      - movies.csv
    to be located one level ABOVE this file (repo root).
    """
    # base directory of this file: .../movie_recomendation_system/set_up
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # assume CSVs are in repo root: .../movie_recomendation_system
    data_dir = os.path.abspath(os.path.join(base_dir, ".."))

    credits_path = os.path.join(data_dir, "credits.csv")
    movies_path = os.path.join(data_dir, "movies.csv")

    # friendly error if files missing
    if not os.path.exists(credits_path) or not os.path.exists(movies_path):
        st.error(
            "❌ Could not find `credits.csv` or `movies.csv`.\n\n"
            "I looked in:\n"
            f"- {credits_path}\n"
            f"- {movies_path}\n\n"
            "✅ Fix:\n"
            "- Move the CSV files to that location, OR\n"
            "- Update `credits_path` and `movies_path` in `load_and_process_data()` "
            "to match your folder structure."
        )
        st.stop()

    # read data
    credits = pd.read_csv(credits_path, encoding="latin1")
    movies = pd.read_csv(movies_path)

    # merge and select columns
    movies = movies.merge(credits, on="title")
    movies = movies[["genres", "id", "keywords", "title", "overview", "cast", "crew"]]

    # drop rows with missing values in these columns
    movies.dropna(inplace=True)

    # parse JSON-like strings into lists
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_top3_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    # split overview to tokens
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # remove spaces inside multi-word tokens
    movies["genres"] = movies["genres"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    movies["keywords"] = movies["keywords"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    movies["cast"] = movies["cast"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    movies["crew"] = movies["crew"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    # combine to tags
    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    # new dataframe with id, title, tags
    new_df = movies[["id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    # vectorization
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    # cosine similarity matrix
    similarity = cosine_similarity(vectors)

    return new_df, similarity


# -------------------------------------------------------------------
# Recommendation logic
# -------------------------------------------------------------------
def recommend(movie_title, new_df, similarity, n_recommendations=5):
    """
    Return a list of recommended movie titles similar to the given movie.
    """
    titles = new_df["title"].values

    if movie_title not in titles:
        return []

    movie_index = new_df[new_df["title"] == movie_title].index[0]
    distances = similarity[movie_index]

    # sort by similarity score, skip first (itself)
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1 : n_recommendations + 1]

    recommended_titles = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_titles


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.title("🎬 Movie Recommendation System")
    st.write(
        "Type or select a movie, and I'll suggest similar movies based on content similarity."
    )

    with st.spinner("Loading data and building recommendation model..."):
        new_df, similarity = load_and_process_data()

    # sidebar info
    st.sidebar.header("About this app")
    st.sidebar.write(
        """
        This is a **content-based** movie recommender:
        - Uses *overview, genres, keywords, cast, and crew*
        - Cleans and tokenizes text
        - Vectorizes with **CountVectorizer (max 5000 features)**
        - Computes similarity using **cosine similarity**
        """
    )

    # movie selection
    movie_list = new_df["title"].values
    selected_movie = st.selectbox(
        "Select a movie",
        movie_list,
        index=0,
        help="Start typing to quickly find a movie.",
    )

    if st.button("🔍 Show Recommendations"):
        recommendations = recommend(selected_movie, new_df, similarity)

        if not recommendations:
            st.error(
                f"Sorry, I couldn't find recommendations for **{selected_movie}**."
            )
        else:
            st.subheader(
                f"Because you watched **{selected_movie}**, you might also like:"
            )
            for i, rec_title in enumerate(recommendations, start=1):
                st.markdown(f"{i}. **{rec_title}**")


if __name__ == "__main__":
    main()
