import ast
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

warnings.filterwarnings("ignore")

ps = PorterStemmer()


# -------------------------------------------------------------------
# Helper functions (same logic as your script, but modular)
# -------------------------------------------------------------------
def convert(obj):
    """Convert JSON-like string of [{'name': ...}, ...] to list of names."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L


def convert_top3_cast(obj):
    """Take only top 3 cast members."""
    L = []
    ct = 0
    for i in ast.literal_eval(obj):
        if ct != 3:
            L.append(i["name"])
            ct += 1
        else:
            break
    return L


def fetch_director(obj):
    """Extract director name from crew list."""
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L


def stem(text):
    """Apply Porter stemming to each word in the text."""
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# -------------------------------------------------------------------
# Data load + preprocessing (cached so Streamlit doesn't redo every run)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
import os
import pandas as pd
import streamlit as st
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# simple stemmer (no NLTK)
# -------------------------------------------------------------------
def stem(text):
    return " ".join(
        word.lower() for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS
    )

# your other helper functions: convert, convert_top3_cast, fetch_director stay the same


@st.cache_resource(show_spinner=True)
def load_and_process_data():
    # 👇 Base directory of this file (…/movie_recomendation_system/set_up)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 👇 Assume CSVs are in the repo root one level up: …/movie_recomendation_system
    data_dir = os.path.abspath(os.path.join(base_dir, ".."))

    credits_path = os.path.join(data_dir, "credits.csv")
    movies_path = os.path.join(data_dir, "movies.csv")

    # Safety check: show a nice error if files are missing
    if not os.path.exists(credits_path) or not os.path.exists(movies_path):
        st.error(
            "❌ Could not find `credits.csv` or `movies.csv`.\n\n"
            "I looked in:\n"
            f"- {credits_path}\n"
            f"- {movies_path}\n\n"
            "✅ Fix:\n"
            "- Either move the CSV files to that location, OR\n"
            "- Update `credits_path` and `movies_path` in `load_and_process_data()` "
            "to point wherever your files actually are."
        )
        st.stop()

    # 👇 Removed `encoding_errors` so it works with all pandas versions
    credits = pd.read_csv(credits_path, encoding="latin1")
    movies = pd.read_csv(movies_path)

    # --- your existing preprocessing logic from before ---

    movies = movies.merge(credits, on="title")
    movies = movies[["genres", "id", "keywords", "title", "overview", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_top3_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


def recommend(movie, new_df, similarity, n_recommendations=5):
    """Return a list of recommended movie titles similar to the given movie."""
    # If movie not in dataset, return empty list
    if movie not in new_df["title"].values:
        return []

    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
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

    with st.spinner("Loading data and building model..."):
        new_df, similarity = load_and_process_data()

    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.write(
        """
        This app uses a content-based recommendation system:
        - Combines **overview, genres, keywords, cast, and crew**
        - Applies **stemming** using NLTK
        - Vectorizes text with **CountVectorizer (max 5000 features)**
        - Uses **cosine similarity** to find similar movies
        """
    )

    # Movie selection
    movie_list = new_df["title"].values
    selected_movie = st.selectbox(
        "Select a movie", movie_list, index=0, help="Start typing to search."
    )

    if st.button("🔍 Show Recommendations"):
        recommendations = recommend(selected_movie, new_df, similarity)

        if not recommendations:
            st.error(f"Sorry, '{selected_movie}' was not found in the dataset.")
        else:
            st.subheader(f"Because you watched **{selected_movie}**, you might also like:")
            for i, rec in enumerate(recommendations, start=1):
                st.markdown(f"{i}. **{rec}**")


if __name__ == "__main__":
    main()
