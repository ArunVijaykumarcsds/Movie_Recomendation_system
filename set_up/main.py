import os
import ast
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# Streamlit config
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
    L = []
    for i in ast.literal_eval(obj):
        L.append(i.get("name", "").strip())
    return L


def convert_top3_cast(obj):
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
    L = []
    for i in ast.literal_eval(obj):
        if i.get("job") == "Director":
            name = i.get("name", "").strip()
            if name:
                L.append(name)
            break
    return L


def stem(text):
    return " ".join(
        word.lower() for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS
    )


# -------------------------------------------------------------------
# Load + process data (CSV in SAME folder as main.py)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_and_process_data():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    credits_path = os.path.join(base_dir, "credits.csv")
    movies_path = os.path.join(base_dir, "movies.csv")

    if not os.path.exists(credits_path) or not os.path.exists(movies_path):
        st.error(
            "❌ Could not find `credits.csv` or `movies.csv`.\n\n"
            "Make sure this folder contains at least:\n\n"
            "- main.py\n"
            "- credits.csv\n"
            "- movies.csv\n"
        )
        st.stop()

    credits = pd.read_csv(credits_path, encoding="latin1")
    movies = pd.read_csv(movies_path)

    movies = movies.merge(credits, on="title")
    movies = movies[["genres", "id", "keywords", "title", "overview", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_top3_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

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

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["id", "title", "tags"]]
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


# -------------------------------------------------------------------
# Recommend movies
# -------------------------------------------------------------------
def recommend(movie_title, new_df, similarity, n=5):
    titles = new_df["title"].values
    if movie_title not in titles:
        return []

    idx = new_df[new_df["title"] == movie_title].index[0]
    distances = similarity[idx]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1 : n + 1]

    return [new_df.iloc[i[0]].title for i in movies_list]


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.title("🎬 Movie Recommendation System")

    # --- Local header image (banner.jpg in same folder) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    banner_path = os.path.join(base_dir, "banner.jpg")

    if os.path.exists(banner_path):
        st.image(
            banner_path,
            caption="Movie Recommendation System",
            use_column_width=True,
        )
    else:
        st.info(
            "ℹ️ Add an image named `banner.jpg` in this folder to display a header banner."
        )

    st.write("Select a movie, and I’ll suggest similar movies based on content similarity.")

    with st.spinner("Loading data…"):
        new_df, similarity = load_and_process_data()

    movie_list = new_df["title"].values
    selected_movie = st.selectbox("Choose a movie", movie_list)

    if st.button("🔍 Get Recommendations"):
        recs = recommend(selected_movie, new_df, similarity)
        if not recs:
            st.error("No recommendations found.")
        else:
            st.subheader(f"Because you watched **{selected_movie}**, you may also like:")
            for i, r in enumerate(recs, 1):
                st.write(f"{i}. **{r}**")


if __name__ == "__main__":
    main()
