import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie and discover other movies "
    "with similar genres, keywords, cast, and plots."
)


# --------------------------------------------------
# 2. LOAD AND PREPROCESS THE DATASET
# --------------------------------------------------

DATA_DIR = Path("data")

MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"


@st.cache_data
def load_data():

    if not MOVIES_FILE.exists() or not CREDITS_FILE.exists():
        raise FileNotFoundError(
            "Dataset files not found. Place both CSV files "
            "inside the data folder."
        )

    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    # Merge the two datasets
    df = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        suffixes=("_movie", "_credit")
    )

    # Select the required columns
    df = df[
        [
            "id",
            "title_movie",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]
    ].copy()

    df.rename(
        columns={"title_movie": "title"},
        inplace=True
    )

    # Remove missing titles and duplicate movies
    df.dropna(
        subset=["title"],
        inplace=True
    )

    df.drop_duplicates(
        subset=["id"],
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    return df


# --------------------------------------------------
# 3. PARSE GENRES, KEYWORDS, CAST, AND DIRECTORS
# --------------------------------------------------

def parse_list(value):

    if not isinstance(value, str):
        return []

    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

        return []

    except (ValueError, SyntaxError):
        return []


def extract_names(value, limit=None):

    items = parse_list(value)
    names = []

    for item in items:

        if isinstance(item, dict):
            name = item.get("name")

            if name:
                names.append(
                    str(name).replace(" ", "")
                )

        if limit and len(names) >= limit:
            break

    return names


def extract_director(value):

    crew = parse_list(value)

    for person in crew:

        if (
            isinstance(person, dict)
            and person.get("job") == "Director"
        ):
            name = person.get("name")

            if name:
                return str(name).replace(" ", "")

    return ""


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

def prepare_features(df):

    df = df.copy()

    df["overview"] = (
        df["overview"]
        .fillna("")
        .astype(str)
    )

    df["genres"] = df["genres"].apply(
        extract_names
    )

    df["keywords"] = df["keywords"].apply(
        extract_names
    )

    df["cast"] = df["cast"].apply(
        lambda value: extract_names(
            value,
            limit=5
        )
    )

    df["director"] = df["crew"].apply(
        extract_director
    )

    # Combine all features into one text field
    df["tags"] = (
        df["genres"].apply(
            lambda values: " ".join(values)
        )
        + " "
        + df["keywords"].apply(
            lambda values: " ".join(values)
        )
        + " "
        + df["cast"].apply(
            lambda values: " ".join(values)
        )
        + " "
        + df["director"]
        + " "
        + df["overview"].str.lower()
    )

    # Remove rows with no usable feature information
    df = df[
        df["tags"].str.strip().ne("")
    ].copy()

    df.reset_index(drop=True, inplace=True)

    return df


# --------------------------------------------------
# 5. TRAIN THE CONTENT-BASED RECOMMENDATION MODEL
# --------------------------------------------------

@st.cache_resource
def build_model():

    df = load_data()
    df = prepare_features(df)

    # Convert text into numerical TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["tags"]
    )

    return df, tfidf_matrix


# --------------------------------------------------
# 6. RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(
    selected_title,
    df,
    tfidf_matrix,
    number_of_recommendations=5,
    genre_filter=None
):

    # Find the selected movie
    matching_indices = df.index[
        df["title"] == selected_title
    ].tolist()

    if not matching_indices:
        return pd.DataFrame()

    movie_index = matching_indices[0]

    # Calculate similarity against every movie
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort movies from most to least similar
    similar_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for index in similar_indices:

        # Exclude the selected movie
        if index == movie_index:
            continue

        movie = df.iloc[index]

        # Apply the optional genre filter
        if genre_filter and genre_filter != "All Genres":

            if genre_filter not in movie["genres"]:
                continue

        recommendations.append(
            {
                "title": movie["title"],
                "overview": movie["overview"],
                "genres": ", ".join(movie["genres"]),
                "director": movie["director"],
                "similarity_score": round(
                    float(similarity_scores[index]),
                    3
                )
            }
        )

        if len(recommendations) >= number_of_recommendations:
            break

    return pd.DataFrame(recommendations)


# --------------------------------------------------
# 7. LOAD THE MODEL
# --------------------------------------------------

try:

    with st.spinner(
        "Loading movie dataset and building recommendation model..."
    ):

        df, tfidf_matrix = build_model()

except Exception as error:

    st.error(str(error))

    st.info(
        "Create a folder named 'data' and place both "
        "TMDB CSV files inside it."
    )

    st.stop()


st.success(
    f"Recommendation system ready! "
    f"Loaded {len(df):,} movies."
)


# --------------------------------------------------
# 8. USER INPUTS
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

number_of_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

all_genres = sorted(
    {
        genre
        for genres in df["genres"]
        for genre in genres
    }
)

genre_filter = st.sidebar.selectbox(
    "Filter by genre",
    ["All Genres"] + all_genres
)

st.subheader("Find Movies You Might Like")

selected_title = st.selectbox(
    "Search and select a movie",
    options=sorted(
        df["title"].unique().tolist()
    )
)


# --------------------------------------------------
# 9. DISPLAY THE SELECTED MOVIE
# --------------------------------------------------

selected_movie = df[
    df["title"] == selected_title
].iloc[0]

st.markdown("### Selected Movie")

st.write(f"**Title:** {selected_movie['title']}")

st.write(
    "**Genres:** "
    + (
        ", ".join(selected_movie["genres"])
        if selected_movie["genres"]
        else "Not available"
    )
)

st.write(
    "**Director:** "
    + (
        selected_movie["director"]
        if selected_movie["director"]
        else "Not available"
    )
)

st.write(
    selected_movie["overview"]
    if selected_movie["overview"]
    else "No plot description available."
)


# --------------------------------------------------
# 10. GENERATE RECOMMENDATIONS
# --------------------------------------------------

if st.button(
    "Recommend Movies",
    type="primary",
    use_container_width=True
):

    recommendations = recommend_movies(
        selected_title=selected_title,
        df=df,
        tfidf_matrix=tfidf_matrix,
        number_of_recommendations=number_of_recommendations,
        genre_filter=genre_filter
    )

    if recommendations.empty:

        st.warning(
            "No recommendations found for these settings. "
            "Try selecting a different movie or genre."
        )

    else:

        st.subheader(
            f"Movies Similar to {selected_title}"
        )

        for rank, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            with st.container(border=True):

                st.markdown(
                    f"### {rank}. {movie['title']}"
                )

                st.write(
                    f"**Genres:** {movie['genres'] or 'Not available'}"
                )

                st.write(
                    f"**Director:** "
                    f"{movie['director'] or 'Not available'}"
                )

                st.write(
                    movie["overview"]
                    if movie["overview"]
                    else "No plot description available."
                )

                st.progress(
                    min(
                        max(
                            float(movie["similarity_score"]),
                            0.0
                        ),
                        1.0
                    )
                )

                st.caption(
                    "Cosine similarity: "
                    f"{movie['similarity_score']:.3f}"
                )


# --------------------------------------------------
# 11. PROJECT INFORMATION
# --------------------------------------------------

with st.expander("How does this recommendation system work?"):

    st.write(
        """
        This project uses content-based filtering.

        1. Movie genres, keywords, cast, directors, and
           plot descriptions are combined into text features.

        2. TF-IDF converts the text into numerical vectors.

        3. Cosine similarity compares the movie vectors.

        4. The system ranks movies by similarity and returns
           the highest-ranked matches.

        The system does not learn individual user preferences
        or use personal viewing histories.

        Similarity scores measure the closeness of feature
        vectors. They are not probabilities that a user will
        enjoy a movie.
        """
    )

st.divider()

st.caption(
    "Movie Recommendation System | "
    "Built with Python, Pandas, Scikit-learn, and Streamlit"
)