import streamlit as st

from src.config import DEFAULT_FEATURES, DEFAULT_WEIGHTS
from src.data_loader import active_preprocessing_key, load_prepared_games
from src.recommender import build_vector_model, normalize_weights
from src.ui import page_setup, require_data


page_setup("Build Recommender")
st.title("Build Recommender")

st.write(
    "Choose which game information should influence similarity, then set how much the final "
    "match score should care about content, rating, popularity, and recency."
)

st.subheader("Content Features")
selected_features = {}
cols = st.columns(3)
for index, (feature, default) in enumerate(DEFAULT_FEATURES.items()):
    selected_features[feature] = cols[index % 3].checkbox(
        feature.replace("_", " ").title(),
        value=st.session_state.get("selected_features", DEFAULT_FEATURES).get(feature, default),
    )

if not any(selected_features.values()):
    st.warning("Select at least one feature to build the recommender.")
    st.stop()

st.subheader("Score Weights")
left, right = st.columns(2)
current_weights = st.session_state.get("score_weights", DEFAULT_WEIGHTS)
weights = {
    "content": left.slider("Content Similarity", 0.0, 1.0, float(current_weights["content"]), 0.05),
    "rating": left.slider("Rating Score", 0.0, 1.0, float(current_weights["rating"]), 0.05),
    "popularity": right.slider("Popularity Score", 0.0, 1.0, float(current_weights["popularity"]), 0.05),
    "recency": right.slider("Recency Score", 0.0, 1.0, float(current_weights["recency"]), 0.05),
}
normalized = normalize_weights(weights)

st.caption(
    f"Normalized weights: content {normalized['content']:.0%}, rating {normalized['rating']:.0%}, "
    f"popularity {normalized['popularity']:.0%}, recency {normalized['recency']:.0%}."
)

if st.button("Build / Update Recommender", type="primary"):
    with st.spinner("Preparing data and building the recommender..."):
        df = load_prepared_games()
        if not require_data(df):
            st.stop()
        data_key = active_preprocessing_key()
        st.session_state["selected_features"] = selected_features
        st.session_state["score_weights"] = normalized
        feature_key = tuple(sorted(selected_features.items()))
        build_vector_model(df, feature_key, data_key)
        st.session_state["recommender_model_ready"] = True
        st.session_state["recommender_model_key"] = (feature_key, data_key)
    st.success("Recommender is ready. Go to the Game Recommender page to test it.")
else:
    st.info("No recommender is built on page load. Adjust the settings, then click Build / Update Recommender when you are ready.")

st.subheader("How The Recommendation Score Is Calculated")
st.markdown(
    """
    The recommender tries to answer one simple question: **which games feel closest to the game the user already likes?**

    **1. First, the app reads the game's content.**  
    It combines the selected fields, such as genres, tags, categories, and short description, into one text profile for each game. For example, a game might be represented by ideas like `Action`, `Co-op`, `MOBA`, `Fantasy`, and words from its description.

    **2. Then, it compares text similarity.**  
    The app uses TF-IDF to notice which words are useful for describing a game. Common words matter less, while more meaningful game-specific words matter more. After that, cosine similarity compares the selected game with other games. Higher similarity means the games use more similar content signals.

    **3. The app also checks direct overlap.**  
    Text similarity is useful, but genre and tag overlap are easier to trust. So the score also checks whether two games share the same genres, Steam tags, and categories. This helps prevent a popular but unrelated game from ranking too high.

    **4. Content is the main score.**  
    The content score combines both signals:

    - most of it comes from TF-IDF text similarity
    - a smaller part comes from shared genres, tags, and categories

    **5. Rating, popularity, and recency are added as support signals.**  
    These do not replace similarity. They only help sort games that are already reasonably relevant:

    - **Rating** rewards games with stronger positive review percentages.
    - **Popularity** uses review count, estimated owners, and peak players, but it is softened so huge games do not dominate everything.
    - **Recency** gives a small boost to newer games.

    **6. Quality and platform compatibility make small final adjustments.**  
    The app gives a small extra boost to games that are both well-rated and reviewed enough to feel reliable. It also gives a small boost when the recommended game supports similar platforms to the selected game.

    **7. The final score is shown as a match percentage.**  
    A higher match score means the game is similar in content and also has decent supporting signals. The score is not a perfect probability; it is a ranking score used to order the recommendations.
    """
)

with st.expander("Technical formula details"):
    st.markdown(
        """
        This is the complete scoring flow used by the recommender.

        **1. Content similarity**  
        The selected game and each candidate game are converted into TF-IDF vectors. Then cosine similarity compares the vectors:

        ```text
        content_similarity = cosine_similarity(selected_game_vector, candidate_game_vector)
        ```

        This produces a value from 0 to 1. Higher means the two games have more similar text features.

        **2. Genre/tag/category overlap**  
        The app also checks direct shared metadata. Genre overlap is weighted more heavily because genre is usually a stronger signal than tags/categories:

        ```text
        genre_overlap = shared_genres / selected_game_genres
        tag_overlap   = shared_tags_and_categories / selected_game_tags_and_categories

        overlap_score = 0.60 * genre_overlap + 0.40 * tag_overlap
        ```

        This helps keep recommendations close to the selected game's actual genre and tag profile.

        **3. Combined content signal**  
        The app combines text similarity with direct metadata overlap:

        ```text
        content_signal = 0.80 * content_similarity + 0.20 * overlap_score
        ```

        Most of the content score comes from TF-IDF similarity, with overlap added as a stabilizer.

        **4. Popularity score**  
        Popularity uses review count, estimated owners, and peak concurrent players. Each value is already scaled between 0 and 1, then softened with square root so very large games do not dominate too strongly:

        ```text
        dampened_value = sqrt(scaled_value)

        popularity_score =
            0.45 * sqrt(total_reviews_scaled)
          + 0.35 * sqrt(owner_midpoint_scaled)
          + 0.20 * sqrt(peak_ccu_scaled)
        ```

        **5. Quality score**  
        Quality is a small supporting adjustment based on rating and review count:

        ```text
        quality_score =
            0.70 * rating_percent_scaled
          + 0.30 * sqrt(total_reviews_scaled)
        ```

        This favors games that are both well-rated and backed by enough reviews.

        **6. Platform score**  
        Platform score checks how many of the selected game's platforms are also supported by the candidate game:

        ```text
        platform_score = shared_platforms / selected_game_platforms
        ```

        For example, if the selected game supports Windows and Linux, and the candidate supports both, the platform score is 1. If it only supports one of them, the score is 0.5.

        **7. Weighted score from user settings**  
        The sliders on this page control how much each main signal matters. If the slider values do not add up to 1, the app normalizes them automatically:

        ```text
        normalized_weight = slider_value / sum_of_all_slider_values
        ```

        Then the main score is calculated:

        ```text
        weighted_score =
            content_weight    * content_signal
          + rating_weight     * rating_percent_scaled
          + popularity_weight * popularity_score
          + recency_weight    * release_year_scaled
        ```

        **8. Final adjusted score**  
        The app gives most of the final result to the weighted score, then adds small quality and platform adjustments:

        ```text
        adjusted_score =
            0.90 * weighted_score
          + 0.06 * quality_score
          + 0.04 * platform_score
        ```

        **9. Match score shown to users**  
        The app converts the final value into a percentage and keeps it between 0 and 100:

        ```text
        match_score = clip(adjusted_score * 100, 0, 100)
        ```

        In short: content similarity is the main driver, rating/popularity/recency help sort relevant games, and quality/platform compatibility make small final corrections.
        """
    )

