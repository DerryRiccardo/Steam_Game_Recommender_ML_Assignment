# Steam Game Recommender

A Streamlit game recommender system using a review-filtered sample of a recent Steam games dataset.

The app recommends games with a content-based approach using genres, Steam tags, categories, descriptions, and optional metadata such as developer and publisher. It also adds a match score using rating, popularity, and release recency signals.

## Pages

1. **Home**  
   Project overview, dataset summary, group members, workflow, and recommendation method.

2. **Exploratory Data Analysis**  
   Explores the review-filtered raw sample, data quality, feature readiness, distributions, and any selected column.

3. **Preprocessing**  
   Shows cleaning options, prepared data, parsed fields, review features, and MinMaxScaler outputs.

4. **Build Recommender**  
   Lets users choose the features and weights used by the recommender.

5. **Game Recommender**  
   Lets users select a game, apply filters, and receive recommendations with match scores and explanations.

## Dataset

Source dataset:

[Steam Games Dataset 2025 by Artemiy Ermilov](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

The original dataset has 89,618 games. The full CSV is too slow and heavy for a Streamlit deployment workflow, so this project intentionally uses a smaller review-filtered sample:

```text
data/games_march2025_1000_reviews.csv
```

The sample contains 6,295 games with more than 1,000 reviews. This keeps the app responsive while preserving enough popular, review-rich games for meaningful EDA and recommendations.

The full Kaggle file can stay local for regeneration or experimentation:

```text
data/games_march2025_cleaned.csv
```

The app does not use the full CSV by default. `src/config.py` points to the sampled CSV.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## How Recommendations Work

The app:

1. Loads the review-filtered Steam sample.
2. Cleans and prepares the dataset.
3. Combines selected metadata fields into a text feature.
4. Converts game text into TF-IDF vectors.
5. Computes cosine similarity for the selected game.
6. Adds genre/tag/category overlap so recommendations stay closer to the selected game's actual taste profile.
7. Blends content similarity with normalized numeric signals:
   - rating score
   - popularity score
   - recency score

The final score is shown as a match score:

```text
match_score =
    content_similarity * content_weight
  + rating_score       * rating_weight
  + popularity_score   * popularity_weight
  + recency_score      * recency_weight
```

Popularity is dampened so very famous but unrelated games do not overpower closer niche matches. Recommendation cards also show shared reasons and a compact score breakdown.

Processed parquet caches are paired with metadata files so stale caches are rebuilt when the source data or preprocessing settings change.

## Preprocessing

Before the recommender is built, the raw sample is cleaned and transformed so it is ready for modeling. The main preprocessing steps are:

- **Remove duplicates**  
  Duplicate rows are removed based on repeated `appid` and repeated game `name`, so one game does not appear multiple times and bias the recommendation results.

- **Remove mature content**  
  Games that indicate adult or sexual content are filtered out using keywords from titles, descriptions, genres, categories, and tags.

- **Parse list-like features**  
  Columns such as `genres`, `tags`, `categories`, `developers`, and `publishers` are converted from raw text into structured lists that can be reused by the recommender.

- **Create rating/review features**  
  The system builds features such as:
  - `total_reviews`
  - `rating_percent`
  - `owner_midpoint`
  - `is_free`
  - `release_year`

- **Scale numeric features with MinMaxScaler**  
  Numeric features used in scoring are normalized into the `0-1` range so large values like review counts do not dominate smaller-scale features like rating percentages.

Scaled features include:

- `rating_percent_scaled`
- `total_reviews_scaled`
- `owner_midpoint_scaled`
- `peak_ccu_scaled`
- `release_year_scaled`

## Model

The recommender uses a content-based ranking pipeline. The main methods are:

- **TF-IDF for text representation**  
  Selected game metadata such as genres, Steam tags, categories, and short descriptions are combined into one text document per game. These documents are then converted into numerical vectors using TF-IDF.

- **Cosine similarity for content similarity**  
  Once every game is represented as a TF-IDF vector, the recommender compares the selected game with candidate games using cosine similarity. A higher cosine value means the games are more similar in content.

- **Overlap score for genre/tag/category matching**  
  In addition to cosine similarity, the system measures explicit overlap in genres, tags, and categories. This helps keep recommendations closer to the selected game's actual taste profile.

- **Weighted ranking for final recommendation**  
  The final recommendation score combines content similarity, rating score, popularity score, and recency score.

The model also adds small adjustments from:

- `quality_score`
- `platform_score`

This makes the final ranking more realistic, because a game should not only be similar in content, but also reasonably good, relevant, and playable on similar platforms.

### Scoring Summary

```text
content_signal =
    0.80 * cosine_similarity
  + 0.20 * overlap_score
```

```text
final_score =
    content_weight    * content_signal
  + rating_weight     * rating_score
  + popularity_weight * popularity_score
  + recency_weight    * recency_score
```

```text
adjusted_score =
    0.90 * final_score
  + 0.06 * quality_score
  + 0.04 * platform_score
```

```text
match_score = adjusted_score * 100
```

With the default configuration, the recommender gives the largest weight to **content similarity**, then uses rating, popularity, and recency as supporting ranking signals.

## Notes For Collaborators

- Commit the review-filtered sample CSV and `dataset_sample_meta.json` for deployment.
- Do not commit the full Kaggle CSV, generated parquet caches, or local path files.
- Keep new shared logic inside `src/`.
- Keep Streamlit page-specific UI inside `Home.py` or `pages/`.
- If recommendation performance becomes slow, reduce `max_features` in `src/recommender.py`.
- Run checks with `pytest -q`.
