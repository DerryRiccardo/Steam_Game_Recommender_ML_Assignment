# Data Folder

This project intentionally uses a deployment-friendly review-filtered sample:

```text
data/games_march2025_1000_reviews.csv
```

The sample is generated from the Kaggle **Steam Games Dataset 2025** and keeps games with more than 1,000 reviews. Metadata about the sample is stored in:

```text
data/dataset_sample_meta.json
```

The full Kaggle CSV can stay on your local machine for regeneration or experimentation:

```text
data/games_march2025_cleaned.csv
```

The app does not use the full CSV by default because loading and preprocessing it makes Streamlit significantly slower. `src/config.py` points to the sampled CSV.

The app may create local processed caches such as:

```text
data/base_clean_games_v1.parquet
data/processed_games_v2.parquet
```

The app may also create matching `.meta.json` files to validate that caches still match the current source data and preprocessing settings. Generated caches and local path files should stay uncommitted.
