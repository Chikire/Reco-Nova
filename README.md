# Reco-Nova

Personalized Product Recommendation Engine for Retail Use Case #05.

## Collaborators

- Chikire Aku-Ibe
- Ssemakula Peter Wasswa

## Project Structure

```text
Reco-Nova/
├── environment.yml
├── docs/
│   └── architecture.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── content_model_notebook.ipynb
├── scripts/
│   └── download_data.sh
├── src/
│   └── reco_nova/
│       ├── api.py
│       ├── app.py
│       ├── content_model.py
│       ├── evaluation.py
│       └── recommender.py
└── tests/
	└── test_content_model.py
```

## Starter Stack

- Python data stack with `pandas`, `scikit-learn`, `scipy`, `mlflow`, and `faiss-cpu`
- Recommendation modeling with `surprise`, `lightfm`, and `sentence-transformers`
- Serving options for `FastAPI` and `Streamlit`

## Quick Start

```bash
conda env create -f environment.yml
conda activate reco-nova
```

## One-Command Run Targets

```bash
make download-data
make preprocess
make train-baseline
make train-hybrid
make train-hybrid-fresh
make evaluate-final
make evaluate-final-fresh
make test
make remove-zip
```

## Kaggle Download Setup

1. Install the Kaggle CLI in your active environment.

```bash
pip install kaggle
```

2. Configure authentication with environment variable. We can create a kaggle token from the settings on your account. 

```bash
export KAGGLE_API_TOKEN="YOUR_KAGGLE_API_KEY" 
```

3. Accept competition rules on Kaggle before downloading:

https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

4. Download and extract official H&M competition files into `data/raw`:

Project command (download + extract) wired in this repo:

```bash
make download-data
```

Direct Kaggle command (download only):

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw
```


5. Manual browser download (no CLI required):

- Open the competition data page:
	https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
- Download the data zip file from the browser.
- Move the zip into `data/raw/`.
- Extract it in place.



macOS/Linux:

```bash
mkdir -p data/raw
unzip data/raw/h-and-m-personalized-fashion-recommendations.zip -d data/raw
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force data/raw
Expand-Archive -Path data/raw/h-and-m-personalized-fashion-recommendations.zip -DestinationPath data/raw -Force
```

- Confirm these files exist in `data/raw/`:
	- `articles.csv`
	- `customers.csv`
	- `transactions_train.csv`
	- `images/`

## Next Steps

1. Add a dataset loader under `src/reco_nova/`.
2. Implement collaborative filtering and content-based feature pipelines.
3. Wire the hybrid ranker into the API and UI entry points.

## Data Pipeline (H&M Dataset)

Data source: Kaggle competition `h-and-m-personalized-fashion-recommendations`.

Place Kaggle files in `data/raw/`:
- `transactions_train.csv`
- `articles.csv`
- `customers.csv`
- `images/` (optional but recommended for multimodal embeddings)

Run preprocessing:

```bash
make preprocess
```

This runs `src/reco_nova/preprocess.py` and does the following:
- Validates required columns in `transactions_train.csv`, `articles.csv`, and `customers.csv`
- Normalizes text fields and parses transaction dates
- Cleans customer metadata, including median imputation for missing `age`
- Builds `item_text` from product/category text columns and `image_path` when matching images exist
- Splits interactions from the most recent 6-month window into:
- `train`: months 1-5 plus weeks 1-2 of month 6
- `val`: week 3 of month 6
- `test`: week 4 of month 6
- Builds `user_idx` and `item_idx` mappings from the training split only to avoid leakage

Generated outputs in `data/processed/`:
- `interactions_train.parquet`
- `interactions_val.parquet`
- `interactions_test.parquet`
- `interactions_clean.parquet`
- `items_clean.parquet`
- `customers_clean.parquet`
- `customer_map.parquet`
- `item_map.parquet`
- `preprocess_summary.json`

## Collaborative Baseline and Offline Evaluation

After preprocessing, train the popularity and collaborative-filtering models:

```bash
make train-baseline
```

The baseline uses randomized truncated SVD over a sparse implicit-feedback
matrix. Repeated purchases receive logarithmically scaled confidence, and
products already seen during training are removed from recommendations. A
global popularity model provides both a comparison baseline and an unknown-user
fallback.

For laptop-friendly iteration, the command defaults to the most recent 500,000
training rows and 1,000 warm validation users. Run the module directly to
change these limits; a value of `0` disables the corresponding cap:

```bash
PYTHONPATH=src python -m reco_nova.train \
  --max-train-rows 1000000 \
  --max-eval-users 5000 \
  --n-components 64 \
  --k 12
```

Generated local artifacts (ignored by Git):

- `artifacts/popularity.joblib`
- `artifacts/collaborative_svd.joblib`
- `artifacts/baseline_metrics.json`

The report compares NDCG@K, MAP@K, and Hit Rate@K. It evaluates only warm
validation users and catalog items because new-user and new-item performance is
reported separately by the cold-start milestone. Duplicate holdout purchases
count as one relevant product, and products previously purchased by that user
are excluded from both recommendations and relevance labels.

Run all automated tests with:

```bash
make test
```

## Databricks MLflow Tracking

Baseline experiments can be recorded in the Databricks-hosted MLflow tracking
server. Databricks Free Edition users can authenticate through browser-based
OAuth and do not need a personal access token.

First install or update the current Databricks CLI. Then copy the workspace URL
from your browser (only the scheme and hostname, without a notebook path) and
create a named OAuth profile:

```bash
databricks auth login --host "https://your-workspace.cloud.databricks.com" --profile RECO_NOVA
```

The command opens a browser for interactive sign-in. On macOS, the short-lived
OAuth credential is stored in Keychain rather than in this repository. Verify
the profile and configure MLflow to use it:

```bash
databricks current-user me --profile RECO_NOVA
export DATABRICKS_CONFIG_PROFILE="RECO_NOVA"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_NAME="/Shared/reco-nova-baselines"
```

Ensure the declared MLflow and Databricks dependencies are installed, then run:

```bash
conda env update -f environment.yml
make train-baseline-databricks
```

If the environment was created before MLflow 3 was declared, remove the stale
conda package before updating:

```bash
conda remove -n reco-nova mlflow databricks-cli -y
conda env update -n reco-nova -f environment.yml
```

Each run records:

- Training limits, SVD dimensions, K, and random seed.
- Training row, user, item, and validation-user counts.
- NDCG@K, MAP@K, and Hit Rate@K for popularity and collaborative SVD.
- The JSON evaluation report and both serialized model artifacts.
- Project, task, and evaluation-scope tags.

The Databricks run and experiment IDs are also added to the local
`artifacts/baseline_metrics.json` report. OAuth is preferred, but if another
workspace uses token authentication, never place its token in the Makefile,
repository, notebooks, or committed `.env` files.

## Content-Based and Hybrid Recommender

Issue #10 combines collaborative and product-metadata signals. The content
model creates TF-IDF word/bigram features from `item_text`, compresses them with
randomized SVD, and represents each user by a weighted centroid of previously
purchased products. The hybrid ranker normalizes collaborative and content
scores per user before blending them.

Train and tune locally:

```bash
make train-hybrid
```

Track the comparison and weight sweep in Databricks MLflow:

```bash
export DATABRICKS_CONFIG_PROFILE="RECO_NOVA"
export MLFLOW_TRACKING_URI="databricks"
make train-hybrid-databricks
```

Defaults compare collaborative weights `0.25`, `0.50`, and `0.75` on a seeded
random sample of 1,000 warm validation users. The run reports popularity,
collaborative SVD, content TF-IDF, and best-hybrid metrics. Customize a run with:

```bash
PYTHONPATH=src python -m reco_nova.train_hybrid \
  --max-train-rows 1000000 \
  --max-eval-users 5000 \
  --n-components 64 \
  --hybrid-weights 0.25,0.5,0.75 \
  --k 12
```

Generated artifacts:

- `artifacts/hybrid/content_tfidf.joblib`
- `artifacts/hybrid/collaborative_svd.joblib`
- `artifacts/hybrid/popularity.joblib`
- `artifacts/hybrid/best_hybrid_config.json`
- `artifacts/hybrid/hybrid_metrics.json`

Enable fresh-item exposure (metadata-only onboarding for unseen products):

```bash
make train-hybrid-fresh
```

Track the same run in Databricks MLflow:

```bash
make train-hybrid-fresh-databricks
```

Direct module command with explicit flags:

```bash
PYTHONPATH=src python -m reco_nova.train_hybrid \
  --max-train-rows 1000000 \
  --max-eval-users 5000 \
  --n-components 64 \
  --hybrid-weights 0.25,0.5,0.75 \
  --k 12 \
  --include-fresh-catalog-items \
  --min-fresh-in-top-k 1
```

When enabled, the report also includes fresh-catalog exposure metrics:

- `fresh_catalog_coverage_at_k`
- `fresh_share_at_k`
- `users_with_fresh_hit_at_k`

For a fair warm-start comparison, all four approaches rank only products seen
in the training catalog. New-item retrieval will be measured separately in the
cold-start and multimodal evaluations.

## Final Held-Out Evaluation

After selecting model settings on validation data, run the frozen comparison
once on `interactions_test.parquet`:

```bash
make evaluate-final
```

This retrains the final models on the development data (train plus validation),
keeps the selected `0.75` collaborative hybrid weight fixed, and reports:

- NDCG@K, MAP@K, and Hit Rate@K.
- Catalog Coverage@K.
- 95% user-bootstrap confidence intervals.
- Popularity, collaborative SVD, content TF-IDF, and hybrid results.

The permanent table is written to `docs/offline_evaluation_report.md`; detailed
intervals and configuration are saved to
`artifacts/final/final_evaluation.json`. Track the final benchmark with:

```bash
make evaluate-final-databricks
```

Evaluate final models with fresh-item exposure enabled:

```bash
make evaluate-final-fresh
```

Track fresh-item final evaluation in Databricks:

```bash
make evaluate-final-fresh-databricks
```

Direct module command with explicit flags:

```bash
PYTHONPATH=src python -m reco_nova.evaluate_final \
  --k 12 \
  --collaborative-weight 0.75 \
  --include-fresh-catalog-items \
  --min-fresh-in-top-k 1
```

## New-User Cold Start

Unknown users follow an explainable fallback hierarchy: recent session items,
age-band and membership popularity, preferred product group, then global
popularity. Evaluate genuinely unseen test users with:

```bash
make evaluate-cold-start
make evaluate-cold-start-databricks
```

The command compares demographic and no-context fallbacks using NDCG, MAP, Hit
Rate, catalog coverage, and bootstrap confidence intervals. It writes a
permanent report with examples to `docs/cold_start_report.md` and detailed
results to `artifacts/cold_start/cold_start_metrics.json`.
