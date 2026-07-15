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

This runs [src/reco_nova/preprocess.py](/Users/chikire/mds/Reco-Nova/src/reco_nova/preprocess.py) and does the following:
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

## Content Model Files

- `src/reco_nova/content_model.py`: Content-based recommendation engine (TF-IDF and embedding retrieval), including name-to-item resolution helpers.
- `notebooks/content_model_notebook.ipynb`: End-to-end experimentation notebook for ID-based and name-based content retrieval with ranked result tables.
- `tests/test_content_model.py`: Unit tests for content-model behavior.

Run content-model tests:

```bash
pytest tests/test_content_model.py
```

