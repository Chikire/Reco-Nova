# Reco-Nova

Personalized Product Recommendation Engine — Hackathon Retail Use Case #05.

> **Team:** Chikire Aku-Ibe · Ssemakula Peter Wasswa

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Project Structure](#2-project-structure)
3. [Prerequisites](#3-prerequisites)
4. [Step-by-Step Setup](#4-step-by-step-setup)
   - [Step 1 — Create the Conda environment](#step-1--create-the-conda-environment)
   - [Step 2 — Configure Kaggle credentials](#step-2--configure-kaggle-credentials)
   - [Step 3 — Download the H&M dataset](#step-3--download-the-hm-dataset)
   - [Step 4 — Preprocess raw data](#step-4--preprocess-raw-data)
   - [Step 5 — Train baseline models](#step-5--train-baseline-models)
   - [Step 6 — Train the hybrid recommender](#step-6--train-the-hybrid-recommender)
   - [Step 7 — Final held-out evaluation](#step-7--final-held-out-evaluation)
   - [Step 8 — Cold-start evaluation](#step-8--cold-start-evaluation)
   - [Step 9 — Policy-impact simulation](#step-9--policy-impact-simulation)
   - [Step 10 — Run the API and UI](#step-10--run-the-api-and-ui)
5. [Databricks MLflow Tracking](#5-databricks-mlflow-tracking)
6. [Conversational Assistant (Ollama)](#6-conversational-assistant-ollama)
7. [Running Tests](#7-running-tests)
8. [Reproduce Everything in One Command](#8-reproduce-everything-in-one-command)
9. [Reference — All Make Targets](#9-reference--all-make-targets)
10. [Reference — Direct Module Flags](#10-reference--direct-module-flags)

---

## 1. What This Project Does

Reco-Nova is a three-layer hybrid recommendation engine trained on the Kaggle
H&M Fashion dataset (7.8 M purchase interactions, 105 K products, 1.4 M customers).

| Layer | Model | Signal |
|---|---|---|
| Collaborative | Truncated SVD (64 components) | User–item purchase matrix |
| Content | TF-IDF bigrams + SVD compression | Product metadata text |
| Hybrid | Score-normalized weighted blend | Best of both, tuned on validation |

Additional capabilities:

- **Cold-start**: 4-level fallback for anonymous users (session → demographics → category → global)
- **Explainability**: per-item `signals`, `evidence_article_ids`, and `reason` text on every response
- **A/B simulation**: position-biased CTR comparison of popularity baseline vs. hybrid
- **GenAI assistant**: conversational shopping via Ollama (regex fallback when Ollama is offline)
- **FastAPI** serving layer + **Streamlit** product-discovery UI

---

## 2. Project Structure

```text
Reco-Nova/
├── Makefile                        ← all run targets
├── environment.yml                 ← conda environment spec
├── requirements.txt
├── docs/
│   ├── architecture.md
│   ├── data_flow.md
│   ├── explainability.md
│   ├── offline_evaluation_report.md
│   ├── cold_start_report.md
│   └── policy_impact_report.md
├── notebooks/
│   ├── EDA_notebook.ipynb
│   └── content_model_notebook.ipynb
├── scripts/
│   ├── download_data.sh
│   └── test_mlflow_connection.py
├── data/
│   ├── raw/                        ← Kaggle CSV + images (git-ignored)
│   └── processed/                  ← parquet outputs (git-ignored)
├── artifacts/                      ← trained models + metrics (git-ignored)
│   ├── hybrid/
│   ├── final/
│   ├── cold_start/
│   └── policy_impact/
├── src/
│   └── reco_nova/
│       ├── api.py                  ← FastAPI app
│       ├── app.py                  ← Streamlit UI
│       ├── assistant.py            ← conversational shopping assistant
│       ├── preprocess.py
│       ├── train.py                ← baseline models
│       ├── train_hybrid.py
│       ├── evaluate_final.py
│       ├── evaluate_cold_start.py
│       ├── evaluate_policy_impact.py
│       ├── evaluate_assistant.py
│       ├── evaluation.py           ← NDCG / MAP / Hit Rate
│       ├── tracking.py             ← MLflow logging
│       └── models/
│           ├── collaborative.py    ← CollaborativeSVD
│           ├── content.py          ← ContentRecommender (TF-IDF)
│           ├── hybrid.py           ← HybridRecommender
│           ├── cold_start.py       ← ColdStartRecommender
│           └── popularity.py       ← PopularityRecommender
└── tests/
```

---

## 3. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| conda | ≥ 23 | Anaconda or Miniconda |
| Python | 3.11 | Pinned in `environment.yml` |
| Kaggle account | — | Free; needed to download dataset |
| Ollama | latest | Only for the GenAI assistant; optional |
| Databricks workspace | — | Only for remote MLflow tracking; optional |

---

## 4. Step-by-Step Setup

### Step 1 — Create the Conda environment

```bash
conda env create -f environment.yml
conda activate reco-nova
```

This installs all Python dependencies: `pandas`, `scikit-learn`, `scipy`,
`faiss-cpu`, `sentence-transformers`, `fastapi`, `uvicorn`, `streamlit`,
`mlflow`, and more.

> **Updating an existing environment**
> ```bash
> conda env update -n reco-nova -f environment.yml --prune
> ```

---

### Step 2 — Configure Kaggle credentials

You need a Kaggle account and a Kaggle API token to download the dataset.

**2a. Get your API token**

1. Log in at [kaggle.com](https://www.kaggle.com)
2. Go to **Account → API → Create New Token**
3. A `kaggle.json` file is downloaded — it contains your username and key

**2b. Set the environment variable** (recommended — keeps secrets out of files)

```bash
export KAGGLE_API_TOKEN="your_api_key_here"
```

Add this to `~/.zshrc` or `~/.bashrc` to make it permanent.

**2c. Accept the competition rules** (one-time, required before download)

Visit: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

Click **Join Competition** and accept the rules.

---

### Step 3 — Download the H&M dataset

```bash
make download-data
```

This runs `scripts/download_data.sh` which downloads and extracts the Kaggle
competition zip into `data/raw/`. After it completes, confirm these files exist:

```
data/raw/
├── articles.csv
├── customers.csv
├── transactions_train.csv
├── sample_submission.csv
└── images/          ← ~105K product JPEG files
```

> **Manual download (no CLI)**
> Download the zip from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data,
> place it in `data/raw/`, then extract:
> ```bash
> unzip data/raw/h-and-m-personalized-fashion-recommendations.zip -d data/raw
> ```

---

### Step 4 — Preprocess raw data

```bash
make preprocess
```

What this does:

- Normalizes text fields and parses transaction timestamps
- Imputes missing customer ages with the training-set median
- Builds `item_text` — a single string per article concatenating all categorical fields
- Resolves `image_path` for articles that have a matching JPEG in `data/raw/images/`
- Performs a **temporal split** on the most-recent 6-month window:
  - `train` — months 1–5 plus weeks 1–2 of month 6
  - `val` — week 3 of month 6
  - `test` — week 4 of month 6 (held out; not touched until final evaluation)
- Builds `customer_map` and `item_map` from the training split only (no leakage)

Outputs written to `data/processed/`:

```
interactions_train.parquet   (~7.8 M rows)
interactions_val.parquet     (~930 K rows)
interactions_test.parquet    (~250 K rows)
interactions_clean.parquet
items_clean.parquet
customers_clean.parquet
customer_map.parquet
item_map.parquet
preprocess_summary.json
```

---

### Step 5 — Train baseline models

```bash
make train-baseline
```

Trains two models on `interactions_train.parquet` and evaluates on `interactions_val.parquet`:

| Model | Algorithm |
|---|---|
| `PopularityRecommender` | Interaction count rank |
| `CollaborativeSVD` | Randomized TruncatedSVD (64 components) on sparse implicit-feedback matrix |

Evaluation uses 1,000 warm validation users by default. Metrics: NDCG@12, MAP@12, Hit Rate@12.

Outputs:

```
artifacts/popularity.joblib
artifacts/collaborative_svd.joblib
artifacts/baseline_metrics.json
```

---

### Step 6 — Train the hybrid recommender

```bash
make train-hybrid
```

Adds a content-based model and tunes the hybrid blend weight:

1. Fits `ContentRecommender` — TF-IDF bigrams (20 K features) → SVD (64 components) on `item_text`
2. Retrains popularity and SVD on the same split
3. Grid-searches `collaborative_weight` over {0.25, 0.50, 0.75} on validation NDCG@12
4. Saves the best weight to `artifacts/hybrid/best_hybrid_config.json`

> **Tip — include fresh (unseen) catalog items:**
> ```bash
> make train-hybrid-fresh
> ```
> This allows the content model to recommend products with no training interactions,
> and adds `fresh_catalog_coverage_at_k`, `fresh_share_at_k`, and
> `users_with_fresh_hit_at_k` to the metrics report.

Outputs:

```
artifacts/hybrid/content_tfidf.joblib
artifacts/hybrid/collaborative_svd.joblib
artifacts/hybrid/popularity.joblib
artifacts/hybrid/best_hybrid_config.json
artifacts/hybrid/hybrid_metrics.json
```

---

### Step 7 — Final held-out evaluation

```bash
make evaluate-final
```

**Run this only once.** It retrains all models on `train + val` (frozen configuration),
then evaluates once on `interactions_test.parquet`. This is the reportable number.

Reports NDCG@12, MAP@12, Hit Rate@12, Catalog Coverage@12, and 95% bootstrap
confidence intervals for all four models (popularity, SVD, content, hybrid).

> **With fresh-item exposure:**
> ```bash
> make evaluate-final-fresh
> ```

Outputs:

```
artifacts/final/collaborative_svd.joblib
artifacts/final/content_tfidf.joblib
artifacts/final/popularity.joblib
artifacts/final/final_evaluation.json
docs/offline_evaluation_report.md       ← human-readable results table
```

---

### Step 8 — Cold-start evaluation

```bash
make evaluate-cold-start
```

Evaluates four fallback strategies on users whose IDs never appeared in train or val:

| Priority | Strategy | Trigger |
|---|---|---|
| 1 | `session_content` | Session article IDs provided |
| 2 | `demographic_popularity` | Age + membership provided, segment ≥ 50 events |
| 3 | `category_popularity` | Preferred product group provided |
| 4 | `global_popularity` | No context at all |

Outputs:

```
artifacts/cold_start/cold_start.joblib
artifacts/cold_start/cold_start_metrics.json
docs/cold_start_report.md
```

---

### Step 9 — Policy-impact simulation

```bash
make evaluate-policy-impact
```

Simulates 200 rounds of user interactions for 1,000 users under two policies
using a position-biased click model, then computes CTR lift:

- **Baseline policy:** global popularity
- **Personalized policy:** hybrid recommender

Outputs:

```
artifacts/policy_impact/policy_impact_report.json
docs/policy_impact_report.md
```

---

### Step 10 — Run the API and UI

Make sure steps 7 and 8 have been completed first (the API loads the `final/`
and `cold_start/` artifacts at startup).

**In terminal 1 — start the API:**

```bash
make run-api
```

The FastAPI server starts on **http://localhost:8000**.  
Interactive API docs: **http://localhost:8000/docs**

**In terminal 2 — start the UI:**

```bash
make run-ui
```

The Streamlit UI starts on **http://localhost:8501**.

**Quick smoke test via curl:**

```bash
# Health check
curl http://localhost:8000/health | python -m json.tool

# Personalized recommendations (known user)
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "00007d2de826758b65a93dd36f3a97869f399168", "limit": 6}' \
  | python -m json.tool

# Cold-start: new user with session context
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"session_article_ids": ["706016001", "759871002"], "limit": 6}' \
  | python -m json.tool

# Cold-start: new user with demographics only
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"age": 27, "club_member_status": "active", "limit": 6}' \
  | python -m json.tool
```

**Streamlit UI modes:**

| Mode | What it shows |
|---|---|
| Personalized | Enter a customer ID → ranked feed with "Recommended because…" explanations |
| Discover | Enter age, membership, product group, or session items → cold-start feed |
| Assistant | Natural-language shopping chat grounded in the live catalog |

---

## 5. Databricks MLflow Tracking

Every pipeline step has a `*-databricks` make target that logs metrics, params,
and model artifacts to a Databricks-hosted MLflow server. **Skip this section
entirely if you only need local runs.**

### 5a — Install the Databricks CLI

```bash
pip install databricks-cli
```

### 5b — Authenticate with your workspace

```bash
databricks auth login \
  --host "https://your-workspace.cloud.databricks.com" \
  --profile RECO_NOVA
```

This opens a browser for OAuth sign-in (Databricks Free Edition uses OAuth;
no personal access token is needed). On macOS the credential is stored in
Keychain, not in this repository.

Verify the profile works:

```bash
databricks current-user me --profile RECO_NOVA
```

### 5c — Set environment variables in your shell

```bash
export DATABRICKS_CONFIG_PROFILE="RECO_NOVA"
export MLFLOW_TRACKING_URI="databricks"
```

Add both lines to `~/.zshrc` or `~/.bashrc` so they persist across sessions.

### 5d — Update the conda environment (if installed before MLflow 3)

```bash
# Remove stale packages first (only if you hit version conflicts)
conda remove -n reco-nova mlflow databricks-cli -y

# Reinstall from spec
conda env update -n reco-nova -f environment.yml --prune
```

### 5e — Run any Databricks-enabled target

Each target below is a drop-in replacement for its local counterpart:

```bash
make train-baseline-databricks           # step 5 with remote tracking
make train-hybrid-databricks             # step 6 with remote tracking
make train-hybrid-fresh-databricks       # step 6 (fresh-item) with remote tracking
make evaluate-final-databricks           # step 7 with remote tracking
make evaluate-final-fresh-databricks     # step 7 (fresh-item) with remote tracking
make evaluate-cold-start-databricks      # step 8 with remote tracking
make evaluate-policy-impact-databricks   # step 9 with remote tracking
```

Each run logs to the experiment path shown in the table below (auto-created if
it does not exist):

| Target | MLflow experiment |
|---|---|
| `train-baseline-databricks` | `/Shared/reco-nova-baselines` |
| `train-hybrid-databricks` | `/Shared/reco-nova-hybrid` |
| `evaluate-final-databricks` | `/Shared/reco-nova-final-evaluation` |
| `evaluate-cold-start-databricks` | `/Shared/reco-nova-cold-start` |
| `evaluate-policy-impact-databricks` | `/Shared/reco-nova-policy-impact` |

Every run records: training configuration, row/user/item counts, NDCG@K /
MAP@K / Hit Rate@K, the JSON report, and serialized model artifacts.

> **Security note:** Never put Databricks tokens in the Makefile, notebooks,
> or committed `.env` files. Always use environment variables or the CLI profile.

---

## 6. Conversational Assistant (Ollama)

The assistant extracts shopping intent from natural language (category, colour,
style, budget, result count) and calls the recommendation engine. It never
invents product IDs — all products come from the live catalog.

**Install Ollama and pull the default model (one-time):**

```bash
brew install ollama
ollama serve &
ollama pull llama3.2:3b
```

**Start the API with assistant support:**

```bash
make run-api
```

**Test the assistant:**

```bash
curl -s -X POST http://localhost:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me casual blue tops under $40", "limit": 6}' \
  | python -m json.tool
```

The assistant falls back to a deterministic regex parser if Ollama is not running.
Override the model with `RECO_NOVA_OLLAMA_MODEL`, or disable LLM calls entirely:

```bash
export RECO_NOVA_LLM_PROVIDER=local
```

Run the reproducible assistant evaluation:

```bash
make evaluate-assistant
```

---


## 7. Running Tests

```bash
make test
```

Runs the full pytest suite covering preprocessing, all model classes, evaluation
metrics, cold-start strategies, hybrid training, API endpoints, the Streamlit
app, tracking integration, and the assistant.

---

## 8. Reproduce Everything in One Command

```bash
make reproduce
```

Runs: `preprocess → train-baseline → train-hybrid → evaluate-final → evaluate-cold-start`
in sequence. Assumes Kaggle data is already in `data/raw/`.

---

## 9. Reference — All Make Targets

```
make download-data                  Download + extract H&M Kaggle files
make remove-zip                     Remove the downloaded zip to free disk space
make preprocess                     Clean raw data and write parquet outputs
make train-baseline                 Train popularity + SVD, evaluate on val
make train-baseline-databricks      Same + log to Databricks MLflow
make train-hybrid                   Train content + hybrid, tune blend weight
make train-hybrid-databricks        Same + log to Databricks MLflow
make train-hybrid-fresh             Hybrid with fresh-catalog item exposure
make train-hybrid-fresh-databricks  Same + log to Databricks MLflow
make evaluate-final                 Frozen retrain on train+val, evaluate on test
make evaluate-final-databricks      Same + log to Databricks MLflow
make evaluate-final-fresh           Final eval with fresh-item exposure
make evaluate-final-fresh-databricks Same + log to Databricks MLflow
make evaluate-cold-start            Evaluate new-user fallback strategies
make evaluate-cold-start-databricks Same + log to Databricks MLflow
make evaluate-policy-impact         Simulate baseline vs hybrid CTR lift
make evaluate-policy-impact-databricks Same + log to Databricks MLflow
make evaluate-assistant             Reproducible intent + guardrail proxy report
make run-api                        Start FastAPI server on :8000
make run-ui                         Start Streamlit UI on :8501
make test                           Run the full pytest suite
make warm-models                    Pre-download SentenceTransformer weights
make reproduce                      Full pipeline in one command

Note: run `make help` for the same list at any time.
```

---

## 10. Reference — Direct Module Flags

Each pipeline step can be run directly for fine-grained control. The `make`
targets are wrappers around these commands.

**Preprocess**

```bash
PYTHONPATH=src python -m reco_nova.preprocess \
  --raw-dir data/raw \
  --processed-dir data/processed
```

**Train baseline**

```bash
PYTHONPATH=src python -m reco_nova.train \
  --processed-dir data/processed \
  --artifacts-dir artifacts \
  --max-train-rows 0 \       # 0 = all rows
  --max-eval-users 1000 \
  --n-components 64 \
  --k 12
```

**Train hybrid**

```bash
PYTHONPATH=src python -m reco_nova.train_hybrid \
  --processed-dir data/processed \
  --artifacts-dir artifacts/hybrid \
  --max-train-rows 0 \
  --max-eval-users 1000 \
  --n-components 64 \
  --hybrid-weights 0.25,0.5,0.75 \
  --k 12 \
  [--include-fresh-catalog-items] \
  [--min-fresh-in-top-k 1]
```

**Final evaluation**

```bash
PYTHONPATH=src python -m reco_nova.evaluate_final \
  --processed-dir data/processed \
  --artifacts-dir artifacts/final \
  --report-path docs/offline_evaluation_report.md \
  --k 12 \
  [--include-fresh-catalog-items] \
  [--min-fresh-in-top-k 1]
```

**Cold-start evaluation**

```bash
PYTHONPATH=src python -m reco_nova.evaluate_cold_start \
  --processed-dir data/processed \
  --artifacts-dir artifacts/cold_start \
  --report-path docs/cold_start_report.md \
  --k 12
```

**Policy-impact simulation**

```bash
PYTHONPATH=src python -m reco_nova.evaluate_policy_impact \
  --processed-dir data/processed \
  --artifacts-dir artifacts/policy_impact \
  --report-path docs/policy_impact_report.md \
  --baseline-policy popularity \
  --max-eval-users 1000 \
  --k 12 \
  --simulation-rounds 200
```
