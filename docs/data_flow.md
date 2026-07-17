# Data Flow: Ingestion to Ranking

## 1. Raw Data Sources

| File | Rows (approx.) | Key Columns |
|---|---|---|
| `transactions_train.csv` | ~31M | customer_id, article_id, t_dat, price, sales_channel_id |
| `articles.csv` | ~105K | article_id, prod_name, product_type_name, product_group_name, colour_group_name, detail_desc |
| `customers.csv` | ~1.4M | customer_id, FN, Active, club_member_status, fashion_news_frequency, age |
| `images/` | ~105K | JPEG product photos, keyed by article_id |

---

## 2. Preprocessing (`preprocess.py`)

```mermaid
flowchart LR
    CSV["transactions_train.csv\narticles.csv\ncustomers.csv"]
    CLEAN["Clean\n• Normalise text\n• Parse timestamps\n• Drop nulls / negatives\n• Impute missing age (median)\n• Resolve image paths"]
    SPLIT["Temporal split\nmost-recent 6-month window:\n• Train  – months 1-5 + wk 1-2 of m6\n• Val    – week 3 of month 6\n• Test   – week 4 of month 6"]
    MAP["Build ID maps\n(train-only, prevents leakage)\ncustomer_map, item_map"]
    TEXT["Build item_text\nconcat all categorical fields\n→ single string per article"]
    OUT["data/processed/\ninteractions_{train,val,test}.parquet\ninteractions_clean.parquet\nitems_clean.parquet\ncustomers_clean.parquet\ncustomer_map.parquet\nitem_map.parquet\npreprocess_summary.json"]
    CSV --> CLEAN --> SPLIT --> MAP --> TEXT --> OUT
```

**Scale after preprocessing:**  
~7.8M train interactions · ~930K val · ~250K test  
49,882 articles with price data in `artifacts/prices.json`

---

## 3. Baseline Training (`train.py`)

```
interactions_train.parquet
    │
    ├─► PopularityRecommender.fit()
    │       count interactions per article, rank by frequency
    │       → artifacts/popularity.joblib
    │
    └─► CollaborativeSVD.fit()
            build sparse user-item matrix (log-scale confidence)
            TruncatedSVD(n_components=64)
            → artifacts/collaborative_svd.joblib
            → artifacts/baseline_metrics.json
```

---

## 4. Hybrid Training (`train_hybrid.py`)

```
interactions_train.parquet + items_clean.parquet
    │
    ├─► ContentRecommender.fit()
    │       TF-IDF(ngram_range=(1,2)) on item_text
    │       TruncatedSVD(n_components=64) compression
    │       user profile = weighted centroid of seen item factors
    │       → artifacts/hybrid/content_tfidf.joblib
    │
    ├─► Retrain PopularityRecommender + CollaborativeSVD
    │       → artifacts/hybrid/{popularity,collaborative_svd}.joblib
    │
    └─► Weight tuning (grid-search: 0.25, 0.50, 0.75)
            evaluate each weight on validation users by NDCG@K
            → artifacts/hybrid/best_hybrid_config.json
            → artifacts/hybrid/hybrid_metrics.json
```

---

## 5. Final Evaluation (`evaluate_final.py`)

```
interactions_train.parquet + interactions_val.parquet (combined)
    │
    ├─► Re-train all models on train+val (frozen configuration)
    │       → artifacts/final/{collaborative_svd,content_tfidf,popularity}.joblib
    │
    └─► Evaluate on warm-start test users
            metrics: NDCG@K, MAP@K, Hit Rate@K, Catalog Coverage@K
            95% bootstrap CIs (1,000 resamples)
            → artifacts/final/final_evaluation.json
```

---

## 6. Cold-Start Evaluation (`evaluate_cold_start.py`)

```
interactions_train.parquet + interactions_val.parquet (combined)
    │
    ├─► Train ContentRecommender + ColdStartRecommender
    │       segment_counts keyed by (age_band, club_member_status)
    │       → artifacts/cold_start/cold_start.joblib
    │
    └─► Evaluate on NEW test users (absent from train+val)
            strategies: global_popularity vs demographic_popularity
            → artifacts/cold_start/cold_start_metrics.json
```

---

## 7. Serving / Online Ranking Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI (api.py)
    participant SVC as RecommendationService
    participant HYB as HybridRecommender
    participant CS as ColdStartRecommender
    participant META as metadata dict

    Client->>API: POST /recommend {user_id, age, session_items, max_budget}
    API->>SVC: check known_users set
    alt user_id in training set
        SVC->>HYB: recommend(user_id, limit)
        HYB->>HYB: CF candidates (10×limit) + Content candidates (10×limit)
        HYB->>HYB: min-max normalise scores
        HYB->>HYB: blend: cf_w * cf_score + (1-cf_w) * content_score
        HYB->>HYB: exclude seen items, sort desc
        HYB-->>SVC: list[HybridScore]
    else unknown user
        SVC->>CS: recommend(age, membership, session_items, product_group)
        CS->>CS: session → demographic → category → global fallback
        CS-->>SVC: ColdStartResult
    end
    SVC->>SVC: filter by max_budget (prices dict)
    SVC->>META: enrich: prod_name, product_group, colour, description, image_path
    SVC->>SVC: build signals + evidence_article_ids (content cosine similarity)
    SVC-->>API: RecommendationResponse
    API-->>Client: JSON
```

---

## 8. Price Catalog Population

```
On first startup (if artifacts/prices.json absent):
    data/processed/interactions_train.parquet
        → group by article_id
        → median(price) × 1000   # normalised raw values → approx USD
        → round to integer
        → write artifacts/prices.json  (822 KB, 49,882 articles)

On subsequent startups:
    artifacts/prices.json  →  loaded directly (skip parquet read)
```

The ×1000 factor converts the normalised price column (range ~0.004–0.7)
to a USD-like scale. It is a proxy only; the actual H&M catalogue prices
are not published.

---

## 9. Artifact Inventory

| Path | Contents | Produced by |
|---|---|---|
| `artifacts/popularity.joblib` | PopularityRecommender | train.py |
| `artifacts/collaborative_svd.joblib` | CollaborativeSVD | train.py |
| `artifacts/baseline_metrics.json` | NDCG/MAP/HR on val | train.py |
| `artifacts/hybrid/` | All three models + weight config + metrics | train_hybrid.py |
| `artifacts/final/` | Re-trained models + final_evaluation.json | evaluate_final.py |
| `artifacts/cold_start/` | ColdStartRecommender + metrics | evaluate_cold_start.py |
| `artifacts/policy_impact/` | policy_impact_report.json | evaluate_policy_impact.py |
| `artifacts/prices.json` | Median price per article (USD-like) | api.py (lazy) |
