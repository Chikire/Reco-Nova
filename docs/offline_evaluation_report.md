# Reco-Nova Offline Evaluation Update

This report captures the latest validation-set outputs shared from:

- `make train-baseline`
- `make train-hybrid`
- `make train-hybrid-fresh`

All runs used full training data (`max_train_rows: 0`) and evaluated 1,000 warm validation users at `K=12`.

## 1) Baseline Validation (Popularity + Collaborative SVD)

### Configuration

- max_train_rows: 0
- max_eval_users: 1000
- n_components: 64
- k: 12
- random_state: 42
- recency_half_life_days: null

### Data

- training_rows: 7,587,803
- training_users: 719,245
- training_items: 49,882
- warm_validation_users: 1,000

### Metrics

| Model | NDCG@K | MAP@K | Hit Rate@K |
|---|---:|---:|---:|
| popularity | 0.002293 | 0.000718 | 0.017000 |
| collaborative_svd | 0.005399 | 0.003121 | 0.022000 |

Evaluation scope: Warm-start validation users and items only; duplicate holdout purchases are one relevant item and items previously purchased by that user are excluded.

### Interpretation

- Collaborative SVD outperforms popularity on all ranking metrics (NDCG, MAP, Hit Rate), indicating that user-item interaction structure provides useful personalization beyond global trends.
- Absolute values remain low, which is expected for sparse implicit-feedback fashion data with strict warm-start filtering and seen-item exclusion.
- This section establishes the minimum bar for model quality: any hybrid strategy should beat collaborative SVD or justify trade-offs through broader catalog exposure.

## 2) Hybrid Validation (No Fresh-Item Exposure)

### Configuration

- max_train_rows: 0
- max_eval_users: 1000
- n_components: 64
- max_text_features: 20000
- k: 12
- random_state: 42
- hybrid_weights: 0.25,0.5,0.75
- best_collaborative_weight: 0.75
- include_fresh_catalog_items: false
- min_fresh_in_top_k: 0
- recency_half_life_days: null

### Data

- training_rows: 7,587,803
- training_users: 719,245
- training_items: 49,882
- catalog_items: 49,882
- fresh_catalog_items: 55,660
- warm_validation_users: 1,000

### Metrics

| Model | NDCG@K | MAP@K | Hit Rate@K |
|---|---:|---:|---:|
| popularity | 0.003144 | 0.001226 | 0.017000 |
| collaborative_svd | 0.006605 | 0.003371 | 0.027000 |
| content_tfidf | 0.001749 | 0.000863 | 0.007000 |
| hybrid_best | 0.006555 | 0.003428 | 0.026000 |

### Hybrid Weight Tuning

| Weight (CF) | NDCG@K | MAP@K | Hit Rate@K |
|---|---:|---:|---:|
| 0.25 | 0.002326 | 0.001158 | 0.010000 |
| 0.50 | 0.005028 | 0.002735 | 0.019000 |
| 0.75 | 0.006555 | 0.003428 | 0.026000 |

Evaluation scope: Seeded random sample of warm-start validation users and items; previously purchased products are excluded.

### Interpretation

- The tuned hybrid (`cf=0.75`) is close to collaborative SVD and slightly better on MAP, but it does not materially improve Hit Rate.
- Pure content performance is much weaker than collaborative in this setup, which suggests text metadata alone is not sufficient for strong warm-user ranking on this split.
- The weight sweep shows a monotonic gain as collaborative weight increases (`0.25 -> 0.75`), meaning collaborative signal is currently the primary driver of relevance, while content acts as a light secondary signal.

## 3) Hybrid Validation (Fresh-Item Exposure Enabled)

### Configuration

- max_train_rows: 0
- max_eval_users: 1000
- n_components: 64
- max_text_features: 20000
- k: 12
- random_state: 42
- hybrid_weights: 0.25,0.5,0.75
- best_collaborative_weight: 0.75
- include_fresh_catalog_items: true
- min_fresh_in_top_k: 1
- recency_half_life_days: null

### Data

- training_rows: 7,587,803
- training_users: 719,245
- training_items: 49,882
- catalog_items: 105,542
- fresh_catalog_items: 55,660
- warm_validation_users: 1,000

### Metrics

| Model | NDCG@K | MAP@K | Hit Rate@K | Fresh Catalog Coverage@K | Fresh Share@K | Users with Fresh Hit@K |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.003144 | 0.001226 | 0.017000 | 0.000000 | 0.000000 | 0.000000 |
| collaborative_svd | 0.006605 | 0.003371 | 0.027000 | 0.000000 | 0.000000 | 0.000000 |
| content_tfidf | 0.001825 | 0.000895 | 0.008000 | 0.042688 | 0.320000 | 0.928000 |
| hybrid_best | 0.005796 | 0.002976 | 0.024000 | 0.019421 | 0.135167 | 1.000000 |

### Hybrid Weight Tuning (Fresh Metrics Included)

| Weight (CF) | NDCG@K | MAP@K | Hit Rate@K | Fresh Catalog Coverage@K | Fresh Share@K | Users with Fresh Hit@K |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.001566 | 0.000731 | 0.008000 | 0.043011 | 0.324833 | 1.000000 |
| 0.50 | 0.004012 | 0.002144 | 0.016000 | 0.035429 | 0.256833 | 1.000000 |
| 0.75 | 0.005796 | 0.002976 | 0.024000 | 0.019421 | 0.135167 | 1.000000 |

Evaluation scope: Seeded random sample of warm-start validation users and items; previously purchased products are excluded.

### Interpretation

- Enabling fresh-item exposure introduces a clear relevance-vs-discovery trade-off: ranking quality drops versus the non-fresh hybrid (`NDCG 0.006555 -> 0.005796`, `MAP 0.003428 -> 0.002976`, `Hit Rate 0.026 -> 0.024`).
- In return, the model guarantees fresh-item presence for all evaluated users (`users_with_fresh_hit_at_k = 1.0`) and allocates meaningful recommendation share to new catalog products (`fresh_share_at_k ~ 0.135` for the selected weight).
- Higher collaborative weights still improve core relevance metrics, while lower collaborative weights increase fresh exposure, giving a controllable operating point depending on business priority.

### Tracking

- run_id: a692236179444ec39487799cf6044e39
- experiment_id: 4313269454646507
- tracking_uri: databricks
- experiment_name: /Shared/reco-nova-baselines

## Overall Interpretation

- For warm-user accuracy, collaborative signal remains the strongest contributor in this pipeline.
- Hybrid blending helps modestly when collaborative dominates, but text-only content has limited standalone quality on this slice.
- Fresh-item onboarding works as designed and provides explicit exposure guarantees, with an expected and measurable impact on ranking relevance.
- Recommended practical default: keep `cf=0.75` for balanced performance, then tune `min_fresh_in_top_k` and/or weight only when business goals require more aggressive new-item discovery.
