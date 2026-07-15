# Reco-Nova Final Offline Evaluation

This report evaluates frozen model choices once on the untouched test split. 
Hybrid weighting was selected on validation data before this evaluation.

## Evaluation setup

- Training interactions: 500,000
- Eligible training users: 90,407
- Eligible catalog items: 29,761
- Warm test users evaluated: 1,000
- Ranking cutoff: K=12
- Frozen collaborative weight: 0.75
- Bootstrap samples: 1000

## Results

| Model | NDCG@K | MAP@K | Hit Rate@K | Catalog Coverage@K |
|---|---:|---:|---:|---:|
| popularity | 0.010933 | 0.005514 | 0.046000 | 0.000538 |
| collaborative_svd | 0.021874 | 0.014120 | 0.067000 | 0.015860 |
| content_tfidf | 0.021955 | 0.014953 | 0.056000 | 0.244111 |
| hybrid_frozen | 0.024083 | 0.015978 | 0.067000 | 0.100702 |

## Interpretation

Metrics are macro-averaged across users. The JSON artifact contains 95% user-bootstrap confidence intervals for NDCG, MAP, and Hit Rate. This benchmark covers warm users and training-catalog products; cold-start performance is evaluated separately under issue #11.

## Reproduce

```bash
make evaluate-final
```

## Databricks MLflow

- Experiment: `/Shared/reco-nova-final-evaluation`
- Experiment ID: `2689369543776540`
- Run ID: `092123a4b4374546813b6391c67f98fb`
