# Reco-Nova

Personalized Product Recommendation Engine for Retail Use Case #05.

# Collaborators

- Chikire Aku-Ibe
- Ssemukula Peter Wasswa

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
├── src/
│   └── reco_nova/
│       ├── api.py
│       ├── app.py
│       ├── evaluation.py
│       └── recommender.py
└── tests/
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

## Next Steps

1. Add a dataset loader under `src/reco_nova/`.
2. Implement collaborative filtering and content-based feature pipelines.
3. Wire the hybrid ranker into the API and UI entry points.
