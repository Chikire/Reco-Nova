# Architecture

```mermaid
flowchart LR
    A[User interactions] --> B[Interaction matrix]
    C[Product metadata] --> D[Content features]
    B --> E[Collaborative filter]
    D --> F[Content-based retrieval]
    E --> G[Hybrid ranker]
    F --> G
    G --> H[Recommended for You feed]
    G --> I[Explainability text]
    H --> J[API / UI]
    I --> J
```

The first version of the project should keep the pipeline simple:
- ingest user-item events and product metadata
- train collaborative and content-based models separately
- merge scores in a hybrid ranker
- expose ranked results through FastAPI or Streamlit
