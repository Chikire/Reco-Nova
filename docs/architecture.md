# Architecture

## System Overview

Reco-Nova is a personalized product recommendation engine for fashion retail
(H&M dataset). It combines collaborative filtering, content-based retrieval,
and a conversational assistant into a single FastAPI service, exposed through
a Streamlit UI.

## High-Level Component Diagram

```mermaid
flowchart TB
    subgraph Offline["Offline Training Pipeline"]
        RAW["Raw Data\n(transactions, articles, customers)"]
        PRE["preprocess.py\n(clean, split, build item_text)"]
        TRAIN_B["train.py\n(Popularity + Collaborative SVD)"]
        TRAIN_H["train_hybrid.py\n(Content TF-IDF + weight tuning)"]
        EVAL_F["evaluate_final.py\n(frozen re-train on train+val)"]
        EVAL_CS["evaluate_cold_start.py\n(new-user test split)"]
        EVAL_PI["evaluate_policy_impact.py\n(simulated CTR/lift)"]
        ARTS["artifacts/\n(*.joblib, best_hybrid_config.json,\nprices.json)"]
        RAW --> PRE
        PRE --> TRAIN_B
        TRAIN_B --> TRAIN_H
        TRAIN_H --> EVAL_F
        TRAIN_H --> EVAL_CS
        EVAL_F --> EVAL_PI
        TRAIN_H --> ARTS
        EVAL_F --> ARTS
        EVAL_CS --> ARTS
    end

    subgraph Serving["Online Serving Layer"]
        API["FastAPI\n(api.py)\nPOST /recommend\nPOST /explain\nPOST /assistant/chat\nGET /health"]
        ASS["ShoppingAssistant\n(assistant.py)\nlocal regex | Ollama LLM"]
        SVC["RecommendationService\n(background-loaded)\nHybridRecommender\nColdStartRecommender\nmetadata + prices"]
        UI["Streamlit UI\n(app.py)\nPersonalized | Discover\nConversational assistant"]
    end

    ARTS -->|joblib.load| SVC
    SVC --> API
    ASS --> API
    UI -->|HTTP POST| API
    API -->|JSON| UI
```

## Request Routing Diagram

```mermaid
flowchart TD
    REQ["Incoming request\n(user_id?, age?, session_items?)"]
    KNOWN{"user_id in\ntraining set?"}
    HYBRID["HybridRecommender\n(CollaborativeSVD + ContentTF-IDF\nblended & ranked)"]
    COLD{"Session items\npresent?"}
    SESSION["ColdStart: session_content\n(content similarity to viewed items)"]
    DEMO{"Demographics\nprovided?"}
    DEMO_POP["ColdStart: demographic_popularity\n(age-band × membership segment)"]
    CAT{"Preferred group\nprovided?"}
    CAT_POP["ColdStart: category_popularity\n(product group head)"]
    GLOBAL["ColdStart: global_popularity\n(platform-wide head)"]
    BUDGET{"max_budget\nset?"}
    FILTER["Price filter\n(article price ≤ budget)"]
    EXPLAIN["Build explanation\n(signals, evidence_article_ids, reason)"]
    RESP["RecommendationResponse"]

    REQ --> KNOWN
    KNOWN -->|Yes| HYBRID
    KNOWN -->|No| COLD
    COLD -->|Yes| SESSION
    COLD -->|No| DEMO
    DEMO -->|Yes| DEMO_POP
    DEMO -->|No| CAT
    CAT -->|Yes| CAT_POP
    CAT -->|No| GLOBAL
    HYBRID --> BUDGET
    SESSION --> BUDGET
    DEMO_POP --> BUDGET
    CAT_POP --> BUDGET
    GLOBAL --> BUDGET
    BUDGET -->|Yes| FILTER
    BUDGET -->|No| EXPLAIN
    FILTER --> EXPLAIN
    EXPLAIN --> RESP
```

## Model Composition

```mermaid
classDiagram
    class HybridRecommender {
        +collaborative: CollaborativeSVD
        +content: ContentRecommender
        +collaborative_weight: float
        +recommend(user_id, limit, fresh_quota) list~HybridScore~
    }
    class CollaborativeSVD {
        +n_components: int = 64
        +user_factors_
        +item_factors_
        +seen_: dict
        +fit(interactions)
        +recommend(user_id, limit) list~tuple~
    }
    class ContentRecommender {
        +n_components: int = 64
        +tfidf: TfidfVectorizer
        +svd: TruncatedSVD
        +item_factors_
        +fit(items, interactions)
        +recommend(user_id, limit) list~tuple~
    }
    class PopularityRecommender {
        +item_counts_: Series
        +fit(interactions)
        +recommend(seen, limit) list~tuple~
    }
    class ColdStartRecommender {
        +content: ContentRecommender
        +popularity: PopularityRecommender
        +segment_counts_: dict
        +recommend(context) ColdStartResult
    }
    HybridRecommender --> CollaborativeSVD
    HybridRecommender --> ContentRecommender
    ColdStartRecommender --> ContentRecommender
    ColdStartRecommender --> PopularityRecommender
    CollaborativeSVD ..> PopularityRecommender : fallback
    ContentRecommender ..> PopularityRecommender : fallback
```

## Hybrid Ranking Implementation

Product metadata is represented with word/bigram TF-IDF compressed to 64 SVD
components. A user's content profile is the normalized, confidence-weighted
centroid of the product factors for all items in their training history —
repeat purchases receive higher weight.

Collaborative SVD and content retrieval each produce a candidate pool of
`10 × limit` items. Scores are min-max normalized per request, then blended:

```text
hybrid_score = cf_weight × normalized_cf_score
             + (1 − cf_weight) × normalized_content_score
```

The weight is tuned on the validation split by grid-search over {0.25, 0.50,
0.75}, selecting the value that maximizes NDCG@K (MAP@K and Hit Rate@K as
deterministic tie-breakers). The best weight found is stored in
`artifacts/hybrid/best_hybrid_config.json` and loaded at serve time.

Previously purchased items are excluded before ranking. Warm-start benchmarks
restrict every model to the catalog observed during training; fresh-item
exposure is measured separately to avoid future-catalog leakage.
