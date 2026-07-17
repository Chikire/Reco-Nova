# Assumptions and Trade-offs

## Data Assumptions

| Assumption | Rationale | Risk if violated |
|---|---|---|
| H&M transaction timestamps represent real purchase order | Temporal train/val/test split depends on chronological ordering | Leakage between splits; metrics would be optimistically biased |
| The most-recent 6-month window is representative of current user preferences | Models trained on this window generalize to future weeks | Concept drift (seasonal shifts, new trends) degrades accuracy |
| Missing customer age can be imputed with the training-set median | Age is missing for ~20% of customers; median is a conservative fill | Demographic cold-start segments become less precise for imputed users |
| Raw price column × 1000 approximates USD | H&M prices are not published; the raw column is normalized (range ~0.004–0.7) | Budget filtering thresholds are not interpretable in real GBP/EUR/USD |
| `article_id` is stable across the raw CSV files | Join key between transactions, articles, and images | Silent mismatches if IDs are remapped in future data exports |

---

## Modeling Assumptions

| Assumption | Rationale | Risk if violated |
|---|---|---|
| Implicit positive-only feedback (purchase = interest) | No explicit ratings available | False positives: one-off gift purchases inflate affinity scores |
| Repeated purchases increase confidence (log-scale weighting) | More purchases = stronger preference signal | Staple items (socks, basics) dominate and suppress novel recommendations |
| 64 SVD components is sufficient for both collaborative and content models | Balances representational capacity with training/serving cost | Underfitting niche taste clusters; overfitting to popular item subspace |
| User profile = normalized centroid of product factors | Simple, interpretable, no recency decay (optional flag exists) | Recent taste shifts are underweighted; stale history dilutes current preferences |
| TF-IDF bigrams on product metadata capture semantic similarity | Low-resource approach; no embedding model required | Synonyms and multi-language descriptions are not handled; content similarity is purely lexical |
| Min-max normalization per request is a fair score combination | Puts both models on a common [0,1] scale before blending | Score range varies by user density; sparse users' CF scores compress poorly |

---

## Architecture Trade-offs

### Hybrid weight is a single global scalar

**Decision:** One weight (found by validation grid-search over {0.25, 0.50, 0.75}) applies to all users.  
**Benefit:** Deterministic; no per-user overfitting; reproducible evaluation.  
**Cost:** Users who only benefit from content (new-ish warm users with few purchases) get the same CF weight as power users.

### Popularity as the universal fallback

**Decision:** Both `CollaborativeSVD` and `ContentRecommender` delegate to `PopularityRecommender` for unknown users, rather than raising an error.  
**Benefit:** The API always returns a non-empty list.  
**Cost:** Unknown users receive globally popular items, not personalized ones. The strategy field signals this clearly to the client.

### No real-time interaction logging

**Decision:** The serving layer does not write interaction events back to any store.  
**Benefit:** Stateless API; no write path to fail or throttle.  
**Cost:** Models cannot be updated incrementally; retraining requires re-running the full offline pipeline from raw data.

### Content model trained on metadata only (no image embeddings)

**Decision:** `item_text` concatenates categorical fields; images are referenced by path but not featurized.  
**Benefit:** No GPU/CLIP dependency; training on CPU in minutes.  
**Cost:** Visually similar items with different text descriptions are not grouped correctly. Image-based similarity was planned as a future milestone.

### Synchronous joblib deserialization in a background thread

**Decision:** Model files (~1.5 GB combined pickle data) are loaded in a daemon thread after the API process starts.  
**Benefit:** `make run-api` returns in ~2 seconds; the terminal is immediately free.  
**Cost:** All recommendation endpoints return HTTP 503 for ~2.5 minutes. Clients (including the UI) must poll `/health` and handle 503 gracefully.

### No request-level caching

**Decision:** Every `/recommend` request re-runs the full ranking computation.  
**Benefit:** Results always reflect the current model state and any per-request context.  
**Cost:** Repeated identical requests recompute the same scores. At demo scale (~10 concurrent users) this is negligible; at production scale an LRU cache on (user_id, limit) would be warranted.

---

## Evaluation Trade-offs

### Warm-start only benchmark

**Decision:** The main NDCG/MAP/Hit Rate evaluation restricts to users present in training.  
**Rationale:** Cold-start users have no ground truth in the standard sense; evaluating them as warm-start users would contaminate the metric.  
**Cost:** The headline numbers do not reflect the ~30% of test users who are new. Cold-start quality is measured separately with its own evaluation script.

### Policy impact uses simulated clicks, not live A/B data

**Decision:** `evaluate_policy_impact.py` generates synthetic click events using a position-bias model against the test interactions.  
**Benefit:** No live traffic required; reproducible.  
**Cost:** The simulation treats any test-set purchase as a "relevant" click, which systematically underestimates CTR for personalized recommendations that surface items outside the narrow historical head. The negative "relevant CTR" metric in the report is an artefact of catalog breadth, not a real performance degradation.

### Bootstrap confidence intervals on 1,000 resamples

**Decision:** 95% CIs are computed by resampling users (not transactions).  
**Benefit:** Correct granularity — variance comes from user heterogeneity.  
**Cost:** CIs assume independent users. Users who interact with the same popular items are not independent; CIs may be slightly too narrow.

---

## Security and Privacy Assumptions

| Assumption | Notes |
|---|---|
| The API is deployed on a private network or behind an authenticated gateway | No authentication is implemented in the API itself. Direct public exposure would allow arbitrary user ID enumeration. |
| `user_id` values are opaque internal identifiers, not PII | The system echoes `user_id` in responses. If user IDs are email addresses or similar, this would leak PII. |
| Ollama runs locally | The assistant POSTs user messages to Ollama. If Ollama is remote, user queries leave the local environment. |
| Prompt injection is mitigated by the `_UNSAFE` regex guard | The guard catches common patterns but is not a comprehensive defense. A model with tool-use capabilities would require a more robust sandbox. |
