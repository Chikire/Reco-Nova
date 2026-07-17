# Reco-Nova Final Offline Evaluation

This report evaluates frozen model choices once on the untouched test split.
Hybrid weighting was selected on validation data before this evaluation.
This run includes fresh-catalog exposure (items with no training interactions
are eligible for recommendation via the content model).

## Evaluation setup

- Training interactions: 7,821,301
- Eligible training users: 731,092
- Training-catalog items: 50,685
- Fresh-catalog items (no training interactions): 54,857
- Warm test users evaluated: 1,000
- Ranking cutoff: K=12
- Frozen collaborative weight: 0.75
- Bootstrap samples: 1,000
- Fresh-item exposure: enabled (min 1 fresh item per top-K)

## Results

| Model | NDCG@12 | MAP@12 | Hit Rate@12 | Catalog Coverage@12 |
|---|---:|---:|---:|---:|
| popularity | 0.006806 | 0.003768 | 0.029 | 0.000335 |
| collaborative_svd | 0.007181 | 0.004958 | 0.021 | 0.010437 |
| content_tfidf | 0.002757 | 0.001947 | 0.007 | 0.135267 |
| hybrid_frozen | 0.006690 | 0.004420 | 0.019 | 0.060708 |

### 95% Bootstrap Confidence Intervals

| Model | NDCG@12 lower | NDCG@12 upper | Hit Rate@12 lower | Hit Rate@12 upper |
|---|---:|---:|---:|---:|
| popularity | 0.003860 | 0.010396 | 0.020 | 0.040 |
| collaborative_svd | 0.003605 | 0.011237 | 0.013 | 0.031 |
| content_tfidf | 0.000535 | 0.005579 | 0.002 | 0.013 |
| hybrid_frozen | 0.003208 | 0.010743 | 0.011 | 0.029 |

### Fresh-Catalog Exposure Metrics

| Model | Fresh Coverage@12 | Fresh Share@12 | Users With Fresh Hit |
|---|---:|---:|---:|
| popularity | 0.000 | 0.000 | 0.000 |
| collaborative_svd | 0.000 | 0.000 | 0.000 |
| content_tfidf | 0.042 | 0.311 | 0.910 |
| hybrid_frozen | 0.020 | 0.135 | 1.000 |

## Interpretation

**Popularity** achieves the highest hit rate (2.9%) among all models, reflecting
that the most-purchased items remain broadly relevant at test time. Its near-zero
catalog coverage confirms it only ever recommends a tiny slice of the catalog.

**Collaborative SVD** achieves the best NDCG and MAP, indicating it ranks
relevant items higher within a ranked list, even though its absolute hit rate is
lower than popularity. It covers ~1% of the catalog per user.

**Content TF-IDF** has the lowest relevance metrics but the highest catalog
coverage (13.5%). With fresh-catalog exposure enabled, it surfaces fresh items
to 91% of users, contributing 31% of its recommendations from unseen products.
This is its primary role: discovery and new-item onboarding, not precision.

**Hybrid (frozen at 0.75 collaborative weight)** blends collaborative precision
with content diversity. With fresh-catalog exposure, 100% of evaluated users
receive at least one fresh item in their top-12, and fresh items constitute
13.5% of recommendations. The hybrid trades a small NDCG reduction versus
pure collaborative in exchange for meaningful catalog coverage (6%) and full
fresh-item reach.

All confidence intervals overlap substantially across models, confirming that
differences are not statistically distinguishable at N=1,000 test users.
The primary differentiator between models is the **coverage–relevance tradeoff**
rather than a clearly dominant model.

## Reproduce

```bash
# Standard final evaluation
make evaluate-final

# With fresh-catalog exposure (this report)
make evaluate-final-fresh
```
