# Reco-Nova Cold-Start Evaluation

This proxy benchmark evaluates users whose IDs never appear in train or validation.
Test purchases are used only as relevance labels.

## Evaluation setup

- New test users available: 1,824
- New test users evaluated: 1,000
- Eligible catalog items: 50,685

## Results

| Strategy | NDCG@K | MAP@K | Hit Rate@K | Coverage@K |
|---|---:|---:|---:|---:|
| global_popularity | 0.005626 | 0.002534 | 0.027000 | 0.000237 |
| demographic_fallback | 0.005704 | 0.002919 | 0.025000 | 0.001026 |

## Example outputs

### demographics

- Strategy: `demographic_popularity`
- Explanation: Popular with shoppers in age band 16-24 and membership active.
- Products: `759871002, 706016001, 741356002, 599580055, 372860002`

### category

- Strategy: `category_popularity`
- Explanation: Popular products in accessories.
- Products: `759465001, 673396002, 759469001, 759482001, 759479001`

### session

- Strategy: `session_content`
- Explanation: Based on products viewed this session.
- Products: `108775044, 538699007, 538699001, 780188001, 780188002`

### no_context

- Strategy: `global_popularity`
- Explanation: Popular products across all shoppers.
- Products: `610776002, 706016001, 610776001, 599580055, 599580038`

## Interpretation

Demographic performance is compared directly with the no-context global fallback. Session and category strategies are demonstrated separately because using test purchases to manufacture those inputs would leak relevance labels.

```bash
make evaluate-cold-start
```
