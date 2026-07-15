# Reco-Nova Cold-Start Evaluation

This proxy benchmark evaluates users whose IDs never appear in train or validation.
Test purchases are used only as relevance labels.

## Evaluation setup

- New test users available: 1,818
- New test users evaluated: 1,000
- Eligible catalog items: 29,761

## Results

| Strategy | NDCG@K | MAP@K | Hit Rate@K | Coverage@K |
|---|---:|---:|---:|---:|
| global_popularity | 0.013669 | 0.006044 | 0.067000 | 0.000403 |
| demographic_fallback | 0.015166 | 0.006961 | 0.071000 | 0.002587 |

## Example outputs

### demographics

- Strategy: `demographic_popularity`
- Explanation: Popular with shoppers in age band 16-24 and membership active.
- Products: `918522001, 448509014, 706016001, 924243001, 915526001`

### category

- Strategy: `category_popularity`
- Explanation: Popular products in accessories.
- Products: `759465001, 673396002, 759482001, 552716001, 893820001`

### session

- Strategy: `session_content`
- Explanation: Based on products viewed this session.
- Products: `538699007, 780188002, 767869001, 767869002, 688463003`

### no_context

- Strategy: `global_popularity`
- Explanation: Popular products across all shoppers.
- Products: `924243001, 751471001, 706016001, 923758001, 448509014`

## Interpretation

Demographic performance is compared directly with the no-context global fallback. Session and category strategies are demonstrated separately because using test purchases to manufacture those inputs would leak relevance labels.

```bash
make evaluate-cold-start
```

## Databricks MLflow

- Experiment: `/Shared/reco-nova-cold-start`
- Experiment ID: `2491952985243196`
- Run ID: `7fe5594282504bfdaebeebe63e0a116a`
