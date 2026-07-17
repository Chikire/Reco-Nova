# Recommendation Explainability

Reco-Nova explanations are derived from model inputs and scores rather than
generated claims.

## Personalized users

For known users, each product includes:

- `signals.collaborative`: the normalized collaborative contribution after the
  configured hybrid weight is applied.
- `signals.content`: the normalized metadata-content contribution after its
  hybrid weight is applied.
- `evidence_article_ids`: the training-history product with the greatest cosine
  similarity to the recommendation in content-factor space.
- `reason`: a “Because you interacted with…” sentence using that product's
  catalog name when available.

The contributions are request-local ranking evidence, not probabilities. A
zero contribution means that source did not place the product in its retrieved
candidate pool.

## Cold-start users

Cold-start explanations name the actual fallback used:

- `session_content`
- `demographic_popularity`
- `category_popularity`
- `global_popularity`

No behavioral history is claimed for an anonymous user. The `signals` mapping
contains the selected fallback with value `1.0`.

## API

Both `POST /recommend` and `POST /explain` return explanation fields. The
dedicated endpoint is intended for clients that want to make the explanation
interaction explicit, while retaining a consistent response schema.

Explanation evidence is covered by unit, API-integration, and real-artifact
smoke tests.
