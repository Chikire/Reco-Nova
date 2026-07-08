"""FastAPI entry point for serving recommendations."""

from fastapi import FastAPI

app = FastAPI(title="Reco-Nova Recommendation API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic readiness check."""
    return {"status": "ok"}
