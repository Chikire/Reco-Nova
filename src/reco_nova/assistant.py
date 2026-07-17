"""Grounded conversational shopping orchestration for Reco-Nova.

The model is deliberately limited to intent extraction. Product selection and
all product facts come from the recommendation service, which prevents invented
catalog items and keeps the assistant useful when Ollama is not running.
"""

from __future__ import annotations

import os
import re
import json
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One bounded conversation turn supplied by the client."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class ShoppingIntent(BaseModel):
    """Shopping constraints understood from natural language."""

    product_group: str | None = None
    style: str | None = None
    colour: str | None = None
    max_budget: float | None = Field(default=None, ge=0)
    limit: int = Field(default=6, ge=1, le=20)


class AssistantRequest(BaseModel):
    """Chat request with optional recommendation context."""

    message: str = Field(min_length=1, max_length=2_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    user_id: str | None = Field(default=None, min_length=1)
    age: float | None = Field(default=None, ge=13, le=120)
    club_member_status: str | None = None
    session_article_ids: list[str] = Field(default_factory=list, max_length=50)


class AssistantResult(BaseModel):
    """Grounded assistant answer plus auditable intent and product results."""

    message: str
    intent: ShoppingIntent
    recommendations: list[dict[str, object]]
    strategy: str | None = None
    mode: Literal["ollama", "local"]
    unsupported_constraints: list[str] = Field(default_factory=list)
    guardrail: str | None = None


_COLOURS = {
    "black", "white", "orange", "red", "blue", "green", "yellow", "pink",
    "purple", "brown", "beige", "grey", "gray", "navy", "cream", "gold", "silver",
}
_STYLES = {
    "casual", "formal", "sporty", "minimal", "classic", "streetwear", "oversized",
    "slim", "elegant", "party", "office", "summer", "winter", "vintage",
}
_UNSAFE = re.compile(
    r"\b(?:ignore (?:all |the )?(?:previous|system)|system prompt|developer message|"
    r"steal|weapon|explosive|self[- ]?harm|suicide)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy)[!.?\s]*$",
    re.IGNORECASE,
)
_BUDGET = re.compile(
    r"(?:under|below|less than|max(?:imum)?|budget(?: of| is)?|up to)\s*"
    r"(?:[$£€]\s*)?(\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_LIMIT = re.compile(
    r"\b(?:show|find|give|recommend)(?:\s+me)?\s+(\d{1,2})\b", re.IGNORECASE
)


def local_intent(text: str, product_groups: list[str] | None = None) -> ShoppingIntent:
    """Extract common constraints deterministically for offline demos and tests."""
    lowered = text.lower()
    budget = _BUDGET.search(text)
    count = _LIMIT.search(text)
    colour = next((value for value in _COLOURS if re.search(rf"\b{value}\b", lowered)), None)
    style = next((value for value in _STYLES if re.search(rf"\b{value}\b", lowered)), None)
    product_group = None
    for group in sorted(product_groups or [], key=len, reverse=True):
        if group.lower() in lowered:
            product_group = group
            break
    return ShoppingIntent(
        product_group=product_group,
        style=style,
        colour="grey" if colour == "gray" else colour,
        max_budget=float(budget.group(1)) if budget else None,
        limit=min(int(count.group(1)), 20) if count else 6,
    )


def extract_intent(
    text: str,
    product_groups: list[str],
    model: str | None = None,
) -> tuple[ShoppingIntent, Literal["ollama", "local"]]:
    """Use Ollama structured output when available, otherwise local parsing."""
    fallback = local_intent(text, product_groups)
    if os.getenv("RECO_NOVA_LLM_PROVIDER", "ollama").lower() != "ollama":
        return fallback, "local"
    try:
        prompt = (
            "Extract only explicit shopping intent. Do not infer sensitive traits. "
            f"product_group must be null or exactly one of: {product_groups}. "
            "Use null for missing style, colour, or budget; limit defaults to 6. "
            f"Return the supplied JSON schema only. Shopping request: {text}"
        )
        body = json.dumps(
            {
                "model": model or os.getenv("RECO_NOVA_OLLAMA_MODEL", "llama3.2:3b"),
                "messages": [{"role": "user", "content": prompt}],
                "format": ShoppingIntent.model_json_schema(),
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        request = Request(
            f"{base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            content = json.loads(response.read().decode("utf-8"))["message"]["content"]
        parsed = ShoppingIntent.model_validate_json(content)
        allowed_groups = {group.casefold(): group for group in product_groups}
        if parsed.product_group:
            parsed.product_group = allowed_groups.get(parsed.product_group.casefold())
        # Numeric constraints are simple to extract exactly. Never accept a
        # model-invented budget or result count that was absent from the text.
        parsed.max_budget = fallback.max_budget
        parsed.limit = fallback.limit
        return parsed, "ollama"
    except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        # Recommendation remains available during model, service, or schema failures.
        pass
    return fallback, "local"


class ShoppingAssistant:
    """Translate conversation into a grounded recommendation-service call."""

    def __init__(
        self,
        recommend: Callable[[dict[str, object]], dict[str, object]],
        product_groups: list[str],
    ) -> None:
        self._recommend = recommend
        self._product_groups = product_groups

    def chat(self, request: AssistantRequest) -> AssistantResult:
        if _UNSAFE.search(request.message):
            return AssistantResult(
                message=(
                    "I can help with safe product discovery, but I can’t follow that request. "
                    "Tell me the category, colour, style, or occasion you are shopping for."
                ),
                intent=ShoppingIntent(),
                recommendations=[],
                mode="local",
                guardrail="unsafe_or_prompt_injection",
            )

        if _GREETING.fullmatch(request.message):
            return AssistantResult(
                message=(
                    "Hi! I’m your Reco—Nova shopping assistant. Tell me what you’re "
                    "looking for—try a product, colour, style, occasion, or budget."
                ),
                intent=ShoppingIntent(),
                recommendations=[],
                mode="local",
            )

        user_context = [
            turn.content
            for turn in request.history[-6:]
            if turn.role == "user" and not _GREETING.fullmatch(turn.content)
        ]
        intent_text = "\n".join([*user_context, request.message])
        intent, mode = extract_intent(intent_text, self._product_groups)
        current_intent = local_intent(request.message, self._product_groups)
        # An explicit category in the newest message replaces older context.
        if current_intent.product_group:
            intent.product_group = current_intent.product_group
        # Ask the recommender for a candidate pool, then enforce text/colour
        # constraints against catalog fields before returning the requested K.
        payload: dict[str, object] = {"limit": 20}
        payload["catalog_query"] = request.message
        for key in ("user_id", "age", "club_member_status", "session_article_ids"):
            value = getattr(request, key)
            if value not in (None, [], ""):
                payload[key] = value
        if intent.product_group:
            payload["preferred_product_group"] = intent.product_group
            payload["use_demographics"] = False

        response = self._recommend(payload)
        products = list(response.get("recommendations", []))
        catalog_search = response.get("strategy") == "catalog_text_search"
        if catalog_search and not current_intent.product_group:
            # A concrete product query (for example, shorts or cups) is more
            # specific than a broad category inferred from older conversation.
            intent.product_group = None
        if intent.product_group and not catalog_search:
            products = [
                product for product in products
                if intent.product_group.lower() in str(product.get("product_group", "")).lower()
            ]
        if intent.colour:
            products = [
                product for product in products
                if intent.colour.lower() in " ".join(
                    str(product.get(field, "")) for field in ("colour", "product_name", "description")
                ).lower()
            ]
        if intent.style:
            products = [
                product for product in products
                if intent.style.lower() in " ".join(
                    str(product.get(field, "")) for field in ("product_name", "description")
                ).lower()
            ]
        products = products[: intent.limit]
        unsupported = ["budget"] if intent.max_budget is not None else []
        constraints = [value for value in (intent.colour, intent.style, intent.product_group) if value]
        qualifier = f" for {' · '.join(constraints)}" if constraints else ""
        message = f"I found {len(products)} grounded recommendation(s){qualifier}."
        if unsupported:
            message += " Price is not available in this catalog, so I could not verify your budget."
        if not products:
            message += " Try a broader category or fewer constraints."
        return AssistantResult(
            message=message,
            intent=intent,
            recommendations=products,
            strategy=str(response.get("strategy")) if response.get("strategy") else None,
            mode=mode,
            unsupported_constraints=unsupported,
        )
