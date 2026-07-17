"""Streamlit product-discovery experience backed by the Reco-Nova API."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


API_URL = os.getenv("RECO_NOVA_API_URL", "http://localhost:8000").rstrip("/")


class APIError(RuntimeError):
    """Raised when the recommendation service cannot fulfill a UI request."""


def api_request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the FastAPI service using only the Python standard library."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{API_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise APIError(f"Recommendation service returned {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise APIError(f"Cannot reach the recommendation service at {API_URL}") from exc


def build_payload(
    mode: str,
    limit: int,
    user_id: str = "",
    age: int | None = None,
    membership: str | None = None,
    product_group: str | None = None,
    session_items: str = "",
) -> dict[str, Any]:
    """Build a compact API request from UI controls."""
    payload: dict[str, Any] = {"limit": limit}
    if mode == "Personalized":
        if user_id.strip():
            payload["user_id"] = user_id.strip()
        return payload
    if age is not None:
        payload["age"] = age
    if membership and membership != "Not specified":
        payload["club_member_status"] = membership.lower()
    if product_group and product_group != "All products":
        payload["preferred_product_group"] = product_group
    session = [item.strip() for item in session_items.split(",") if item.strip()]
    if session:
        payload["session_article_ids"] = session[:50]
    return payload


@st.cache_data(show_spinner=False)
def product_groups(processed_dir: str = "data/processed") -> list[str]:
    path = Path(processed_dir) / "items_clean.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path, columns=["product_group_name"])
    return sorted(
        value for value in frame["product_group_name"].dropna().astype(str).unique()
        if value
    )


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        :root { --orange:#ff6b00; --orange2:#ff9a3d; --ink:#0b0b0d; --muted:#6f7178; --paper:#f7f7f5; }
        .stApp { background: var(--paper); color: var(--ink); font-family:'DM Sans',sans-serif; }
        [data-testid="stSidebar"] { background:#0b0b0d; border-right:1px solid #252529; }
        [data-testid="stSidebar"] * { color:#f8f8f5 !important; }
        [data-testid="stSidebar"] .stButton button { background:var(--orange); border:0; color:white; }
        h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
        .block-container { max-width:1440px; padding:1.5rem 3rem 4rem; }
        .rn-nav { display:flex; justify-content:space-between; align-items:center; padding:.4rem 0 1.4rem; }
        .rn-brand { display:flex; align-items:center; gap:.7rem; font:800 1.15rem Manrope; }
        .rn-mark { width:38px;height:38px;border-radius:12px;background:var(--orange);color:white;display:grid;place-items:center;font-weight:800;box-shadow:0 8px 22px #ff6b0040; }
        .rn-live { padding:.45rem .75rem;border:1px solid #d9d9d5;border-radius:999px;font-size:.78rem;font-weight:700;background:white; }
        .rn-hero { position:relative;overflow:hidden;background:#0b0b0d;color:white;border-radius:30px;padding:3.5rem 3.6rem;margin-bottom:1.6rem;box-shadow:0 24px 70px #12121222; }
        .rn-hero:after { content:'';position:absolute;width:410px;height:410px;border-radius:50%;right:-80px;top:-160px;background:radial-gradient(circle,#ff7a18 0,#ff6b00 38%,transparent 70%);opacity:.85; }
        .rn-eyebrow { color:#ff9a3d;font-size:.76rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;margin-bottom:1rem; }
        .rn-hero h1 { color:white;font-size:clamp(2.7rem,5vw,5.3rem);line-height:.94;max-width:850px;margin:0 0 1.2rem;position:relative;z-index:1; }
        .rn-hero p { color:#c9c9ca;font-size:1.05rem;max-width:610px;line-height:1.65;position:relative;z-index:1; }
        .rn-pills { display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1.5rem;position:relative;z-index:1; }
        .rn-pill { border:1px solid #ffffff2b;background:#ffffff10;border-radius:999px;padding:.55rem .85rem;font-size:.8rem;font-weight:600; }
        .rn-section { display:flex;justify-content:space-between;align-items:end;margin:2.2rem 0 1rem; }
        .rn-section h2 { margin:0;font-size:2rem; }
        .rn-section p { margin:.35rem 0 0;color:var(--muted); }
        [data-testid="stVerticalBlockBorderWrapper"] { background:white;border:1px solid #e7e6e1 !important;border-radius:20px !important;box-shadow:0 10px 30px #1111110a;overflow:hidden; }
        [data-testid="stImage"] img { border-radius:15px;aspect-ratio:3/4;object-fit:cover;background:#efefeb; }
        .rn-card-kicker { color:var(--orange);text-transform:uppercase;letter-spacing:.1em;font-size:.68rem;font-weight:800; }
        .rn-product-name { font:700 1.02rem Manrope;margin:.3rem 0 .15rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
        .rn-group { color:var(--muted);font-size:.8rem;min-height:1.2rem; }
        .rn-reason { background:#fff3e8;border-left:3px solid var(--orange);padding:.7rem .8rem;border-radius:8px;color:#58321c;font-size:.78rem;line-height:1.4;margin-top:.7rem; }
        .rn-empty { min-height:260px;border-radius:15px;background:linear-gradient(145deg,#171719,#ff6b00);display:grid;place-items:center;color:white;font:800 2rem Manrope; }
        .stButton button[kind="primary"], .stFormSubmitButton button { background:var(--orange)!important;color:white!important;border:0!important;border-radius:12px!important;font-weight:700!important;min-height:46px;box-shadow:0 10px 24px #ff6b0030; }
        .stButton button:hover, .stFormSubmitButton button:hover { background:#e85f00!important;transform:translateY(-1px); }
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input { border-radius:12px!important; }
        @media(max-width:760px){.block-container{padding:1rem}.rn-hero{padding:2.2rem 1.5rem;border-radius:22px}.rn-hero h1{font-size:2.6rem}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, dict[str, Any]]:
    st.sidebar.markdown("## Discovery controls")
    st.sidebar.caption("Shape the feed with as much—or as little—context as you like.")
    mode = st.sidebar.radio("Shopping mode", ["Discover", "Personalized"], horizontal=True)
    limit = st.sidebar.slider("Number of products", 4, 20, 8)
    values: dict[str, Any] = {"limit": limit}
    if mode == "Personalized":
        values["user_id"] = st.sidebar.text_input(
            "Customer ID", placeholder="Paste a known customer ID"
        )
    else:
        values["age"] = st.sidebar.slider("Age", 16, 90, 28)
        values["membership"] = st.sidebar.selectbox(
            "Membership", ["Not specified", "Active", "Pre-create", "Left club"]
        )
        groups = ["All products", *product_groups()]
        values["product_group"] = st.sidebar.selectbox("Product group", groups)
        values["session_items"] = st.sidebar.text_area(
            "Recently viewed article IDs",
            placeholder="108775044, 706016001",
            help="Comma-separated IDs personalize this anonymous session.",
        )
    st.sidebar.divider()
    try:
        health = api_request("/health")
        if health.get("models_ready"):
            st.sidebar.success("Recommendation engine online")
        else:
            st.sidebar.warning("API online · models unavailable")
    except APIError:
        st.sidebar.error("API offline · run `make run-api`")
    return mode, values


def render_product(product: dict[str, Any], rank: int) -> None:
    with st.container(border=True):
        image_path = product.get("image_path")
        if image_path and Path(image_path).exists():
            st.image(image_path, use_container_width=True)
        else:
            st.markdown('<div class="rn-empty">RN</div>', unsafe_allow_html=True)
        name = escape(str(product.get("product_name") or f"Product {product['article_id']}"))
        group = escape(str(product.get("product_group") or "Curated find"))
        article_id = escape(str(product["article_id"]))
        reason = escape(str(product.get("reason", "Selected for you")))
        st.markdown(
            f'<div class="rn-card-kicker">Pick {rank:02d} · {article_id}</div>'
            f'<div class="rn-product-name">{name}</div>'
            f'<div class="rn-group">{group}</div>'
            f'<div class="rn-reason">✦ {reason}</div>',
            unsafe_allow_html=True,
        )
        signals = product.get("signals", {})
        if signals:
            with st.expander("Why this pick?"):
                for label, value in signals.items():
                    st.caption(label.replace("_", " ").title())
                    st.progress(min(max(float(value), 0.0), 1.0))
                evidence = product.get("evidence_article_ids", [])
                if evidence:
                    st.caption(f"Evidence product: {evidence[0]}")


def render_assistant(mode: str, values: dict[str, Any]) -> None:
    """Render the conversational layer while keeping product facts API-grounded."""
    st.markdown(
        '<div class="rn-section"><div><h2>Ask Reco—Nova</h2>'
        '<p>Describe a category, colour, style, occasion, or budget in your own words.</p></div>'
        '<span class="rn-live">GENAI ASSISTANT</span></div>',
        unsafe_allow_html=True,
    )
    history = st.session_state.setdefault("assistant_history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    prompt = st.chat_input("Try: Find me six casual black pieces under $60")
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    payload = build_payload(mode, **values)
    payload.update({"message": prompt, "history": history[:-1][-12:]})
    try:
        with st.spinner("Understanding your shopping intent…"):
            answer = api_request("/assistant/chat", payload)
    except APIError as exc:
        st.error(str(exc))
        return
    history.append({"role": "assistant", "content": answer["message"]})
    st.session_state["recommendations"] = {
        "explanation": answer["message"],
        "strategy": answer.get("strategy") or "conversational_discovery",
        "recommendations": answer.get("recommendations", []),
    }
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Reco-Nova · Find your next favorite",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    st.markdown(
        '<div class="rn-nav"><div class="rn-brand"><span class="rn-mark">R</span> RECO—NOVA</div>'
        '<span class="rn-live">● LIVE DISCOVERY</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<section class="rn-hero"><div class="rn-eyebrow">Personal style · powered by intelligence</div>
        <h1>Less scrolling.<br>More finding.</h1>
        <p>A recommendation experience that understands your signals, explains every pick, and gets useful from the very first visit.</p>
        <div class="rn-pills"><span class="rn-pill">✦ Explainable picks</span><span class="rn-pill">◉ Cold-start ready</span><span class="rn-pill">↗ Adaptive discovery</span></div></section>""",
        unsafe_allow_html=True,
    )
    mode, values = render_sidebar()
    render_assistant(mode, values)
    st.divider()
    with st.form("discovery_form", border=False):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(
                f"### {'Your personal edit' if mode == 'Personalized' else 'Start your discovery'}"
            )
            st.caption("Tune the controls, then let Reco-Nova build the edit.")
        with right:
            submitted = st.form_submit_button("Create my edit  →", type="primary", use_container_width=True)

    if submitted:
        payload = build_payload(mode, **values)
        if mode == "Personalized" and not payload.get("user_id"):
            st.warning("Add a customer ID or switch to Discover mode.")
        else:
            try:
                with st.spinner("Curating products around your signals…"):
                    st.session_state["recommendations"] = api_request("/explain", payload)
            except APIError as exc:
                st.error(str(exc))
                st.info("Start the backend in another terminal with `make run-api`.")

    response = st.session_state.get("recommendations")
    if response:
        st.markdown(
            f'<div class="rn-section"><div><h2>Your Reco-Nova edit</h2><p>{response["explanation"]}</p></div>'
            f'<span class="rn-live">{response["strategy"].replace("_", " ").upper()}</span></div>',
            unsafe_allow_html=True,
        )
        products = response.get("recommendations", [])
        for start in range(0, len(products), 4):
            columns = st.columns(4, gap="medium")
            for offset, product in enumerate(products[start : start + 4]):
                with columns[offset]:
                    render_product(product, start + offset + 1)
    else:
        st.markdown(
            '<div class="rn-section"><div><h2>Built for every first impression</h2>'
            '<p>Choose your signals in the sidebar and create a product edit that feels intentional.</p></div></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
