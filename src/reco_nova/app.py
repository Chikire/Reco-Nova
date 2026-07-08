"""Streamlit entry point for the product discovery UI."""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Reco-Nova", page_icon="🛍️", layout="wide")
    st.title("Reco-Nova")
    st.subheader("Personalized Product Recommendation Engine")
    st.write(
        "Use this app to prototype a personalized homepage feed and "
        "explanation layer."
    )


if __name__ == "__main__":
    main()
