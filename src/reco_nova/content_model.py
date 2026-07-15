"""Content-Based Recommendation Engine using TF-IDF and Sentence Embeddings."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
from difflib import SequenceMatcher

class ContentRecommender:
    def __init__(self, items_df: pd.DataFrame):
        """
        Initializes the recommender with the cleaned items catalog.
        Requires 'article_id', 'item_text', and 'prod_name' columns.
        """
        # Drop items with missing text to prevent crashes
        self.items_df = items_df.dropna(subset=['item_text']).reset_index(drop=True)
        self.article_ids = self.items_df['article_id'].values
        self.item_texts = self.items_df['item_text'].astype(str).tolist()
        
        # Internal dictionaries for fast lookups
        self.id_to_idx = {art_id: idx for idx, art_id in enumerate(self.article_ids)}
        
        # Placeholders for our models
        self.tfidf_matrix = None
        self.faiss_index = None

    def _norm_text(self, value) -> str:
        """Normalize text for consistent matching."""
        return str(value).strip().lower()

    def resolve_name_to_article_id(self, name_query: str) -> dict | None:
        """Resolve a name query to the best matching article_id and keep ID canonical."""
        query_norm = self._norm_text(name_query)
        work = self.items_df[["article_id", "prod_name", "item_text"]].copy()
        work["prod_name_norm"] = work["prod_name"].fillna("").map(self._norm_text)

        # 1) Exact name match
        exact = work[work["prod_name_norm"] == query_norm]
        if not exact.empty:
            best = exact.iloc[0]
            return {
                "query_name": name_query,
                "article_id": best["article_id"],
                "prod_name": best["prod_name"],
                "match_type": "exact",
                "name_match_score": 1.0,
            }

        # 2) Contains match
        contains = work[work["prod_name_norm"].str.contains(query_norm, na=False)]
        if not contains.empty:
            contains = contains.copy()
            contains["name_match_score"] = contains["prod_name_norm"].map(
                lambda x: SequenceMatcher(None, query_norm, x).ratio()
            )
            best = contains.sort_values("name_match_score", ascending=False).iloc[0]
            return {
                "query_name": name_query,
                "article_id": best["article_id"],
                "prod_name": best["prod_name"],
                "match_type": "contains",
                "name_match_score": float(best["name_match_score"]),
            }

        # 3) Fuzzy fallback on name similarity
        scored = work.copy()
        scored["name_match_score"] = scored["prod_name_norm"].map(
            lambda x: SequenceMatcher(None, query_norm, x).ratio()
        )
        best = scored.sort_values("name_match_score", ascending=False).iloc[0]

        if best["name_match_score"] < 0.35:
            return None

        return {
            "query_name": name_query,
            "article_id": best["article_id"],
            "prod_name": best["prod_name"],
            "match_type": "fuzzy",
            "name_match_score": float(best["name_match_score"]),
        }

    def build_tfidf(self):
        """Build TF-IDF features."""
        print("Building TF-IDF Matrix...")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = vectorizer.fit_transform(self.item_texts)
        print(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")

    def build_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Generate sentence embeddings for products."""
        print(f"Loading SentenceTransformer: {model_name}...")
        model = SentenceTransformer(model_name)
        
        print("Encoding item texts (this may take a few minutes)...")
        embeddings = model.encode(self.item_texts, show_progress_bar=True, batch_size=256)
        
        print("Building FAISS Index for fast similarity search...")
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        
        self.faiss_index = faiss.IndexFlatIP(dimension) 
        self.faiss_index.add(embeddings)
        print("FAISS Index successfully built.")

    def get_similar_tfidf(self, article_id: str, top_n: int = 5) -> list[str]:
        """Compute similarity and return Top-N (TF-IDF)."""
        if self.tfidf_matrix is None:
            raise ValueError("Call build_tfidf() first.")
        if article_id not in self.id_to_idx:
            return []
            
        idx = self.id_to_idx[article_id]
        target_vector = self.tfidf_matrix[idx]
        
        similarities = cosine_similarity(target_vector, self.tfidf_matrix).flatten()
        
        top_indices = np.argpartition(similarities, -(top_n + 1))[-(top_n + 1):]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        return [self.article_ids[i] for i in top_indices if i != idx][:top_n]

    def get_similar_tfidf_df(self, article_id: str, top_n: int = 5) -> pd.DataFrame:
        """Return a ranked DataFrame with TF-IDF similarity scores and item details."""
        if self.tfidf_matrix is None:
            raise ValueError("Call build_tfidf() first.")
        if article_id not in self.id_to_idx:
            return pd.DataFrame()

        idx = self.id_to_idx[article_id]
        target_vector = self.tfidf_matrix[idx]
        similarities = cosine_similarity(target_vector, self.tfidf_matrix).flatten()

        top_indices = np.argpartition(similarities, -(top_n + 1))[-(top_n + 1):]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        top_indices = [i for i in top_indices if i != idx][:top_n]

        result = self.items_df.iloc[top_indices][["article_id", "item_text"]].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result.insert(1, "score", similarities[top_indices])
        result.insert(0, "query_article_id", article_id)
        result.insert(1, "query_item_text", self.items_df.iloc[idx]["item_text"])
        return result.reset_index(drop=True)

    def get_similar_embeddings(self, article_id: str, top_n: int = 5) -> list[str]:
        """Compute similarity and return Top-N (Embeddings)."""
        if self.faiss_index is None:
            raise ValueError("Call build_embeddings() first.")
        if article_id not in self.id_to_idx:
            return []
            
        idx = self.id_to_idx[article_id]
        
        target_vector = np.expand_dims(self.faiss_index.reconstruct(idx), axis=0)
        distances, indices = self.faiss_index.search(target_vector, top_n + 1)
        
        return [self.article_ids[i] for i in indices[0] if i != idx][:top_n]

    def get_similar_embeddings_df(self, article_id: str, top_n: int = 5) -> pd.DataFrame:
        """Return a ranked DataFrame with embedding similarity scores and item details."""
        if self.faiss_index is None:
            raise ValueError("Call build_embeddings() first.")
        if article_id not in self.id_to_idx:
            return pd.DataFrame()

        idx = self.id_to_idx[article_id]
        target_vector = np.expand_dims(self.faiss_index.reconstruct(idx), axis=0)
        distances, indices = self.faiss_index.search(target_vector, top_n + 1)

        rows = []
        for rank, (score, candidate_idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if candidate_idx == idx:
                continue
            rows.append({
                "query_article_id": article_id,
                "query_item_text": self.items_df.iloc[idx]["item_text"],
                "rank": len(rows) + 1,
                "score": float(score),
                "article_id": self.article_ids[candidate_idx],
                "item_text": self.items_df.iloc[candidate_idx]["item_text"],
            })
            if len(rows) >= top_n:
                break

        return pd.DataFrame(rows)