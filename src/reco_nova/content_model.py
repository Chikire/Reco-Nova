"""Content-Based Recommendation Engine using TF-IDF and Sentence Embeddings."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

class ContentRecommender:
    def __init__(self, items_df: pd.DataFrame):
        """
        Initializes the recommender with the cleaned items catalog.
        Requires 'article_id' and 'item_text' columns.
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

    def build_tfidf(self):
        """Build TF-IDF features."""
        print("Building TF-IDF Matrix...")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,  # Cap features to save memory
            ngram_range=(1, 2)  # Capture pairs like "blue shirt"
        )
        self.tfidf_matrix = vectorizer.fit_transform(self.item_texts)
        print(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")

    def build_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Generate sentence embeddings for products."""
        print(f"Loading SentenceTransformer: {model_name}...")
        # all-MiniLM-L6-v2 is the industry standard for fast, high-quality embeddings
        model = SentenceTransformer(model_name)
        
        print("Encoding item texts (this may take a few minutes)...")
        embeddings = model.encode(self.item_texts, show_progress_bar=True, batch_size=256)
        
        print("Building FAISS Index for fast similarity search...")
        # L2 normalize embeddings so Inner Product equals Cosine Similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        
        # FAISS prevents the OOM crashes that happen if you try to do a 105k x 105k cosine matrix
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
        
        # Compute cosine similarity between the target item and all other items
        similarities = cosine_similarity(target_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top N scores (excluding the item itself)
        # argpartition is much faster than sorting the entire array
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
        
        # Reconstruct the vector from the FAISS index
        target_vector = np.expand_dims(self.faiss_index.reconstruct(idx), axis=0)
        
        # FAISS search is lightning fast. We ask for top_n + 1 to account for the item itself
        distances, indices = self.faiss_index.search(target_vector, top_n + 1)
        
        return [self.article_ids[i] for i in indices[0] if i != idx][:top_n]

    def get_similar_embeddings_df(
        self,
        article_id: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Return a ranked DataFrame with embedding similarity scores and item details."""
        if self.faiss_index is None:
            raise ValueError("Call build_embeddings() first.")
        if article_id not in self.id_to_idx:
            return pd.DataFrame()

        idx = self.id_to_idx[article_id]
        target_vector = np.expand_dims(self.faiss_index.reconstruct(idx), axis=0)
        distances, indices = self.faiss_index.search(target_vector, top_n + 1)

        rows = []
        for rank, (score, candidate_idx) in enumerate(
            zip(distances[0], indices[0]),
            start=1,
        ):
            if candidate_idx == idx:
                continue
            rows.append(
                {
                    "query_article_id": article_id,
                    "query_item_text": self.items_df.iloc[idx]["item_text"],
                    "rank": len(rows) + 1,
                    "score": float(score),
                    "article_id": self.article_ids[candidate_idx],
                    "item_text": self.items_df.iloc[candidate_idx]["item_text"],
                }
            )
            if len(rows) >= top_n:
                break

        return pd.DataFrame(rows)
