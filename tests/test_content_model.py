"""Tests for the Content-Based Recommender.

Run this suite from the root of your project using: 
pytest tests/test_content_model.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Tell Python to look in the src directory relative to this test file
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reco_nova.content_model import ContentRecommender


@pytest.fixture
def mock_items_df():
    """Provides a small dummy catalog for testing."""
    return pd.DataFrame({
        "article_id": ["item_1", "item_2", "item_3", "item_4"],
        "item_text": [
            "red cotton shirt",      # item 1
            "blue cotton shirt",     # item 2 (Very similar to 1)
            "black leather shoes",   # item 3 (Completely different)
            None,                    # item 4 (Should be dropped)
        ]
    })


def test_recommender_initialization_drops_missing_text(mock_items_df):
    """Verifies that items without text are safely removed to prevent crashes."""
    recommender = ContentRecommender(mock_items_df)
    
    assert len(recommender.items_df) == 3, "Should have dropped item_4"
    assert "item_4" not in recommender.id_to_idx
    assert recommender.id_to_idx["item_1"] == 0


def test_build_and_get_similar_tfidf(mock_items_df):
    """
    Verifies that the TF-IDF matrix is built correctly and that 
    keyword overlap successfully pulls the most similar items.
    """
    recommender = ContentRecommender(mock_items_df)
    recommender.build_tfidf()
    
    # The matrix should have 3 rows (for the 3 valid items)
    assert recommender.tfidf_matrix.shape[0] == 3
    
    # Get top 1 similar item for 'item_1' ("red cotton shirt")
    # It should return 'item_2' ("blue cotton shirt") because of the word overlap.
    recommendations = recommender.get_similar_tfidf("item_1", top_n=1)
    
    assert len(recommendations) == 1
    assert recommendations[0] == "item_2"
    
    # Ensure it doesn't crash on an unknown item
    assert recommender.get_similar_tfidf("unknown_item") == []


# Notice the patch target now points to content_model
@patch("reco_nova.content_model.SentenceTransformer")
def test_build_and_get_similar_embeddings(mock_st_class, mock_items_df):
    """
    Verifies the FAISS index pipeline. 
    We use @patch to fake the SentenceTransformer so the test runs instantly 
    without downloading a massive neural network.
    """
    # 1. Setup the fake embeddings
    # We create a fake 3D vector space where item 1 and 2 are close, and 3 is far away.
    dummy_embeddings = np.array([
        [1.0, 0.0, 0.0], # item_1 vector
        [0.9, 0.1, 0.0], # item_2 vector (Close to item 1)
        [0.0, 1.0, 0.0], # item_3 vector (Different direction)
    ], dtype=np.float32)
    
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = dummy_embeddings
    mock_st_class.return_value = mock_model_instance
    
    # 2. Run the pipeline
    recommender = ContentRecommender(mock_items_df)
    recommender.build_embeddings()
    
    # 3. Assertions
    # Ensure the model was actually called
    mock_st_class.assert_called_once_with('all-MiniLM-L6-v2')
    assert recommender.faiss_index.ntotal == 3, "FAISS index should contain 3 vectors"
    
    # Test similarities
    recommendations = recommender.get_similar_embeddings("item_1", top_n=1)
    
    assert len(recommendations) == 1
    assert recommendations[0] == "item_2", "Should recommend the mathematically closest vector"