import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_retriever(mocker):
    """Fixture to mock heavy vector retrieval calls."""
    mock = mocker.patch("RAG_pipeline.rag_retriever.retriever.DenseRerankRetriever")
    instance = mock.return_value
    instance.retrieve.return_value = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_101",
            "content": "Sample WebSphere documentation content.",
            "rerank_score": 3.45,
            "metadata": {"title": "WebSphere Fix"}
        }
    ]
    return instance

def test_retriever_mocked_output(mock_retriever):
    """Test logic using mocked retrieval results."""
    results = mock_retriever.retrieve("WebSphere memory leak")
    assert len(results) == 1
    assert results[0]["document_id"] == "doc_101"
    assert results[0]["rerank_score"] > 0.0