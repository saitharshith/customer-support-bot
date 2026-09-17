import pytest
from pydantic import ValidationError
from backend.schemas import QueryRequest

def test_valid_query_request():
    """Test valid query payload parsing."""
    payload = QueryRequest(question="How to fix DB2 timeout?", thread_id="thread_1")
    assert payload.question == "How to fix DB2 timeout?"
    assert payload.thread_id == "thread_1"

def test_invalid_short_question():
    """Test min_length validation failure."""
    with pytest.raises(ValidationError):
        QueryRequest(question="hi")  # Fails min_length constraint (< 3 chars)