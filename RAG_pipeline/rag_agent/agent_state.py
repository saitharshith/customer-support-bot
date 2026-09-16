from typing import List, Dict, Any, TypedDict, Optional

class AgentState(TypedDict):
    question: str
    context: List[Dict[str, Any]]
    max_confidence_score: float
    confidence_passed: bool
    generation: Optional[str]
    escalated: bool
    error: Optional[str]