from langgraph.graph import StateGraph, START, END
from RAG_pipeline.rag_agent.agent_state import AgentState
from RAG_pipeline.rag_agent.agent_nodes import AgentNodes

def decide_next_step(state: AgentState) -> str:
    """Conditional Edge Evaluator based on confidence score."""
    if state["confidence_passed"]:
        return "generate_answer"
    else:
        return "fallback_escalate"

def build_graph():
    nodes = AgentNodes()
    workflow = StateGraph(AgentState)
    # 1. Add Nodes
    workflow.add_node("retrieve", nodes.retrieve_and_filter_node)
    workflow.add_node("generate_answer", nodes.generate_answer_node)
    workflow.add_node("fallback_escalate", nodes.fallback_escalation_node)
    # 2. Add Standard Edges
    workflow.add_edge(START, "retrieve")
    # 3. Add Conditional Edge Routing
    workflow.add_conditional_edges(
        "retrieve",
        decide_next_step,
        {
            "generate_answer": "generate_answer",
            "fallback_escalate": "fallback_escalate"
        }
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("fallback_escalate", END)
    return workflow.compile()