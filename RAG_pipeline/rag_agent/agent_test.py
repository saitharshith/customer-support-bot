import os
import warnings
from RAG_pipeline.rag_agent.state_graph import build_graph

# Suppress HuggingFace/LiteLLM warnings for a cleaner CLI
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    app = build_graph()
    print("==================================================")
    print("🤖 RAG Support Agent | Interactive Mode")
    print("Type 'quit' or 'exit' to stop.")
    print("==================================================")

    while True:
        user_input = input("\n[User Question]: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue

        initial_state = {
            "question": user_input,
            "context": [],
            "max_confidence_score": 0.0,
            "confidence_passed": False,
            "generation": None,
            "escalated": False,
            "error": None
        }

        print("\n[Agent Trace]:")
        final_answer = ""
        is_escalated = False
        
        # Stream the graph execution events
        for event in app.stream(initial_state):
            for node_name, node_state in event.items():
                print(f" ⚙️  Executed Node: {node_name}")
                
                # Capture the final output from either generation or escalation nodes
                if "generation" in node_state:
                    final_answer = node_state["generation"]
                    is_escalated = node_state.get("escalated", False)

        print("\n[Agent Response]:")
        print(final_answer)
        
        if is_escalated:
            print("\n🚨 [SYSTEM TICKET TAGGED FOR HUMAN ROUTING]")
        print("-" * 50)