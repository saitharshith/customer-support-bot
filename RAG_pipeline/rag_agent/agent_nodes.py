import os
import litellm
from typing import Dict, Any
from dotenv import load_dotenv
from RAG_pipeline.rag_agent.agent_state import AgentState
from RAG_pipeline.rag_retriever.retriever import DenseRerankRetriever

load_dotenv()
LITELLM_MODEL = os.getenv("LITELLM_MODEL")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    litellm.groq_key = GROQ_API_KEY
litellm.drop_params = True  
litellm.telemetry = False

class AgentNodes:
    def __init__(self):
        self.retriever = DenseRerankRetriever(candidate_top_n=20, rerank_top_k=5)

    def retrieve_and_filter_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Executes retrieval and calculates maximum candidate rerank confidence score."""
        question = state["question"]
        print(f"\n[Node: Retrieve] Searching database for query: '{question}'...")
        
        raw_docs = self.retriever.retrieve(question)
        
        if not raw_docs:
            return {
                "context": [],
                "max_confidence_score": -99.0,
                "confidence_passed": False
            }
            
        # Top rank score represents highest system confidence
        max_score = raw_docs[0]["rerank_score"]
        
        # Filter chunks using the confidence threshold
        valid_docs = [doc for doc in raw_docs if doc["rerank_score"] > CONFIDENCE_THRESHOLD]
        confidence_passed = len(valid_docs) > 0
        
        print(f"[Node: Retrieve] Top Score: {max_score:.4f} | Valid Chunks: {len(valid_docs)}/{len(raw_docs)}")
        
        return {
            "context": valid_docs,
            "max_confidence_score": max_score,
            "confidence_passed": confidence_passed
        }

    def generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Generates an answer using LiteLLM on valid contexts."""
        print("[Node: Generate] Generating answer using retrieved context...")
        
        context_str = "\n\n".join([
            f"--- Document ID: {doc['document_id']} ---\n{doc['content']}" 
            for doc in state["context"]
        ])
        
        prompt = f"""You are an expert technical support engineer. Answer the question below using ONLY the provided documentation chunks. 
            If the information is not sufficient, state clearly that you do not know.

                    ### Documentation Context:
                    {context_str}

                    ### Question:
                    {state['question']}

                    ### Response:"""

        response = litellm.completion(
            model=LITELLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        return {"generation": answer, "escalated": False}

    def fallback_escalation_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Triggers fallback when confidence score drops below threshold."""
        print("[Node: Escalate] Confidence score below threshold. Routing to human support queue...")
        
        escalation_msg = (
            f"SYSTEM ESCALATION: Unconfident retrieval (Top score: {state['max_confidence_score']:.4f}). "
            f"Ticket has been automatically queued for human tier-2 support."
        )
        return {
            "generation": escalation_msg,
            "escalated": True
        }