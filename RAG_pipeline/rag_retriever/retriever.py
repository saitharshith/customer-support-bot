import os
from typing import List, Dict, Any
import psycopg2
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT")),
}
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME"
)

class DenseRerankRetriever:
    def __init__(self, candidate_top_n: int = 20, rerank_top_k: int = 5):
        self.candidate_top_n = candidate_top_n
        self.rerank_top_k = rerank_top_k
        
        print("Initializing Embeddings and Cross-Encoder...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def _get_db_connection(self):
        return psycopg2.connect(**DB_CONFIG)

    def dense_search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        """Stage 1: Fetch top N candidates using pgvector cosine distance."""
        query_vector = self.embeddings.embed_query(query)
        
        sql = """
        SELECT 
            id,
            document_id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
        """
        
        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(query_vector), str(query_vector), top_n))
                rows = cur.fetchall()
                
        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "document_id": row[1],
                "content": row[2],
                "metadata": row[3],
                "dense_score": float(row[4])
            })
        return results

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Stage 2: Rescore candidates with Cross-Encoder."""
        if not candidates:
            return []
            
        pairs = [[query, doc["content"]] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for idx, score in enumerate(rerank_scores):
            candidates[idx]["rerank_score"] = float(score)
            
        # Sort by Cross-Encoder score descending
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Main entry point for pipeline."""
        candidates = self.dense_search(query, top_n=self.candidate_top_n)
        reranked_docs = self.rerank(query, candidates, top_k=self.rerank_top_k)
        return reranked_docs


# LangGraph Node Interface wrapper
def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph state node wrapper."""
    query = state["question"]
    retriever = DenseRerankRetriever(candidate_top_n=20, rerank_top_k=5)
    retrieved_docs = retriever.retrieve(query)
    threshold = 0.0 
    filtered_docs = [doc for doc in  retrieved_docs if doc['rerank_score'] > threshold]
    if not filtered_docs:
        return {
            **state,
            'context':None
            }
    
    return {
        **state,
        "context": retrieved_docs
    }