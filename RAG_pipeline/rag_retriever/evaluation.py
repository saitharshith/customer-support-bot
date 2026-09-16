import math
from typing import List, Set, Dict
# from retriever import retrieval_node


class IREvaluator:
    @staticmethod
    def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
        if not relevant_doc_ids:
            return 0.0
        retrieved_k = set(retrieved_doc_ids[:k])
        hits = retrieved_k.intersection(relevant_doc_ids)
        return len(hits) / len(relevant_doc_ids)

    @staticmethod
    def reciprocal_rank_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
        for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
        if not relevant_doc_ids:
            return 0.0
            
        retrieved_k = retrieved_doc_ids[:k]
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k):
            rel = 1 if doc_id in relevant_doc_ids else 0
            dcg += rel / math.log2(i + 2) 
            
        # Ideal DCG: best possible ranking where relevant docs appear first
        ideal_hits = min(len(relevant_doc_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        
        if idcg == 0.0:
            return 0.0
            
        return dcg / idcg

    def evaluate_batch(self, test_cases: List[Dict], retriever_instance, k_values: List[int] = [3, 5]) -> Dict:
        """Runs batch benchmark across test queries and calculates mean metrics."""
        summary_results = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
        
        for case in test_cases:
            query = case["query"]
            relevant_ids = set(case["relevant_document_ids"])
            
            # Fetch dense + reranked results
            results = retriever_instance.retrieve(query)
            retrieved_ids = [doc["document_id"] for doc in results]
            
            for k in k_values:
                recall = self.recall_at_k(retrieved_ids, relevant_ids, k)
                mrr = self.reciprocal_rank_at_k(retrieved_ids, relevant_ids, k)
                ndcg = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
                
                summary_results[k]["recall"].append(recall)
                summary_results[k]["mrr"].append(mrr)
                summary_results[k]["ndcg"].append(ndcg)
                
        # Calculate averages
        metrics_report = {}
        for k in k_values:
            metrics_report[f"k={k}"] = {
                "Mean Recall": sum(summary_results[k]["recall"]) / len(test_cases),
                "MRR": sum(summary_results[k]["mrr"]) / len(test_cases),
                "Mean nDCG": sum(summary_results[k]["ndcg"]) / len(test_cases)
            }
            
        return metrics_report