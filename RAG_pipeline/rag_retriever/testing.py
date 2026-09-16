from retriever import DenseRerankRetriever
from evaluation import IREvaluator
from generate_test_cases import generate_dynamic_test_cases

if __name__ == "__main__":
    print("--- Starting Retrieval Evaluation Suite ---")
    
    # Generate test cases from real database entries
    TEST_CASES = generate_dynamic_test_cases(limit=10)
    
    if not TEST_CASES:
        print("Error: No test cases found. Ensure database ingestion is complete.")
        exit()

    retriever = DenseRerankRetriever(candidate_top_n=20, rerank_top_k=5)
    evaluator = IREvaluator()
    
    # 1. Inspect single query result
    sample_case = TEST_CASES[0]
    print(f"\n[Test Query]: '{sample_case['query']}'")
    print(f"[Expected Doc ID]: {sample_case['relevant_document_ids']}")
    
    docs = retriever.retrieve(sample_case['query'])
    
    print("\n--- Top 3 Retrieved Documents ---")
    for rank, doc in enumerate(docs[:3], start=1):
        match_flag = "MATCH!" if doc['document_id'] in sample_case['relevant_document_ids'] else ""
        print(f"Rank {rank} | Doc ID: {doc['document_id']} {match_flag} | Score: {doc['rerank_score']:.4f}")
        print(f"Snippet: {doc['content'][:100]}...\n")

    # 2. Compute Metrics across batch
    print("--- Computing IR Metrics across Test Suite ---")
    results = evaluator.evaluate_batch(TEST_CASES, retriever, k_values=[3, 5])
    
    print("\n================ EVALUATION REPORT ================")
    for cutoff, scores in results.items():
        print(f"\nMetrics @ {cutoff}:")
        for metric_name, score in scores.items():
            print(f"  - {metric_name}: {score:.4f}")
    print("===================================================")