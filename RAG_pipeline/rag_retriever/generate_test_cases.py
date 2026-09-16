import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT")),
}

def generate_dynamic_test_cases(limit: int = 5):
    """Fetches real document IDs and titles/snippets from the DB to form valid evaluation ground truth."""
    sql = """
    SELECT document_id, metadata->>'title' as title, content 
    FROM document_chunks 
    WHERE metadata->>'title' IS NOT NULL AND length(metadata->>'title') > 10
    LIMIT %s;
    """
    
    test_cases = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
            
            for doc_id, title, content in rows:
                # Use the document's actual title as the search query
                test_cases.append({
                    "query": title,
                    "relevant_document_ids": [doc_id]
                })
                
    return test_cases

if __name__ == "__main__":
    cases = generate_dynamic_test_cases(3)
    print("Generated Ground Truth Test Cases:")
    for c in cases:
        print(f"Query: {c['query']}")
        print(f"Target Doc ID: {c['relevant_document_ids']}\n")