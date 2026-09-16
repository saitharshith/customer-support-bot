import os
import sys
import time
import psycopg2
import psycopg2.extras
import ijson
import json
import uuid
import torch 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch.ao.quantization as ao_quant

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 1. Device Selection & Model Initialization
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Loading embedding model on device: '{device}'...")

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': device}
)

# Apply quantization ONLY if running on CPU (PyTorch qint8 is CPU-specific)
if device == "cpu":
    print("Quantizing model for CPU execution...")
    raw_model = embeddings_model._client[0].auto_model  
    quantized_model = ao_quant.quantize_dynamic(
        model=raw_model,
        qconfig_spec={torch.nn.Linear},  
        dtype=torch.qint8
    )
    embeddings_model._client[0].auto_model = quantized_model
    print("Quantization complete.")

# 2. Text Splitter Setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100,
    separators=["\n\n", "\n", "```", ".", " ", ""]
)

hostname = os.getenv("DB_HOST", "localhost")
database = os.getenv("DB_NAME", "techqa")
username = os.getenv("DB_USER", "postgres")
pwd = os.getenv("DB_PASSWORD", "PVatjtyW")
port_id = int(os.getenv("DB_PORT", 5432))

def initialize_database(cursor):
    """Activates vector extension and creates schema if it doesn't exist."""
    print("Checking database schema...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'document_chunks'
        );
    """)
    if not cursor.fetchone()[0]:
        print("Table 'document_chunks' not found. Creating table and indexes...")
        create_script = '''
        CREATE TABLE document_chunks (
            id UUID PRIMARY KEY,
            document_id VARCHAR(255),
            content TEXT,
            embedding VECTOR(384),
            metadata JSONB,
            chunk_index INTEGER
        );
        '''
        cursor.execute(create_script)
        cursor.execute("CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);")
        cursor.execute("CREATE INDEX content_gin_idx ON document_chunks USING GIN (to_tsvector('english', content));")
        print("Schema and indexes created successfully.")
    else:
        print("Schema already exists. Skipping creation.")

def flush_batch(cursor, conn, batch_data):
    """Helper function to execute bulk insert and commit."""
    if not batch_data:
        return 0
    
    insert_query = '''
    INSERT INTO document_chunks (id, document_id, content, embedding, metadata, chunk_index)
    VALUES %s
    ON CONFLICT (id) DO NOTHING;
    '''
    psycopg2.extras.execute_values(cursor, insert_query, batch_data)
    conn.commit()
    return len(batch_data)

conn = None
cur = None

BATCH_SIZE = 1000  # Number of chunks to buffer before flushing to DB

try:
    print("Connecting to database...")
    conn = psycopg2.connect(
        host=hostname, dbname=database, user=username, password=pwd, port=port_id
    )
    cur = conn.cursor()
    
    initialize_database(cur)
    conn.commit()

    json_file_path = r"IBM-techqa-dataset\TechQA\TechQA\technote_corpus\full_technote_collection.sections.json"
    print("Streaming full dataset using ijson...")

    start_time = time.time()
    total_docs_processed = 0
    total_chunks_inserted = 0
    db_batch_buffer = []

    with open(json_file_path, 'rb') as f:
        parser = ijson.kvitems(f, '')
        
        for doc_key, doc in parser:
            total_docs_processed += 1
            
            doc_id = doc.get('id', doc_key)
            title = doc.get('title', 'Untitled')
            content = doc.get('text', doc.get('document_text', ''))
            
            if not content:
                continue
                
            chunks = text_splitter.split_text(content)
            if not chunks:
                continue
                
            # Batch embedding computation for all chunks in the document
            vectors = embeddings_model.embed_documents(chunks)
            
            for chunk_idx, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                metadata = json.dumps({"title": title, "source": "ibm_techqa"})
                
                db_batch_buffer.append((
                    str(uuid.uuid4()),
                    doc_id,
                    chunk_text,
                    str(vector),
                    metadata,
                    chunk_idx
                ))

            # Flush batch when threshold reached
            if len(db_batch_buffer) >= BATCH_SIZE:
                inserted_count = flush_batch(cur, conn, db_batch_buffer)
                total_chunks_inserted += inserted_count
                db_batch_buffer = []

            # Progress Logging
            if total_docs_processed % 500 == 0:
                elapsed = time.time() - start_time
                docs_per_sec = total_docs_processed / elapsed
                print(f"[Progress] Docs Processed: {total_docs_processed} | Total Chunks Ingested: {total_chunks_inserted} | Speed: {docs_per_sec:.2f} docs/sec")

        # Flush any remaining items in the buffer
        if db_batch_buffer:
            inserted_count = flush_batch(cur, conn, db_batch_buffer)
            total_chunks_inserted += inserted_count

        elapsed_total = time.time() - start_time
        print("\n==================================================")
        print("✅ FULL INGESTION COMPLETE")
        print(f"Total Documents Processed: {total_docs_processed}")
        print(f"Total Chunks Uploaded:   {total_chunks_inserted}")
        print(f"Total Time Elapsed:       {elapsed_total / 60:.2f} minutes")
        print("==================================================")

except Exception as error:
    print(f"\n❌ Error occurred during execution: {error.__class__.__name__} - {error}")
    if conn:
        conn.rollback()

finally:
    if cur:
        cur.close()
    if conn:
        conn.close()
    print("Database connection closed.")