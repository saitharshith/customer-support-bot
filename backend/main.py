import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Add project root directory to sys.path to resolve imports cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas import QueryRequest, QueryResponse, DocumentChunk
from RAG_pipeline.rag_agent.state_graph import build_graph

app_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_graph
    print("Initializing LangGraph execution engine for Version 1...")
    app_graph = build_graph()
    yield
    print("Shutting down application...")

app = FastAPI(
    title="Technical Support Agent",
    version="1.0.0",
    lifespan=lifespan
)

# 1. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev environment; restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Serve index.html at the root URL '/'
@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(ROOT_DIR, "frontend/index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found in root directory.")
    return FileResponse(index_path)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """System health check endpoint for container orchestrators (Docker/K8s)."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_rag_agent(payload: QueryRequest):
    """
    Executes the RAG agent pipeline directly:
    1. Builds initial state.
    2. Runs the LangGraph retrieval & generation workflow.
    3. Returns the output payload.
    """
    global app_graph
    if not app_graph:
        raise HTTPException(status_code=500, detail="Agent graph is not initialized.")

    initial_state = {
        "question": payload.question,
        "context": [],
        "max_confidence_score": 0.0,
        "confidence_passed": False,
        "generation": None,
        "escalated": False,
        "error": None
    }

    try:
        final_state = app_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent graph execution failed: {str(e)}")

    # Map raw context rows to Pydantic chunks schema
    formatted_context = [
        DocumentChunk(
            chunk_id=str(doc.get("chunk_id", "")),
            document_id=str(doc.get("document_id", "")),
            content=doc.get("content", ""),
            metadata=doc.get("metadata", {}),
            rerank_score=float(doc.get("rerank_score", 0.0))
        )
        for doc in final_state.get("context", [])
    ]

    response_data = {
        "question": payload.question,
        "generation": final_state.get("generation", "No generation output produced."),
        "escalated": final_state.get("escalated", False),
        "max_confidence_score": float(final_state.get("max_confidence_score", 0.0)),
        "context": [chunk.model_dump() for chunk in formatted_context]
    }

    return QueryResponse(**response_data)