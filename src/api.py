import time
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingest import ingest_documents
from src.graph import graph
from src.config import index, INDEX_NAME

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal_rag_api")

app = FastAPI(
    title="Legal RAG API",
    description="Legal document question-answering system powered by LangGraph, Gemini, and Pinecone",
    version="1.0.0",
)

# CORS middleware for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The legal question to answer based on the document corpus.",
        example="What is the notice period in the Bluecrest Analytics employment agreement?"
    )


class CitationItem(BaseModel):
    source: str = Field(..., description="Source document filename")


class RetrievalScoreItem(BaseModel):
    source: str = Field(..., description="Source document filename")
    score: float = Field(..., description="Cosine similarity score")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer or fallback response")
    citations: List[CitationItem] = Field(default_factory=list, description="Source document citations")
    found: bool = Field(..., description="Whether sufficient relevant evidence was found")
    retrieval_scores: List[RetrievalScoreItem] = Field(default_factory=list, description="Scores of top retrieved chunks")
    response_time_ms: float = Field(..., description="Execution time in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks: int


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, summary="Ask a question")
async def ask(request: QuestionRequest):
    start_time = time.time()
    logger.info("Received question request: %s", request.question)

    initial_state = {
        "question": request.question,
        "chunks": [],
        "quality_check": "",
        "answer": "",
        "citations": [],
    }

    try:
        # Use graph.ainvoke for non-blocking async execution
        final_state = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 5}
        )
    except Exception as e:
        logger.error("Error executing RAG graph: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

    retrieval_scores = [
        {
            "source": chunk["source"],
            "score": chunk["score"]
        }
        for chunk in final_state.get("chunks", [])
    ]

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logger.info("Answer generated in %.2f ms (found=%s)", elapsed_ms, final_state.get("quality_check") == "good")

    return {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "found": final_state.get("quality_check") == "good",
        "retrieval_scores": retrieval_scores,
        "response_time_ms": elapsed_ms,
    }


@app.post("/ingest", response_model=IngestResponse, summary="Ingest document corpus")
async def ingest(
    force: bool = Query(
        False,
        description="Force re-ingestion even if index is already populated"
    )
):
    try:
        if not force:
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            if total_vectors > 0:
                return {
                    "status": "skipped",
                    "message": f"Index '{INDEX_NAME}' already contains {total_vectors} vectors. Use ?force=true to re-ingest.",
                    "chunks": total_vectors,
                }

        result = ingest_documents()
        return {
            "status": "success",
            "message": result["message"],
            "chunks": result["chunks"]
        }
    except Exception as e:
        logger.error("Error during document ingestion: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )