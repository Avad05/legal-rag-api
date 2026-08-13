from fastapi import FastAPI
from pydantic import BaseModel
from src.ingest import ingest_documents

from src.graph import graph


app = FastAPI(
    title="Legal RAG API",
    description="Legal document question-answering API",
    version="1.0.0",
)


# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


# ------------------------------------------------------------
# Ask endpoint
# ------------------------------------------------------------

@app.post("/ask")
async def ask(request: QuestionRequest):

    initial_state = {
        "question": request.question,
        "chunks": [],
        "quality_check": "",
        "answer": "",
        "citations": [],
    }

    final_state = graph.invoke(
        initial_state,
        config={
            "recursion_limit": 5
        }
    )
    retrieval_scores = [
    {
        "source": chunk["source"],
        "score": chunk["score"]
    }
    for chunk in final_state["chunks"]
    ]

    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "found": final_state["quality_check"] == "good",
        "retrieval_scores": retrieval_scores
    }

@app.post("/ingest")
async def ingest():

    result = ingest_documents()

    return {
        "status": "success",
        "message": result["message"],
        "chunks": result["chunks"]
    }