import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from src.config import index, embeddings, llm, SCORE_THRESHOLD, TOP_K

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal_rag_graph")


def _extract_text(response) -> str:
    """
    Extract plain text from an LLM response.

    Handles both:
    - response.content as a plain str
    - response.content as a list of content parts
      (e.g. [{"type": "text", "text": "...", ...}])
    - Fallback to response.text or str(response)
    """
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        if parts:
            return " ".join(parts).strip()

    # Final fallback
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    return str(response).strip()


# ============================================================
# LangGraph State
# ============================================================

class RAGState(TypedDict):
    question: str
    chunks: List[dict]
    quality_check: str
    answer: str
    citations: List[dict]


# ============================================================
# Retrieve Node
# ============================================================

def retrieve(state: RAGState):
    """
    Take the user's question, embed it, search Pinecone,
    and return the top relevant chunks.
    """
    question = state["question"]
    logger.info("Retrieving chunks for question: '%s'", question)

    # 1. Convert question into an embedding
    query_vector = embeddings.embed_query(question)

    # 2. Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    # 3. Extract chunks, sources and similarity scores
    chunks = []
    for match in results["matches"]:
        chunks.append(
            {
                "chunk_id": match["metadata"]["chunk_id"],
                "text": match["metadata"]["text"],
                "source": match["metadata"]["source"],
                "score": match["score"],
            }
        )

    logger.info("Retrieved %d chunks from Pinecone.", len(chunks))

    return {
        "chunks": chunks
    }


# ============================================================
# Quality Check Node
# ============================================================

def check_quality(state: RAGState):
    """
    Check whether the retrieved chunks are good enough
    to proceed to answer generation.

    First:
        Use Pinecone similarity score as a cheap filter.

    Second:
        If the score is good enough, ask Gemini whether
        the retrieved context actually contains enough
        information to answer the question.
    """
    chunks = state["chunks"]

    # 1. Handle case where nothing was retrieved
    if not chunks:
        logger.warning("No chunks retrieved.")
        return {"quality_check": "bad"}

    # 2. Score-based check
    highest_score = max(chunk["score"] for chunk in chunks)
    logger.info("Highest retrieval score: %.4f (Threshold: %.2f)", highest_score, SCORE_THRESHOLD)

    if highest_score < SCORE_THRESHOLD:
        logger.info("Score check: BAD - Below threshold")
        return {"quality_check": "bad"}

    logger.info("Score check: PASSED")

    # Filter for relevant chunks for context evaluation
    relevant_chunks = [c for c in chunks if c["score"] >= SCORE_THRESHOLD]
    context_chunks = relevant_chunks if relevant_chunks else chunks

    # 3. Build context for Gemini
    context = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in context_chunks
    )

    # 4. Create quality-check prompt
    prompt = f"""
    You are evaluating retrieved evidence for a legal question.

    Question:
    {state["question"]}

    Retrieved context:
    {context}

    Does the retrieved context contain enough information
    to answer the question accurately?

    Return ONLY one word:
    good
    or
    bad
    """

    # 5. Ask Gemini
    response = llm.invoke(prompt)
    decision = _extract_text(response).lower()

    logger.info("Gemini quality decision: '%s'", decision)

    # 6. Safety fallback
    if decision not in ["good", "bad"]:
        logger.warning("Unexpected decision '%s', falling back to 'bad'", decision)
        decision = "bad"

    return {
        "quality_check": decision
    }


# ============================================================
# Generate Answer Node
# ============================================================

def generate_answer(state: RAGState):
    question = state["question"]
    chunks = state["chunks"]

    # Filter out low-scoring chunks below threshold for cleanest context
    relevant_chunks = [c for c in chunks if c["score"] >= SCORE_THRESHOLD]
    if not relevant_chunks:
        relevant_chunks = chunks  # Fallback to all retrieved if none exceed threshold

    context = "\n\n".join(
        f"Source: {chunk['source']}\n"
        f"Chunk ID: {chunk['chunk_id']}\n"
        f"{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
You are a legal document question-answering assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.
Do not invent facts.

Question:
{question}

Context:
{context}

Provide a concise and accurate answer.
"""

    response = llm.invoke(prompt)
    answer = _extract_text(response)

    # Collect unique citations from relevant chunks ordered by score
    seen_sources = set()
    citations = []
    for chunk in sorted(relevant_chunks, key=lambda x: x["score"], reverse=True):
        source = chunk["source"]
        if source not in seen_sources:
            seen_sources.add(source)
            citations.append({"source": source})

    logger.info("Generated answer with %d citation(s).", len(citations))

    return {
        "answer": answer,
        "citations": citations
    }


# ============================================================
# Fallback No-Answer Node
# ============================================================

def no_answer(state: RAGState):
    logger.info("Routing to no_answer fallback.")
    return {
        "answer": "I cannot find sufficient information to answer this question in the provided documents.",
        "citations": []
    }


def route_quality(state: RAGState):
    return state["quality_check"]


# ============================================================
# Connecting Nodes through LangGraph
# ============================================================

workflow = StateGraph(RAGState)

workflow.add_node("retriever", retrieve)
workflow.add_node("qualityChecker", check_quality)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("no_answer", no_answer)

workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "qualityChecker")
workflow.add_conditional_edges(
    "qualityChecker",
    route_quality,
    {
        "good": "generate_answer",
        "bad": "no_answer"
    }
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("no_answer", END)

graph = workflow.compile()


# ============================================================
# CLI test (only runs when executed directly)
# ============================================================

if __name__ == "__main__":
    test_input = {
        "question": "Northfield offered to pay 70% of open invoices if who drops the damage counterclaim?",
        "chunks": [],
        "quality_check": "",
        "answer": "",
        "citations": []
    }

    final_state = graph.invoke(test_input, config={"recursion_limit": 5})

    print("\nFINAL ANSWER:")
    print(final_state["answer"])

    print("\nQUALITY:")
    print(final_state["quality_check"])

    print("\nCITATIONS:")
    for citation in final_state["citations"]:
        print(citation)