import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from pinecone import Pinecone


# ============================================================
# Configuration
# ============================================================

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# ============================================================
# Initialize Pinecone
# ============================================================

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"
index = pc.Index(index_name)


# ============================================================
# Initialize Gemini
# ============================================================

# Used for converting questions into vectors
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Used for reasoning / quality checking
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


# ============================================================
# LangGraph State
# ============================================================

class RAGState(TypedDict):
    question: str
    chunks: List[dict]
    quality_check: str
    answer: str
    citations: List[str]


# ============================================================
# Retrieve Node
# ============================================================

def retrieve(state: RAGState):
    """
    Take the user's question, embed it, search Pinecone,
    and return the top 5 relevant chunks.
    """

    question = state["question"]

    # --------------------------------------------------------
    # 1. Convert question into an embedding
    # --------------------------------------------------------

    query_vector = embeddings.embed_query(question)

    # --------------------------------------------------------
    # 2. Search Pinecone
    # --------------------------------------------------------

    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True,
    )

    # --------------------------------------------------------
    # 3. Extract chunks, sources and similarity scores
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. Return chunks to the graph state
    # --------------------------------------------------------

    return {
        "chunks": chunks
    }


# ============================================================
# Quality Check Node
# ============================================================

SCORE_THRESHOLD = 0.60


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

    # --------------------------------------------------------
    # 1. Handle case where nothing was retrieved
    # --------------------------------------------------------

    if not chunks:
        print("No chunks retrieved.")

        return {
            "quality_check": "bad"
        }

    # --------------------------------------------------------
    # 2. Score-based check
    # --------------------------------------------------------

    highest_score = max(
        chunk["score"]
        for chunk in chunks
    )

    print("Highest retrieval score:", highest_score)

    if highest_score < SCORE_THRESHOLD:

        print("Score check: BAD")

        return {
            "quality_check": "bad"
        }

    print("Score check: PASSED")

    # --------------------------------------------------------
    # 3. Build context for Gemini
    # --------------------------------------------------------

    context = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in chunks
    )

    # --------------------------------------------------------
    # 4. Create quality-check prompt
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. Ask Gemini
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    print("RAW RESPONSE:", response)
    print("RESPONSE TEXT:", response.text)

    decision = response.text.strip().lower()

    print("Gemini quality decision:", decision)

    # --------------------------------------------------------
    # 6. Safety fallback
    # --------------------------------------------------------

    if decision not in ["good", "bad"]:
        decision = "bad"

    return {
        "quality_check": decision
    }

def generate_answer(state: RAGState):

    question = state["question"]
    chunks = state["chunks"]

    context = "\n\n".join(
        f"Source: {chunk['source']}\n"
        f"Chunk ID: {chunk['chunk_id']}\n"
        f"{chunk['text']}"
        for chunk in chunks
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

    answer = response.text.strip()

    # Use the highest-scoring chunk as the primary citation
    best_chunk = max(
        chunks,
        key=lambda chunk: chunk["score"]
    )

    citations = [
        {
            "source": best_chunk["source"]
        }
    ]

    return {
        "answer": answer,
        "citations": citations
    }

def no_answer(state: RAGState):

    return {
        "answer": "I cannot find sufficient information to answer this question in the provided documents.",
        "citations": []
    }

def route_quality(state: RAGState):

    return state["quality_check"]

#========================================================
# Connecting Nodes through Langgraph
#========================================================

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