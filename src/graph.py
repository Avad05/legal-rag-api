import os
from typing import TypedDict, List

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


# ============================================================
# Manual Testing
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "Who claims Orbix delivered a defective enterprise resource planning suite?",

        "A commercial suit above five lakh fictional rupees must go to what for 30 days",

        "What is the capital of France?",
    ]

    for question in test_questions:

        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        # ----------------------------------------------------
        # Create initial state
        # ----------------------------------------------------

        state: RAGState = {
            "question": question,
            "chunks": [],
            "quality_check": "",
            "answer": "",
            "citations": [],
        }

        # ----------------------------------------------------
        # Run retrieve node
        # ----------------------------------------------------

        retrieved = retrieve(state)

        state["chunks"] = retrieved["chunks"]

        # ----------------------------------------------------
        # Run quality check node
        # ----------------------------------------------------

        quality = check_quality(state)

        # ----------------------------------------------------
        # Print final result
        # ----------------------------------------------------

        if state["chunks"]:

            highest_score = max(
                chunk["score"]
                for chunk in state["chunks"]
            )

        else:
            highest_score = 0

        print("\nFINAL RESULT")
        print("Highest score:", highest_score)
        print("Quality:", quality["quality_check"])
        print("-" * 80)