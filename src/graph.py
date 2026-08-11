import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone


# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"
index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


# -----------------------------
# LangGraph State
# -----------------------------

class RAGState(TypedDict):
    question: str
    chunks: List[dict]
    quality_check: str
    answer: str
    citations: List[str]


# -----------------------------
# Retrieve Node
# -----------------------------

def retrieve(state: RAGState):

    question = state["question"]

    # Convert question into an embedding
    query_vector = embeddings.embed_query(question)

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    # Extract retrieved chunks
    chunks = []

    for match in results["matches"]:
        chunks.append({
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "score": match["score"]
        })

    return {
        "chunks": chunks
    }


# -----------------------------
# Test Retrieve Node
# -----------------------------

question = (
    "Who vacated Unit 4B on 31 March 2025"
)

initial_state: RAGState = {
    "question": question,
    "chunks": [],
    "quality_check": "",
    "answer": "",
    "citations": []
}

result = retrieve(initial_state)

print("\nRetrieved chunks:\n")

for chunk in result["chunks"]:
    print("Source:", chunk["source"])
    print("Score:", chunk["score"])
    print("Text:", chunk["text"])
    print("-" * 80)
