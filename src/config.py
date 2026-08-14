import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# ============================================================
# Constants
# ============================================================

INDEX_NAME = "langchain-test-index"
EMBED_DIM = 3072
SCORE_THRESHOLD = 0.60
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5


# ============================================================
# Pinecone
# ============================================================

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(INDEX_NAME)


# ============================================================
# Gemini
# ============================================================

# Used for converting questions / documents into vectors
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Used for reasoning / quality checking
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)
