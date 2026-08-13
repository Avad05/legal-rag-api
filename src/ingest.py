import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"
EMBED_DIM = 3072


def ingest_documents():

    # -----------------------------
    # Create Pinecone index
    # -----------------------------

    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

    index = pc.Index(index_name)

    # -----------------------------
    # Load & split documents
    # -----------------------------

    loader = DirectoryLoader(
        "corpus",
        glob="./*.md",
        loader_cls=TextLoader
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # -----------------------------
    # Embed documents
    # -----------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    texts = [doc.page_content for doc in docs]

    vectors_list = embeddings.embed_documents(texts)

    # -----------------------------
    # Check embedding dimension
    # -----------------------------

    if len(vectors_list[0]) != EMBED_DIM:
        raise ValueError(
            f"Embedding dim {len(vectors_list[0])} doesn't match "
            f"index dim {EMBED_DIM}. "
            "Recreate the index with the correct dimension."
        )

    # -----------------------------
    # Build Pinecone records
    # -----------------------------

    records = []

    for i, (doc, vector) in enumerate(zip(docs, vectors_list)):

        chunk_id = hashlib.sha256(
            f"{doc.metadata['source']}-{i}".encode()
        ).hexdigest()

        records.append({
            "id": chunk_id,
            "values": vector,
            "metadata": {
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "source": doc.metadata["source"],
            },
        })

    # -----------------------------
    # Upsert in batches
    # -----------------------------

    batch_size = 100

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert(vectors=batch)

    return {
        "message": (
            f"Successfully added {len(docs)} chunks "
            "from directory into Pinecone."
        ),
        "chunks": len(docs)
    }


# -----------------------------
# CLI execution
# -----------------------------

if __name__ == "__main__":

    result = ingest_documents()

    print(result["message"])