from pinecone import ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

from src.config import (
    pc, embeddings,
    INDEX_NAME, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
)


def ingest_documents():

    # -----------------------------
    # Create Pinecone index
    # -----------------------------

    
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

    index = pc.Index(INDEX_NAME)

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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(documents)

    # -----------------------------
    # Embed documents
    # -----------------------------

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