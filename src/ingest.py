import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"
if pc.has_index(index_name):
    pc.delete_index(index_name)

EMBED_DIM = 3072

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# --- Load & split docs ---
loader = DirectoryLoader("../corpus", glob="./*.md", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# --- Embed manually ---
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

texts = [doc.page_content for doc in docs]
vectors_list = embeddings.embed_documents(texts)


# Sanity check the dimension actually matches the index
if len(vectors_list[0]) != EMBED_DIM:
    raise ValueError(
        f"Embedding dim {len(vectors_list[0])} doesn't match index dim {EMBED_DIM}. "
        "Recreate the index with the correct dimension."
    )

# --- Build upsert payload ---
records = []
for i, (doc, vector) in enumerate(zip(docs, vectors_list)):
    records.append({
        "id": hashlib.sha256(f"{doc.metadata['source']}-{i}".encode()).hexdigest(),
        "values": vector,
        "metadata": {
            "text": doc.page_content,
            **doc.metadata,
        },
    })

# --- Upsert in batches (Pinecone recommends <=100 per batch) ---
batch_size = 100
for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    index.upsert(vectors=batch)

print(f"Successfully added {len(docs)} chunks from directory into Pinecone.")

question = "Forced buyout claim"

query_vector = embeddings.embed_query(question)

results = index.query(
    vector=query_vector,
    top_k=1,
    include_metadata=True
)

print(results)