import numpy as np
from sentence_transformers import SentenceTransformer

import sys
############# importing sqlite >3.35 before chroma db which requires ir
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import sqlite3
print(sqlite3.sqlite_version)

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
SIMILARITY_THRESHOLD = 0.3  # Lower threshold
QUERY_FILE = "Query3.txt"  # File containing queries
TEXT_FILE = "Transcription.txt"  # File containing the transcript
CHROMA_DB_PATH = "/home/pgupt60/RagBasedVideoAnalysis/RagBasedVideoAnalysis/mm_vdb"

# Load the embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create or load a collection
collection = chroma_client.get_or_create_collection(
    name="hybrid_search",
    metadata={"distance_function": "cosine"}
)

# Load and process the transcript
with open(TEXT_FILE, "r", encoding="utf-8") as file:
    large_text = file.read()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = text_splitter.split_text(large_text)

# Convert text chunks into embeddings
chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

# Store text chunks and embeddings in ChromaDB
for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings.tolist())):
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("📌 Number of stored elements in ChromaDB:", len(collection.get()["ids"]))

# Load queries from the query file
with open(QUERY_FILE, "r", encoding="utf-8") as file:
    queries = [line.strip() for line in file.readlines() if line.strip()]

# Process each query
matched_queries = []
unmatched_queries = []

for query_text in queries:
    print(f"\n🔍 **Processing Query:** {query_text}")
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10  # Retrieve more results
    )
    
    # Check if results are valid
    if results and results["embeddings"]:
        retrieved_embeddings = np.array(results["embeddings"][0])
        query_embedding_np = np.array(query_embedding)
        
        # Calculate similarity scores
        similarity_scores = np.dot(retrieved_embeddings, query_embedding_np) / (
            np.linalg.norm(retrieved_embeddings, axis=1) * np.linalg.norm(query_embedding_np)
        )
        
        # Filter and sort results
        filtered_results = sorted(
            [(doc, score) for doc, score in zip(results["documents"][0], similarity_scores) if score >= SIMILARITY_THRESHOLD],
            key=lambda x: x[1], reverse=True
        )[:3]  # Retrieve top 3 matches
        
        # Store results
        if filtered_results:
            for doc, score in filtered_results:
                matched_queries.append((query_text, doc, score))
        else:
            unmatched_queries.append(query_text)
    else:
        print("⚠️ No embeddings found for query.")
        unmatched_queries.append(query_text)

# Save results to files
with open("matched_queries.txt", "w", encoding="utf-8") as file:
    for query, doc, score in matched_queries:
        file.write(f"Query: {query}\nMatch: {doc}\nScore: {score:.4f}\n\n")

with open("unmatched_queries.txt", "w", encoding="utf-8") as file:
    for query in unmatched_queries:
        file.write(f"Query: {query}\n")

print("\n✅ Results saved to `matched_queries.txt` and `unmatched_queries.txt`")
