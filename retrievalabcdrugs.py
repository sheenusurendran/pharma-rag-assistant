import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 3. Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# 4. Get collection
collection = client.get_collection(
    name="pharma_knowledge"
)

# 5. User question
question = "What is the starting dose of Drug ABC?"

# 6. Convert question into embedding
question_embedding = embedding_model.encode([question])

# 7. Retrieve Top 4 candidates from ChromaDB
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=4,
    include=["documents", "distances"]
)

# 8. Get retrieved chunks
chunks = results["documents"][0]

print("\nQuestion:")
print(question)

print("\nChromaDB retrieved:")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)

# 9. Create question + chunk pairs
pairs = [[question, chunk] for chunk in chunks]

# 10. Rerank
scores = reranker.predict(pairs)

# 11. Combine chunks with scores
ranked_results = list(zip(chunks, scores))

# 12. Sort highest score first
ranked_results.sort(key=lambda x: x[1], reverse=True)

print("\n\nAfter Reranking:")

for i, (chunk, score) in enumerate(ranked_results):
    print("\n-------------------------")
    print("Rank:", i + 1)
    print("Reranker score:", score)
    print("Chunk:")
    print(chunk)