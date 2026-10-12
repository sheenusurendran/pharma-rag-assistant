import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Read knowledge base
with open("pharma_knowledge.txt", "r") as file:
    knowledge = file.read()

# 3. Split knowledge into chunks
chunks = knowledge.split("\n\n")

print("Number of chunks:", len(chunks))

# 4. Create embeddings for all chunks
embeddings = model.encode(chunks)

# 5. Create persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# 6. Create/get collection
collection = client.get_or_create_collection(
    name="pharma_knowledge"
)

# 7. Store chunks + embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Knowledge stored in ChromaDB.")
print("Documents stored:", collection.count())