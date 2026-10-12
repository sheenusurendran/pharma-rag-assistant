from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read the knowledge base
with open("pharma_knowledge.txt", "r") as file:
    knowledge = file.read()

# Create chunks
chunks = knowledge.split("\n\n")

# Create embeddings for all chunks
embeddings = model.encode(chunks)

question= "How does the ABC drug affect blood pressure?"
# Create embeddings for questions
embeddings_question = model.encode([question])

# Display the results
for i, embedding in enumerate(embeddings):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunks[i])
    print("Embedding:")
    print(embedding)

    # Compare question vector with every chunk vector
similarities = cosine_similarity(
    question_embedding,
    embeddings
)

print("Similarity scores:")
print(similarities)
