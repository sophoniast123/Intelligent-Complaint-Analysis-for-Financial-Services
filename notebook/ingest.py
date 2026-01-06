# ingest.py
import pandas as pd
import chromadb

# Load dataset
df = pd.read_parquet("data/complaint_embeddings.parquet")

# Initialize Chroma
client = chromadb.Client()
collection = client.create_collection(
    name="complaints",
    metadata={"hnsw:space": "cosine"}
)

# Insert data (batching is important)
batch_size = 1000

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]

    collection.add(
        ids=batch.index.astype(str).tolist(),
        embeddings=batch["embedding"].tolist(),
        documents=batch["chunk_text"].tolist(),
        metadatas=batch[[
            "product_category",
            "issue",
            "sub_issue",
            "company",
            "state",
            "date_received"
        ]].to_dict("records")
    )

print("✅ Vector database built successfully")
