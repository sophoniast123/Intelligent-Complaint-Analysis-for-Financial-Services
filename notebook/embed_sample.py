# embed_sample.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chunking import chunk_text

# Load stratified sample
df = pd.read_parquet("data/cfpb_sample.parquet")

model = SentenceTransformer("all-MiniLM-L6-v2")

records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    chunks = chunk_text(row["clean_narrative"])

    for idx, chunk in enumerate(chunks):
        records.append({
            "complaint_id": row["complaint_id"],
            "product": row["product"],
            "issue": row["issue"],
            "sub_issue": row["sub_issue"],
            "chunk_text": chunk,
            "chunk_index": idx
        })

chunk_df = pd.DataFrame(records)

# Generate embeddings
embeddings = model.encode(
    chunk_df["chunk_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

chunk_df["embedding"] = embeddings.tolist()

# Save for inspection
chunk_df.to_parquet("data/sample_embeddings.parquet", index=False)

print("✅ Task 2 complete: Chunking + Embeddings built")
