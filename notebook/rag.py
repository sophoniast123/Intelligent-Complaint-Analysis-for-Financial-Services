# rag.py
import chromadb
from sentence_transformers import SentenceTransformer
import openai

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma
client = chromadb.Client()
collection = client.get_collection("complaints")

openai.api_key = "YOUR_API_KEY"

def retrieve_complaints(query, product_filter=None, k=5):
    query_embedding = embedder.encode(query).tolist()

    where_clause = None
    if product_filter:
        where_clause = {"product_category": product_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause
    )

    return results["documents"][0]

def generate_answer(question, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are an internal AI assistant for a financial company.
Answer the question ONLY using the customer complaints below.
Summarize recurring themes and root causes clearly.

Customer Complaints:
{context}

Question:
{question}

Answer:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
