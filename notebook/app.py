# app.py
from rag import retrieve_complaints, generate_answer

print("💬 Intelligent Complaint Analysis Chatbot")
print("Type 'exit' to quit\n")

while True:
    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    product = input("Filter by product (or press Enter): ")
    product = product if product else None

    docs = retrieve_complaints(question, product_filter=product)
    answer = generate_answer(question, docs)

    print("\n🧠 Answer:")
    print(answer)
    print("-" * 80)
