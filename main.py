from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from document_processor import load_pdf_with_page_numbers
from retriever import VectorStore
from generator import generate_answer

app = FastAPI()

# Enable CORS so frontend (React) can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev — restrict in production!
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body format
class QueryRequest(BaseModel):
    question: str

# Load PDF and create the vector store once on server start
chunks = load_pdf_with_page_numbers("data/Virtualization.pdf")
store = VectorStore(chunks)

# Define the API endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    # question = request.question

    # # Retrieve most relevant chunks
    # relevant_chunks = store.search(question)

    # # Generate answer from those chunks
    # answer = generate_answer(relevant_chunks, question)

    # return {"answer": answer}  # Return as JSON
    question = request.question
    print(f"📥 Received question: {question}")

    try:
        # Search for relevant document chunks
        relevant_chunks = store.search(question)
        print(f"🔍 Found {len(relevant_chunks)} relevant chunks")

        # Generate a final answer using LLM or Gemini/DeepSeek API
        answer = generate_answer(relevant_chunks, question)
        print(f"💬 Generated answer: {answer}")

        return {"answer": answer}
    except Exception as e:
        print("❌ Error processing question:", str(e))
        return {"answer": "Something went wrong while answering the question."}

