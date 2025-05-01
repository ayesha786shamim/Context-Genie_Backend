import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

# Configure Gemini with your API key
genai.configure(api_key="#################################")

# Automatically select a valid model that supports generateContent
def get_supported_model():
    models = list(genai.list_models())
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Using model: {m.name}")
            # Check for deprecated models and avoid them
            if "gemini-1.0" in m.name.lower():
                print(f"❌ Model {m.name} is deprecated. Skipping.")
                continue
            if "gemini-1.5-flash" in m.name.lower():  
                return genai.GenerativeModel(m.name)
    raise Exception("❌ No supported model found that is not deprecated.")

# Initialize model
model = get_supported_model()

# Classify the type of query
def classify_query(query):
    query = query.lower().strip()
    if "?" in query or len(query.split()) <= 5:
        return "question_answering"
    elif "generate" in query or "create" in query:
        return "content_generation"
    else:
        return "question_answering"  # Fallback to QA for unknown types

# Generate answer using Gemini
def generate_answer(context_chunks, query):
    try:
        task_type = classify_query(query)
        print(f"🧠 Detected task: {task_type}")

        if task_type == "question_answering":
            context_text = "\n\n".join([f"(Page {c['page']}): {c['text']}" for c in context_chunks])

            prompt = f"""
            Answer the question based on the context below:

            Context:
            {context_text}

            Question: {query}
            Answer:
            """

            # Get the response from the model
            response = model.generate_content(prompt)
            answer = response.text.strip()

            # Integrated page numbers directly in the response
            formatted_answer = f"{answer} (Referenced from pages: " + ", ".join([str(c['page']) for c in context_chunks]) + ")"

            return formatted_answer

        elif task_type == "content_generation":
            response = model.generate_content(query)
            return response.text.strip()

        else:
            return "❌ Sorry, I cannot process this request."
    except Exception as e:
        print(f"❌ Error: {e}")
        return "❌ An error occurred while generating the answer."

# FastAPI application instance
app = FastAPI()

# Pydantic model for incoming requests
class QueryRequest(BaseModel):
    context_chunks: list
    query: str

# Route for generating answers
@app.post("/generate-answer")
async def generate_answer_route(request: QueryRequest):
    return {"answer": generate_answer(request.context_chunks, request.query)}
