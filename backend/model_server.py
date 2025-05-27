import json
from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag_pipeline import SimpleRAG
from rag.rag_with_llm import generate_answer

app = FastAPI()

# Ladda chunks och modell en gång vid start
with open("chunking/best_chunking.json", "r", encoding="utf-8") as f:
    data = json.load(f)
chunks = data["chunks"]
rag_model = SimpleRAG(chunks)

class QuestionRequest(BaseModel):
    message: str
    history: list

@app.post("/answer")
def answer_question(req: QuestionRequest):
    context = rag_model.query(req.message, top_k=3)
    answer = generate_answer(req.message, context)
    return {"answer": answer}