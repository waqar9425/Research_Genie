from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import traceback

app = FastAPI()

graph = None  # lazy init

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []
    retry_count: int = 0
    max_retries: int = 2

@app.on_event("startup")
def load_graph():
    global graph
    try:
        from graph import build_graph
        graph = build_graph()
        print("✅ LangGraph loaded successfully")
    except Exception:
        print("❌ Failed to load LangGraph")
        traceback.print_exc()
        graph = None

@app.get("/")
def root():
    return {
        "status": "running",
        "graph_loaded": graph is not None
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if graph is None:
        return {"error": "Graph not loaded"}

    result = graph.invoke({
        "query": req.query,
        "chat_history": req.chat_history,
        "retry_count": req.retry_count,
        "max_retries": req.max_retries
    })

    return {
  "answer": result.get("draft_answer", ""),
  "eval": result.get("critique")
}
