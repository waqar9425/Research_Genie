from typing import TypedDict, List, Dict
import faiss

class GraphState(TypedDict, total=False):
    query: str
    rewritten_query: str
    chat_history: List[str]

    papers: List[Dict]
    pdf_texts: List[str]
    chunks: List[str]
    metadata: List[Dict]

    index: "faiss.IndexFlatL2"
    retriever: any
    docs: List[str]

    answer: str
    eval_result: str

    retry_count: int
    max_retries: int
