from typing import TypedDict, List, Dict

class GraphState(TypedDict, total=False):
    query: str
    rewritten_query: str

    chat_history: List[Dict[str, str]]

    papers: List[Dict]
    pdf_texts: List[str]
    chunks: List[str]
    metadata: List[Dict]

    index: any
    retriever: any
    docs: List[str]

    draft_answer: str
    final_answer: str
    critique: str

    retry_count: int
    max_retries: int

