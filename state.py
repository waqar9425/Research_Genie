from typing import TypedDict, List, Dict
import faiss

class GraphState(TypedDict, total=False):
    # existing
    query: str
    rewritten_query: str
    docs: List[str]

    # agent outputs
    draft_answer: str
    critique: str
    final_answer: str

    retry_count: int
    max_retries: int
