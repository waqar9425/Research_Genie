from langgraph.graph import StateGraph, END
from state import GraphState
from tools import *

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("fetch_papers", fetch_papers_node)
    workflow.add_node("extract_texts", extract_texts_node)
    workflow.add_node("chunk_texts", chunk_texts_node)
    workflow.add_node("build_index", build_index_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("evaluate", evaluation_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("no_docs", no_docs_node)

    workflow.set_entry_point("fetch_papers")

    workflow.add_edge("fetch_papers", "extract_texts")
    workflow.add_edge("extract_texts", "chunk_texts")
    workflow.add_edge("chunk_texts", "build_index")
    workflow.add_edge("build_index", "rewrite_query")
    workflow.add_edge("rewrite_query", "retriever")

    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "generate": "generator",
            "no_docs": "no_docs"
        }
    )

    workflow.add_edge("generator", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "end": END,
            "retry": "retry",
            "no_docs": "no_docs"
        }
    )

    workflow.add_edge("retry", "rewrite_query")
    workflow.add_edge("no_docs", END)

    return workflow.compile()
