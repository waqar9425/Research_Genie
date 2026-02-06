from langgraph.graph import StateGraph, END
from state import GraphState
from tools import *

def build_graph():
    workflow = StateGraph(GraphState)

    # --- Nodes ---
    workflow.add_node("fetch_papers", fetch_papers_node)
    workflow.add_node("extract_texts", extract_texts_node)
    workflow.add_node("chunk_texts", chunk_texts_node)
    workflow.add_node("build_index", build_index_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retriever", retriever_node)

    # Step-2 agents
    workflow.add_node("answer_agent", generator_node)
    workflow.add_node("critic_agent", evaluation_node)
    workflow.add_node("improve_agent", retry_node)

    workflow.add_node("no_docs", no_docs_node)

    # --- Entry ---
    workflow.set_entry_point("fetch_papers")

    # --- Retrieval pipeline ---
    workflow.add_edge("fetch_papers", "extract_texts")
    workflow.add_edge("extract_texts", "chunk_texts")
    workflow.add_edge("chunk_texts", "build_index")
    workflow.add_edge("build_index", "rewrite_query")
    workflow.add_edge("rewrite_query", "retriever")

    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "generate": "answer_agent",
            "no_docs": "no_docs"
        }
    )

    # --- Step-2 reflection loop ---
    workflow.add_edge("answer_agent", "critic_agent")

    workflow.add_conditional_edges(
        "critic_agent",
        route_after_evaluation,
        {
            "end": END,
            "retry": "improve_agent"
        }
    )

    workflow.add_edge("improve_agent", "answer_agent")

    workflow.add_edge("no_docs", END)

    return workflow.compile()
