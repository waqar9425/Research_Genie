#Define graph
from typing import TypedDict, List, Dict
import faiss
import numpy as np

from dowload_papers import *
from text_extraction import *
from llm_config import llm, embeddings_model
from state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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



# tools data collections

def rewrite_query_node(state: GraphState) -> dict:
    history = "\n".join(state.get("chat_history", []))
    query = state["query"]

    prompt = f"""
You are a query rewriting assistant for document retrieval.

Conversation so far:
{history}

User question:
{query}

Rewrite the question to be:
- precise
- standalone
- optimized for retrieving academic papers

Rewritten query:
"""

    rewritten = llm.invoke(prompt).content.strip()
    print("[INFO] Rewritten query:", rewritten)

    return {"rewritten_query": rewritten}



def fetch_papers_node(state: GraphState):
    # Do arXiv fetch/download (same as your function)
    papers = fetch_arxiv_papers(query=state["query"], max_results=1)
    print("[INFO] fetched papers:", [p["local_pdf"] for p in papers if "local_pdf" in p])
    return {"papers": papers}

def extract_texts_node(state: GraphState):
    texts = []
    for p in state["papers"]:
        path = p.get("local_pdf")
        if path and os.path.exists(path):
            text = extract_text_pdfplumber(path)
            print(f"[INFO] extracted text length for {path}: {len(text) if text else 0}")
            if text:
                texts.append(text)
    return {"pdf_texts": texts}


# tools for chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

# 3. Chunk Text
def chunk_texts_node(state: GraphState):
    all_chunks, all_meta = [], []
    for idx, txt in enumerate(state["pdf_texts"]):
        chks = text_splitter.split_text(txt)
        all_chunks.extend(chks)
        all_meta.extend([{"source": state["papers"][idx]["title"]}] * len(chks))
        print(f"[INFO] chunked document {idx} into {len(chks)} chunks")
    return {"chunks": all_chunks, "metadata": all_meta}



def build_index_node(state: GraphState):
    chunks = state.get("chunks", [])
    metas = state.get("metadata", [])
    if not chunks:
        print("[WARN] No chunks to index")
        return {"retriever": None, "index": None}

    vectorstore = FAISS.from_texts(
        texts=chunks, embedding=embeddings_model, metadatas=metas
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print(f"[INFO] FAISS index built with {len(chunks)} chunks")

    return {"index": vectorstore, "retriever": retriever}



def retriever_node(state: GraphState):
    retr = state.get("retriever")
    if retr is None:
        print("[ERROR] Retriever is missing! State keys:", list(state.keys()))
        return {"docs": []}
    #docs = retr.get_relevant_documents(state["query"])
    docs = retr.invoke(state["query"])

    print(f"[INFO] Retrieved {len(docs)} chunks")
    return {"docs": [d.page_content for d in docs]}


# additional node to handle retrieval fallbacks
def route_after_retrieval(state: GraphState) -> str:
    if not state["docs"]:
        return "no_docs"
    return "generate"

def no_docs_node(state: GraphState) -> dict:
    return {
        "answer": "I couldn't find relevant documents to answer your question."
    }


def generator_node(state: GraphState):
    docs = state.get("docs", [])
    q = state["query"]

    if not docs:
        return {"draft_answer": ""}

    context = "\n\n".join(docs)

    prompt = f"""
You are an expert research assistant.

Question:
{q}

Answer using ONLY the context below.
Do not hallucinate.

Context:
{context}

Draft Answer:
"""

    resp = llm.invoke(prompt)
    return {"draft_answer": resp.content}


def evaluation_node(state: GraphState) -> dict:
    context = "\n\n".join(state["docs"])

    prompt = f"""
You are a strict reviewer.

Question:
{state['query']}

Context:
{context}

Answer:
{state['draft_answer']}

Evaluate the answer for:
1. Grounding in context
2. Completeness
3. Clarity

Reply in this format:

PASS
or
FAIL: <short explanation of what is wrong or missing>
"""

    critique = llm.invoke(prompt).content.strip()

    return {
        "critique": critique
    }



def route_after_evaluation(state: GraphState) -> str:
    critique = state["critique"]

    if critique.startswith("PASS"):
        return "end"

    if state["retry_count"] < state["max_retries"]:
        return "retry"

    return "end"

def retry_node(state: GraphState) -> dict:
    retries = state.get("retry_count", 0) + 1
    print(f"[WARN] Improving answer (attempt {retries})")

    prompt = f"""
Improve the answer based on reviewer feedback.

Question:
{state['query']}

Previous Answer:
{state['draft_answer']}

Critique:
{state['critique']}

Improved Answer:
"""

    improved = llm.invoke(prompt).content

    return {
        "draft_answer": improved,
        "retry_count": retries
    }


