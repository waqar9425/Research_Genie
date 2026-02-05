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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"[INFO] FAISS index built with {len(chunks)} chunks")

    return {"index": vectorstore, "retriever": retriever}



def retriever_node(state: GraphState):
    retr = state.get("retriever")
    if retr is None:
        return {"docs": []}

    query = state.get("rewritten_query", state["query"])
    docs = retr.get_relevant_documents(query)

    print(f"[INFO] Retrieved {len(docs)} chunks using query: {query}")
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
        return {"answer": "Sorry, I couldn't find relevant documents."}

    context = "\n\n".join(docs)
    prompt = f"""You are a helpful assistant.
Question: {q}

Help me answer using only the context below:

{context}

Answer:"""

    resp = llm.invoke(prompt)
    return {"answer": resp.content}


#from langchain.prompts import ChatPromptTemplate

EVAL_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating an AI answer.

Question:
{question}

Context:
{context}

Answer:
{answer}

Evaluate if the answer is:
1. Grounded in the context
2. Non-empty
3. Helpful

Reply ONLY with one word:
PASS or FAIL
""")

def evaluation_node(state: GraphState) -> dict:
    context = "\n\n".join(state["docs"])
    prompt = EVAL_PROMPT.format(
        question=state["query"],
        context=context,
        answer=state["answer"]
    )

    result = llm.invoke(prompt).content.strip().upper()
    return {"eval_result": result}




def route_after_evaluation(state: GraphState) -> str:
    if state["eval_result"] == "PASS":
        return "end"

    if state["retry_count"] < state["max_retries"]:
        return "retry"

    return "no_docs"

def retry_node(state: GraphState) -> dict:
    retries = state.get("retry_count", 0) + 1
    print(f"[WARN] Retry attempt {retries}")

    return {
        "retry_count": retries,
        "rewritten_query": ""  # force rewrite again
    }

