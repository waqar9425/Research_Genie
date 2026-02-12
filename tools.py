#Define graph

from dowload_papers import *
from text_extraction import *
from llm_config import llm, embeddings_model
from state import GraphState



def rewrite_query_node(state: GraphState) -> dict:
    #history = "\n".join(state.get("chat_history", []))
    history = format_chat_history(state.get("chat_history", []))

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
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

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



# def build_index_node(state: GraphState):
#     chunks = state.get("chunks", [])
#     metas = state.get("metadata", [])
#     if not chunks:
#         print("[WARN] No chunks to index")
#         return {"retriever": None, "index": None}

#     vectorstore = FAISS.from_texts(
#         texts=chunks, embedding=embeddings_model, metadatas=metas
#     )
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
#     print(f"[INFO] FAISS index built with {len(chunks)} chunks")

#     return {"index": vectorstore, "retriever": retriever}
import time
from langchain_community.vectorstores import FAISS

def build_index_node(state: GraphState):
    chunks = state.get("chunks", [])
    metas = state.get("metadata", [])
    
    if not chunks:
        return {"retriever": None, "index": None}

    batch_size = 5  # Small batches for the free tier
    
    # Initialize the vectorstore with just the first batch
    vectorstore = FAISS.from_texts(
        texts=chunks[:batch_size], 
        embedding=embeddings_model, 
        metadatas=metas[:batch_size]
    )

    # Loop through the rest of the chunks with a delay
    for i in range(batch_size, len(chunks), batch_size):
        print(f"[INFO] Embedding batch {i} to {i+batch_size}...")
        
        batch_texts = chunks[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)
        
        # 1-2 second sleep prevents the "Resource Exhausted" error
        time.sleep(1.5) 

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print(f"[INFO] FAISS index built with {len(chunks)} chunks")

    return {"index": vectorstore, "retriever": retriever}


def retriever_node(state: GraphState):
    retr = state.get("retriever")

    if retr is None:
        print("[ERROR] Retriever is missing! State keys:", list(state.keys()))
        return {**state, "docs": []}

    docs = retr.invoke(state["rewritten_query"])  # use rewritten query
    print(f"[INFO] Retrieved {len(docs)} chunks")

    return {**state, "docs": docs}




# additional node to handle retrieval fallbacks
def route_after_retrieval(state):
    docs = state.get("docs", [])
    print("ROUTER DOC COUNT:", len(docs))

    if docs:
        print("ROUTING → generate")
        return "generate"
    else:
        print("ROUTING → no_docs")
        return "no_docs"



def no_docs_node(state: GraphState) -> dict:
    return {
        "answer": "I couldn't find relevant documents to answer your question."
    }


def generator_node(state: GraphState):
    docs = state.get("docs", [])
    q = state["query"]
    chat = format_chat_history(state.get("chat_history", []))

    if not docs:
        return {"draft_answer": ""}

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful research assistant.

Conversation so far:
{chat}

Current question:
{q}

Answer using ONLY the context below.

Context:
{context}

Draft Answer:
"""
    resp = llm.invoke(prompt)
    return {"draft_answer": resp.content}


def evaluation_node(state: GraphState) -> dict:
    docs = state.get("docs", [])
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a easy going reviewer.
Return ONLY one of the following:

PASS
PASS

Question:
{state['query']}

Context:
{context}

Answer:
{state['draft_answer']}
...
"""
    critique = llm.invoke(prompt).content.strip()
    return {"critique": critique}



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



def format_chat_history(chat_history):
    if not chat_history:
        return ""

    formatted = []
    for msg in chat_history:
        role = msg["role"].capitalize()
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def update_chat_history_node(state: GraphState) -> dict:
    chat = state.get("chat_history", []).copy()

    chat.append({
        "role": "user",
        "content": state["query"]
    })

    chat.append({
        "role": "assistant",
        "content": state["draft_answer"]
    })

    return {"chat_history": chat}
