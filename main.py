from langchain_huggingface import HuggingFaceEmbeddings
# 1. Imports
import os
import json
import feedparser
import time
import re
import requests
import pdfplumber
import faiss
import numpy as np
from tqdm import tqdm

from typing import TypedDict, List

# LangGraph & LangChain components
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from dowload_papers import *
# 2. Global Config
DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "papers")
META_DIR = os.path.join(DATA_DIR, "metadata")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

import os



from tools import *

from graph import build_graph

graph = build_graph()

# response = graph.invoke({"query": "explain attention in large language model?"})
# print("Answer:", response["answer"])
response = graph.invoke({
    "query": "explain attention in large language model?",
    "chat_history": [
        "User asked about transformers earlier",
        "Assistant explained self-attention briefly"
    ],
    "retry_count": 0,
    "max_retries": 2
})

print("Answer:", response["answer"])