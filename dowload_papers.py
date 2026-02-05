#data collection utility functions
#DOWNLOAD PDF AND METADATA FROM ARCHIV

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

# 2. Global Config
DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "papers")
META_DIR = os.path.join(DATA_DIR, "metadata")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def sanitize_filename(name):
    name = name.replace("\n", " ").replace("\r", " ")  # remove newlines
    name = re.sub(r'[\\/*?:"<>|]', "", name)           # remove invalid characters
    name = re.sub(r'\s+', '_', name.strip())           # collapse spaces
    return name[:80]  # limit length for safety


def download_pdf_with_retry(pdf_url, save_path, retries=3, delay=5):
    for i in range(retries):
        try:
            response = requests.get(pdf_url, timeout=20)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"[WARN] Got status {response.status_code}, retrying...")
        except Exception as e:
            print(f"[ERROR] Attempt {i+1}: {e}")
        time.sleep(delay)
    return False

def fetch_arxiv_papers(query="natural language processing", max_results=5):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query.replace(' ', '+')}&start=0&max_results={max_results}"

    print(f"[INFO] Fetching arXiv results for query: {query}")
    feed = feedparser.parse(base_url + search_query)
    papers = []

    for entry in tqdm(feed.entries):
        paper = {
            "title": entry.title.strip(),
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary.strip(),
            "published": entry.published,
            "pdf_url": entry.link.replace("abs", "pdf"),
            "arxiv_id": entry.id.split("/abs/")[-1],
            "link": entry.link,
            "categories": entry.tags[0]['term'] if 'tags' in entry and entry.tags else None
        }

        filename = sanitize_filename(paper["title"]) + ".pdf"
        pdf_path = os.path.join(PDF_DIR, filename)

        # 🛑 Skip download if file already exists
        if os.path.exists(pdf_path):
            print(f"[INFO] Skipping download, file already exists: {filename}")
            paper["local_pdf"] = pdf_path
        else:
            success = download_pdf_with_retry(paper["pdf_url"], pdf_path)
            if success:
                paper["local_pdf"] = pdf_path
            else:
                print(f"[ERROR] Skipping paper due to download failure: {paper['title']}")
                continue

        # Save paper metadata
        meta_path = os.path.join(META_DIR, sanitize_filename(paper["title"]) + ".json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(paper, f, indent=4)

        papers.append(paper)
        time.sleep(2)  # avoid rate limiting

    print(f"[INFO] Fetched and saved {len(papers)} papers.")
    return papers