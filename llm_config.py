from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model


# Load .env file
load_dotenv()
# Access environment variables
openai_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")


embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)