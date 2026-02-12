
import os
from dotenv import load_dotenv



# Load .env file
load_dotenv()
# Access environment variables
openai_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
# 1. Replace the OpenAI line with this:
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# embeddings_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)