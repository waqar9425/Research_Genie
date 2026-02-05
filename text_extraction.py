# data extraction from pdf
import pdfplumber

def extract_text_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"[ERROR] Could not extract text from {pdf_path}: {e}")
        return None