from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "sanity ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}
