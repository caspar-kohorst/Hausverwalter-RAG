import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

DATA_DIR = "data"
DB_PATH = "db/faiss_index.bin"
META_PATH = "db/meta.pkl"


def extract_pdf_chunks(pdf_path, chunk_size=500, overlap=50):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    # Simple chunking by character count
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end]
        meta = {"page_range": f"{start}-{end}", "source": os.path.basename(pdf_path)}
        chunks.append({"text": chunk, "meta": meta})
        start += chunk_size - overlap
    return chunks


def get_embeddings(texts):
    client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        embeddings.append(resp.data[0].embedding)
    return embeddings


def main():
    os.makedirs("db", exist_ok=True)
    all_chunks = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            print(f"Processing {pdf_path} ...")
            chunks = extract_pdf_chunks(pdf_path)
            all_chunks.extend(chunks)
    texts = [c["text"] for c in all_chunks]
    meta = [c["meta"] for c in all_chunks]
    embeddings = get_embeddings(texts)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    arr = np.array(embeddings, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] > 0 and arr.shape[1] > 0:
        index.add(arr)
    faiss.write_index(index, DB_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"meta": meta, "texts": texts}, f)
    print("Ingestion complete.")

if __name__ == "__main__":
    main() 