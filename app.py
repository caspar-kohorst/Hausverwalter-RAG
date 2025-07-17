import streamlit as st
import faiss
import numpy as np
import pickle
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o"  # or "gpt-4"

DB_PATH = "db/faiss_index.bin"
META_PATH = "db/meta.pkl"

@st.cache_resource
def load_db():
    index = faiss.read_index(DB_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def get_embedding(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(resp.data[0].embedding, dtype="float32")

def search(query, index, meta, k=10):
    emb = get_embedding(query)
    D, I = index.search(np.array([emb]), k)
    results = []
    for idx in I[0]:
        results.append({
            "text": meta["texts"][idx],
            "meta": meta["meta"][idx]
        })
    return results

def llm_answer(query, context):
    client = OpenAI(api_key=OPENAI_API_KEY)
    context_str = "\n\n".join([f"Chunk {i+1} (Seitenbereich: {c['meta'].get('page_range', '-')}, Quelle: {c['meta'].get('source', '-')})\n{c['text']}" for i, c in enumerate(context)])
    prompt = f"""Du bist ein Experte im Thema Hausverwaltung mit jahrelanger Erfahrung in Deutschland. Nutze die folgenden Auszüge als Kontext zusätzlich zu deinem eigenen parametrical Wissen, 
                um die Frage zu beantworten. In einem ersten Schritt abstrahiere bitte ob es sich um Gemeinschaftseigentum oder um Sondereigentum handelt. Anschließend spezifiziere die Antwort auf die Frage.
                Antworte auf Deutsch. Antworte präzise ohne dabei wichtige Details zu vergessen."\n\nKontext:\n{context_str}\n\nFrage: {query}\n\nAntwort:"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content

def main():
    st.title("Hausverwaltung RAG Chatbot")
    st.write("Stelle eine Frage zum Thema Hausverwaltung.")

    index, meta = load_db()
    user_input = st.text_input("Deine Frage:", "")

    if st.button("Antworten") and user_input.strip():
        with st.spinner("Suche relevante Passagen..."):
            context = search(user_input, index, meta)
        with st.spinner("Generiere Antwort..."):
            answer = llm_answer(user_input, context)
        st.markdown("### Antwort")
        st.write(answer)
        st.markdown("---")
        st.markdown("### Kontext (Top 10 Passagen)")
        for i, c in enumerate(context):
            st.markdown(f"""
<div style='border:1px solid #e6e6e6; border-radius:8px; padding:16px; margin-bottom:16px; background-color:#f9f9f9;'>
<b>Quelle:</b> <span style='color:#3366cc'>{c['meta'].get('source', '-')}</span> &nbsp; | &nbsp; <b>Seitenbereich:</b> <span style='color:#3366cc'>{c['meta'].get('page_range', '-')}</span><br>
<small><i>Snippet:</i></small><br>
<span style='font-size: 0.98em;'>{c['text'][:500].replace(chr(10), ' ') + ('...' if len(c['text']) > 500 else '')}</span>
</div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 