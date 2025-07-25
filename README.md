<<<<<<< HEAD
# WEG Recht RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about WEG-Recht using the latest data from a local PDF (WEG.pdf).

## Features
- Embeds WEG-Recht PDF (chunked) into a local FAISS vector database
- Retrieves top 5 relevant passages for each user query
- Uses OpenAI LLM for answer generation
- Streamlit web interface displays both answer and context

## Setup

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Set up your environment variables:**
   - Create a `.env` file and add your OpenAI API key as `OPENAI_API_KEY=...`
3. **Place your WEG PDF in the folder as `WEG.pdf`**
4. **Run the data ingestion script manually:**
   - Run `data_ingest.py` to (re-)embed the PDF:
   ```bash
   python data_ingest.py
   ```
   (This extracts, chunks, and embeds the PDF. Re-run after updating the PDF.)
5. **Start the chatbot UI:**
   ```
   streamlit run app.py
   ```

## Deployment

### Free Hosting (Streamlit Cloud)
- Push your code to a public GitHub repo
- Go to [Streamlit Cloud](https://streamlit.io/cloud), sign in, and deploy your repo
- Add your `OPENAI_API_KEY` as a secret in the Streamlit Cloud settings
- Upload your PDF and run ingestion as above

### AWS Hosting
- Launch an EC2 instance (Ubuntu recommended)
- Install Python 3.8+, pip, and git
- Clone your repo and follow the setup steps above
- Open port 8501 in your security group
- Run `streamlit run app.py`
- Manually run `data_ingest.py` after updating the PDF

## References
- [gesetze-aus-dem-internet](https://github.com/nfelger/gesetze-aus-dem-internet)
- [RAG-nificent Example](https://github.com/MaxMLang/RAG-nificent)
- [RAG in Action Tutorial](https://python.plainenglish.io/rag-in-action-build-an-ai-that-answers-like-an-expert-using-your-own-data-9b4d890a1031) 
=======
#   H a u s v e r w a l t e r - R A G  
 