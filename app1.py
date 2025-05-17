import os
import io
import time
import uuid
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from qdrant_client.http.models import PointStruct

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "session_docs"

# Setup clients
client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI setup
st.set_page_config(page_title="📄 Ask My PDF")
st.title("📚 Ask Your PDF Documents (OpenChat)")

# Session state
if "docs_uploaded" not in st.session_state:
    st.session_state.docs_uploaded = []
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

# File upload
uploaded_files = st.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for pdf_file in uploaded_files:
        file_name = pdf_file.name
        if any(doc["file_name"] == file_name for doc in st.session_state.docs_uploaded):
            continue

        doc_id = str(uuid.uuid4())
        reader = PdfReader(io.BytesIO(pdf_file.read()))
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Chunk text
        sentences = full_text.split(". ")
        chunks, chunk = [], ""
        for sentence in sentences:
            if len(chunk) + len(sentence) < 500:
                chunk += sentence + ". "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + ". "
        if chunk:
            chunks.append(chunk.strip())
        chunks = [c for c in chunks if c.strip()]

        # Vectorize and store
        vectors = embed_model.encode(chunks).tolist()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={"text": chunks[i], "doc_id": doc_id, "file_name": file_name}
            ) for i in range(len(chunks))
        ]

        if COLLECTION_NAME not in [col.name for col in client_qdrant.get_collections().collections]:
            client_qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 384, "distance": "Cosine"}
            )

        client_qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        st.session_state.docs_uploaded.append({"file_name": file_name, "doc_id": doc_id})
        st.session_state.chat_histories[doc_id] = []

        st.success(f"✅ {file_name} uploaded and indexed.")

# Document selector
doc_options = {doc["file_name"]: doc["doc_id"] for doc in st.session_state.docs_uploaded}
selected_doc = st.selectbox("📑 Select a document:", list(doc_options.keys()) if doc_options else [])

# QA Section
if selected_doc:
    selected_doc_id = doc_options[selected_doc]
    query_key = f"query_{selected_doc_id}"
    query = st.text_input("💬 Ask a question:", key=query_key)

    if st.button("🔍 Get Answer") and query:
        query_vector = embed_model.encode(query).tolist()
        results = client_qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
            query_filter=Filter(must=[{"key": "doc_id", "match": {"value": selected_doc_id}}])
        )
        context = "\n".join([r.payload["text"] for r in results])

        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

        with st.spinner("💬 OpenChat is thinking..."):
            try:
                res = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "openchat", "prompt": prompt, "stream": False}
                )
                answer = res.json().get("response", "").strip()
            except Exception as e:
                st.error(f"⚠️ OpenChat error: {e}")
                answer = ""

        if answer:
            st.session_state.chat_histories[selected_doc_id].append((query, answer))
            st.subheader("💬 OpenChat's Answer")
            placeholder = st.empty()
            typed = ""
            for char in answer:
                typed += char
                placeholder.markdown(typed)
                time.sleep(0.01)

    if st.button("📜 View Chat History"):
        history = st.session_state.chat_histories[selected_doc_id]
        if history:
            with st.expander("📜 Chat History", expanded=True):
                for i, (q, a) in enumerate(history, 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a}")
                    st.markdown("---")
        else:
            st.info("ℹ️ No chat history yet.")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_histories[selected_doc_id] = []
        st.rerun()
