import os
import io
import time
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from qdrant_client.http.models import PointStruct
from dotenv import load_dotenv
import uuid

# --- Load environment variables ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "session_docs"

# --- Setup Clients ---
client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embed_model = SentenceTransformer("all-MiniLM-L6-v2",device=None)

# --- Streamlit UI setup ---
st.set_page_config(page_title="📄 Ask My PDF")
st.title("📚 Ask Your PDF Documents")

# --- Session State Setup ---
if "docs_uploaded" not in st.session_state:
    st.session_state.docs_uploaded = []

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

# --- File Upload ---
uploaded_files = st.file_uploader("📂 Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for pdf_file in uploaded_files:
        file_name = pdf_file.name
        if any(doc["file_name"] == file_name for doc in st.session_state.docs_uploaded):
            continue  # skip already uploaded file

        doc_id = str(uuid.uuid4())

        with st.spinner(f"📖 Extracting from {file_name}..."):
            reader = PdfReader(io.BytesIO(pdf_file.read()))
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Chunking text
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

        chunks = [c for c in chunks if c.strip()]  # remove empty

        with st.spinner("🔎 Indexing to Qdrant..."):
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

        # Save doc metadata
        st.session_state.docs_uploaded.append({"file_name": file_name, "doc_id": doc_id})
        st.session_state.chat_histories[doc_id] = []

        # Toast-like success message
        placeholder = st.empty()
        placeholder.success(f"✅ {file_name} uploaded and indexed.")
        time.sleep(2)
        placeholder.empty()

# --- Document Selector ---
# Ensure only existing uploaded files are listed
current_uploaded_names = [file.name for file in uploaded_files] if uploaded_files else []
st.session_state.docs_uploaded = [
    doc for doc in st.session_state.docs_uploaded if doc["file_name"] in current_uploaded_names
]
doc_options = {doc["file_name"]: doc["doc_id"] for doc in st.session_state.docs_uploaded}
selected_doc = st.selectbox("📑 Select a document to ask about:", list(doc_options.keys()) if doc_options else [])

# --- If a document is selected ---
if selected_doc:
    selected_doc_id = doc_options[selected_doc]
    # --- Use document-specific key for question input ---
    query_key = f"query_{selected_doc_id}"
    query = st.text_input("💬 Ask a question:", key=query_key)

    # --- Answer logic ---
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

        prompt = f"""
        Use the context below to answer the user's question.

        Context:
        {context}

        Question:
        {query}
        """

        model = genai.GenerativeModel("models/gemini-1.5-pro")

        with st.spinner("💬 Gemini is thinking..."):
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=512
                )
            )

        answer = response.text.strip()
        st.session_state.chat_histories[selected_doc_id].append((query, answer))

        # Typing animation
        st.subheader("💬 Gemini's Answer (typing...)")
        placeholder = st.empty()
        typed = ""
        for char in answer:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.01)

    # --- Show history only if button clicked ---
    show_history_now = st.button("📜 View Chat History")

    if show_history_now:
        history = st.session_state.chat_histories[selected_doc_id]
        if history:
            with st.expander("📜 Previous Q&A History", expanded=True):
                for i, (q, a) in enumerate(history, 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a}")
                    st.markdown("---")
        else:
            st.info("ℹ️ No chat history available for this document.")

    # --- Clear Chat ---
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_histories[selected_doc_id] = []
        st.rerun()