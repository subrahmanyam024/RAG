# 📚 Ask Your PDF Documents (Gemini-Powered)

This is a Streamlit app that lets you upload PDF documents and ask questions about them. It uses:
- **Google Gemini** for answering questions,
- **Qdrant** for vector storage and retrieval,
- **Sentence Transformers** for generating embeddings.

---

## 🚀 Features

- Upload and index multiple PDFs.
- Ask natural language questions about any document.
- Answers are generated based on document content.
- Chat history and Q&A stored per document.
- Clear chat and re-index anytime.

---

## 🛠 Requirements

- Python 3.10 or later
- Virtual environment (recommended)

---

## 📦 Installation

1. **Clone the Repository or Download Files**

```bash
git clone https://github.com/your-username/pdf-gemini-app.git
cd pdf-gemini-app

Set up a virtual environment

bash

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies

bash

pip install -r requirements.txt

🔑 Environment Variables
Create a .env file in the root folder and add:

env
Copy code
GEMINI_API_KEY=your_google_gemini_api_key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
Get your Gemini API key from: https://makersuite.google.com/app/apikey

Use Qdrant Cloud or run Qdrant locally.

▶️ Run the App
bash
Copy code
streamlit run app.py

🧠 Example Use Case
Upload a PDF (e.g., research paper, textbook).

Ask: "What is the main conclusion in section 3?"

Get a smart, context-based answer powered by Gemini!

🤖 Tech Stack
Streamlit UI

Google Gemini (google-generativeai)

Qdrant Vector DB

Sentence Transformers (all-MiniLM-L6-v2)

PyPDF2 for PDF parsing

💡 Tips
Gemini API has daily free tier limits. Be aware of rate limits: Rate Limits Docs

For better performance, keep PDF sizes reasonable (~50 pages max per file recommended).

📜 License
MIT License

🙌 Acknowledgements
Streamlit

Google Generative AI

Qdrant

Hugging Face

yaml


---

Let me know if you'd like me to tailor it to Hugging Face or OpenChat/Ollama instead.





