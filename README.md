# 🩺 Medical Chatbot

A smart AI-powered chatbot designed to answer medical-related questions using Retrieval-Augmented Generation (RAG). It combines the power of LLMs ( Gemini) with vector-based document search to provide accurate, context-aware answers from trusted medical sources.

---

## 🚀 Features

- 🔍 **RAG-based QA**: Combines LLMs with vector search for reliable answers.
- 💬 **Conversational Interface**: Accepts natural language questions.
- 🧠 **LLM Integration**: Supports OpenAI GPT or Google Gemini models.
- 📚 **Custom Medical Knowledge**: Uses your own documents (PDFs, articles, etc.) for retrieval.
- ⚙️ **Embeddings via HuggingFace or Google**.
- 🗂️ **Vector Store**: Supports Pinecone, FAISS, or other LangChain-compatible backends.

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **OpenAI** / **Google Gemini**
- **HuggingFace Transformers**
- **Pinecone / FAISS** (for vector storage)
- **Streamlit / CLI / Notebook UI** (choose one for interaction)
- **dotenv** for environment configuration

---
# .env file

PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
GOOGLE_API_KEY=your-gemini-api-key

# Running
Run store_index.py after loading the pdf documents, ensure you have the indexer from PINECONE file then run app.py,

The project should run on localhost port 8080



