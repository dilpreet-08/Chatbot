# NIT Jalandhar RAG-Based University Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot designed to answer queries about NIT Jalandhar admissions, academic programs, placements, campus facilities, and more using official documents.

---

## 🛠️ Tech Stack & Architecture

- **Web Framework:** Flask
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (`faiss-cpu`)
- **Document Extractors:** `PyPDF2`, `python-docx`
- **LLM Engine:** LLaMA 3 via Ollama (`http://127.0.0.1:11434`)

---

## 🚀 How to Run the Project in the Future

### 1. Install Dependencies

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Install and Run Ollama (When Ready)

1. Download and install Ollama from [https://ollama.com](https://ollama.com).
2. Download and launch the LLaMA 3 model:
   ```bash
   ollama run llama3
   ```
*(Note: If Ollama is not running, the chatbot will still provide retrieved knowledge directly from NIT Jalandhar documents via fallback).*

### 3. Start the Web Application

Run the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
├── app.py                  # Flask entry point and HTTP endpoints
├── chat_engine.py          # Query routing, guardrails, and RAG orchestrator
├── chatbot_logic.py        # Rule-based fallback logic
├── data_loader.py          # Document loader (PDF, DOCX, TXT) with error-safe extraction
├── vector_store.py         # FAISS vector store with local disk caching
├── llama_engine.py         # Ollama LLaMA-3 integration with resilient fallback
├── knowledge_base/         # University documents and brochures
├── templates/              # HTML templates (index.html)
├── static/                 # CSS styling and JavaScript
├── requirements.txt        # Python package dependencies
└── README.md               # Setup and usage guide
```
