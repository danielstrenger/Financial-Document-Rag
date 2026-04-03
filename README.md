# Financial Document RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system designed to parse, index, and query complex financial reports. 

This project demonstrates the implementation of an LLM architecture using the following tools:
* **Language:** Python 3.10+
* **Orchestration:** LangChain (LCEL)
* **LLM Engine:** OpenAI API (`gpt-3.5-turbo`)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB (Local SQLite Persistence)
* **Frontend:** Chainlit

## System Architecture

This system utilizes a two-tier architecture to ensure modularity and scalability:

1. **The Ingestion Pipeline (`ingest.py`):** A backend ETL script that utilizes LangChain to parse raw PDF financial documents. It implements `RecursiveCharacterTextSplitter` for semantic chunking and utilizes local HuggingFace embeddings (`all-MiniLM-L6-v2`) to project text into a high-dimensional vector space. The vectors are persisted locally using **ChromaDB**.
2. **The Application Interface (`app.py`):** An asynchronous React-based chat application built with **Chainlit**. It operates as a read-only client to the vector database. The underlying reasoning engine uses **LangChain Expression Language (LCEL)** to explicitly separate retrieval logic from prompt formatting and LLM generation, streaming tokens directly to the UI for a highly responsive user experience.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Linux/macOS
source .venv/bin/activate 
# On Windows
.venv\Scripts\activate
```
### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```
### 4. Environment Variables
This application requires an OpenAI API key for the language model. Export it to your terminal before running the application:
```bash
# On Linux/macOS
export OPENAI_API_KEY="your-api-key-here"
# On Windows (cmd.exe)
set OPENAI_API_KEY=your-api-key-here
```

## Usage Guide
### Phase 1: Ingesting Data
Before querying, you must ingest your financial documents into the vector database. By default, the script will read the PDF files in `./documents/`, but you can redirect it to any other folder (or single PDF file) using the `--path` argument.
```bash
python -m ingest --path ./documents
```
### Phase 2: Launching the Interface
Once the database is populated with the information from your documents, launch the Chainlit server:
```bash
chainlit run app.py
```
You can access the chat interface by navigating to `http://localhost:8000` in your browser.