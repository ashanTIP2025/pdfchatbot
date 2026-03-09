# PDF Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDF documents and ask conversational questions about their content. Answers are grounded exclusively in your uploaded documents, with built-in PII guardrails to protect sensitive information.

## Features

- Upload one or more PDFs and chat with their contents
- Conversational memory — follow-up questions use prior context
- **PII guardrails** — blocks questions containing sensitive data and blocks responses that would reveal personal information
- Clear conversation button to reset and start fresh

## How It Works

```
Upload PDFs → Extract text → Chunk & embed → Store in FAISS
                                                    ↓
User question → [Input PII check] → Retrieve chunks → GPT-4o → [Output PII check] → Display
```

## Setup

### 1. Clone the repo and create a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# or
source .venv/bin/activate       # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Do not install `spacy` — it conflicts with pydantic 2.12+ in this environment and is not required.

### 3. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run the app

```bash
streamlit run app.py
```

## PII Guardrails

The app has two layers of PII protection using regex-based detection:

| Rail | Trigger | Behaviour |
|------|---------|-----------|
| **Input** | User's question contains PII | Question is blocked; LLM is never called |
| **Output** | LLM's answer contains PII | Answer is blocked; not saved to chat history |

### Detected PII types

| Type | Example |
|------|---------|
| Email address | user@example.com |
| Phone number | (555) 123-4567 |
| SSN | 123-45-6789 |
| Credit card | 4111 1111 1111 1111 |
| IP address | 192.168.1.1 |

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `PyPDF2` | PDF text extraction |
| `langchain-classic` | ConversationalRetrievalChain |
| `langchain-openai` | GPT-4o + OpenAI embeddings |
| `langchain-community` | FAISS vector store integration |
| `langchain-text-splitters>=0.3.2` | Text chunking |
| `faiss-cpu` | In-memory vector similarity search |
| `python-dotenv` | Load `.env` variables |

## Limitations

- PDFs must be text-based (scanned image PDFs are not supported)
- PII detection is regex-based — it catches structured patterns but not names
- The FAISS index is in-memory only; PDFs must be re-processed on each app restart
