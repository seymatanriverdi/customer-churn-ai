# LangChain PDF Chat — Local RAG with Ollama and FAISS

A local Retrieval-Augmented Generation (RAG) application that answers questions about a PDF document using **LangChain**, **FAISS**, **Hugging Face embeddings** and a **local Llama 3.2 model served by Ollama**. No cloud API is required: documents and questions never leave the machine.

This project is the LangChain version of my earlier from-scratch implementation, [local-PDF-chat — RAG with Ollama and FAISS](https://github.com/seymatanriverdi/local-PDF-chat---RAG-with-Ollama-and-FAISS). The earlier version builds each RAG step by hand (chunking loop, embedding, FAISS index, search). This version rebuilds the same pipeline with LangChain's standard components (document loaders, text splitters, vector stores, retrievers and prompt templates), which makes it shorter and easier to extend.

## Features

- PDF loading with LangChain `PyPDFLoader` (page metadata kept)
- Recursive character text splitting (chunk size 500, overlap 100)
- Sentence-Transformers embeddings (`all-MiniLM-L6-v2`) via `HuggingFaceEmbeddings`
- FAISS vector store and top-k retriever (`k=3`)
- Grounded prompt with `ChatPromptTemplate`:
  - answers only from the retrieved context
  - uses only the definition of the term asked about
  - replies *"Bu bilgi dokümanda bulunamadı."* ("This information was not found in the document.") instead of guessing
- Deterministic generation (`temperature=0`) with a local Llama 3.2 3B model
- Retrieved context is printed with each answer for transparency and debugging

## Architecture

```text
PDF ──> PyPDFLoader ──> RecursiveCharacterTextSplitter ──> chunks
                                                            │
                                                            v
                                         HuggingFaceEmbeddings (MiniLM)
                                                            │
                                                            v
                                                    FAISS vector store
                                                            │
User question ──> Retriever (top-3 chunks) <────────────────┘
                        │
                        v
          ChatPromptTemplate (grounded instructions + context + question)
                        │
                        v
              ChatOllama (Llama 3.2 3B, temperature 0)
                        │
                        v
                 Answer (Turkish) + retrieved context
```

## Tech Stack

- Python
- LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`)
- LangChain Hugging Face integration + Sentence Transformers
- FAISS (`faiss-cpu`)
- Ollama + Llama 3.2 3B (`langchain-ollama`)
- pypdf

## Project Structure

```text
langchain-pdf-chat/
├── data/                    # put your PDF here (PDFs are git-ignored)
├── pdf_chat_langchain.py    # main application
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Install [Ollama](https://ollama.com) and download the model:

   ```bash
   ollama pull llama3.2:3b
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. Create a `data/` folder, put your PDF in it, and update `PDF_PATH` in `pdf_chat_langchain.py` to match the file name.

## Usage

```bash
python pdf_chat_langchain.py
```

Type a question and press Enter. The app prints the retrieved context and the answer. Type `exit` to quit.

```text
Ask a question or type 'exit': Güverte nedir?

Retrieved Context:
...

Answer:
...
```

## Design Decisions

- **Why LangChain after a manual version?** Writing the pipeline by hand first made each RAG step clear. Rebuilding it with LangChain shows how the same steps map to standard, swappable components (for example, changing FAISS to another vector store or Ollama to a hosted LLM is a one-line change).
- **Recursive splitting instead of fixed-size slicing:** `RecursiveCharacterTextSplitter` tries to split on paragraphs and sentences first, so chunks are less likely to cut a definition in half.
- **Llama 3.2 3B instead of 1B:** The 1B model was tried first (still visible as a commented-out option in the code). The 3B model followed the grounding rules more reliably, at the cost of slower answers.
- **Strict prompt rules:** Small local models tend to mix neighbouring definitions in glossary-style documents. The prompt limits the answer to the requested term and makes "not found" the default when the context does not support an answer.

## Limitations

- Small local models can still occasionally mix up nearby definitions or ignore a prompt rule.
- Answer quality depends on how well the PDF text is extracted (scanned PDFs need OCR first).
- The FAISS index is rebuilt in memory on every run and is not persisted.
- No conversation memory: each question is answered independently.
