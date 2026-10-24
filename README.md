---
title: AST RAG
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# AST-1 / AST-2 RAG

Retrieval-augmented question answering over two MLOps assignment PDFs.

- Embeddings: `BAAI/bge-small-en-v1.5`, FAISS vector store
- LLM: `Qwen/Qwen2.5-1.5B-Instruct` (override with the `LLM_MODEL` Space variable)
- Upload `AST-1.pdf` and `AST-2.pdf` next to `app.py`.
