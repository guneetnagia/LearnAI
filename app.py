"""Gradio RAG app over AST-1.pdf / AST-2.pdf — Hugging Face Spaces entry point."""
import os
import re
from pathlib import Path

import gradio as gr
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# ---- settings (the defaults chosen in the notebook) ---------------------------------
BASE_DIR = Path(__file__).resolve().parent  # PDFs are found next to app.py, wherever you run it from
PDF_FILES = ["AST-1.pdf", "AST-2.pdf"]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CHUNK_SIZE, CHUNK_OVERLAP = 800, 120
DEFAULT_K, DEFAULT_MMR = 6, True
MMR_FETCH_K, MMR_LAMBDA = 20, 0.75  # λ <= 0.25 dropped recall 0.98 -> 0.79 in the notebook run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_text(text: str) -> str:
    text = text.replace("●", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# ---- index -------------------------------------------------------------------------
documents = []
for name in PDF_FILES:
    path = BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put {name} next to app.py (rename e.g. AST-1_1.pdf to AST-1.pdf).")
    # PyMuPDF keeps words whole; PyPDFLoader split some ("c orresponding", "Pytes t").
    for page in PyMuPDFLoader(str(path)).load():
        page.page_content = clean_text(page.page_content)
        page.metadata["source"] = path.stem
        documents.append(page)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
).split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# ---- LLM ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


def llm_generate(messages, max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    out = generator(prompt, generation_config=gen_cfg, return_full_text=False)
    return out[0]["generated_text"].strip()


def to_messages(prompt_value):
    role = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role[m.type], "content": m.content} for m in prompt_value.to_messages()]


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about two course assignment documents "
    "(AST-1: moving an ML model from notebook to production code; AST-2: testing with pytest and packaging). "
    "Use ONLY the context below. If the answer is not in the context, reply exactly: "
    "\"I don't know based on the provided documents.\" Be concise and use bullet points for lists. "
    "Copy file names and commands exactly as they are written in the context; never invent or alter a name."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
    ("human", "{question}"),
])
generation_chain = prompt | RunnableLambda(to_messages) | RunnableLambda(llm_generate) | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']} p.{d.metadata['page'] + 1}]\n{d.page_content}" for d in docs)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~])")


def quote_literally(text):
    """Show PDF text as-is: escape Markdown so '3. data_manger.py' is not a list and __init__.py not bold."""
    lines = [_MD_SPECIAL.sub(r"\\\1", line.strip()) for line in text.splitlines()]
    return "\n".join(f"> {line}  " if line else ">" for line in lines)


def answer(question, use_mmr, k):
    if not question.strip():
        return "Please enter a question.", ""
    if use_mmr:
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": int(k), "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA})
    else:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})
    docs = retriever.invoke(question)
    reply = generation_chain.invoke({"context": format_docs(docs), "question": question})
    ctx = "\n\n".join(
        f"**{d.metadata['source']} p.{d.metadata['page'] + 1}**\n\n{quote_literally(d.page_content)}" for d in docs
    )
    return reply, ctx


EXAMPLES = [
    "What is the purpose of the config.yml file in the production project?",
    "What does data_manager.py do?",
    "What are the three types of tests and how do they differ?",
    "Which files are related to packaging the model?",
    "What commands build the package, and which folders are created afterwards?",
]

with gr.Blocks(title="AST-1 / AST-2 RAG") as demo:
    gr.Markdown("# Ask the MLOps assignments (AST-1 & AST-2)\n"
                "Retrieval-augmented answers from the two assignment PDFs.")
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Question", lines=2,
                                  placeholder="e.g. Which files are used for packaging?")
            ask_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            use_mmr = gr.Checkbox(label="Use MMR (diverse passages)", value=DEFAULT_MMR)
            k = gr.Slider(1, 8, value=DEFAULT_K, step=1, label="Chunks to retrieve (k)")
    reply = gr.Markdown()
    with gr.Accordion("Retrieved context", open=False):
        context = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=question)
    ask_btn.click(answer, [question, use_mmr, k], [reply, context])
    question.submit(answer, [question, use_mmr, k], [reply, context])

if __name__ == "__main__":
    demo.launch()