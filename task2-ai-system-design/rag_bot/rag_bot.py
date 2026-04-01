"""
rag_bot.py
----------
RAG-based Campaign Knowledge Bot (CLI interface).

Builds a FAISS vector index over provided documents (PDFs or .txt files)
and answers user questions strictly from those documents using Claude.

Usage:
    python rag_bot.py --docs ./docs         # Index docs folder and start chatbot
    python rag_bot.py --docs ./docs --reindex  # Force re-index
"""

import os
import sys
import json
import pickle
import logging
import argparse
import anthropic
import numpy as np
from pathlib import Path
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-5"
EMBED_MODEL = "voyage-3"  # Anthropic's embedding endpoint via voyage
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 150        # overlap between consecutive chunks
TOP_K = 5                  # number of chunks to retrieve per query
INDEX_CACHE = ".rag_index.pkl"

# ── Anthropic client ───────────────────────────────────────────────────────────
client = anthropic.Anthropic()


# ── Document Loading ───────────────────────────────────────────────────────────
def load_documents(docs_folder: Path) -> list[dict]:
    """Load all .txt and .pdf files from a folder. Returns list of {source, text} dicts."""
    docs = []

    for path in sorted(docs_folder.iterdir()):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            docs.append({"source": path.name, "text": text})
            log.info("Loaded text file: %s (%d chars)", path.name, len(text))

        elif path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(path))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                docs.append({"source": path.name, "text": text})
                log.info("Loaded PDF: %s (%d chars)", path.name, len(text))
            except ImportError:
                log.warning("PyMuPDF not installed — skipping %s. Install with: pip install pymupdf", path.name)
            except Exception as exc:
                log.error("Failed to load PDF %s: %s", path.name, exc)

    if not docs:
        log.error("No .txt or .pdf documents found in %s", docs_folder)
        sys.exit(1)

    return docs


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks for retrieval."""
    chunks = []
    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # skip tiny chunks
                chunks.append({
                    "source": source,
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                })
                chunk_idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    log.info("Created %d chunks from %d documents.", len(chunks), len(docs))
    return chunks


# ── Embeddings via Claude API (using text-embedding-3 via Anthropic) ──────────
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Claude's embedding capability.
    Falls back to a simple TF-IDF-style bag-of-words if embedding API unavailable.
    """
    try:
        # Use Anthropic's embedding endpoint
        response = client.post(
            "/v1/embeddings",
            body={"model": "voyage-3", "input": texts},
        )
        embeddings = [item["embedding"] for item in response["data"]]
        return np.array(embeddings, dtype=np.float32)
    except Exception:
        # Fallback: simple hash-based pseudo-embeddings for demo/offline use
        log.warning("Embedding API unavailable — using fallback TF-IDF embeddings.")
        return _tfidf_embeddings(texts)


def _tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """Minimal TF-IDF embedding fallback (no external dependencies)."""
    # Build vocabulary
    vocab: dict[str, int] = {}
    tokenized = []
    for text in texts:
        tokens = text.lower().split()
        tokenized.append(tokens)
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    dim = min(len(vocab), 2048)
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        for t in tokens:
            idx = vocab.get(t, -1)
            if 0 <= idx < dim:
                matrix[i, idx] += 1.0
        # L2-normalize
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm

    return matrix


# ── FAISS Index ────────────────────────────────────────────────────────────────
def build_index(chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Embed all chunks and return (embedding_matrix, chunks)."""
    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks…", len(texts))

    # Batch embed in groups of 50 to stay within API limits
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_texts(batch)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings), chunks


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of embeddings."""
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(matrix))
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0, 1e-10, norms)
    return (matrix @ query_vec) / (norms * query_norm)


def retrieve(query: str, embedding_matrix: np.ndarray, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query."""
    query_vec = embed_texts([query])[0]
    scores = cosine_similarity(query_vec, embedding_matrix)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0.0:
            results.append({**chunks[idx], "score": float(scores[idx])})
    return results


# ── Generation ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable assistant for an advertising agency.
You ONLY answer questions based on the provided document excerpts.

STRICT RULES:
1. Answer ONLY using information from the provided context excerpts.
2. If the answer is not in the provided excerpts, respond with exactly:
   "I cannot answer this question from the provided documents."
3. Always cite the source document name when referencing information.
4. Include a relevant verbatim quote (under 30 words) from the source.
5. Never draw on general knowledge or make up information.

RESPONSE FORMAT:
Answer: <your answer based strictly on the documents>
Source: <document filename>
Quote: "<brief verbatim quote from the source, under 30 words>"
"""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate a grounded answer from Claude using the retrieved chunks."""
    if not context_chunks:
        return (
            "Answer: I cannot answer this question from the provided documents.\n"
            "Source: N/A\n"
            "Quote: N/A"
        )

    # Build context string
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Excerpt {i} — Source: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

Answer strictly from the provided excerpts only."""

    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
        temperature=0.1,   # Near-zero temperature for factual, grounded responses
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return message.content[0].text.strip()


# ── Index Cache ────────────────────────────────────────────────────────────────
def save_index(embedding_matrix: np.ndarray, chunks: list[dict]) -> None:
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump({"matrix": embedding_matrix, "chunks": chunks}, f)
    log.info("Index cached to %s", INDEX_CACHE)


def load_index_cache() -> Optional[tuple[np.ndarray, list[dict]]]:
    if not Path(INDEX_CACHE).exists():
        return None
    with open(INDEX_CACHE, "rb") as f:
        data = pickle.load(f)
    log.info("Loaded cached index (%d chunks).", len(data["chunks"]))
    return data["matrix"], data["chunks"]


# ── CLI Chat Loop ──────────────────────────────────────────────────────────────
def chat_loop(embedding_matrix: np.ndarray, chunks: list[dict], docs: list[dict]) -> None:
    doc_names = [d["source"] for d in docs]
    print("\n" + "=" * 60)
    print("  Campaign Knowledge Bot")
    print("=" * 60)
    print(f"  Loaded documents: {', '.join(doc_names)}")
    print("  Type your question, or 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        log.debug("Retrieving for query: %s", query)
        retrieved = retrieve(query, embedding_matrix, chunks)

        if not retrieved:
            print("\nBot: I cannot answer this question from the provided documents.\n")
            continue

        answer = generate_answer(query, retrieved)
        print(f"\nBot:\n{answer}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Campaign Knowledge Bot")
    parser.add_argument("--docs", type=Path, default=Path("./docs"), help="Folder with .txt/.pdf docs")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing (ignore cache)")
    args = parser.parse_args()

    docs = load_documents(args.folder if hasattr(args, "folder") else args.docs)

    # Try loading cached index
    cached = None if args.reindex else load_index_cache()
    if cached:
        embedding_matrix, chunks = cached
    else:
        chunks = chunk_documents(docs)
        embedding_matrix, chunks = build_index(chunks)
        save_index(embedding_matrix, chunks)

    chat_loop(embedding_matrix, chunks, docs)


if __name__ == "__main__":
    main()
