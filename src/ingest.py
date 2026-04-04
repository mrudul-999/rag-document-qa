# src/ingest.py

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────
# All magic numbers in one place — easier to explain in interviews
# "I centralised config so tuning is one place, not scattered"

CHUNK_SIZE = 512        # max tokens per chunk
CHUNK_OVERLAP = 50      # overlap between chunks to preserve boundary context
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

# ── STEP 1: LOAD ─────────────────────────────────────────
def load_document(file_path: str) -> list:
    """
    Load a PDF and return a list of Document objects.
    Each Document has:
      - page_content: the raw text of that page
      - metadata: {"source": filepath, "page": page_number}
    """
    print(f"📄 Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"   Loaded {len(pages)} pages")
    return pages


# ── STEP 2: SPLIT ─────────────────────────────────────────
def split_documents(pages: list) -> list:
    """
    Split pages into chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks,
    then sentences, then words — it's 'recursive' because it tries each
    separator in order until chunks are small enough.

    chunk_overlap ensures sentences that fall on a boundary aren't
    cut in half — last 50 tokens of chunk N appear at start of chunk N+1.
    """
    print(f"✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        # tries these separators in order — prefers paragraph breaks
    )
    chunks = splitter.split_documents(pages)
    print(f"   Created {len(chunks)} chunks")

    # Show a sample so you can see what a chunk looks like
    if chunks:
        print(f"\n   Sample chunk (first 200 chars):")
        print(f"   '{chunks[0].page_content[:200]}...'")
        print(f"   Metadata: {chunks[0].metadata}\n")

    return chunks


# ── STEP 3: EMBED + INDEX ─────────────────────────────────
def build_vectorstore(chunks: list) -> FAISS:
    """
    Convert chunks to vectors and store in FAISS.

    HuggingFaceEmbeddings runs the model LOCALLY — no API call,
    no cost, no rate limits. all-MiniLM-L6-v2 produces 384-dim vectors.

    FAISS.from_documents:
      - calls embeddings.embed_documents() on every chunk
      - builds an index for fast similarity search
      - stores both the vectors AND the original text
    """
    print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~80MB — subsequent runs use cache)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
        # normalize = vectors are unit length → cosine similarity
        # is equivalent to dot product → faster computation
    )

    index_path = Path(FAISS_INDEX_PATH)
    if index_path.exists() and (index_path / "index.faiss").exists():
        print(f"🗂️  Found existing FAISS index! Adding {len(chunks)} new chunks to it...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
        print(f"   Index updated ✅")
    else:
        print(f"🗂️  Building NEW FAISS index from {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"   Index built ✅")

    return vectorstore, embeddings


# ── STEP 4: SAVE ──────────────────────────────────────────
def save_vectorstore(vectorstore: FAISS, path: str = FAISS_INDEX_PATH):
    """
    Persist FAISS index to disk.
    Saves two files:
      - faiss_index/index.faiss  → the actual vectors
      - faiss_index/index.pkl    → the text chunks + metadata
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(path)
    print(f"💾 FAISS index saved to '{path}/'")


# ── STEP 5: LOAD BACK (verify it works) ───────────────────
def load_vectorstore(path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    Must use the SAME embedding model used during ingestion.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
        # this flag acknowledges you trust the source of the index file
    )
    print(f"📂 FAISS index loaded from '{path}/'")
    return vectorstore


# ── MAIN: run the full pipeline ────────────────────────────
def ingest(file_path: str):
    """
    Full ingestion pipeline:
    PDF → load → split → embed → FAISS index → save to disk
    """
    print("\n" + "="*50)
    print("RAG INGESTION PIPELINE")
    print("="*50 + "\n")

    pages = load_document(file_path)
    chunks = split_documents(pages)
    vectorstore, embeddings = build_vectorstore(chunks)
    save_vectorstore(vectorstore)

    # ── VERIFY: do a test search ──────────────────────────
    print("\n🔍 Verification — test similarity search:")
    test_query = "what does a cow eat"
    results = vectorstore.similarity_search_with_score(test_query, k=3)

    for i, (doc, score) in enumerate(results):
        print(f"\n   Result {i+1} (similarity score: {score:.4f}):")
        print(f"   '{doc.page_content[:150]}...'")
        print(f"   Source: page {doc.metadata.get('page', 'unknown')}")

    print(f"\n✅ Ingestion complete — {len(chunks)} chunks indexed")
    return vectorstore


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        print(f"   Put a PDF in the data/ folder and run:")
        print(f"   python src/ingest.py data/yourfile.pdf")
        sys.exit(1)

    ingest(file_path)