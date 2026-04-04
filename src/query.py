# src/query.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
TOP_K_CHUNKS = 4          # how many chunks to retrieve per query
TEMPERATURE = 0.1         # low = factual, high = creative


# ── PROMPT TEMPLATE ───────────────────────────────────────
# This is what gets sent to the LLM.
# {context} = the retrieved chunks
# {question} = the user's question
#
# Interview point: prompt engineering matters a lot here.
# "Answer ONLY from the context" prevents hallucination.
# "If you don't know, say so" prevents confident wrong answers.

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions.

Context:
{context}

Question: {question}

Instructions:
- Answer ONLY using information from the context above
- If the context doesn't contain enough information, say "I don't have
  enough information in the provided documents to answer this"
- Be concise and direct
- Do not make up information

Answer:"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# ── LOAD VECTORSTORE ────────────────────────────────────── 
def load_vectorstore() -> FAISS:
    """Load the FAISS index built during ingestion."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# ── BUILD QA CHAIN ────────────────────────────────────────
def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Wire together: retriever + prompt + LLM into a RetrievalQA chain.

    chain_type="stuff" means:
    - retrieve k chunks
    - "stuff" them all into one prompt
    - send to LLM in a single call

    Alternative chain types (know these for interviews):
    - "map_reduce": summarise each chunk separately, then combine
      → good for very long documents that exceed context window
    - "refine": answer iteratively, refining with each chunk
      → more accurate but much slower (k LLM calls instead of 1)
    - "map_rerank": answer from each chunk, pick highest scoring
      → good when chunks are very different quality
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=TEMPERATURE,
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    from langchain_huggingface import ChatHuggingFace
    llm = ChatHuggingFace(llm=endpoint)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",       # cosine similarity search
            search_kwargs={"k": TOP_K_CHUNKS}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,       # return which chunks were used
    )
    return qa_chain


# ── QUERY FUNCTION ────────────────────────────────────────
def query_documents(qa_chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the RAG pipeline.
    Returns answer + source chunks used to generate it.
    """
    # ── INTENT ROUTER (Small Talk Intercept) ──
    greetings = ["hi", "hii", "hello", "hey", "how are you", "what's up", "hey how are you"]
    if question.lower().strip() in greetings:
        return {
            "question": question,
            "answer": "Hello! I am your Document AI Assistant. I am doing great! Please ask me a question about the documents you've uploaded.",
            "sources": [],
            "num_chunks_used": 0,
        }

    # ── DEFAULT RAG PATH ──
    result = qa_chain.invoke({"query": question})

    # Format source documents for clean output
    sources = []
    for doc in result["source_documents"]:
        sources.append({
            "content": doc.page_content[:300],   # preview of chunk
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
        })

    return {
        "question": question,
        "answer": result["result"].strip(),
        "sources": sources,
        "num_chunks_used": len(sources),
    }


# ── QUICK TEST — run this file directly to verify ─────────
if __name__ == "__main__":
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("Building QA chain...")
    qa_chain = build_qa_chain(vectorstore)

    test_questions = [
        "What is the attention mechanism?",
        "What are the main components of the Transformer architecture?",
        "What training data was used?",
    ]

    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Q: {question}")
        result = query_documents(qa_chain, question)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['num_chunks_used']} chunks from pages "
              f"{[s['page'] for s in result['sources']]}")



