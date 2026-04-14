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
        streaming=True,
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


import asyncio
from langchain.callbacks.base import BaseCallbackHandler

async def stream_query_documents(qa_chain: RetrievalQA, question: str):
    """
    Run the RAG pipeline asynchronously and stream tokens back using native astream().
    """
    # ── INTENT ROUTER ──
    greetings = ["hi", "hii", "hello", "hey", "heyy", "heyyy", "how are you", "what's up", "hey how are you"]
    if question.lower().strip() in greetings:
        yield {"type": "token", "content": "Hello! I am your Document AI Assistant. "}
        yield {"type": "token", "content": "Please ask me a question about the documents you've uploaded."}
        yield {"type": "sources", "sources": []}
        return

    try:
        # 1. Retrieve context
        retriever = qa_chain.retriever
        docs = retriever.invoke(question)  # use sync invoke to avoid faiss async limitations
        
        # 2. Combine document context
        context = "\n\n".join([d.page_content for d in docs])
        
        # 3. Format Prompt
        prompt_chain = qa_chain.combine_documents_chain.llm_chain
        formatted_prompt = prompt_chain.prompt.format(context=context, question=question)
        
        # 4. Stream from LLM via modern LCEL
        llm = prompt_chain.llm
        
        async for chunk in llm.astream(formatted_prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield {"type": "token", "content": content}

        # 5. Yield source metadata
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:300],
                "page": doc.metadata.get("page", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
            })
        yield {"type": "sources", "sources": sources}

    except Exception as e:
        yield {"type": "error", "content": str(e)}

# ── QUICK TEST — run this file directly to verify ─────────
if __name__ == "__main__":
    pass



