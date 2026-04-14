# Document Q&A RAG System: Interview Upgrade Log

This document tracks the advanced features and architectural improvements made to the repository to make it a more impressive, production-ready project for technical interviews.

## 📝 Planned Upgrades

We will be iterating on the system based on the following planned features:

### 1. Advanced RAG Techniques
- [x] **Cross-Encoder Re-ranking**: Retrieve more documents initially, then re-rank them using a Cross-Encoder to significantly boost answer relevance.
- [ ] **Hybrid Search**: Combine Dense (FAISS) and Sparse (BM25) retrieval using Reciprocal Rank Fusion.
- [ ] **Smart Chunking**: Replace naive character splitting with Semantic/Document Hierarchy chunking.

### 2. Production Architecture & Backend
- [x] **Streaming Responses**: Implement Server-Sent Events (SSE) in FastAPI to stream LLM tokens directly to the Gradio interface in real-time.
- [ ] **Conversational Memory**: Add `ConversationBufferWindowMemory` to allow for multi-turn conversational context.
- [ ] **Background Processing**: Setup `FastAPI BackgroundTasks` for asynchronous document ingestion.
- [x] **Dockerization**: Add a `Dockerfile` and `docker-compose.yml` for instant setup.

### 3. UI/UX Polishing
- [x] **Premium Theming**: Integrated a custom Google Font (Inter), modern color palettes, gradient titles, and improved Chat UI layout using Gradio custom CSS.
- [ ] **Source Highlighting**: Add dynamic citations and expandible source text upon hovering/clicking citations.

### 4. Evaluation & Observability
- [ ] **Tracing**: Integrate LangSmith or Langfuse to log LLM inputs, latency, and costs.
- [ ] **Evaluation Framework**: Add a Ragas script to mathematically score the RAG pipeline's faithfulness and ground truth accuracy.

---

## 🛠️ Execution Log
*(We will log updates here as we complete them)*

**Date: April 13-14, 2026**
* **UI/UX Upgrade:** Completely refactored `frontend.py` to use `gr.themes.Soft` with custom indigo/blue styling, Google Fonts, and custom CSS for a centralized, modern look.
* **Streaming Responses:** Implemented an asynchronous streaming pipeline to drastically improve apparent latency:
  * **Langchain Async Handlers (`src/query.py`)**: Added an `AsyncStreamCallbackHandler` that intercepts LLM token generation and queues them into a non-blocking `asyncio.Queue` background task.
  * **Server-Sent Events (`src/api.py`)**: Designed a new `/stream` REST endpoint utilizing FastAPI's `StreamingResponse` to continuously pull tokens from the queue and flush them to the client instantly as Server-Sent Events (SSE). 
  * **Generator Functions (`frontend.py`)**: Migrated the Gradio `ChatInterface` to consume the stream via generator (`yield`) over `requests.iter_lines()`, giving the chat a fluid "typewriter" effect just like ChatGPT.
* **Cross-Encoder Re-ranking:** Replaced standard FAISS top-k retrieval with a `ContextualCompressionRetriever`. The pipeline now over-fetches 15 chunks (for high recall), then scores each chunk against the user's query using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model, filtering the context window down to the exactly 4 most definitively relevant chunks to feed into the generative model.
* **Dockerization:** Wrote a multi-service `docker-compose.yml` architecture with a unified `Dockerfile` to instantiate the FastAPI backend (`api` container) and Gradio frontend (`frontend` container) cleanly. Mounted local volumes to persist vector embeddings (`faiss_index`) explicitly, and passed `.env` data securely. Re-wired the UI logic to allow dynamic API service-name networking inside the Docker bridge network.
