# test_setup.py — Mini RAG Playground
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Initialize Embeddings
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Define a small knowledge base
test_texts = [
    "HikariCP is a high-performance JDBC connection pool.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "LangChain is a framework for developing applications powered by large language models.",
    "Retrieval-Augmented Generation (RAG) improves LLM responses by adding retrieved context.",
    "Python is a versatile programming language widely used in AI and data science."
]

# 3. Create Vector Store
print("Creating vector store...")
vectorstore = FAISS.from_texts(test_texts, embeddings)

# 4. Perform a search (with scores!)
query = "what is hikariCP"
k = 2
results = vectorstore.similarity_search_with_score(query, k=k)

print(f"\n🔍 Query: '{query}'")
print(f"Top {len(results)} results (Lower score = More similar):")
for i, (res, score) in enumerate(results):
    print(f" {i+1}. [Score: {score:.4f}] {res.page_content}")

# 💡 Observation: For math or random facts NOT in your docs, 
# you'll notice the 'Score' is much higher (meaning less similar).

# 💡 IDEA: You can add more texts above or try different queries!