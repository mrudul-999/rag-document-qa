import gradio as gr
import requests

API_URL = "http://localhost:8000"

def upload_pdf(file) -> str:
    if file is None:
        return "Please select a file first."
    
    print(f"Uploading {file.name} to API...")
    
    # Open the file and send it to FastAPI
    with open(file.name, "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": f})
        
    if response.status_code == 200:
        data = response.json()
        return f"✅ Success! {data['filename']} loaded ({data['chunks_created']} chunks indexed)."
    else:
        return f"❌ Error: {response.text}"

def ask_question(message: str, history: list) -> str:
    payload = {
        "question": message,
        "k": 4  # Retrieve the top 4 chunks
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            
            # Format the source pages creatively!
            source_pages = set([str(s["page"]) for s in data["sources"]])
            sources_text = "\n\n*(Sources: Page(s) " + ", ".join(source_pages) + ")*"
            
            return answer + sources_text
        elif response.status_code == 503:
            return "⚠️ Please upload a PDF file on the left first!"
        else:
            return f"Error: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to the FastAPI backend. Is it running?"

# Create the layout
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Document Q&A RAG System")
    gr.Markdown("Upload a PDF to build the knowledge base, then ask questions about it!")
    
    with gr.Row():
        # LEFT COLUMN (Upload area)
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Ingest PDF", variant="primary")
            status_output = gr.Textbox(label="System Status")
            
            # Connect the button click to the upload function
            upload_btn.click(
                fn=upload_pdf, 
                inputs=file_input, 
                outputs=status_output,
                api_name=False
            )
            
        # RIGHT COLUMN (Chat area)
        with gr.Column(scale=3):
            gr.Markdown("### 2. Chat with Document")
            # Gradio's built-in chat interface maps directly to our ask_question function
            chat = gr.ChatInterface(
                fn=ask_question, 
                show_progress="hidden"
            )

# Run the app!
if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=8501)
