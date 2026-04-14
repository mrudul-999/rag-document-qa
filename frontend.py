import gradio as gr
import requests

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

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

import json

def ask_question(message: str, history: list):
    payload = {
        "question": message,
        "k": 4  # Retrieve the top 4 chunks
    }
    
    try:
        with requests.post(f"{API_URL}/stream", json=payload, stream=True) as response:
            if response.status_code == 200:
                answer = ""
                sources_text = ""
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data = json.loads(decoded_line[6:])
                            
                            if data["type"] == "token":
                                answer += data["content"]
                                yield answer
                            elif data["type"] == "sources":
                                if data["sources"]:
                                    source_pages = set([str(s["page"]) for s in data["sources"]])
                                    sources_text = "\n\n*(Sources: Page(s) " + ", ".join(source_pages) + ")*"
                                yield answer + sources_text
                            elif data["type"] == "error":
                                yield f"❌ Error from model: {data['content']}"
            elif response.status_code == 503:
                yield "⚠️ Please upload a PDF file on the left first!"
            else:
                yield f"Error: {response.text}"
                
    except requests.exceptions.ConnectionError:
        yield "❌ Error: Could not connect to the FastAPI backend. Is it running?"

# Create a custom theme
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="*shadow_sm",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
)

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
.header-text {
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #4f46e5, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
}
.subtitle-text {
    text-align: center;
    color: #64748b;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css) as app:
    with gr.Column():
        gr.HTML('''
            <h1 class="header-text">🧠 AI Document Intelligence</h1>
            <p class="subtitle-text">Upload your PDF and extract insights instantly powered by RAG</p>
        ''')
        
        with gr.Row():
            # LEFT COLUMN (Upload area)
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Knowledge Base")
                gr.Markdown("Upload a document to provide context to the AI.")
                with gr.Group():
                    file_input = gr.File(label="Upload File", file_types=[".pdf"])
                    upload_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")
                status_output = gr.Textbox(label="System Status", show_copy_button=False)
                
                # Connect the button click to the upload function
                upload_btn.click(
                    fn=upload_pdf, 
                    inputs=file_input, 
                    outputs=status_output,
                    api_name=False
                )
                
            # RIGHT COLUMN (Chat area)
            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    fn=ask_question,
                    chatbot=gr.Chatbot(height=500, label="Agent", show_copy_button=True),
                    textbox=gr.Textbox(placeholder="Ask a question about the document...", container=False, scale=7),
                )

# Run the app!
if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=8501)
