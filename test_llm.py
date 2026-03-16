import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv()
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=10)
chat = ChatHuggingFace(llm=llm)
print(chat.invoke("Hello"))
