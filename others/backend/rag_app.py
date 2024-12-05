import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import openai
import streamlit as st
import json
import os


# import the openai api key
home_dir = os.environ['HOME']

with open(f'{home_dir}/.api-keys.json') as f:
    keys = json.load(f)

# Your API key
API_KEY = keys['OPENAI_API_KEY']

# API_KEY = 'YOUR OPENAI API KEY'

class DenseRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        """Build a FAISS index for the given documents."""
        self.documents = documents
        embeddings = self.model.encode(documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        """Retrieve the top-k most relevant documents for a query."""
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, k)
        results = [(self.documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
        return results

class RAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        openai.api_key = API_KEY  # use your own api key

    def generate_response(self, query, k=5):
        """Generate a response by combining retrieval and generation."""
        retrieved_docs = self.retriever.retrieve(query, k)
        context = "\n".join([f"{i+1}. {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
        prompt = (
            f"Contextual documents:\n{context}\n\n"
            f"Query: {query}\n"
            f"Response:"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # can change the model base on your task
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response['choices'][0]['message']['content'].strip(), retrieved_docs

def process_uploaded_file(file):
    """Extract text from different document types."""
    if file.type == "text/plain":
        return file.read().decode("utf-8").split("\n")
    elif file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.split("\n")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.split("\n")
    else:
        raise ValueError("Unsupported file type!")

# Streamlit UI
st.title("Contextual Retrieval-Augmented Generation (RAG) System")
uploaded_file = st.file_uploader("Upload a document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

retriever = DenseRetriever()
rag_system = RAGSystem(retriever)

if uploaded_file:
    try:
        documents = process_uploaded_file(uploaded_file)
        retriever.build_index(documents)
        st.success("Document index built successfully!")
    except ValueError as e:
        st.error(str(e))

query = st.text_input("Enter your question:")
if query and retriever.index:
    response, retrieved_docs = rag_system.generate_response(query)
    st.subheader("Response")
    st.write(response)

    st.subheader("Retrieved Documents")
    for i, (doc, score) in enumerate(retrieved_docs):
        st.write(f"{i+1}. {doc} (Score: {score})")


## conda activate dsan5800
# streamlit run ./backend/rag_app.py