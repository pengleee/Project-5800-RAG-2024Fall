import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import openai
import streamlit as st
from sentence_transformers import CrossEncoder

class DenseRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", reranker_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.reranker = CrossEncoder(reranker_model_name)

    def build_index(self, documents):
        """Build a FAISS index for the given documents."""
        self.documents = documents
        embeddings = self.model.encode(documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        """Retrieve the top-k most relevant documents for a query using FAISS."""
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, k)
        results = [(self.documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
        return results

    def rerank(self, query, retrieved_docs):
        """Re-rank the retrieved documents using a cross-encoder."""
        pairs = [(query, doc[0]) for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        return [(doc[0], score) for doc, score in reranked]

class RAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever

    def generate_response(self, query, model_name="gpt-3.5-turbo", k=5):
        """Generate a response by combining retrieval and generation."""
        if not openai.api_key:
            raise ValueError("API key is not set. Please enter your OpenAI API key.")
        
        retrieved_docs = self.retriever.retrieve(query, k)
        reranked_docs = self.retriever.rerank(query, retrieved_docs)
        context = "\n".join([f"{i+1}. {doc}" for i, (doc, _) in enumerate(reranked_docs)])
        prompt = (
            f"Contextual documents:\n{context}\n\n"
            f"Query: {query}\n"
            f"Response:"
        )
        response = openai.ChatCompletion.create(
            model=model_name, # change model based on user selection
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response['choices'][0]['message']['content'].strip(), reranked_docs

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

st.set_page_config(page_title="Contextual Retrieval-Augmented Generation (RAG) System", page_icon="📖", layout="wide")

# Sidebar Instructions & API Key Input
st.sidebar.header("App Instructions")
st.sidebar.write("""
This is a Contextual Retrieval-Augmented Generation (RAG) system. 
1. Upload a document (TXT, PDF, DOCX) using the file uploader.
2. After uploading, input a question about the content of the document.
3. The system will retrieve relevant documents and generate a response using a language model.
""")

# API key input with reminder and question mark
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password", 
    help="You can get a key at https://platform.openai.com/account/api-keys."
)

if openai_api_key:
    openai.api_key = openai_api_key

# Model Selection
model_name = st.sidebar.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4o"])

# Main Header
st.header("📖 Contextual Retrieval-Augmented Generation (RAG) System")

# File Upload
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
    try:
        # Pass the user-selected model to generate_response
        response, retrieved_docs = rag_system.generate_response(query, model_name=model_name)
        
        st.subheader("Response")
        st.write(response)

        st.subheader("Retrieved Documents")
        for i, (doc, score) in enumerate(retrieved_docs):
            st.write(f"{i+1}. {doc} (Score: {score})")
    except ValueError as e:
        st.error(str(e))

## conda activate dsan5800
## streamlit run rag_app_new.py