import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="SmartDoc AI - RAG System")
st.title("SmartDoc AI - Similarity Search & RAG System")

# Load Embedding Model (Lightweight + Stable)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
                text += page_text.replace("\n", " ")


    st.success("PDF Loaded Successfully!")

    # Split text into chunks
    def chunk_text(text, chunk_size=500):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks

    chunks = chunk_text(text)

    # Generate embeddings for document chunks
    doc_embeddings = model.encode(chunks)

    st.success("Document Indexed!")

    # User Question
    query = st.text_input("Ask a question about this document")

    if query:

        # Create embedding for query
        query_embedding = model.encode([query])

        # Calculate similarity
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Get most relevant chunk
        best_index = np.argmax(scores)
        best_chunk = chunks[best_index]

        st.subheader("Most Relevant Context:")
        st.write(best_chunk)
