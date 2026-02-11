import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="SmartDoc AI - Similarity Search & RAG System")
st.title("SmartDoc AI - Similarity Search & RAG System")

# Load embedding model (lightweight + stable)
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
        text += page.extract_text()

    st.success("PDF Loaded Successfully!")

    # Split text into chunks
    def chunk_text(text, chunk_size=500):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks

    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = model.encode(chunks)

    # Convert to numpy float32
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    st.success("Document Indexed!")

    # Ask Question
    question = st.text_input("Ask a question about this document")

    if question:
        query_embedding = model.encode([question])
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = index.search(query_embedding, k=3)

        st.subheader("Top Relevant Chunks:")
        context = ""

        for idx in I[0]:
            st.write(chunks[idx])
            context += chunks[idx] + "\n\n"

        st.subheader("Answer:")
        st.success(context)
