import streamlit as st
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartDoc AI - RAG System")
st.title("SmartDoc AI - Similarity Search & RAG System")

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

    st.success("Document Indexed!")

    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)

    # Ask Question
    question = st.text_input("Ask a question about this document")

    if question:

        question_vector = vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, vectors)
        most_similar_index = np.argmax(similarities)

        st.subheader("Most Relevant Context:")
        st.write(chunks[most_similar_index])
