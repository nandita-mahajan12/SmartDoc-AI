import streamlit as st
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartDoc AI - RAG System")
st.title("SmartDoc AI - Similarity Search & RAG System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Read PDF
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content.replace("\n", " ")

    st.success("PDF Loaded Successfully!")

    # Split into chunks
    def chunk_text(text, chunk_size=500):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    chunks = chunk_text(text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)

    st.success("Document Indexed!")

    query = st.text_input("Ask a question about this document")

    if query:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, vectors).flatten()
        best_match_index = np.argmax(similarities)

        st.subheader("Most Relevant Context:")
        st.write(chunks[best_match_index])
