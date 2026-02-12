import streamlit as st
import numpy as np
from PyPDF2 import PdfReader

st.set_page_config(page_title="SmartDoc AI - Similarity Search")

st.title("SmartDoc AI - Similarity Search (Fully Stable)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content.replace("\n", " ")

    st.success("PDF Loaded Successfully!")

    # Split into chunks
    def chunk_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = chunk_text(text)

    st.success("Document Indexed!")

    query = st.text_input("Ask a question about this document")

    if query:

        # Simple keyword-based similarity
        scores = []

        query_words = set(query.lower().split())

        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            common_words = query_words.intersection(chunk_words)
            scores.append(len(common_words))

        best_match_index = np.argmax(scores)

        st.subheader("Most Relevant Context:")
        st.write(chunks[best_match_index])
