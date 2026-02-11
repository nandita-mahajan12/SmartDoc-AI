SmartDoc AI – Similarity Search & RAG System

An AI-powered Similarity Search & Retrieval-Augmented Generation (RAG) system that allows users to upload a PDF and ask questions based on its content.
It uses Sentence Transformers + FAISS for semantic search and retrieves the most relevant document chunks to generate intelligent answers.

Features

 Upload any PDF document
 Semantic similarity search using embeddings
 FAISS vector database indexing
 RAG-based Question Answering
 Lightweight & fast (MiniLM embedding model)
 Streamlit web interface

Tech Stack

 Python
 Streamlit
 Sentence-Transformers
 FAISS (Vector Search)
 NumPy
 PyPDF

How It Works

 Upload a PDF
 Text is extracted from the document
 Text is split into smaller chunks
 Each chunk is converted into embeddings
 FAISS indexes the embeddings
 When user asks a question:
 Question is converted into embedding

Most similar chunks are retrieved

Retrieved context is used to generate answer
