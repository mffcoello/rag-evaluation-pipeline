# RAG Evaluation Pipeline

A retrieval-augmented generation (RAG) system with built-in evaluation metrics, built from scratch using LangChain, ChromaDB, and Hugging Face embeddings.

## What it does
Takes a set of documents, embeds them into a vector database, retrieves the most relevant chunks for any question, and evaluates retrieval quality using precision@k and user feedback.

## Why I built it
To understand how RAG systems work under the hood and how to measure whether retrieval is actually returning useful results — a core skill for production ML systems.

## How it works
1. Documents are split into chunks and embedded using `all-MiniLM-L6-v2` from Hugging Face
2. Embeddings are stored in ChromaDB
3. Questions are embedded and matched against stored chunks using cosine similarity
4. Results are scored with precision@k and logged with user feedback

## How to run it
Open `notebooks/rag_evaluation.ipynb` in Google Colab and run cells top to bottom.
