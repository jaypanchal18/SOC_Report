import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF parsing
import openai
import streamlit as st
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key securely from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of the MiniLM embeddings
index = faiss.IndexFlatL2(dimension)

# Connect to SQLite database
conn = sqlite3.connect('knowledge_base.db')
cursor = conn.cursor()

# Create table for storing context
cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS context_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence TEXT,
        embedding BLOB
    )
''')
conn.commit()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to generate embeddings for the extracted text
def generate_embeddings(sentences):
    embeddings = embedding_model.encode(sentences)
    return embeddings

# Function to store sentences and embeddings in SQLite and FAISS
def store_in_knowledge_base(sentences, embeddings):
    for sentence, embedding in zip(sentences, embeddings):
        embedding_blob = embedding.tobytes()
        cursor.execute(
            "INSERT INTO context_data (sentence, embedding) VALUES (?, ?)", (sentence, embedding_blob))
        conn.commit()

    # Add embeddings to FAISS index
    index.add(embeddings)

# Function to retrieve relevant sentences using FAISS
def retrieve_from_knowledge_base(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1:  # Ensure it's a valid index
            cursor.execute(
                "SELECT sentence FROM context_data WHERE id=?", (idx + 1,))
            result = cursor.fetchone()
            if result:
                results.append(result[0])
    return results

# Function to generate a response using OpenAI GPT-4
def generate_response(prompt, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nAnswer the following question: {prompt}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit Interface
st.title("RAG System with Vector Database Integration")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Create a directory for storing uploaded files if it doesn't exist
upload_folder = "uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Extract text, generate embeddings, and store them in the vector database
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        text = extract_text_from_pdf(file_path)
        sentences = text.split('.')
        embeddings = generate_embeddings(sentences)
        store_in_knowledge_base(sentences, embeddings)
    st.success("Documents processed and stored in the knowledge base.")

    # Now, display the query input field after files are uploaded
    query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        # Retrieve context from the knowledge base
        retrieved_context = retrieve_from_knowledge_base(query)
        combined_context = " ".join(retrieved_context)

        # Generate a response using the retrieved context
        answer = generate_response(query, combined_context)
        st.write("Answer:", answer)

# Close the database connection when the app stops
conn.close()
