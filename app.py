import os
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from flask import Flask, request, jsonify
from langchain.text_splitter import CharacterTextSplitter  # Added this import

# New imports for Hugging Face Transformers
from transformers import pipeline

app = Flask(__name__)

# Global Variables (Initialized when the API starts)
vectorstore = None
llm = None

# LLM Model Selection (Choose a more powerful model)
LLM_MODEL_NAME = "google/flan-ul2"  # Or "google/flan-t5-xl", or "google/flan-ul2" - try these!
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"

# Data Loading Functions (Modified)


def load_pdf(pdf_path):
    """Loads text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        print(f"Loaded PDF: {pdf_path}")
        return text
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return None


def clean_transcription(text):
    """Cleans the transcription by removing timestamps and extra whitespace."""
    text = re.sub(r'\d+:\d+:\d+.\d+ --> \d+:\d+:\d+.\d+', '', text)  # Remove timestamps
    text = re.sub(r'\[\d+:\d+:\d+]', '', text) # Remove timestamps like [00:00:00]
    text = re.sub(r'\(Refer Slide Time: \d+:\d+\)', '', text)  # Remove (Refer Slide Time: ...)
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

# Modified chunk_text function
def chunk_text(text, chunk_size=800, chunk_overlap=100):
    """Splits text into smaller chunks, attempting to respect sentence boundaries."""
    text_splitter = CharacterTextSplitter( #Changed from RecursiveCharacterTextSplitter to CharacterTextSplitter
        separator="\n", # This tells the splitter to split based on new lines which will keep sections together
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_embeddings(chunks, embeddings_model_name=EMBEDDINGS_MODEL_NAME):
    """Creates embeddings for text chunks using Hugging Face Sentence Transformers."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    print(f"Using embeddings model: {embeddings_model_name}")
    return embeddings

def create_vectorstore(chunks, embeddings, persist_directory="chroma_db"):
    """Creates a Chroma vector store from text chunks and embeddings."""
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Vectorstore created and persisted to: {persist_directory}")
    return vectorstore

def load_vectorstore(persist_directory="chroma_db", embeddings_model_name=EMBEDDINGS_MODEL_NAME):
    """Loads an existing Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(f"Vectorstore loaded from: {persist_directory}")
    return vectorstore

# Function to initialize the Hugging Face pipeline
def initialize_llm(model_name=LLM_MODEL_NAME):
    """Initializes the Hugging Face pipeline for text generation."""
    global llm
    try:
        llm = pipeline("text2text-generation", model=model_name, device="cuda:0" if app.config['USE_CUDA'] else "cpu") #Added device
        print(f"Hugging Face pipeline initialized with model: {model_name}")
    except Exception as e:
        print(f"Error initializing Hugging Face pipeline: {e}")
        llm = None

# Modified query_llm function
def query_llm(query, context):
    """Queries the Hugging Face pipeline with a context and a question."""
    global llm
    if llm:
        prompt = f"""You are an assistant that answers questions about software engineering based on lecture slides. Use the following context from lecture slides to answer the question at the end. If you cannot answer based on the context, respond with I don't know the answer.

        Context:
        {context}

        Question: {query}
        Answer:
        """
        try:
            result = llm(prompt, max_length=500, do_sample=False, temperature=0.1) #Added temperature
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error querying Hugging Face pipeline: {e}")
            return "Sorry, I encountered an error while generating the answer."
    else:
        return "The Hugging Face pipeline is not initialized. Please create or load the knowledge base first."

# API Endpoints
@app.route('/create_knowledge_base', methods=['POST'])
def create_knowledge_base():
    """
    Creates the knowledge base (vectorstore) from a transcript file and PDF paths.
    Expects JSON payload with 'transcript_path' and 'pdf_paths' (list).
    """
    global vectorstore
    try:
        data = request.get_json()
        transcript_path = data.get('transcript_path')
        pdf_paths = data.get('pdf_paths', [])  # Default to empty list if not provided

        if not transcript_path:
            return jsonify({"error": "Transcript path is required"}), 400

        # Transcript Processing
        #transcript_text = load_transcript(transcript_path) #USE THE BELOW CODE INSTEAD
        transcript_text = load_pdf(transcript_path) #USE THE pdf loader here
        if transcript_text:
            cleaned_transcript = clean_transcription(transcript_text)
            transcript_chunks = chunk_text(cleaned_transcript)
        else:
            print("Transcript loading failed.")
            transcript_chunks = []  # Empty list if loading fails

        # PDF Processing
        pdf_chunks = []
        for pdf_path in pdf_paths:
            pdf_text = load_pdf(pdf_path)
            if pdf_text:
                pdf_chunks.extend(chunk_text(pdf_text))

        # Combine Chunks
        all_chunks = transcript_chunks + pdf_chunks

        # Create Embeddings and Vectorstore
        if all_chunks:
            embeddings = create_embeddings(all_chunks)
            vectorstore = create_vectorstore(all_chunks, embeddings)
            initialize_llm() #Initialize LLM here
            return jsonify({"message": "Knowledge base created successfully"}), 200
        else:
            return jsonify({"error": "No text chunks to process"}), 400

    except Exception as e:
        print(f"Error creating knowledge base: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/load_knowledge_base', methods=['POST'])
def load_knowledge_base():
    """Loads the knowledge base (vectorstore) from the persist directory."""
    global vectorstore
    try:
        vectorstore = load_vectorstore()
        initialize_llm() # Initialize LLM here too
        return jsonify({"message": "Knowledge base loaded successfully"}), 200

    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_chatbot():
    """
    Queries the chatbot.
    Expects JSON payload with 'query' field.
    """
    global vectorstore
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if vectorstore:
            # Perform similarity search to retrieve relevant context
            results = vectorstore.similarity_search(query, k=4)  # Retrieve top 4 most similar chunks
            context = "\n".join([doc.page_content for doc in results])  # Concatenate the contexts
            answer = query_llm(query, context) # Call LLM with context
            return jsonify({"answer": answer}), 200
        else:
            return jsonify({"error": "Knowledge base not initialized. Call /create_knowledge_base or /load_knowledge_base first."}), 500
    except Exception as e:
        print(f"Error querying chatbot: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.config['USE_CUDA'] = False  # Set to False if you don't have CUDA
    app.run(debug=True) # Remove debug=True for production
