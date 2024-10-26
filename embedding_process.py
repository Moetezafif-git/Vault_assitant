from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import time

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "doc"
OLLAMA_BASE_URL = "http://localhost:11434"

def main():
    print("Starting the process...")
    start_time = time.time()
    generate_data_store()
    end_time = time.time()
    print(f"Process completed in {end_time - start_time:.2f} seconds.")

def generate_data_store():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    print("Splitting documents into chunks...")
    chunks = split_text(documents)
    
    print("Saving chunks to Chroma...")
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.mdx")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared existing Chroma database at {CHROMA_PATH}.")

    # Initialize OllamaEmbeddings
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url=OLLAMA_BASE_URL
    )

    # Create a new Chroma instance
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Set a reasonable batch size
    batch_size = 500

    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        db.add_texts(texts=texts, metadatas=metadatas)
        print(f"Processed batch {i//batch_size + 1} of {len(chunks)//batch_size + 1}")

    # Persist the database
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
