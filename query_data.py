from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

CHROMA_PATH = "chroma"
OLLAMA_BASE_URL = "http://localhost:11434"

def main():
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",  # or another suitable embedding model
        base_url=OLLAMA_BASE_URL
    )
    
    # Initialize Chroma with Ollama embeddings
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Set up retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})
    print(retriever)

    # Set up language model using Ollama
    llm = Ollama(base_url=OLLAMA_BASE_URL, model="llama2")

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Ask a question
    question = "how can i use CSI provider ?"
    try:
        result = qa_chain.invoke({"query": question})
        print("Answer:", result["result"])
        print("\nSources:")
        for source in result["source_documents"]:
            print(f"- {source.metadata['source']}")
    except Exception as e:
        print(f"Error during question answering: {e}")

if __name__ == "__main__":
    main()