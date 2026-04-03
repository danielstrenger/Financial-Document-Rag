from langchain_community.document_loaders import PyPDFLoader
import os
import argparse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import hashlib

def parse_document(file_path: str) -> list[Document]:
    """
    Loads a PDF document and splits it into semantic chunks for vectorization.

    Utilizes LangChain's PyPDFLoader for ingestion and RecursiveCharacterTextSplitter 
    to create overlapping text chunks, ensuring contextual integrity is maintained 
    across chunk boundaries.

    Args:
        file_path (str): The absolute or relative path to the target PDF file.

    Returns:
        list[Document]: A list of LangChain Document objects containing the chunked 
                        text and associated metadata.
    """
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(document)
    return split_documents

def embed_documents(documents: list[Document]) -> None:
    """
    Embeds text chunks and persists them into a local Chroma vector database.

    Initializes a HuggingFace embedding model ('all-MiniLM-L6-v2') and computes 
    vector representations for the provided documents. It generates deterministic 
    SHA-256 hashes based on chunk content and source metadata to serve as unique 
    identifiers, preventing duplicate entries on subsequent executions.

    Args:
        documents (list[Document]): A list of chunked LangChain Document objects 
                                    ready for vectorization.
    """
    vector_store = Chroma(
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory="./chroma_db"
    )

    # use hash values as ids to prevent overwriting of existing embeddings when script is run again with new documents.
    ids = [
        hashlib.sha256((doc.page_content + doc.metadata.get('source', '')).encode()).hexdigest() 
        for doc in documents
    ]

    vector_store.add_documents(
        documents = documents,
        ids = ids
    )
    return vector_store

def main() -> None:
    """
    Executes the command-line ETL pipeline for document ingestion.

    Parses the '--path' argument to locate target PDF files (either a single file 
    or a directory). Orchestrates the parsing, chunking, and embedding processes, 
    and handles file-not-found exceptions. Outputs execution metrics to the console.
    """
    parser = argparse.ArgumentParser(description="Ingest PDFs into a Chroma vector store.")
    parser.add_argument(
        "--path",
        required=False,
        help="Path to a PDF file or a directory containing PDF files.",
        default="./documents"
    )
    args = parser.parse_args()

    docs_path = args.path

    # check if the path exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Path not found: {docs_path}")

    # check if the path points to a directory or a single pdf file
    if os.path.isdir(docs_path):
        # get all the pdf files in the directory
        pdf_files = [
            os.path.join(docs_path, f)
            for f in os.listdir(docs_path)
            if f.endswith(".pdf")
        ]

        if len(pdf_files) == 0:
            raise FileNotFoundError(f"No pdf files found in the directory: {docs_path}")

    elif os.path.isfile(docs_path):
        pdf_files = [docs_path]
    else:
        raise FileNotFoundError(f"Not a file or directory: {docs_path}")

    # split the documents
    documents: list[Document] = []
    for pdf_file in pdf_files:
        documents = documents.extend(parse_document(pdf_file))

    embed_documents(documents)
    print(f"Successfully embedded {len(documents)} chunk(s) from {len(pdf_files)} documents into ./chroma_db.")


if __name__ == "__main__":
    main()