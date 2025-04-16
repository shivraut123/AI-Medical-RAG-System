import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# --- UPDATED LINE BELOW ---
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Setup the path
DATA_PATH = "data/"

def create_vector_db():
    print("⏳ Loading PDF... (This might take a few minutes)")
    
    # Load the PDF file from the data folder
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages.")

    # 2. Split Text into Chunks
    # The AI cannot read the whole book at once, so we split it into small pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(texts)} chunks.")

    # 3. Create Embeddings
    # This turns text into numbers so the computer can understand it
    print("⏳ Creating Embeddings... (Downloading model on first run)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # 4. Create Vector DB (FAISS)
    print("⏳ Building Vector Database...")
    db = FAISS.from_documents(texts, embeddings)
    
    # 5. Save Locally
    db.save_local("vectorstore/db_faiss")
    print("🎉 Success! Database saved to 'vectorstore/db_faiss'")

if __name__ == "__main__":
    create_vector_db()