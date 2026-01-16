from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    # Using BAAI/bge-large-en-v1.5 which produces 1024-dimensional embeddings to match Pinecone index
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return embeddings