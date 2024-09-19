import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('A Handbook on Employee Relations and Labour Laws in India.pdf')
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
final_documents = text_splitter.split_documents(documents)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import faiss

# Create the vectorDb directory for storing the FAISS index
vector_db_dir = "vectorDb"
os.makedirs(vector_db_dir, exist_ok=True)

# Define the path for the FAISS index
faiss_file = os.path.join(vector_db_dir, "faiss_index.index")

# Check if the FAISS index file exists and load it, otherwise create and save it
if os.path.exists(faiss_file):
    # Load the FAISS index from file
    index = faiss.read_index(faiss_file)
    vectorDb = FAISS(index=index)
else:
    # Create a new FAISS index
    vectorDb = FAISS.from_documents(final_documents, OpenAIEmbeddings())
    
    # Save the FAISS index to a file
    index = vectorDb.index  # Access the underlying FAISS index
    faiss.write_index(index, faiss_file)

from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = vectorDb.as_retriever()
)

question = "What is the Document about?"
response = qa_chain.invoke({"query": question})
print(response["result"])
