import os
from dotenv import load_dotenv
import regex as re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
api_key = GOOGLE_API_KEY
if not api_key:
    raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")

from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Replace the path with your file path
pdf_text = load_pdf(file_path="A Handbook on Employee Relations and Labour Laws in India.pdf")

def split_text(text: str) -> List[str]:
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

chunked_text = split_text(text=pdf_text)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    collections = chroma_client.list_collections()
    if name in [collection.name for collection in collections]:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    else:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))
    return db, name

db, name = create_chroma_db(documents=chunked_text, path="vectorstore", name="rag_ex")

def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db

db = load_chroma_collection(path="vectorstore", name="rag_ex")

def get_relevant_passage(query, db, n_results):
    try:
        results = db.query(query_texts=[query], n_results=n_results)
        if 'documents' in results and results['documents']:
            passage = results['documents'][0]
            print(passage)
            return passage
        else:
            raise ValueError("No documents found for the query.")
    except Exception as e:
        print(f"Error retrieving passage: {e}")
        return None

def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are Insaaf Insight, an AI-driven legal assistant designed to help users with their legal queries in Indian courts. 
    Your demeanor is professional and informative. You will answer users' questions with your knowledge and the context provided. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering incorrectly. 
    If you don't know the answer to a question, please don't share false information. Be open about your capabilities and limitations.
    Do not say thank you and do not mention that you are an AI Assistant \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    return prompt

def generate_ans(prompt):
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def generate_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    if not relevant_text:
        return "No relevant text found for the query."
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = generate_ans(prompt)
    return answer

# Example usage
if __name__ == "__main__":
    try:
        db = load_chroma_collection(path="vectorstore", name="rag_ex")
        answer = generate_answer(db, query="What is Labour law?")
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")
