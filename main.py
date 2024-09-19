import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import Ollama

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load PDF documents
loader = PyPDFLoader('A Handbook on Employee Relations and Labour Laws in India.pdf')
documents = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(documents)

# Create the FAISS vector store from documents
vectorDb = FAISS.from_documents(final_documents, OpenAIEmbeddings())

# Initialize the LLM (Ollama)
llm = Ollama(model="llama3")

# Define a custom prompt template
def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (f"You are Insaaf Insight, an AI-driven legal assistant designed to help users with their legal queries in Indian courts. "
              f"Your demeanor is professional and informative. You will answer users' questions with your knowledge and the context provided. "
              f"If a question does not make any sense, or is not factually coherent, explain why instead of answering incorrectly. "
              f"If you don't know the answer to a question, please don't share false information. Be open about your capabilities and limitations. "
              f"Do not say thank you and do not mention that you are an AI Assistant. "
              f"If the passage is irrelevant to the answer, you may ignore it.\n\n"
              f"QUESTION: '{query}'\n"
              f"PASSAGE: '{escaped}'\n\n"
              f"ANSWER:")
    return prompt

# Function to get relevant passages from the FAISS database
def get_relevant_passages(query, n_results=3):
    try:
        results = vectorDb.similarity_search(query, n_results=n_results)
        if results:
            # Extract text from Document objects
            return " ".join([doc.page_content for doc in results])  # Adjust based on actual attribute
        else:
            raise ValueError("No relevant text found for the query.")
    except Exception as e:
        print(f"Error retrieving passages: {e}")
        return None

# Function to generate answers
def generate_answer(query):
    relevant_passage = get_relevant_passages(query)
    if not relevant_passage:
        return "No relevant text found for the query."
    
    prompt = make_rag_prompt(query, relevant_passage)
    
    # Directly invoke the model with the prompt string
    response = llm(prompt)  # Call the LLM with the formatted prompt
    return response  # Return the generated answer

# Example usage
if __name__ == "__main__":
    try:
        question = "Introduce yourself"
        answer = generate_answer(question)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")
