## Insaaf Insight 
Insaaf Insight is an AI-powered legal assistant built using a Retrieval-Augmented Generation (RAG) technique to answer questions from a pdf named: 

📄 “A Handbook on Employee Relations and Labour Laws in India”

It uses:

- Llama 3 served locally through Ollama.
- Hugging Face embeddings.
- FAISS for fast semantic search.

This is a project, built using open-source LLM and embedding model easy to understand and scale.

## 🚀 Features

1. ✔️ Retrieval-Augmented Generation (RAG)

- Retrieves relevant sections from the labour law handbook

- Ensures grounded, citation-backed answers

- Reduces hallucinations

2. ✔️ Fully Local & Open-Source

- No external APIs

- Runs entirely on your machine

- Uses Llama 3 + FAISS + HF embeddings

3. ✔️ Very Simple Project Structure

- Great for learning and showcasing RAG fundamentals.

## 🛒 Requirements
Python 3.11 (REQUIRED)



## Project Structure
```
├── app.py
├── main.py
├── ui/
├── requirements.txt
├── README.md
└── A Handbook on Employee Relations and Labour Laws in India.pdf
```
The main files in the project are following:

- `app.py`: Contains the frontend code to interact with users and display the information.
- `main.py`: Contains the backend logic to process requests and fetch relevant data.
- `requirements.txt`: Contains the list of dependencies required to run the project.

## Prerequisites

To run this project, you need to have the following installed:

- Python 3.11: This project is tested and built for Python 3.11. Other versions may cause library conflicts.

## Installation & Setup
1. Install Ollama inorder to run Llama 3 locally <a href="https://ollama.com/download"><strong>Ollama</strong>
  

2.  Download Llama3 model
   ```bash
   Ollama pull llama3
   ```

3. Clone the Repo
   ```bash
   git clone https://github.com/ishaan-bhalla/Insaaf-Insight-Labour-Laws-Handbook.git
   cd Insaaf-Insight-Labour-Laws-Handbook
   ```
   
4. Create Python virtual environment
   ```bash
   python -m venv venv
   ```
5. Activate Python environment
   For Windows
   ```bash
   venv\Scripts\activate
   ```
   For Mac
   ```bash
   source venv/bin/activate
   ```
6. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```
## ▶️ Running the Project

1. Run the main file
   ```bash
   python main.py
   ```

2. Run the project using Streamlit
   ```bash
   streamlit run app.py
   ```

## 🎯 Why This Project Exists

- Learn RAG fundamentals
- Build a clean legal assistant
- Use fully local, open-source models
- Keep everything simple and understandable
    
