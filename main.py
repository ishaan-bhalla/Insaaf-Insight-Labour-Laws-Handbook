import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = "A Handbook on Employee Relations and Labour Laws in India.pdf"
TOP_K = 3  # passages retrieved per question

# LangSmith tracing is optional: switch it on only when a key is provided.
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"


@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:
    """Load the handbook, split it into overlapping chunks and index them (built once per process)."""
    documents = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


GROQ_URL = "https://api.groq.com/openai/v1"


def pick_groq_model(api_key: str) -> str:
    """Groq renames and retires models, so choose a Llama chat model it currently serves.
    Prefers small, fast ones; set GROQ_MODEL to pin a specific model instead."""
    import httpx
    ids = [m["id"] for m in httpx.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {api_key}"},
                                      timeout=20).raise_for_status().json()["data"]]
    llama = [i for i in ids if "llama" in i.lower() and not any(x in i.lower() for x in ("guard", "prompt"))]
    for pref in ("instant", "8b", "versatile", "70b"):
        for model in llama:
            if pref in model.lower():
                return model
    if llama:
        return llama[0]
    raise RuntimeError(f"No Llama model available on Groq; set GROQ_MODEL to one of: {', '.join(ids)}")


@lru_cache(maxsize=1)
def get_llm():
    """Llama via Groq when GROQ_API_KEY is set (hosted demo), otherwise via a local Ollama server."""
    if os.getenv("GROQ_API_KEY"):
        from langchain_openai import ChatOpenAI
        key = os.getenv("GROQ_API_KEY")
        # Groq exposes an OpenAI-compatible API.
        return ChatOpenAI(base_url=GROQ_URL, api_key=key, model=os.getenv("GROQ_MODEL") or pick_groq_model(key),
                          temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=0)


def make_rag_prompt(query: str, passages: list) -> str:
    context = "\n\n".join(f"[Page {doc.metadata.get('page', 0) + 1}] {doc.page_content}" for doc in passages)
    return (
        "You are Insaaf Insight, an AI-driven legal assistant that answers questions about Indian labour law "
        "using passages from 'A Handbook on Labour Laws of India'. Your demeanor is professional and informative.\n"
        "Answer only from the passages below, and cite the page numbers you used, e.g. (p. 12). "
        "If the passages don't contain the answer, say so instead of guessing. "
        "Do not say thank you and do not mention that you are an AI assistant.\n\n"
        f"PASSAGES:\n{context}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


def get_relevant_passages(query: str, k: int = TOP_K) -> list:
    return get_vector_store().similarity_search(query, k=k)


def generate_answer(query: str) -> tuple[str, list[int]]:
    """Return the answer and the handbook pages it was based on."""
    passages = get_relevant_passages(query)
    if not passages:
        return "No relevant text found for the query.", []
    response = get_llm().invoke(make_rag_prompt(query, passages))
    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in passages})
    return response.content, pages


if __name__ == "__main__":
    question = "What is the maximum number of working hours per week for an adult worker in a factory?"
    answer, pages = generate_answer(question)
    print(answer)
    print("Sources: handbook page" + ("s " if len(pages) > 1 else " ") + ", ".join(map(str, pages)))
