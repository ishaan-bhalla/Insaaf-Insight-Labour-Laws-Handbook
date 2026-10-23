import os
import time
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


def pick_groq_models(api_key: str) -> list[str]:
    """Groq renames and retires models, so choose chat models it currently serves, best first.
    Prefers a Llama model, then the open-weight gpt-oss and Qwen models; set GROQ_MODEL to pin the first."""
    import httpx
    ids = [m["id"] for m in httpx.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {api_key}"},
                                      timeout=20).raise_for_status().json()["data"]]
    # Skip safety classifiers and speech models: they can't answer questions.
    chat = [i for i in ids if not any(x in i.lower() for x in ("guard", "whisper", "orpheus", "tts"))]
    if not chat:
        raise RuntimeError(f"No chat model available on Groq; set GROQ_MODEL to one of: {', '.join(ids)}")
    ranked = []
    for pref in ("llama", "gpt-oss-20b", "gpt-oss-120b", "qwen"):
        ranked += [m for m in chat if pref in m.lower() and m not in ranked]
    return ranked + [m for m in chat if m not in ranked]


@lru_cache(maxsize=1)
def get_llm():
    """Llama via Groq when GROQ_API_KEY is set (hosted demo), otherwise via a local Ollama server.
    On Groq, a second model takes over if the first errors or is rate-limited after retries."""
    if os.getenv("GROQ_API_KEY"):
        from langchain_openai import ChatOpenAI
        key = os.getenv("GROQ_API_KEY")
        models = pick_groq_models(key)
        if os.getenv("GROQ_MODEL"):
            models = [os.getenv("GROQ_MODEL")] + [m for m in models if m != os.getenv("GROQ_MODEL")]

        def groq(model):
            # Groq exposes an OpenAI-compatible API.
            return ChatOpenAI(base_url=GROQ_URL, api_key=key, model=model, temperature=0, timeout=60, max_retries=3)

        primary = groq(models[0])
        return primary.with_fallbacks([groq(models[1])]) if len(models) > 1 else primary
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=0)


def make_rag_prompt(query: str, passages: list) -> str:
    context = "\n\n".join(f"[Page {doc.metadata.get('page', 0) + 1}] {doc.page_content}" for doc in passages)
    return (
        "You are Insaaf Insight, a legal assistant for Indian labour law. Answer the question using only the "
        "passages below, which come from 'A Handbook on Labour Laws of India'.\n"
        "- Start with the answer itself, in plain words. Do not open with 'According to the passage' or a page number.\n"
        "- Put the source at the end of the sentence it supports, e.g. '...48 hours a week (Section 51, p. 14).'\n"
        "- Keep it short: one to three sentences unless the question needs more.\n"
        "- If the passages don't contain the answer, say you couldn't find it in the handbook instead of guessing.\n\n"
        f"PASSAGES:\n{context}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


def get_relevant_passages(query: str, k: int = TOP_K) -> list:
    return get_vector_store().similarity_search(query, k=k)


def answer_with_stats(query: str) -> dict:
    """Answer a question and report the pages used, retrieval and LLM time, and token counts."""
    started = time.perf_counter()
    passages = get_relevant_passages(query)
    retrieval_ms = (time.perf_counter() - started) * 1000
    retrieved_pages = sorted({doc.metadata.get("page", 0) + 1 for doc in passages})
    stats = {"retrieved_pages": retrieved_pages, "retrieval_ms": round(retrieval_ms),
             "llm_ms": None, "input_tokens": None, "output_tokens": None, "model": None}
    if not passages:
        return {"answer": "No relevant text found for the query.", "pages": [], "found": False, **stats}

    started = time.perf_counter()
    reply = get_llm().invoke(make_rag_prompt(query, passages))
    stats["llm_ms"] = round((time.perf_counter() - started) * 1000)
    usage = getattr(reply, "usage_metadata", None) or {}
    stats.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                 model=reply.response_metadata.get("model_name") or reply.response_metadata.get("model"))

    answer = reply.content.strip()
    # Only list sources when the handbook actually answered the question.
    found = not any(phrase in answer.lower() for phrase in ("couldn't find", "couldn\u2019t find", "could not find"))
    return {"answer": answer, "pages": retrieved_pages if found else [], "found": found, **stats}


def generate_answer(query: str) -> tuple[str, list[int]]:
    """Return the answer and the handbook pages it was based on."""
    result = answer_with_stats(query)
    return result["answer"], result["pages"]


if __name__ == "__main__":
    question = "What is the maximum number of working hours per week for an adult worker in a factory?"
    answer, pages = generate_answer(question)
    print(answer)
    print("Sources: handbook page" + ("s " if len(pages) > 1 else " ") + ", ".join(map(str, pages)))
