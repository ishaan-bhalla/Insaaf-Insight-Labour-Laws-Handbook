<div align="center">

<img src="bot.png" alt="Insaaf Insight logo" width="120">

# Insaaf Insight

**A retrieval-augmented (RAG) legal assistant that answers questions about Indian labour law from a 96-page handbook, and shows the pages each answer came from.**

[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://insaaf-insight.streamlit.app/)
![Llama 3](https://img.shields.io/badge/Llama_3-Groq_or_Ollama-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FAISS-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

</div>

## Example

> **Q:** What is the maximum number of working hours per week for an adult worker in a factory?
>
> **A:** According to the passage on Page 14, a worker cannot be employed for more than 48 hours in a week. (Section 51)
>
> *Sources: handbook page 14*

> **Q:** Is a weekly holiday compulsory for factory workers?
>
> **A:** According to the passage on Page 14, a weekly holiday is compulsory. If a worker is asked to work on their weekly holiday, they should have a full holiday on one of three days immediately or after the normal day of holiday. [Section 52(1)].
>
> *Sources: handbook page 14*

Both are real outputs (Llama 3, temperature 0), and both match page 14 of the handbook.

## Contents

1. [How it works](#1-how-it-works)
2. [Run it](#2-run-it)
3. [Deploying the demo](#3-deploying-the-demo)
4. [Limitations](#4-limitations)
5. [Team](#5-team)
6. [Source document](#6-source-document)

## 1. How it works

[![Architecture diagram of ishaan-bhalla/insaaf-insight-labour-laws-handbook](https://gitdiagram.com/ishaan-bhalla/insaaf-insight-labour-laws-handbook/diagram.png)](https://gitdiagram.com/ishaan-bhalla/insaaf-insight-labour-laws-handbook?utm_source=readme&utm_medium=picture)

*Architecture diagram (GitDiagram; click to explore it interactively).*

- **Retrieval:** LangChain's `PyPDFLoader` and `RecursiveCharacterTextSplitter`, Hugging Face sentence embeddings, and a FAISS similarity search. The index is built once per process.
- **Generation:** Llama 3 runs through **Groq** when `GROQ_API_KEY` is set (as on the hosted demo), or through a local **Ollama** server otherwise. Both run at temperature 0.
- **Grounding:** the prompt passes each passage with its page number. The app lists the pages behind every answer, so it can be checked against the handbook.

## 2. Run it

Tested with Python 3.12.

```bash
git clone https://github.com/ishaan-bhalla/Insaaf-Insight-Labour-Laws-Handbook.git
cd Insaaf-Insight-Labour-Laws-Handbook
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then pick a model:

- **Fully local (free, no API keys):** install [Ollama](https://ollama.com/download) and run `ollama pull llama3`.
- **Hosted Llama 3 (free tier):** create a key at [console.groq.com](https://console.groq.com) and put `GROQ_API_KEY=...` in a `.env` file.

```bash
python main.py           # answers one example question in the terminal
streamlit run app.py     # the chat app
```

Optional settings: `GROQ_MODEL` (default `llama-3.1-8b-instant`), `OLLAMA_MODEL` (default `llama3`), and `LANGCHAIN_API_KEY` to turn on LangSmith tracing.

## 3. Deploying the demo

Streamlit Community Cloud can't run Ollama, so the hosted app uses Groq. In the app's **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_..."
```

## 4. Limitations

- **One document.** It only knows this handbook, which predates India's 2019-2020 labour codes (the file dates from 2014, and the latest year it mentions is 2006). It can't speak to later law, other statutes or case law.
- **No accuracy benchmark.** Answers have been spot-checked against the handbook, not scored on a labelled question set.
- **Retrieval can miss.** Only the 3 most similar chunks are used. A question phrased differently from the handbook may retrieve the wrong passages, and the answer is only as good as those passages.
- **Page citations come from retrieval, not the model.** The listed pages are what was retrieved; the model can still misread them.
- **Not legal advice.** It's an information tool.

## 5. Team

B.Tech final-year project (Group 88), Department of CSE & IT, The NorthCap University, supervised by Dr. Nitin Malik (May 2024). Responsibilities as recorded in the project report (Chapter 13):

| Task | Responsible |
|---|---|
| Project planning and research | Ishaan, Priyanshu & Mukul |
| Environment setup | Priyanshu |
| Data collection and preprocessing | Ishaan & Mukul |
| Model selection and training | Priyanshu & Ishaan |
| Model evaluation and optimisation | Ishaan & Mukul |
| UI development | Mukul |
| Front end | Mukul & Ishaan |
| Back end | Priyanshu |
| Testing | Mukul |
| Documentation | Ishaan |

Team: **Ishaan Bhalla**, **Priyanshu Mishra** and **Mukul**. In 2026 Ishaan updated the code so the hosted demo runs on Groq, made tracing optional and added page citations to every answer.

## 6. Source document

The knowledge base is *A Handbook on Labour Laws of India* by **Rajkumar S. Adukia** (B.Com (Hons.), LL.B, AICWA, FCA). All credit for its content belongs to the author; it is included here only so the project runs out of the box.
