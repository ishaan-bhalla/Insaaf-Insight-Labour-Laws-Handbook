<div align="center">

<img src="bot.png" alt="Insaaf Insight logo" width="120">

# Insaaf Insight

**A retrieval-augmented (RAG) legal assistant that answers questions about Indian labour law from a 96-page handbook, and shows the pages each answer came from.**

[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://insaaf-insight.streamlit.app/)
![Open-weight LLMs](https://img.shields.io/badge/Open--weight_LLMs-Groq_or_Ollama-0467DF?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-FAISS-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

</div>

## Example

> **Q:** What is the maximum number of working hours per week for an adult worker in a factory?
>
> **A:** An adult worker in a factory cannot be employed for more than 48 hours in a week (Section 51).
>
> *Sources: handbook page 14*

> **Q:** What is the minimum wage in Karnataka in 2026?
>
> **A:** I couldn't find the answer in the handbook. The passages provided do not mention the minimum wage in Karnataka in 2026.

Real outputs from Llama 3 via Ollama at temperature 0. The first matches page 14 of the handbook. The second shows it declining a question the handbook can't answer, instead of guessing.

## Contents

1. [How it works](#1-how-it-works)
2. [Run it](#2-run-it)
3. [Deploying the demo](#3-deploying-the-demo)
4. [Evaluation](#4-evaluation)
5. [Limitations](#5-limitations)
6. [Team](#6-team)
7. [Source document](#7-source-document)

## 1. How it works

[![Architecture diagram of ishaan-bhalla/insaaf-insight-labour-laws-handbook](https://gitdiagram.com/ishaan-bhalla/insaaf-insight-labour-laws-handbook/diagram.png)](https://gitdiagram.com/ishaan-bhalla/insaaf-insight-labour-laws-handbook?utm_source=readme&utm_medium=picture)

*Architecture diagram (GitDiagram; click to explore it interactively).*

- **Retrieval:** LangChain's `PyPDFLoader` and `RecursiveCharacterTextSplitter`, Hugging Face sentence embeddings, and a FAISS similarity search. The index is built once per process.
- **Generation:** an open-weight model on **Groq** when `GROQ_API_KEY` is set (as on the hosted demo; the app picks a chat model Groq currently serves, preferring Llama, then gpt-oss), or Llama 3 on a local **Ollama** server otherwise. Both run at temperature 0.
- **Grounding:** the prompt passes each passage with its page number. The app lists the pages behind every answer, so it can be checked against the handbook.
- **Fallback:** on Groq a second model takes over if the first errors or stays rate-limited after 3 retries (60 s timeout per call). If both fail, the app says the model is busy instead of showing a raw error.

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

Optional settings: `GROQ_MODEL` (by default the app picks a chat model Groq currently serves), `OLLAMA_MODEL` (default `llama3`), and `LANGCHAIN_API_KEY` to turn on LangSmith tracing.

## 3. Deploying the demo

Streamlit Community Cloud can't run Ollama, so the hosted app uses Groq. In the app's **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_..."
```

## 4. Evaluation

`eval/questions.jsonl` holds 24 factual questions spread across the handbook (pages 6-89), each with its gold page(s), a verbatim supporting snippet and short answer keys, plus 6 questions the handbook doesn't cover (UK redundancy pay, the US minimum wage, Indian tax slabs and others). `python eval/run_eval.py` scores the app on them.

Run on 23 Oct 2026 with `openai/gpt-oss-20b` on Groq's free tier, top 3 chunks:

| Metric | Result |
|---|---|
| Retrieval: a gold page among the 3 retrieved | **21/24** |
| Answer contains the right fact | **19/24** (18/21 when retrieval found the page) |
| Out-of-scope questions declined | **6/6** |
| Answers citing a page that wasn't retrieved | **0/24** |
| Wrongly said "couldn't find it" | 3/24 |
| LLM time per answer (median / 95th pct) | 6.7 s / 10.7 s |
| Retrieval time (median) | 55 ms |
| Tokens per answer (average in / out) | 816 / 172 |
| Cost | $0 (Groq free tier) |

The misses are worth reading: two answers used outside knowledge rather than the passages (60 weekly hours including overtime, where the handbook says 48; a 10% provident fund rate the handbook says rose to 12% in 1997), and three were refusals, two of them because retrieval missed the page. The first answers took 0.6-1.1 s; the median is higher because the free tier throttled later calls. Results are in `eval/results/`. Answer keys are matched as text, so a correct answer in unusual wording would count as a miss.

## 5. Limitations

- **One document.** It only knows this handbook, which predates India's 2019-2020 labour codes (the file dates from 2014, and the latest year it mentions is 2006). It can't speak to later law, other statutes or case law.
- **Small eval set.** 30 questions written from the handbook (a script checks every supporting snippet appears on its gold page). It shows the failure types but is too small for fine-grained comparisons.
- **Retrieval can miss.** Only the 3 most similar chunks are used. A question phrased differently from the handbook may retrieve the wrong passages, and the answer is only as good as those passages.
- **Page citations come from retrieval, not the model.** The listed pages are what was retrieved; the model can still misread them.
- **Not legal advice.** It's an information tool.

## 6. Team

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

## 7. Source document

The knowledge base is *A Handbook on Labour Laws of India* by **Rajkumar S. Adukia** (B.Com (Hons.), LL.B, AICWA, FCA). All credit for its content belongs to the author; it is included here only so the project runs out of the box.
