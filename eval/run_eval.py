"""
Score Insaaf Insight on eval/questions.jsonl.

    python eval/run_eval.py            # uses GROQ_API_KEY if set, else local Ollama

Reports retrieval hit@k (a gold page among the retrieved pages), answer accuracy
(the answer contains one of the answer keys), refusals on out-of-scope questions,
whether pages cited inside the answer were actually retrieved, and latency and tokens.
"""
import json
import os
import re
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from main import TOP_K, answer_with_stats, get_vector_store


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    return round(values[min(len(values) - 1, int(p * len(values)))]) if values else None


def normalise(text: str) -> str:
    """Lower-case and replace the narrow/no-break spaces and non-breaking hyphens models often emit."""
    for odd, plain in (("\u202f", " "), ("\u00a0", " "), ("\u2009", " "), ("\u2011", "-"), ("\u2010", "-")):
        text = text.replace(odd, plain)
    return " ".join(text.split()).lower()


def cited_pages(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\b(?:p|pp|page|pages)\.?\s*(\d{1,3})", answer, re.IGNORECASE)}


with open(os.path.join(ROOT, "eval", "questions.jsonl")) as f:
    items = [json.loads(line) for line in f if line.strip()]

get_vector_store()  # build the index before timing anything
rows = []
for item in items:
    try:
        result = answer_with_stats(item["question"])
    except Exception as e:
        print(f"{item['id']}: ERROR {e}")
        rows.append({"id": item["id"], "type": item["type"], "error": str(e)})
        continue
    answer = normalise(result["answer"])
    row = {
        "id": item["id"],
        "type": item["type"],
        "answer": result["answer"],
        "found": result["found"],
        "retrieved_pages": result["retrieved_pages"],
        "llm_ms": result["llm_ms"],
        "retrieval_ms": result["retrieval_ms"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "model": result["model"],
    }
    if item["type"] == "fact":
        row["retrieval_hit"] = any(p in result["retrieved_pages"] for p in item["gold_pages"])
        row["correct"] = result["found"] and any(normalise(k) in answer for k in item["answer_keys"])
        cites = cited_pages(result["answer"])
        row["cites_unretrieved_page"] = bool(cites - set(result["retrieved_pages"]))
    else:
        row["correct"] = not result["found"]
    rows.append(row)
    print(f"{item['id']}: {'ok ' if row['correct'] else 'MISS'} found={result['found']} pages={result['retrieved_pages']} "
          f"{result['llm_ms']} ms | {result['answer'][:90]!r}")

facts = [r for r in rows if r["type"] == "fact" and "error" not in r]
oos = [r for r in rows if r["type"] == "out_of_scope" and "error" not in r]
answered = [r for r in rows if "error" not in r]
summary = {
    "date": date.today().isoformat(),
    "model": next((r["model"] for r in answered if r["model"]), None),
    "top_k": TOP_K,
    "questions": len(items),
    "errors": len(rows) - len(answered),
    "retrieval_hit_at_k": f"{sum(r['retrieval_hit'] for r in facts)}/{len(facts)}",
    "answer_accuracy": f"{sum(r['correct'] for r in facts)}/{len(facts)}",
    "accuracy_when_retrieval_hit": f"{sum(r['correct'] for r in facts if r['retrieval_hit'])}/{sum(r['retrieval_hit'] for r in facts)}",
    "false_refusals": f"{sum(not r['found'] for r in facts)}/{len(facts)}",
    "out_of_scope_declined": f"{sum(r['correct'] for r in oos)}/{len(oos)}",
    "answers_citing_unretrieved_pages": f"{sum(r['cites_unretrieved_page'] for r in facts)}/{len(facts)}",
    "llm_ms_p50": pct([r["llm_ms"] for r in answered], 0.5),
    "llm_ms_p95": pct([r["llm_ms"] for r in answered], 0.95),
    "retrieval_ms_p50": pct([r["retrieval_ms"] for r in answered], 0.5),
    "avg_input_tokens": round(sum(r["input_tokens"] or 0 for r in answered) / max(len(answered), 1)),
    "avg_output_tokens": round(sum(r["output_tokens"] or 0 for r in answered) / max(len(answered), 1)),
}
out = os.path.join(ROOT, "eval", "results", f"run_{summary['date']}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"Saved {out}")
