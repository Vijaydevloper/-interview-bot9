"""Multiple-choice practice: a curated 10-question bank plus optional Groq-generated sets.

A question is {"topic", "question", "options": [4 strings], "answer": index, "explanation", "source"}.
Option order is shuffled when a quiz starts so the right answer is not always in the same slot.
"""
import json
import random
from pathlib import Path

BANK_PATH = Path(__file__).resolve().parent / "data" / "mcq_bank.json"
LETTERS = "ABCD"


def validate(q: dict) -> dict | None:
    """Return a cleaned question, or None if it is unusable (used for model-written questions too)."""
    try:
        opts = [str(o).strip() for o in q["options"]]
        ans = int(q["answer"])
        text = str(q["question"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if not text or len(opts) != 4 or len(set(opts)) != 4 or not all(opts) or not 0 <= ans < 4:
        return None
    return {"topic": str(q.get("topic") or "general"), "question": text, "options": opts, "answer": ans,
            "explanation": str(q.get("explanation", "")).strip(), "source": str(q.get("source", "")).strip()}


def load_bank(path: Path = BANK_PATH) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [q for q in (validate(i) for i in items) if q]


def prepare(questions: list[dict], seed: int | None = None) -> list[dict]:
    """Shuffle question order and each question's options; `answer` follows the shuffle."""
    rng = random.Random(seed)
    out = []
    for q in questions:
        order = list(range(4))
        rng.shuffle(order)
        out.append({**q, "options": [q["options"][i] for i in order], "answer": order.index(q["answer"])})
    rng.shuffle(out)
    return out


def grade(questions: list[dict], picks: list[int | None]) -> dict:
    """picks[i] is the chosen option index (or None if unanswered)."""
    rows = [{"correct": p == q["answer"], "pick": p} for q, p in zip(questions, picks)]
    return {"rows": rows, "correct": sum(r["correct"] for r in rows), "total": len(questions)}
