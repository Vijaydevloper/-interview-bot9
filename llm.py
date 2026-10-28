"""LLM layer: question generation, rubric evaluation, explanations, revision sheets.

With GROQ_API_KEY set, calls Groq's OpenAI-compatible chat API (stdlib HTTP, no SDK).
Without it, simple offline fallbacks keep the app usable for testing; evaluations are
then clearly labelled as a keyword heuristic, not a real assessment.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

import config  # noqa: F401  (loads GROQ_API_KEY etc. from .env)

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

RUBRIC = {"correctness": 4, "coverage": 3, "clarity": 2, "example": 1}


def online() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def model_name() -> str:
    return os.environ.get("GROQ_MODEL", MODEL)


_NOT_CHAT = ("whisper", "orpheus", "guard", "safeguard", "tts", "embed")
_PREFERRED = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def available_models() -> list[str]:
    """Chat models this API key can actually use (GET /models). Empty list if the call fails."""
    if not online():
        return []
    req = urllib.request.Request(
        os.environ.get("GROQ_BASE_URL", BASE_URL).rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "User-Agent": "interview-drill/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError):
        return []
    return sorted(i for i in ids if not any(w in i.lower() for w in _NOT_CHAT))


def _pick_model(models: list[str]) -> str | None:
    return next((m for m in _PREFERRED if m in models), models[0] if models else None)


def _post(body: dict) -> dict:
    req = urllib.request.Request(
        os.environ.get("GROQ_BASE_URL", BASE_URL).rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "User-Agent": "interview-drill/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=int(os.environ.get("GROQ_TIMEOUT", "60"))) as r:
        return json.loads(r.read())


def _complete(system: str, user: str, max_tokens: int = 1500, json_mode: bool = False,
              temperature: float = 0.3, messages: list | None = None) -> str:
    body = {
        "model": model_name(),
        "messages": messages or [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    waited = 0
    switched = False
    while True:
        try:
            data = _post(body)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 404 and "model" in detail and not switched:
                # This key can't use the configured model: switch to one it can use, once.
                switched = True
                models = available_models()
                pick = _pick_model([m for m in models if m != body["model"]])
                if pick:
                    os.environ["GROQ_MODEL"] = pick
                    body["model"] = pick
                    continue
                detail += f" Models available to your key: {', '.join(models) or 'could not be listed'}."
            if e.code == 429 and waited < 30:  # free-tier rate limit: wait as told, then retry
                try:
                    pause = min(float(e.headers.get("retry-after", 5)), 15)
                except (TypeError, ValueError):
                    pause = 5
                time.sleep(pause)
                waited += pause
                continue
            if e.code == 400 and json_mode and "response_format" in detail:
                body.pop("response_format")  # model without JSON mode: the prompt still asks for JSON
                continue
            hints = {401: " Check GROQ_API_KEY.", 404: " Check GROQ_MODEL.",
                     429: " Rate limit reached; wait a minute and retry."}
            raise RuntimeError(f"Groq error {e.code}: {detail}{hints.get(e.code, '')}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise RuntimeError(f"Cannot reach Groq ({e}). Check your connection.") from e
    return (data["choices"][0]["message"].get("content") or "").strip()


def _fmt(chunks) -> str:
    return "\n\n".join(
        f"[{i}] ({c.ref}{' - ' + c.section if c.section else ''})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )


# ---------- quiz question ----------
def generate_question(chunks, focus: str | None = None, difficulty: str = "Intermediate",
                      mode: str = "Mixed") -> str:
    if online():
        hint = f"The student asked to focus on: {focus}\n\n" if focus else ""
        kind = {"Conceptual": "a conceptual", "Coding": "a hands-on coding"}.get(mode, "a")
        return _complete(
            "You are a data-science interviewer. Use ONLY the study notes provided. "
            "If they are insufficient, say so instead of inventing details.",
            hint + f"Write ONE {difficulty.lower()} {kind} interview question (not a yes/no question) "
            "that can be answered from these notes. Do not give the answer. "
            "Return only the question text.\n\n" + _fmt(chunks),
            max_tokens=250,
            temperature=0.5,
        )
    return _offline_question(chunks)


_GENERIC = re.compile(r"^(\d+(\.\d+)*\.?\s*)?(introduction|overview|summary|conclusion|contents|quick comparison)$", re.I)


def _offline_question(chunks) -> str:
    """Pick the most specific heading available; avoid 'Introduction'-style titles."""
    for c in chunks:
        sec = re.sub(r"^\d+(\.\d+)*\.?\s*", "", c.section or "").strip()
        if sec and not _GENERIC.match(c.section.strip()):
            return f"Explain in your own words ({c.topic}): {sec}"
    c = chunks[0]
    first = next((ln for ln in c.text.splitlines() if ln.strip()), "this topic")
    return f"Explain in your own words ({c.topic}): {first.lstrip('#').strip()[:100]}"


# ---------- evaluation ----------
_EVAL_SYSTEM = (
    "You are a fair, consistent technical interviewer grading an answer against the "
    "candidate's own study notes. Accept equivalent wording; do not penalise answers "
    "that are correct but phrased differently from the notes. Respond with JSON only."
)


def _clamp(v, hi) -> float:
    try:
        return max(0.0, min(float(v), hi))
    except (TypeError, ValueError):
        return 0.0


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    return [str(x) for x in v]


def _finalise(raw: dict, mode: str) -> dict:
    """Enforce the rubric caps in code instead of trusting the model's arithmetic."""
    ex_ok = bool(raw.get("example_applicable", True))
    parts = {k: _clamp(raw.get(k, 0), hi) for k, hi in RUBRIC.items()}
    if not ex_ok:
        parts["example"] = 0.0
    max_score = sum(RUBRIC.values()) - (0 if ex_ok else RUBRIC["example"])
    return {
        "parts": parts,
        "score": sum(parts.values()),
        "max_score": max_score,
        "correct_points": _as_list(raw.get("correct_points")),
        "missing_points": _as_list(raw.get("missing_points")),
        "model_answer": str(raw.get("model_answer", "")),
        "feedback": str(raw.get("feedback", "")),
        "mode": mode,
    }


def evaluate(question: str, answer: str, chunks) -> dict:
    if online():
        prompt = (
            f"Question: {question}\n\nCandidate answer:\n{answer}\n\nStudy notes:\n{_fmt(chunks)}\n\n"
            "Score with this rubric: correctness 0-4, coverage 0-3 (key concepts), "
            "clarity 0-2, example 0-1. Set example_applicable=false if an example would be "
            "unnecessary for this question. Return JSON with keys: correctness, coverage, "
            "clarity, example, example_applicable, correct_points (list), missing_points "
            "(list), model_answer (grounded in the notes), feedback (2-3 sentences)."
        )
        raw = None
        for _ in range(3):  # models occasionally break the JSON; retry
            text = _complete(_EVAL_SYSTEM, prompt, max_tokens=1200, json_mode=True, temperature=0.1)
            m = re.search(r"\{.*\}", text, re.S)
            try:
                raw = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                raw = None
            if isinstance(raw, dict) and any(k in raw for k in RUBRIC):
                break
            raw = None
        if raw is None:
            raise ValueError("The model did not return a parsable evaluation. Please retry.")
        return _finalise(raw, "llm")

    # Offline heuristic: keyword overlap with the retrieved notes.
    words = re.findall(r"[a-zA-Z_]{5,}", " ".join(c.text for c in chunks).lower())
    keys = [w for w, _ in Counter(words).most_common(12)]
    got = [k for k in keys if k in answer.lower()]
    ratio = len(got) / len(keys) if keys else 0
    n = len(answer.split())
    raw = {
        "correctness": round(4 * ratio),
        "coverage": round(3 * ratio),
        "clarity": 2 if n >= 30 else 1 if n >= 10 else 0,
        "example": 1 if re.search(r"for example|e\.g\.|such as", answer.lower()) else 0,
        "correct_points": [f"mentioned: {k}" for k in got],
        "missing_points": [f"not mentioned: {k}" for k in keys if k not in got],
        "model_answer": chunks[0].text if chunks else "",
        "feedback": "Offline keyword-overlap heuristic (no API key). Set GROQ_API_KEY "
        "for a real rubric-based evaluation.",
    }
    return _finalise(raw, "offline-heuristic")


# ---------- explain ----------
def explain(question: str, notes_chunks, web_docs) -> str:
    if not online():
        parts = ["_Offline mode: showing retrieved passages only._"]
        parts += [f"**{c.ref}**\n\n{c.text}" for c in notes_chunks]
        parts += [f"**{d['url']}**\n\n{d['text'][:800]}" for d in web_docs]
        return "\n\n".join(parts)
    web = "\n\n".join(f"[W{i}] {d['url']}\n{d['text']}" for i, d in enumerate(web_docs, 1))
    return _complete(
        "You are a patient data-science tutor. Keep information from the student's notes "
        "clearly separate from information from external documentation. If neither source "
        "answers the question, say so plainly rather than guessing.",
        f"Question: {question}\n\nStudent notes:\n{_fmt(notes_chunks) or '(none found)'}\n\n"
        f"External documentation:\n{web or '(none retrieved)'}\n\n"
        "Answer with sections 'From your notes' and 'From official docs' (omit a section "
        "that has nothing). Cite sources as [n] or [Wn].",
    )


# ---------- revision sheet ----------
def revision_sheet(topic: str, avg: float | None, chunks, mistakes: list[dict], focus: str = "") -> str:
    level = f"current average {avg}/10" if avg is not None else "not practised yet"
    miss = "\n".join(
        f"- Q: {m['question']}\n  Feedback: {m['feedback']}" for m in mistakes if m.get("feedback")
    )
    if not online():
        body = "\n\n".join(f"**{c.ref}**\n\n{c.text}" for c in chunks) or "_No notes found._"
        return f"## {topic} ({level})\n\n{body}"
    return _complete(
        "You write concise interview revision sheets grounded in the student's notes. "
        "Use only the notes; do not add unsupported facts.",
        f"Topic: {topic} ({level})\nFocus: {focus or 'core concepts'}\n\nNotes:\n{_fmt(chunks)}\n\n"
        f"Recent weak attempts:\n{miss or '(none recorded)'}\n\n"
        "Write markdown with: Key concepts, Common mistakes, Corrected explanation, and one "
        "sample interview answer.",
        max_tokens=1800,
    )


# ---------- multiple-choice generation ----------
def generate_mcqs(chunks, n: int = 10, difficulty: str = "Intermediate") -> list[dict]:
    """Ask Groq for n four-option questions grounded in the notes. Needs a key; returns only valid ones."""
    import mcq

    if not online():
        raise RuntimeError("Generating new questions needs a Groq API key (put it in .env).")
    prompt = (
        f"Write {n} {difficulty.lower()} multiple-choice interview questions that can be answered "
        "ONLY from the study notes below. Each has exactly 4 distinct options, one correct, and "
        "plausible wrong options. Spread the correct answer across positions. Return JSON: "
        '{"questions": [{"topic": str, "question": str, "options": [4 strings], '
        '"answer": index 0-3, "explanation": str, "source": str}]}. '
        "Use the note's file name as the source and its topic folder name as the topic.\n\n" + _fmt(chunks)
    )
    for _ in range(3):
        text = _complete("You write accurate, unambiguous exam questions from supplied notes. JSON only.",
                         prompt, max_tokens=3500, json_mode=True, temperature=0.4)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            items = json.loads(m.group(0)).get("questions", []) if m else []
        except (json.JSONDecodeError, AttributeError):
            items = []
        good = [q for q in (mcq.validate(i) for i in items if isinstance(i, dict)) if q]
        if good:
            return good[:n]
    raise ValueError("The model did not return usable questions. Please retry.")


# ---------- chatbot practice ----------
_CHAT_SYSTEM = (
    "You are a friendly but rigorous mock interviewer for a data-science candidate. Ground every "
    "technical statement ONLY in the study notes below; if the notes don't cover something, say so "
    "instead of guessing.\n\n"
    "How to behave:\n"
    "- If the conversation is empty: greet in one short sentence and ask ONE {difficulty} question.\n"
    "- If the candidate answered your last question: give brief feedback (what was right, what is "
    "missing, a score out of 10), then ask ONE new question. Do not repeat earlier questions.\n"
    "- If the candidate asks for a hint, an explanation or has their own question: answer from the "
    "notes in plain words, then invite them to continue.\n"
    "- Never reveal the full answer before they have tried, unless they ask for it.\n"
    "Keep replies under 170 words. Use short markdown. Topic focus: {topic}.\n\nSTUDY NOTES:\n{notes}"
)


def chat_reply(history: list[dict], chunks, topic: str = "", difficulty: str = "Intermediate") -> str:
    """history: [{"role": "user"|"assistant", "content": str}, ...]; returns the interviewer's next message."""
    if online():
        system = _CHAT_SYSTEM.format(difficulty=difficulty.lower(), topic=topic or "any",
                                     notes=_fmt(chunks) or "(no matching notes found)")
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
        if not history:
            msgs.append({"role": "user", "content": "Start the interview."})
        return _complete("", "", max_tokens=600, temperature=0.4, messages=msgs)

    # Offline: no real conversation possible. Ask heading-based questions and keyword-check answers.
    if not history or history[-1]["role"] != "user":
        return "_Offline mode (no Groq key), so I can only ask basic questions._\n\n**" + _offline_question(chunks) + "**"
    last_q = next((m["content"] for m in reversed(history[:-1]) if m["role"] == "assistant"), "")
    res = evaluate(last_q, history[-1]["content"], chunks)
    missing = "; ".join(p.removeprefix("not mentioned: ") for p in res["missing_points"][:5])
    return (f"_Offline keyword check, not a real assessment:_ **{res['score']:g}/{res['max_score']:g}**\n\n"
            f"Terms from your notes you didn't use: {missing or 'none'}.\n\n"
            f"**Next question:** {_offline_question(chunks)}")
