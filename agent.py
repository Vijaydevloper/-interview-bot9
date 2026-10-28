r"""LangGraph agent behind every tab of the app.

One graph, six tasks (the last two are `mcq` and `chat`). The entry edge picks a branch from state["task"]:

  question : [pick_topic] -> prepare_question -> END
  evaluate : evaluate -> log_attempt -> END
  explain  : retrieve --notes cover it / web off--> answer -> END
                   \--notes fall short--> lookup_docs -> answer -> END
  revise   : revise -> END
  mcq      : mcq (sample notes -> Groq writes 4-option questions) -> END
  chat     : chat (search notes for the last message -> interviewer reply) -> END

Nodes are plain functions (state in, partial state out). Every data access is an MCP
tool call through tools.call (rag.*, docs.*, progress.*); the graph only decides which
tool runs next, and the LLM (Groq) is called from llm.py.

state["trace"] records each step so the UI can show how a result was produced.
"""
from typing import TypedDict

import llm
import rag
import scheduler
import tools

NOTES_SUFFICIENT = 0.12  # min top TF-IDF similarity to treat the notes as covering a question
SMART_PICK = "Smart pick"


class State(TypedDict, total=False):
    task: str            # question | evaluate | explain | revise
    topics: list[str]
    topic: str | None
    use_web: bool
    # practice
    difficulty: str
    mode: str            # Conceptual | Coding | Mixed
    focus: str
    question: str
    answer_text: str     # the candidate's answer (evaluate)
    context: list[dict]  # chunks the question was generated from / the answer is graded against
    result: dict         # rubric evaluation
    attempt_id: int
    notice: str
    # explain
    notes: list[dict]
    covered: bool
    web_docs: list[dict]
    answer: str
    sources: list[dict]
    # revise
    sheet: str
    # mcq / chat
    count: int
    questions: list[dict]
    history: list[dict]  # chat turns so far: [{"role": "user"|"assistant", "content": str}]
    reply: str
    trace: list[str]


def _t(state: State, line: str) -> list[str]:
    return list(state.get("trace", [])) + [line]


def _chunks(items: list[dict]) -> list[rag.Chunk]:
    return [rag.Chunk(**c) for c in items]


def _labels(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in _chunks(items):
        label = c.ref + (f" - {c.section}" if c.section else "")
        if label not in seen:
            seen.add(label)
            out.append({"kind": "notes", "label": label})
    return out


def _label(topic: str) -> str:
    return topic.replace("_", " ").title()


# ---------------- practice: question ----------------
def pick_topic_node(state: State) -> dict:
    stats = tools.call("progress", "get_topic_stats")
    topic = scheduler.pick_topic(state.get("topics", []), stats)
    return {"topic": topic, "trace": _t(state, f"MCP progress.get_topic_stats -> smart pick: {topic}")}


def prepare_question_node(state: State) -> dict:
    topic, focus = state.get("topic"), (state.get("focus") or "").strip()
    trace = list(state.get("trace", []))
    chunks: list[dict] = []
    if focus:
        query = f"{_label(topic or '')} {focus}".strip()
        chunks = tools.call("rag", "search_notes", query=query, k=4, topic=topic) if topic \
            else tools.call("rag", "search_notes", query=query, k=4)
        trace.append(f"MCP rag.search_notes('{focus}') -> {len(chunks)} chunk(s)")
    if not chunks:  # no focus, or nothing matched: varied material from the topic
        chunks = tools.call("rag", "sample_notes", topic=topic or "", k=3)
        trace.append(f"MCP rag.sample_notes -> {len(chunks)} chunk(s)")
    if not chunks:
        return {"question": "", "context": [], "notice": "No readable text was found for this topic. "
                "Check that the document contains selectable text.", "trace": trace}
    question = llm.generate_question(_chunks(chunks), focus or None,
                                     state.get("difficulty", "Intermediate"), state.get("mode", "Mixed"))
    trace.append(f"question generated ({'Groq' if llm.online() else 'offline'})")
    return {"question": question, "context": chunks, "topic": chunks[0]["topic"], "trace": trace}


# ---------------- practice: evaluate ----------------
def evaluate_node(state: State) -> dict:
    result = llm.evaluate(state["question"], state["answer_text"], _chunks(state["context"]))
    return {"result": result,
            "trace": _t(state, f"answer graded ({'Groq rubric' if result['mode'] == 'llm' else 'offline heuristic'})"
                               f": {result['score']:g}/{result['max_score']:g}")}


def log_attempt_node(state: State) -> dict:
    res = state["result"]
    src = "; ".join(s["label"] for s in _labels(state["context"]))
    try:
        out = tools.call(
            "progress", "log_answer", topic=state["topic"],
            score=res["score"], max_score=res["max_score"], question=state["question"], user_answer=state["answer_text"],
            feedback=res["feedback"], source=src, mode=state.get("mode", ""),
            difficulty=state.get("difficulty", ""),
        )
        return {"attempt_id": out["attempt_id"],
                "trace": _t(state, f"MCP progress.log_answer -> attempt #{out['attempt_id']}")}
    except Exception as e:  # the score is still shown; the failure is not hidden
        return {"notice": f"Scored, but could not save progress: {e}",
                "trace": _t(state, "MCP progress.log_answer failed")}


# ---------------- explain ----------------
def retrieve_node(state: State) -> dict:
    args = {"query": state["question"], "k": 6}
    if state.get("topic"):
        args["topic"] = state["topic"]
    notes = tools.call("rag", "search_notes", **args)
    if not notes and state.get("topic"):  # topic guess was too narrow: search everything
        notes = tools.call("rag", "search_notes", query=state["question"], k=6)
    covered = bool(notes) and notes[0]["score"] >= NOTES_SUFFICIENT
    top = f"top score {notes[0]['score']:.2f}" if notes else "no hits"
    return {
        "notes": notes,
        "covered": covered,
        "trace": _t(state, f"MCP rag.search_notes -> {len(notes)} chunk(s), {top}, "
                           f"{'covered' if covered else 'not covered'}"),
    }


def lookup_docs_node(state: State) -> dict:
    trace = _t(state, "notes fall short -> official docs")
    web_docs: list[dict] = []
    notice = ""
    try:
        hits = tools.call("docs", "search_docs", query=state["question"], max_results=3)
        if isinstance(hits, dict) and "error" in hits:
            notice = f"Docs lookup failed: {hits['error']}."
        else:
            trace.append(f"MCP docs.search_docs -> {len(hits)} result(s)")
            for h in hits[:2]:
                d = tools.call("docs", "fetch_doc", url=h["url"])
                if "error" not in d:
                    web_docs.append(d)
                    trace.append(f"MCP docs.fetch_doc -> {d['url']}")
            if not web_docs:
                notice = "No official documentation matched this question."
    except Exception as e:  # failures are shown to the user, never papered over
        notice = f"Docs lookup failed: {e}"
    return {"web_docs": web_docs, "notice": notice, "trace": trace}


def answer_node(state: State) -> dict:
    covered = state.get("covered")
    notes = _chunks(state.get("notes", [])) if covered else []
    web_docs = state.get("web_docs", [])
    notice = state.get("notice", "")
    if not notes and not web_docs:
        text = ("I couldn't find this in your notes or the official docs, so I won't guess. "
                "Add a note about it or try different wording.")
        return {"answer": f"{text} {notice}".strip(), "sources": [],
                "trace": _t(state, "no source -> declined to guess")}
    try:
        text = llm.explain(state["question"], notes, web_docs)
    except Exception as e:
        text = f"Could not generate an explanation: {e}"
    if notice:
        text = f"_{notice}_\n\n{text}"
    sources = _labels(state.get("notes", [])) if covered else []
    sources += [{"kind": "docs", "label": d["url"]} for d in web_docs]
    return {"answer": text, "sources": sources,
            "trace": _t(state, f"answer ({'Groq' if llm.online() else 'offline passages'})")}


# ---------------- revise ----------------
def revise_node(state: State) -> dict:
    topic, focus = state["topic"], (state.get("focus") or "").strip()
    trace = list(state.get("trace", []))
    query = f"{_label(topic)} key concepts definitions examples {focus}".strip()
    chunks = tools.call("rag", "search_notes", query=query, k=8, topic=topic)
    trace.append(f"MCP rag.search_notes -> {len(chunks)} chunk(s)")
    if not chunks:
        chunks = tools.call("rag", "sample_notes", topic=topic, k=6)
        trace.append(f"MCP rag.sample_notes -> {len(chunks)} chunk(s)")
    if not chunks:
        return {"sheet": "", "notice": "No text is available for this topic.", "trace": trace}
    stats = tools.call("progress", "get_topic_stats")
    avg = next((s["average_score"] for s in stats if s["topic"] == topic), None)
    hist = tools.call("progress", "get_history", limit=5, topic=topic)
    trace.append(f"MCP progress.get_topic_stats / get_history -> {len(hist)} recent attempt(s)")
    sheet = llm.revision_sheet(topic, avg, _chunks(chunks), hist, focus)
    trace.append(f"revision sheet written ({'Groq' if llm.online() else 'offline passages'})")
    return {"sheet": sheet, "context": chunks, "sources": _labels(chunks), "trace": trace}


# ---------------- multiple choice ----------------
def mcq_node(state: State) -> dict:
    topic = state.get("topic") or ""
    trace = list(state.get("trace", []))
    chunks: list[dict] = []
    for _ in range(3):  # a few random passages so the set covers different parts of the notes
        chunks += tools.call("rag", "sample_notes", topic=topic, k=3)
    trace.append(f"MCP rag.sample_notes x3 -> {len(chunks)} chunk(s)")
    if not chunks:
        return {"questions": [], "notice": "No readable notes were found for this topic.", "trace": trace}
    questions = llm.generate_mcqs(_chunks(chunks), state.get("count", 10), state.get("difficulty", "Intermediate"))
    trace.append(f"{len(questions)} question(s) generated (Groq)")
    return {"questions": questions, "context": chunks, "trace": trace}


# ---------------- chatbot practice ----------------
def chat_node(state: State) -> dict:
    topic = state.get("topic") or None
    history = state.get("history", [])
    trace = list(state.get("trace", []))
    chunks: list[dict] = []
    if history and history[-1]["role"] == "user":  # find notes for what the candidate just said
        last_q = next((m["content"] for m in reversed(history[:-1]) if m["role"] == "assistant"), "")
        query = f"{history[-1]['content']} {last_q}"[:600]
        args = {"query": query, "k": 4}
        if topic:
            args["topic"] = topic
        chunks = tools.call("rag", "search_notes", **args)
        trace.append(f"MCP rag.search_notes -> {len(chunks)} chunk(s)")
    if not chunks:  # first turn, or nothing matched: varied material from the topic
        chunks = tools.call("rag", "sample_notes", topic=topic or "", k=3)
        trace.append(f"MCP rag.sample_notes -> {len(chunks)} chunk(s)")
    reply = llm.chat_reply(history, _chunks(chunks), _label(topic) if topic else "", state.get("difficulty", "Intermediate"))
    trace.append(f"reply written ({'Groq' if llm.online() else 'offline'})")
    return {"reply": reply, "context": chunks, "sources": _labels(chunks), "trace": trace}


# ---------------- edges ----------------
def entry(state: State) -> str:
    task = state.get("task", "")
    if task == "question":
        return "pick_topic" if state.get("topic") in (None, "", SMART_PICK) else "prepare_question"
    if task in ("evaluate", "explain", "revise", "mcq", "chat"):
        return {"evaluate": "evaluate", "explain": "retrieve", "revise": "revise",
                "mcq": "mcq", "chat": "chat"}[task]
    raise ValueError(f"unknown task: {task!r}")


def after_retrieve(state: State) -> str:
    return "answer" if state.get("covered") or not state.get("use_web", True) else "lookup_docs"


# ---------------- graph ----------------
def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(State)
    for name, fn in [("pick_topic", pick_topic_node), ("prepare_question", prepare_question_node),
                     ("evaluate", evaluate_node), ("log_attempt", log_attempt_node),
                     ("retrieve", retrieve_node), ("lookup_docs", lookup_docs_node),
                     ("answer", answer_node), ("revise", revise_node),
                     ("mcq", mcq_node), ("chat", chat_node)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, entry, {n: n for n in
                            ("pick_topic", "prepare_question", "evaluate", "retrieve", "revise",
                             "mcq", "chat")})
    g.add_edge("pick_topic", "prepare_question")
    g.add_edge("prepare_question", END)
    g.add_edge("evaluate", "log_attempt")
    g.add_edge("log_attempt", END)
    g.add_conditional_edges("retrieve", after_retrieve,
                            {"answer": "answer", "lookup_docs": "lookup_docs"})
    g.add_edge("lookup_docs", "answer")
    g.add_edge("answer", END)
    g.add_edge("revise", END)
    g.add_edge("mcq", END)
    g.add_edge("chat", END)
    return g.compile()


_graph = None


def run(task: str, **state) -> State:
    """Run one task through the graph and return the final state.

    run("question", topic="sql", topics=[...], difficulty="Beginner", mode="Coding")
    run("evaluate", topic=..., question=..., answer_text=..., context=[...], mode=..., difficulty=...)
    run("explain", question="...", topic=None, use_web=True)
    run("revise", topic="sql", focus="joins")
    run("mcq", topic="sql", count=10, difficulty="Beginner")
    run("chat", topic="sql", history=[{"role": "user", "content": "..."}], difficulty="Beginner")
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke({"task": task, "trace": [], **state})
