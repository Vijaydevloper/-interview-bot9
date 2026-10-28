"""Interview Prep Coach: Streamlit front end.  Run: streamlit run app.py

Every task (question, evaluate, explain, revise) runs through the LangGraph agent in
agent.py; data access goes over MCP (progress, rag, docs, notes servers).
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import agent
import config  # noqa: F401  (loads .env)
import llm
import mcq
import rag
import tools
import ui

st.set_page_config(page_title="Interview Prep Coach", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(ui.HERO, unsafe_allow_html=True)

ss = st.session_state
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
MODES = ["Conceptual", "Coding", "Mixed"]
ALL_TOPICS = "All topics"


@st.cache_resource(show_spinner="Indexing study documents...")
def get_index(signature: tuple) -> rag.Index:
    return rag.Index.build()


idx = get_index(rag.folder_signature())
topics = idx.topics or ["python"]
label = lambda t: t.replace("_", " ").title()  # noqa: E731
smart_label = lambda t: "🎯 Smart pick (weakest first)" if t == agent.SMART_PICK else label(t)  # noqa: E731


# ---------------- progress data over MCP (re-read after each logged answer) ----------------
def _cached(key: str, fn, default):
    pv = ss.get("pv", 0)
    if ss.get(f"{key}_pv") != pv:
        try:
            ss[key] = fn()
        except Exception as e:
            ss[key] = default
            st.error(f"Could not read your progress: {e}")
        ss[f"{key}_pv"] = pv
    return ss[key]


def history() -> list[dict]:
    return _cached("hist", lambda: tools.call("progress", "get_history", limit=200), [])


def weak() -> list[dict]:
    return _cached("weak", lambda: tools.call("progress", "get_weak_topics"), [])


def run_agent(spinner: str, task: str, **state):
    """Run one graph task; show a readable error instead of a traceback."""
    with st.spinner(spinner):
        try:
            return agent.run(task, **state)
        except Exception as e:
            st.error(str(e))
            return None


def trace_expander(trace: list[str]) -> None:
    if trace:
        with st.expander("How this was produced"):
            st.markdown("\n".join(f"{n}. {t}" for n, t in enumerate(trace, 1)))


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## ⚙️ Study settings")
    with st.expander("Groq API key", expanded=not llm.online()):
        st.caption("Set `GROQ_API_KEY` in the `.env` file next to `app.py` (see `.env.example`), then "
                   "restart the app. Or paste a key here for this session only.")
        key = st.text_input("GROQ_API_KEY", type="password", value=ss.get("groq_key", ""),
                            help="Kept in this browser session only. Free keys: console.groq.com")
        if key:
            ss["groq_key"] = key
            os.environ["GROQ_API_KEY"] = key
    if llm.online():
        if "groq_models" not in ss:
            ss["groq_models"] = llm.available_models()
        models = ss["groq_models"]
        if models:
            cur_model = llm.model_name()
            picked = st.selectbox("Groq model", models if cur_model in models else [cur_model] + models,
                                  index=0 if cur_model not in models else models.index(cur_model))
            os.environ["GROQ_MODEL"] = picked
        else:
            st.caption(f"Groq model: `{llm.model_name()}`")
    else:
        st.warning("Offline mode: questions come from headings and scores are a keyword check. "
                   "Add a Groq API key to `.env` for real questions, scoring, chat and explanations.")
    st.caption("Your progress stays on this machine. When Groq is on, the note passages used for a "
               "question are sent to Groq to write it.")
    if st.button("🔄 Refresh document index"):
        get_index.clear()
        ss.pop("library", None)
        st.rerun()
    st.divider()
    st.markdown("### 📚 Library")
    st.metric("Indexed text chunks", len(idx.chunks))
    st.metric("Available topics", len(idx.topics))
    st.caption("Supported: DOCX, PPTX, PDF, TXT, MD")

if not idx.chunks:
    st.warning("No text was indexed. Add study files in the Document library tab "
               "(or to `data/docs/<topic>/`), then refresh the index.")

# ---------------- score tracker ----------------
hist = history()
df = pd.DataFrame(hist)
if not df.empty:
    df["score10"] = df["score"] * 10 / df["max_score"]
c1, c2, c3 = st.columns(3)
c1.metric("🧠 Practice attempts", len(df))
c2.metric("📈 Average score", f"{df['score10'].mean():.1f}/10" if not df.empty else "0/10")
c3.metric("📂 Study topics", len(idx.topics))

tabs = st.tabs(["✨ Practice", "✅ MCQ practice", "💬 Chatbot practice", "📖 Explain a concept", "📝 Revision notes", "📊 My progress",
                "🗂️ Document library"])

# ======================= Practice =======================
with tabs[0]:
    left, right = st.columns([1, 2], gap="large")
    with left:
        topic = st.selectbox("Choose a topic", [agent.SMART_PICK] + topics, format_func=smart_label,
                             key="quiz_topic")
        difficulty = st.select_slider("Difficulty", options=DIFFICULTIES, value="Beginner")
        mode = st.radio("Practice type", MODES)
        focus = st.text_input("Focus (optional)", placeholder="e.g., window functions")
        generate = st.button("✨ Generate interview question", type="primary")
    with right:
        st.markdown("#### Your practice workspace")
        st.info("Choose a topic and generate a question. Your answer will be scored against the "
                "retrieved study notes.")

    if generate:
        out = run_agent("Creating a question from your notes...", "question", topic=topic, topics=topics,
                        difficulty=difficulty, mode=mode, focus=focus)
        if out is not None:
            if not out.get("question"):
                st.error(out.get("notice") or "Could not create a question.")
            else:
                ss["cur"] = {k: out[k] for k in ("question", "context", "topic", "trace")}
                ss["cur"].update(mode=mode, difficulty=difficulty)
                ss["result"] = None
                ss["answer"] = ""  # set before the text area below is created

    cur = ss.get("cur")
    if cur:
        st.markdown("### 💬 Your question")
        st.markdown(ui.tags([label(cur["topic"]), cur["difficulty"], cur["mode"]]), unsafe_allow_html=True)
        st.markdown(ui.question(cur["question"]), unsafe_allow_html=True)
        trace_expander(cur["trace"])
        answer = st.text_area("Write your answer", key="answer", height=160,
                              placeholder="Explain in your own words. Include an example or code if relevant.")
        if st.button("🧑‍🏫 Evaluate my answer", type="primary"):
            if not answer.strip():
                st.warning("Write an answer first.")
            else:
                out = run_agent("Reviewing your answer against the notes...", "evaluate",
                                topic=cur["topic"], question=cur["question"], context=cur["context"],
                                answer_text=answer, mode=cur["mode"], difficulty=cur["difficulty"])
                if out is not None:
                    ss["result"] = {"res": out["result"], "notice": out.get("notice", ""),
                                    "trace": out["trace"], "saved": "attempt_id" in out,
                                    "sources": out["context"]}
                    ss["pv"] = ss.get("pv", 0) + 1  # progress changed: re-read it
                    st.rerun()
        r = ss.get("result")
        if r:
            res = r["res"]
            st.markdown("### 🧾 Feedback")
            st.markdown(ui.scoreboard(res["score"], res["max_score"], res["parts"], llm.RUBRIC),
                        unsafe_allow_html=True)
            if res["mode"] != "llm":
                st.info("This score is a keyword check, not a real assessment. "
                        "Add a Groq API key for rubric scoring.")
            st.write(res["feedback"])
            g1, g2 = st.columns(2)
            g1.markdown("**What you got right**\n\n" + (
                "\n".join(f"- {p}" for p in res["correct_points"]) or "_Nothing identified yet._"))
            g2.markdown("**What to add**\n\n" + (
                "\n".join(f"- {p}" for p in res["missing_points"]) or "_Nothing missing._"))
            with st.expander("Model answer"):
                st.markdown(res["model_answer"] or "_Not available._")
            st.markdown(ui.sources([{"kind": "notes", "label": s["source"] + (f" - {s['section']}" if s["section"] else "")}
                                    for s in r["sources"]]), unsafe_allow_html=True)
            if r["saved"]:
                st.success(f"Progress saved · Score: {res['score']:g}/{res['max_score']:g}")
            if r["notice"]:
                st.warning(r["notice"])
            trace_expander(r["trace"])

# ======================= MCQ practice =======================
with tabs[1]:
    st.markdown("#### Choose the right option")
    m1, m2 = st.columns([1, 2], gap="large")
    with m1:
        mcq_source = st.radio("MCQ source", ["Curated set (10 questions)", "Fresh set from my notes (Groq)"])
        mcq_topic = st.selectbox("MCQ topic (fresh sets)", [ALL_TOPICS] + topics,
                                 format_func=lambda t: t if t == ALL_TOPICS else label(t), key="mcq_topic",
                                 disabled=mcq_source.startswith("Curated"))
        start_mcq = st.button("▶️ Start MCQ practice", type="primary")
    with m2:
        st.info("Answer all 10 questions, then submit. Every answer is saved to your progress "
                "(10/10 for correct, 0 for wrong) under the question's topic.")

    if start_mcq:
        qs, notice = None, ""
        if mcq_source.startswith("Curated"):
            try:
                qs = mcq.load_bank()
            except Exception as e:
                st.error(f"Could not read the question bank: {e}")
        else:
            out = run_agent("Writing 10 questions from your notes...", "mcq",
                            topic=None if mcq_topic == ALL_TOPICS else mcq_topic, count=10)
            if out is not None:
                qs, notice = out.get("questions"), out.get("notice", "")
                if not qs:
                    st.error(notice or "No questions were generated.")
        if qs:
            ss["mcq"] = {"id": ss.get("mcq_n", 0) + 1, "qs": mcq.prepare(qs), "done": None}
            ss["mcq_n"] = ss["mcq"]["id"]

    quiz = ss.get("mcq")
    if quiz and quiz["done"] is None:
        with st.form(f"mcq_form_{quiz['id']}"):
            for i, q in enumerate(quiz["qs"]):
                head, _, code = q["question"].partition("\n\n")
                st.markdown(f"**Q{i + 1}. {head}**")
                st.caption(label(q["topic"]))
                if code:
                    st.code(code, language="python" if q["topic"] in ("python", "pandas") else None)
                st.radio(f"Answer {i + 1}", [0, 1, 2, 3], index=None, key=f"mcq_{quiz['id']}_{i}",
                         format_func=lambda j, q=q: f"{mcq.LETTERS[j]}. {q['options'][j]}",
                         label_visibility="collapsed")
                st.divider()
            submitted = st.form_submit_button("✅ Submit answers", type="primary")
        if submitted:
            picks = [ss.get(f"mcq_{quiz['id']}_{i}") for i in range(len(quiz["qs"]))]
            if any(p is None for p in picks):
                st.warning(f"Please answer every question ({picks.count(None)} left).")
            else:
                graded = mcq.grade(quiz["qs"], picks)
                saved = True
                try:
                    tools.call("progress", "log_answers", attempts=[{
                        "topic": q["topic"], "score": 10 if r["correct"] else 0, "max_score": 10,
                        "question": q["question"], "user_answer": q["options"][r["pick"]],
                        "feedback": q["explanation"], "source": q["source"], "mode": "MCQ"}
                        for q, r in zip(quiz["qs"], graded["rows"])])
                    ss["pv"] = ss.get("pv", 0) + 1
                except Exception as e:
                    saved = False
                    st.error(f"Scored, but could not save progress: {e}")
                quiz["done"] = {**graded, "saved": saved}
                st.rerun()
    if quiz and quiz["done"] is not None:
        d = quiz["done"]
        st.metric("🎯 Your score", f"{d['correct']}/{d['total']}")
        if d["saved"]:
            st.success("Progress saved.")
        by_topic: dict[str, list[int]] = {}
        for q, r in zip(quiz["qs"], d["rows"]):
            by_topic.setdefault(q["topic"], []).append(int(r["correct"]))
        missed = [label(t) for t, v in by_topic.items() if sum(v) < len(v)]
        if missed:
            st.warning("Revisit: " + ", ".join(missed))
        for i, (q, r) in enumerate(zip(quiz["qs"], d["rows"])):
            icon = "✅" if r["correct"] else "❌"
            with st.expander(f"{icon} Q{i + 1}. {q['question'].splitlines()[0][:90]}", expanded=not r["correct"]):
                st.markdown(f"**Your answer:** {mcq.LETTERS[r['pick']]}. {q['options'][r['pick']]}")
                if not r["correct"]:
                    st.markdown(f"**Correct answer:** {mcq.LETTERS[q['answer']]}. {q['options'][q['answer']]}")
                st.write(q["explanation"])
                if q["source"]:
                    st.markdown(ui.sources([{"kind": "notes", "label": q["source"]}]), unsafe_allow_html=True)
        if st.button("🔁 Try again (new order)"):
            ss["mcq"] = {"id": quiz["id"] + 1, "qs": mcq.prepare(quiz["qs"]), "done": None}
            ss["mcq_n"] = ss["mcq"]["id"]
            st.rerun()

# ======================= Chatbot practice =======================
with tabs[2]:
    st.markdown("#### Mock interview chat")
    c_topic_col, c_level_col, c_btn_col = st.columns([1, 1, 1], gap="large")
    with c_topic_col:
        chat_topic = st.selectbox("Interview topic", [ALL_TOPICS] + topics,
                                  format_func=lambda t: t if t == ALL_TOPICS else label(t), key="chat_topic")
    with c_level_col:
        chat_level = st.select_slider("Interview level", options=DIFFICULTIES, value="Beginner")
    with c_btn_col:
        st.write("")
        start_chat = st.button("▶️ Start new interview", type="primary")
    chat_topic_arg = None if chat_topic == ALL_TOPICS else chat_topic
    if not llm.online():
        st.info("Offline mode: the bot can only ask basic questions and do a keyword check. Add your Groq key "
                "to `.env` for a real back-and-forth.")

    def chat_turn(history: list[dict]) -> bool:
        out = run_agent("Interviewer is typing...", "chat", topic=chat_topic_arg, history=history,
                        difficulty=chat_level)
        if out is not None:
            ss["chat"] = history + [{"role": "assistant", "content": out["reply"]}]
            ss["chat_sources"], ss["chat_trace"] = out.get("sources", []), out.get("trace", [])
        return out is not None

    if start_chat:
        ss["chat"] = []
        if chat_turn([]):
            st.rerun()

    for msg in ss.get("chat", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if ss.get("chat_sources"):
        st.markdown(ui.sources(ss["chat_sources"]), unsafe_allow_html=True)
        trace_expander(ss.get("chat_trace", []))

    q1, q2, q3, _ = st.columns([1, 1, 1, 2])
    quick = None
    if q1.button("💡 Hint"):
        quick = "Can I have a hint? Please don't give the full answer."
    if q2.button("⏭️ Next question"):
        quick = "Skip this one and ask me the next question."
    if q3.button("📖 Show answer"):
        quick = "Please show me the answer to your last question."
    typed = st.chat_input("Type your answer, or ask the interviewer anything...")
    user_text = typed or quick
    if user_text:
        history = ss.get("chat", []) + [{"role": "user", "content": user_text}]
        ss["chat"] = history  # shown even if the reply fails
        if chat_turn(history):
            st.rerun()

# ======================= Explain =======================
with tabs[3]:
    st.markdown("#### Understand a topic using your own materials")
    e1, e2 = st.columns([1, 2])
    with e1:
        explain_topic = st.selectbox("Topic", [ALL_TOPICS] + topics,
                                     format_func=lambda t: t if t == ALL_TOPICS else label(t), key="explain_topic")
    with e2:
        question = st.text_input("What would you like explained?",
                                 placeholder="e.g., Explain INNER JOIN vs LEFT JOIN with an example")
    use_web = st.toggle("Check official docs when my notes fall short", value=True)
    if st.button("📖 Explain", key="explain_btn", type="primary"):
        if not question.strip():
            st.warning("Enter a concept or question.")
        else:
            out = run_agent("Preparing an explanation from your notes...", "explain", question=question,
                            topic=None if explain_topic == ALL_TOPICS else explain_topic, use_web=use_web)
            ss["explain_out"] = out
    out = ss.get("explain_out")
    if out:
        st.markdown(out["answer"])
        if out.get("sources"):
            st.markdown(ui.sources(out["sources"]), unsafe_allow_html=True)
        if out.get("notes") and out.get("covered"):
            with st.expander("View retrieved source notes"):
                for c in out["notes"]:
                    st.markdown(f"**{c['source']}**" + (f" · {c['section']}" if c["section"] else ""))
                    st.text(c["text"])
        trace_expander(out.get("trace", []))

# ======================= Revision notes =======================
with tabs[4]:
    st.markdown("#### Build a compact revision sheet")
    r1, r2 = st.columns([1, 2])
    with r1:
        revise_topic = st.selectbox("Topic to revise", topics, format_func=label, key="revise_topic")
    with r2:
        revision_focus = st.text_input("Optional focus",
                                       placeholder="e.g., definitions, formulas, common interview mistakes")
    if st.button("📝 Create revision notes", type="primary"):
        out = run_agent("Creating revision notes...", "revise", topic=revise_topic, focus=revision_focus)
        if out is not None and not out.get("sheet"):
            st.warning(out.get("notice") or "No text is available for this topic.")
        elif out is not None:
            ss["revise_out"] = {**out, "topic": revise_topic}
    rv = ss.get("revise_out")
    if rv:
        st.markdown(rv["sheet"])
        if rv.get("sources"):
            st.markdown(ui.sources(rv["sources"]), unsafe_allow_html=True)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Download revision sheet", rv["sheet"],
                           file_name=f"{rv['topic']}_revision_notes.md", mime="text/markdown")
        if d2.button("💾 Save to exports folder"):
            try:
                saved = tools.call("notes", "save_summary", title=f"Revision: {rv['topic']}", content=rv["sheet"])
                st.success(f"Saved to {saved['path']}")
            except Exception as e:
                st.error(f"Could not save the sheet: {e}")
        trace_expander(rv.get("trace", []))

# ======================= My progress =======================
with tabs[5]:
    st.markdown("#### Your learning history")
    if df.empty:
        st.info("Your attempts will appear here after you answer and evaluate your first question.")
    else:
        view = pd.DataFrame({
            "Date": pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M"),
            "Topic": df["topic"].map(label),
            "Mode": df.get("mode", ""),
            "Level": df.get("difficulty", ""),
            "Score": df["score10"],
            "Question": df["question"],
        })
        st.dataframe(view, hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f")})
        st.markdown("#### Score over time")
        st.line_chart(df.iloc[::-1].reset_index(drop=True)["score10"])
        weak_rows = weak()
        if weak_rows:
            st.markdown("#### Topics to revisit")
            for row in weak_rows:
                st.markdown(ui.meter(label(row["topic"]), row["average_score"],
                                     f"{row['attempt_count']} attempt(s)"), unsafe_allow_html=True)
            weak_df = pd.DataFrame(weak_rows).rename(columns={
                "topic": "Topic", "average_score": "Average score", "attempt_count": "Attempts"})
            st.bar_chart(weak_df.set_index("Topic")["Average score"])
            st.dataframe(weak_df[["Topic", "Average score", "Attempts"]], hide_index=True)
        else:
            st.success("No weak topics: every practised topic averages 7/10 or better.")

# ======================= Document library =======================
with tabs[6]:
    st.markdown("#### Indexed study materials")
    sig = rag.folder_signature()
    if ss.get("library_sig") != sig or "library" not in ss:
        try:
            with st.spinner("Reading the library..."):
                ss["library"] = tools.call("rag", "library_summary")  # MCP rag server
            ss["library_sig"] = sig
        except Exception as e:
            ss["library"] = {}
            st.error(f"Could not read the library: {e}")
    for t, d in sorted(ss["library"].items()):
        with st.expander(f"📁 {label(t)} · {len(d['files'])} file(s) · {d['chunks']} chunks"):
            for f in sorted(d["files"]):
                st.write(f"• {f}")
    with st.expander("➕ Add documents"):
        up_topic = st.text_input("Topic folder", value="python")
        files = st.file_uploader("DOCX, PPTX, PDF, TXT or MD", accept_multiple_files=True,
                                 type=["docx", "pptx", "pdf", "txt", "md"])
        if st.button("Save to library") and files and up_topic.strip():
            folder = "".join(ch for ch in up_topic.strip().lower().replace(" ", "_") if ch.isalnum() or ch == "_")
            dest = Path(rag.DOCS_DIR) / (folder or "general")
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                (dest / Path(f.name).name).write_bytes(f.getbuffer())  # basename only: no path tricks
            get_index.clear()
            st.rerun()
    st.caption("Images such as diagrams aren't searchable. Add a short Markdown note describing them "
               "if you want their content to appear in questions.")
