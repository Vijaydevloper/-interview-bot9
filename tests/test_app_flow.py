"""Smoke test of the Streamlit app's control flow using a stand-in `streamlit` module.

app.py runs top to bottom (as Streamlit does) against the real data/docs library and
real MCP servers; button clicks, reruns and session state are simulated. This checks the
app's own logic (state handling, agent calls, tab flows). It does NOT check Streamlit's
real widget API, so also run `streamlit run app.py` once by hand.
"""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from langgraph_stub import use_langgraph_or_stub  # noqa: E402


class Rerun(Exception):
    pass


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Ctx:
    """`with` target that forwards widget calls (col.metric, sidebar.x, ...) to the stand-in."""

    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.inputs: dict = {}     # label -> value for selectbox / text_input / radio / slider
        self.clicks: set = set()   # button labels clicked in the current run
        self.log: list = []        # (widget, text) in render order
        self.sidebar = Ctx(self)
        self.column_config = types.SimpleNamespace(ProgressColumn=lambda *a, **k: None)

    # ---- output widgets: just record ----
    def _rec(name):
        def f(self, *a, **k):
            self.log.append((name, a[0] if a else ""))
        return f

    for _n in ("set_page_config", "markdown", "caption", "warning", "info", "success", "error", "write",
               "text", "divider", "line_chart", "bar_chart", "dataframe", "metric"):
        locals()[_n] = _rec(_n)

    def metric(self, label, value, *a, **k):
        self.log.append(("metric", f"{label}={value}"))

    # ---- containers ----
    def expander(self, *a, **k):
        return Ctx(self)

    def spinner(self, *a, **k):
        return Ctx(self)

    def columns(self, spec, **k):
        return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, names):
        return [Ctx(self) for _ in names]

    # ---- inputs ----
    def selectbox(self, label, options, format_func=str, key=None, **k):
        for o in options:
            format_func(o)  # exercise the formatter
        return self.inputs.get(label, options[0])

    def radio(self, label, options, key=None, index=0, **k):
        if key is not None and index is None:  # keyed radio with no default (MCQ answers)
            return self.session_state.get(key)
        return self.inputs.get(label, list(options)[0] if index is not None else None)

    def form(self, *a, **k):
        return Ctx(self)

    def form_submit_button(self, label, **k):
        return label in self.clicks

    def code(self, *a, **k):
        self.log.append(("code", a[0] if a else ""))

    def chat_message(self, role):
        self.log.append(("chat_message", role))
        return Ctx(self)

    def chat_input(self, label, **k):
        return self.inputs.pop(label, None)  # like Streamlit: a submitted message is seen by one run only

    def select_slider(self, label, options, value=None, **k):
        return self.inputs.get(label, value)

    def text_input(self, label, value="", key=None, **k):
        return self.inputs.get(label, value)

    def text_area(self, label, key=None, **k):
        return self.session_state.get(key, "")

    def toggle(self, label, value=False, **k):
        return self.inputs.get(label, value)

    def button(self, label, key=None, **k):
        return label in self.clicks

    def download_button(self, *a, **k):
        return False

    def file_uploader(self, *a, **k):
        return []

    def cache_resource(self, *dargs, **dkw):
        def deco(fn):
            memo = {}

            def wrapper(*a):
                if a not in memo:
                    memo[a] = fn(*a)
                return memo[a]

            wrapper.clear = memo.clear
            return wrapper

        return deco(dargs[0]) if dargs and callable(dargs[0]) else deco

    def rerun(self):
        raise Rerun()


def run_app(st, clicks=(), inputs=None, **state):
    """One Streamlit 'interaction': set state, run the script, follow reruns."""
    st.session_state.update(state)
    st.inputs = dict(inputs or {})
    st.clicks = set(clicks)
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    for _ in range(6):
        st.log.clear()
        try:
            exec(compile(src, "app.py", "exec"), {"__name__": "__main__"})
            return
        except Rerun:
            st.clicks = set()  # a click only applies to the run it happened in
    raise AssertionError("rerun loop")


def _topics():
    return sorted(p.name for p in (ROOT / "data" / "docs").iterdir() if p.is_dir())


def _metrics(st):
    return dict(m[1].split("=", 1) for m in st.log if m[0] == "metric")


def test_app_end_to_end_offline(tmp_path, monkeypatch):
    for k in ("GROQ_API_KEY",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PROGRESS_DB", str(tmp_path / "p.db"))
    monkeypatch.setenv("USE_MCP", "1")
    monkeypatch.delenv("STUDY_DOCS", raising=False)
    use_langgraph_or_stub()
    old_cwd, old_mod = os.getcwd(), sys.modules.get("streamlit")
    os.chdir(ROOT)
    st = FakeStreamlit()
    sys.modules["streamlit"] = st
    try:
        # 1. first load: five study topics indexed, no attempts, offline notice, library via MCP
        run_app(st)
        m = _metrics(st)
        assert m["🧠 Practice attempts"] == "0" and m["📂 Study topics"] == str(len(_topics()))
        assert any(w[0] == "warning" and "Offline mode" in w[1] for w in st.log)
        assert "sql" in st.session_state["library"] and st.session_state["library"]["sql"]["files"]

        # 2. generate a question (topic SQL, coding, advanced)
        run_app(st, clicks=["✨ Generate interview question"],
                inputs={"Choose a topic": "sql", "Difficulty": "Advanced", "Practice type": "Coding"})
        cur = st.session_state["cur"]
        assert cur["topic"] == "sql" and cur["question"] and cur["context"] and cur["mode"] == "Coding"
        assert any("rag.sample_notes" in t for t in cur["trace"])

        # 3. evaluate: result stored, progress logged, tracker refreshed after the rerun
        run_app(st, clicks=["🧑‍🏫 Evaluate my answer"],
                answer="An INNER JOIN returns rows with matching keys; a LEFT JOIN keeps all left rows.")
        res = st.session_state["result"]
        assert res["saved"] and 0 <= res["res"]["score"] <= 10
        assert _metrics(st)["🧠 Practice attempts"] == "1"
        run_app(st)  # a plain rerun keeps the feedback on screen
        assert any(x[0] == "markdown" and "board" in x[1] for x in st.log)

        # 4. explain from the notes (toggle off: no docs server call)
        run_app(st, clicks=["📖 Explain"], inputs={
            "What would you like explained?": "Explain LEFT JOIN vs INNER JOIN",
            "Check official docs when my notes fall short": False})
        out = st.session_state["explain_out"]
        assert out["covered"] and any("joins" in s["label"].lower() or "sql" in s["label"].lower()
                                      for s in out["sources"])

        # 5. revision notes
        run_app(st, clicks=["📝 Create revision notes"], inputs={"Topic to revise": "ml"})
        assert st.session_state["revise_out"]["sheet"] and st.session_state["revise_out"]["topic"] == "ml"

        # 5b. MCQ: curated set of 10, graded and logged in one batch
        run_app(st, clicks=["▶️ Start MCQ practice"])
        quiz = st.session_state["mcq"]
        assert len(quiz["qs"]) == 10 and quiz["done"] is None
        n = quiz["id"]
        run_app(st, clicks=["✅ Submit answers"], **{f"mcq_{n}_0": 0})  # unanswered questions: warned, not graded
        assert st.session_state["mcq"]["done"] is None
        answers = {f"mcq_{n}_{i}": (q["answer"] if i < 7 else (q["answer"] + 1) % 4)
                   for i, q in enumerate(quiz["qs"])}
        run_app(st, clicks=["✅ Submit answers"], **answers)
        done = st.session_state["mcq"]["done"]
        assert done["correct"] == 7 and done["total"] == 10 and done["saved"]
        assert _metrics(st)["🎯 Your score"] == "7/10"
        assert _metrics(st)["🧠 Practice attempts"] == "11"  # 1 written answer + 10 MCQ answers

        # 5c. chatbot: start, then answer (offline replies are keyword-check based)
        run_app(st, clicks=["▶️ Start new interview"], inputs={"Interview topic": "sql"})
        assert [m["role"] for m in st.session_state["chat"]] == ["assistant"]
        run_app(st, inputs={"Type your answer, or ask the interviewer anything...":
                            "A LEFT JOIN keeps every row from the left table and fills NULLs."})
        roles = [m["role"] for m in st.session_state["chat"]]
        assert roles == ["assistant", "user", "assistant"]
        assert "Next question" in st.session_state["chat"][-1]["content"]
        run_app(st, clicks=["💡 Hint"])
        assert [m["role"] for m in st.session_state["chat"]][-2:] == ["user", "assistant"]

        # 6. progress tab shows the history and the weak topic
        run_app(st)
        assert any(x[0] == "dataframe" for x in st.log)
    finally:
        os.chdir(old_cwd)
        if old_mod is not None:
            sys.modules["streamlit"] = old_mod
        else:
            sys.modules.pop("streamlit", None)
