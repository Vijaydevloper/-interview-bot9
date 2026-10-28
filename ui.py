"""Look and feel: theme CSS and small HTML helpers for the Streamlit app.

Palette follows the Interview Prep Coach layout: ink #152238, a navy-to-teal hero
gradient (#14213d, #244b75, #397b8c), soft slate borders, and one light-teal accent
for scores. System fonts only, so the app renders the same offline.
"""
import html

CSS = """
<style>
:root { --ink:#152238; --muted:#728096; --line:#d8e1ed; --accent:#7fd6e6; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
.hero { padding: 1.5rem 1.7rem; border-radius: 22px; background: linear-gradient(120deg,#14213d,#244b75 65%,#397b8c);
  color: white; margin-bottom: 1.1rem; }
.hero h1 { color: white; margin: 0; font-size: 2.2rem; padding: 0; }
.hero p { color: #e6f2ff; margin: .45rem 0 0 0; font-size: 1.05rem; }
div[data-testid="stMetric"] { border: 1px solid rgba(128,145,170,.25); padding: 1rem; border-radius: 16px;
  background: rgba(128,145,170,.06); }
.stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 600; }
.stButton button { width: 100%; }
div[data-testid="stTabs"] button { font-weight: 650; }
.small-muted { color: var(--muted); font-size: .88rem; }
button:focus-visible, textarea:focus-visible, input:focus-visible { outline: 3px solid #244b75 !important; outline-offset: 2px; }
textarea { font-size: 1rem !important; line-height: 1.5 !important; }

/* tags and sources */
.tags { display: flex; gap: .4rem; flex-wrap: wrap; margin: .2rem 0 .6rem 0; }
.tag { border: 1px solid var(--line); background: rgba(128,145,170,.10); padding: 2px 12px; border-radius: 999px;
  font-size: .82rem; font-weight: 600; }
.tag.notes { background: rgba(57,123,140,.14); border-color: rgba(57,123,140,.35); }
.tag.docs { background: rgba(230,167,0,.16); border-color: rgba(230,167,0,.45); }
.q-text { font-size: 1.45rem; font-weight: 700; line-height: 1.3; margin: .2rem 0 1rem 0; }

/* scoreboard */
.board { background: linear-gradient(120deg,#14213d,#244b75); color: #E8F0F8; border-radius: 18px; padding: 22px 26px;
  margin: 1.1rem 0; display: flex; gap: 2.2rem; flex-wrap: wrap; align-items: center; }
.board .digits { font-size: 4.6rem; font-weight: 800; line-height: .95; color: var(--accent); }
.board .outof { font-size: 1.5rem; color: #9DB4CC; margin-left: .35rem; }
.board .verdict { font-size: 1rem; margin-top: .4rem; font-weight: 600; }
.board .parts { flex: 1; min-width: 240px; display: grid; gap: .55rem; }
.part { display: grid; grid-template-columns: 8.5rem 1fr 2.6rem; gap: .7rem; align-items: center; font-size: .92rem; }
.part .bar { height: 8px; background: rgba(255,255,255,.18); border-radius: 99px; overflow: hidden; }
.part .bar i { display: block; height: 100%; background: var(--accent); border-radius: 99px; }
.part b { text-align: right; color: #FFFFFF; }

/* meters (topics to revisit) */
.meter { display: grid; grid-template-columns: 9rem 1fr 4.6rem; gap: .8rem; align-items: center; padding: .35rem 0; }
.meter .m-label { font-weight: 600; }
.meter .m-track { height: 10px; background: rgba(128,145,170,.25); border-radius: 99px; overflow: hidden; }
.meter .m-fill { display: block; height: 100%; border-radius: 99px; }
.m-weak { background: #D6453D; } .m-mid { background: #E6A700; } .m-good { background: #2E8B62; }
.meter .m-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.hint { color: var(--muted); font-size: .9rem; }

@media (max-width: 640px) {
  .hero h1 { font-size: 1.8rem; } .board .digits { font-size: 3.6rem; }
  .part { grid-template-columns: 6.5rem 1fr 2.4rem; } .meter { grid-template-columns: 6rem 1fr 3.8rem; }
}
@media (prefers-reduced-motion: no-preference) {
  .board { animation: drop .45s cubic-bezier(.2,.8,.2,1); }
  @keyframes drop { from { transform: translateY(-8px); opacity: 0; } to { transform: none; opacity: 1; } }
}
</style>
"""

HERO = """
<div class="hero">
  <h1>🎯 Interview Prep Coach</h1>
  <p>Your document-grounded study partner for Python, SQL, data science, ML, deep learning and agentic AI.</p>
</div>
"""


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def verdict(score: float, max_score: float) -> str:
    r = score / max_score if max_score else 0
    if r >= 0.8:
        return "Interview-ready answer"
    if r >= 0.6:
        return "Close. Tighten the gaps below"
    return "Needs another pass"


_PART_LABELS = {"correctness": "Correctness", "coverage": "Key concepts", "clarity": "Clarity", "example": "Example"}


def scoreboard(score: float, max_score: float, parts: dict, caps: dict) -> str:
    rows = ""
    for key, label in _PART_LABELS.items():
        cap = caps.get(key, 0)
        if key == "example" and max_score == sum(caps.values()) - caps.get("example", 0):
            continue  # example not applicable for this question
        pct = 100 * parts.get(key, 0) / cap if cap else 0
        rows += (
            f'<div class="part"><span>{label}</span>'
            f'<div class="bar"><i style="width:{pct:.0f}%"></i></div>'
            f'<b>{parts.get(key, 0):g}/{cap}</b></div>'
        )
    return (
        '<div class="board" role="group" aria-label="Score">'
        f'<div><div><span class="digits">{score:g}</span><span class="outof">/ {max_score:g}</span></div>'
        f'<div class="verdict">{esc(verdict(score, max_score))}</div></div>'
        f'<div class="parts">{rows}</div></div>'
    )


def tags(items: list[str], kind: str = "") -> str:
    inner = "".join(f'<span class="tag {kind}">{esc(i)}</span>' for i in items if i)
    return f'<div class="tags">{inner}</div>'


def question(text: str) -> str:
    return f'<div class="q-text">{esc(text)}</div>'


def meter(label: str, avg: float, detail: str = "") -> str:
    tone = "m-weak" if avg < 5 else "m-mid" if avg < 7 else "m-good"
    return (
        f'<div class="meter"><span class="m-label">{esc(label)}</span>'
        f'<div class="m-track"><i class="m-fill {tone}" style="width:{max(2, min(avg, 10)) * 10:.0f}%"></i></div>'
        f'<span class="m-val">{avg:.1f}/10</span></div>'
        + (f'<div class="hint">{esc(detail)}</div>' if detail else "")
    )


def sources(items: list[dict]) -> str:
    """items: {"kind": "notes" | "docs", "label": str}"""
    inner = "".join(
        f'<span class="tag {i["kind"]}">{"Your notes" if i["kind"] == "notes" else "Official docs"}: {esc(i["label"])}</span>'
        for i in items
    )
    return f'<div class="tags">{inner}</div>'
