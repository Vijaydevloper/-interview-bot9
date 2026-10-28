"""RAG foundation: ingestion, chunking and retrieval over study documents.

Retrieval uses TF-IDF (scikit-learn) so the project runs with no embedding
model or API key. The Index class is the only thing the app touches, so it can
later be swapped for embeddings + a vector DB without changing the UI.

Topic label = name of the sub-folder a file lives in (data/docs/<topic>/file).
Files placed directly in the docs folder get the topic "general".
"""
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = os.environ.get("STUDY_DOCS", "data/docs")
SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".pptx"}
_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.M)


@dataclass
class Chunk:
    text: str
    source: str  # file name
    topic: str
    page: int | None = None
    score: float = 0.0
    section: str = ""  # nearest heading above the chunk

    @property
    def ref(self) -> str:
        return f"{self.source}" + (f", p.{self.page}" if self.page else "")


# ---------- ingestion ----------
def _docx_text(path: Path) -> str:
    """Body text in document order: headings become '#' lines, tables become 'a | b' rows."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text or text.lower() == "table of contents":
                continue
            try:
                style = para.style.name if para.style is not None else ""
            except Exception:
                style = ""
            if style.startswith("Heading") or style == "Title":
                digits = re.findall(r"\d", style)
                level = min(int(digits[0]), 3) if digits else 1
                text = "#" * level + " " + text
            blocks.append(text)
        elif tag == "tbl":
            rows = []
            for row in Table(child, doc).rows:
                cells = []
                for cell in row.cells:
                    t = cell.text.strip().replace("\n", " ")
                    if not cells or t != cells[-1]:  # skip merged-cell repeats
                        cells.append(t)
                rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _pptx_slides(path: Path) -> list[tuple[int | None, str]]:
    """Slide text without python-pptx: read <a:t> runs per paragraph from each slide's XML.
    The first line of a slide becomes its heading so quiz questions can name the slide."""
    import zipfile
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    out = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        for name in sorted(names, key=lambda n: int(re.findall(r"\d+", n)[0])):
            root = ET.fromstring(z.read(name))
            paras = []
            for p in root.iter(ns + "p"):
                t = "".join(r.text or "" for r in p.iter(ns + "t")).strip()
                if t:
                    paras.append(t)
            if paras:
                out.append((int(re.findall(r"\d+", name)[0]), f"# {paras[0]}\n\n" + "\n\n".join(paras[1:])))
    return out


def _read_pages(path: Path) -> list[tuple[int | None, str]]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == ".docx":
        return [(None, _docx_text(path))]
    if ext == ".pptx":
        return _pptx_slides(path)
    return []


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking. A heading always travels with the paragraph after it."""
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras: list[str] = []
    for p in raw:
        if paras and paras[-1].startswith("#") and "\n" not in paras[-1] and len(paras[-1]) < 120:
            paras[-1] = paras[-1] + "\n" + p
        else:
            paras.append(p)
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > size * 2:  # a single huge paragraph
            chunks.append(cur[:size])
            cur = cur[size - overlap :]
    if cur.strip():
        chunks.append(cur)
    return chunks


def load_chunks(docs_dir: str = DOCS_DIR) -> list[Chunk]:
    root = Path(docs_dir)
    out: list[Chunk] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(root)
        topic = rel.parts[0] if len(rel.parts) > 1 else "general"
        try:
            pages = _read_pages(path)
        except Exception:
            continue  # unreadable file: skip rather than crash the app
        section = ""
        for page, text in pages:
            for piece in chunk_text(text):
                found = _HEADING.findall(piece)
                if found:
                    section = found[-1].strip()
                out.append(Chunk(piece, path.name, topic, page, 0.0, section))
    return out


def folder_signature(docs_dir: str = DOCS_DIR) -> tuple:
    """Cheap fingerprint used to rebuild the index when files change."""
    root = Path(docs_dir)
    if not root.exists():
        return ()
    return tuple(
        (str(p), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


# ---------- retrieval ----------
_TOKEN = re.compile(r"[a-z0-9_]+")


def _stem(w: str) -> str:
    """Very light stemmer so 'joins' matches 'JOIN' and 'queries' matches 'query'."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def analyze(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    toks = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.topics = sorted({c.topic for c in chunks})
        self._vec = self._matrix = None
        if chunks:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
            # the section heading is repeated so "SQL joins" finds the JOIN section first
            self._matrix = self._vec.fit_transform(
                [f"{c.section} {c.section} {c.text}" for c in chunks]
            )

    @classmethod
    def build(cls, docs_dir: str = DOCS_DIR) -> "Index":
        return cls(load_chunks(docs_dir))

    def topic_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.topic] = counts.get(c.topic, 0) + 1
        return counts

    def search(self, query: str, k: int = 4, topic: str | None = None) -> list[Chunk]:
        if not self.chunks or not query.strip():
            return []
        from sklearn.metrics.pairwise import linear_kernel

        if topic:  # inside one topic its own name ("sql") says nothing about which passage fits
            trimmed = re.sub(rf"\b{re.escape(topic)}\b", " ", query, flags=re.I)
            if _TOKEN.search(trimmed.lower()):
                query = trimmed
        sims = linear_kernel(self._vec.transform([query]), self._matrix).ravel()
        order = sims.argsort()[::-1]
        hits = []
        for i in order:
            if sims[i] <= 0:
                break
            if topic and self.chunks[i].topic != topic:
                continue
            c = self.chunks[i]
            hits.append(Chunk(c.text, c.source, c.topic, c.page, float(sims[i]), c.section))
            if len(hits) == k:
                break
        return hits

    def sample_context(self, topic: str | None = None, k: int = 3) -> list[Chunk]:
        """A random chunk from the topic plus its nearest neighbours."""
        pool = [c for c in self.chunks if not topic or c.topic == topic]
        if not pool:
            return []
        seed = random.choice(pool)
        related = self.search(seed.text, k=k, topic=seed.topic)
        return related or [seed]
