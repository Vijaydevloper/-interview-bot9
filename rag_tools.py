"""RAG tools: search the study notes. Exposed over MCP by rag_server.py."""
import json
from dataclasses import asdict

import rag

_index = None


def _get_index() -> "rag.Index":
    global _index
    if _index is None:
        _index = rag.Index.build()
    return _index


def search_notes(query: str, k: int = 4, topic: str | None = None) -> str:
    """Search the student's study notes (TF-IDF). Returns chunks with text, source, topic, section, score."""
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    hits = _get_index().search(query, k=k, topic=topic or None)
    return json.dumps([asdict(c) for c in hits])


def list_topics() -> str:
    """List study topics with the number of indexed chunks in each."""
    return json.dumps(_get_index().topic_counts())


def sample_notes(topic: str = "", k: int = 3) -> str:
    """A random passage from the topic (or any topic) plus its nearest neighbours: varied quiz material."""
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    return json.dumps([asdict(c) for c in _get_index().sample_context(topic or None, k=k)])


def library_summary() -> str:
    """Per topic: number of chunks and the list of source files."""
    out: dict = {}
    for c in _get_index().chunks:
        t = out.setdefault(c.topic, {"chunks": 0, "files": []})
        t["chunks"] += 1
        if c.source not in t["files"]:
            t["files"].append(c.source)
    return json.dumps(out)
