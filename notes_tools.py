"""Notes tools: export revision sheets as markdown files."""
import datetime as dt
import json
import os
import re


def save_summary(title: str, content: str) -> str:
    """Save a revision sheet as a markdown file and return its path."""
    if not content.strip():
        raise ValueError("content must not be empty")
    out_dir = os.environ.get("EXPORT_DIR", "exports")
    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "revision"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{stamp}-{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content.strip()}\n")
    return json.dumps({"path": path})
