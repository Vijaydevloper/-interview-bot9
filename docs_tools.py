"""Web/Docs tools: search and fetch from a whitelist of official documentation.

Search uses DuckDuckGo's HTML endpoint restricted to trusted sites. It is a
best-effort convenience; failures return an {"error": ...} payload so the app
can say so instead of inventing an answer.
"""
import json
import re
from urllib.parse import parse_qs, unquote, urlparse

TRUSTED_DOMAINS = [
    "pandas.pydata.org",
    "docs.python.org",
    "numpy.org",
    "scikit-learn.org",
    "scipy.org",
    "www.sqlite.org",
    "www.postgresql.org",
    "matplotlib.org",
    "seaborn.pydata.org",
    "pytorch.org",
]
_HEADERS = {"User-Agent": "Mozilla/5.0 (interview-prep-bot)"}


def _trusted(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def search_docs(query: str, max_results: int = 5) -> str:
    """Search trusted official documentation. Returns title, url, snippet."""
    import httpx
    from bs4 import BeautifulSoup

    sites = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS)
    try:
        r = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": f"{query} ({sites})"},
            headers=_HEADERS,
            timeout=15,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception as e:  # network errors, blocks, timeouts
        return json.dumps({"error": f"search failed: {e}"})

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for res in soup.select(".result"):
        a = res.select_one("a.result__a")
        if not a or not a.get("href"):
            continue
        href = a["href"]
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        if href.startswith("//"):
            href = "https:" + href
        if not _trusted(href):
            continue
        snip = res.select_one(".result__snippet")
        results.append(
            {
                "title": a.get_text(" ", strip=True),
                "url": href,
                "snippet": snip.get_text(" ", strip=True) if snip else "",
            }
        )
        if len(results) >= max_results:
            break
    return json.dumps(results)


def fetch_doc(url: str, max_chars: int = 4000) -> str:
    """Fetch a trusted documentation page and return cleaned text plus URL."""
    if not _trusted(url):
        return json.dumps({"error": "domain is not in the trusted documentation list"})
    import httpx
    from bs4 import BeautifulSoup

    try:
        r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return json.dumps({"error": f"fetch failed: {e}"})
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))
    return json.dumps({"url": url, "text": text[:max_chars]})
