"""Lightweight spaced repetition: weak or neglected topics come up more often.

Not full SM-2. Each topic gets a weight from its average score and how long
ago it was last practised, and the next quiz topic is drawn by weight.
"""
import datetime as dt
import random

NEW_TOPIC_WEIGHT = 3.0  # never practised: worth covering, but below clearly weak topics
TARGET = 7.0            # average (/10) we call "on track"


def days_since(iso: str, now: dt.datetime | None = None) -> int:
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        then = dt.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return 0
    if then.tzinfo is None:
        then = then.replace(tzinfo=dt.timezone.utc)
    return max(0, (now - then).days)


def weight(avg: float, days: int) -> float:
    """Lower scores and longer gaps both raise the weight (gap counts up to two weeks)."""
    return 0.5 + max(0.0, TARGET - avg) + 0.25 * min(days, 14)


def topic_weights(all_topics: list[str], stats: list[dict], now: dt.datetime | None = None) -> dict[str, float]:
    by_topic = {s["topic"]: s for s in stats}
    out = {}
    for t in all_topics:
        s = by_topic.get(t)
        out[t] = (
            weight(s["average_score"], days_since(s["last_practiced_at"], now)) if s else NEW_TOPIC_WEIGHT
        )
    return out


def pick_topic(all_topics: list[str], stats: list[dict], rng: random.Random | None = None) -> str | None:
    if not all_topics:
        return None
    w = topic_weights(all_topics, stats)
    r = rng or random
    return r.choices(list(w), weights=list(w.values()), k=1)[0]
