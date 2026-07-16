from __future__ import annotations


def mt_reward(hypothesis: str, reference: str) -> float:
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    h = set(hypothesis.lower().split())
    r = set(reference.lower().split())
    return len(h.intersection(r)) / max(1, len(r))
