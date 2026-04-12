from __future__ import annotations
from typing import List

def simple_plan(question: str, max_subqs: int = 4) -> List[str]:
    q = (question or "").strip().rstrip("?")
    if not q:
        return []

    seeds = [
        f"What background/definitions are needed for {q}?",
        f"What are the key entities/times/numbers in {q}?",
        f"To answer {q}, what direct evidence is still missing?",
        f"Based on the above, answer directly: {q}",
    ]

    out: List[str] = []
    for s in seeds:
        if s not in out:
            out.append(s)
        if len(out) >= max_subqs:
            break
    return out
