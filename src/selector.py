
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

def simple_token_len(text: str) -> int:
    if not text: return 0
    return max(1, int(len(text.split()) * 1.3))

def mmr_rerank(cands: List[Dict[str, Any]], lambda_mm: float = 0.6, topn: int = 20) -> List[Dict[str, Any]]:
    items = cands[:]
    S = np.stack(
        [d["_emb_norm"] if "_emb_norm" in d else d["_emb"] for d in items if d.get("_emb") is not None] or np.zeros(
            (1, 128)), axis=0)
    if S.ndim == 2 and S.shape[0] == len(items):
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        sim = S @ S.T
    else:
        n = len(items)
        sim = np.eye(n) * 1.0
        for i in range(n):
            for j in range(i+1, n):
                s = 1.0 if (items[i].get("title")==items[j].get("title")) else 0.1
                sim[i,j] = sim[j,i] = s

    selected = []
    remain = list(range(len(items)))
    if not remain: return []
    seed = max(remain, key=lambda i: items[i].get("fusion_score", items[i].get("retrieval_score", 0.0)))
    selected.append(seed); remain.remove(seed)
    while remain and len(selected) < topn:
        best_i = None; best_val = -1e9
        for i in remain:
            rel = items[i].get("fusion_score", items[i].get("retrieval_score", 0.0))
            div = max(sim[i, j] for j in selected) if selected else 0.0
            val = lambda_mm * rel - (1 - lambda_mm) * div
            if val > best_val:
                best_val = val; best_i = i
        selected.append(best_i); remain.remove(best_i)
    return [items[i] for i in selected]

def select_documents(candidates: List[Dict[str, Any]], budget_tokens: int = 1500, lambda_mm: float = 0.6) -> List[Dict[str, Any]]:
    for d in candidates:
        d["token_len"] = d.get("token_len") or simple_token_len(d.get("text",""))
        if d.get("_emb") is not None:
            v = d["_emb"]; d["_emb_norm"] = v / (np.linalg.norm(v) + 1e-8)
    topn = max(5, min(len(candidates), budget_tokens // 120 * 4))
    reranked = mmr_rerank(candidates, lambda_mm=lambda_mm, topn=topn)
    out = []; used = 0
    for d in reranked:
        if used + d["token_len"] <= budget_tokens:
            out.append(d); used += d["token_len"]
    return out
