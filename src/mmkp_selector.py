from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .selector import simple_token_len

def _get_emb(d: Dict[str, Any]) -> np.ndarray:
    if "_emb_norm" in d and d["_emb_norm" ] is not None:
        return np.asarray(d["_emb_norm"], dtype=np.float32)
    if "_emb" in d and d["_emb"] is not None:
        v = np.asarray(d["_emb"], dtype=np.float32)
        n = np.linalg.norm(v) + 1e-8
        return v / n
    return np.zeros((128,), dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

def _greedy_cluster(cands: List[Dict[str, Any]], sim_threshold: float = 0.82) -> List[List[int]]:
    embs = [_get_emb(d) for d in cands]
    used = [False] * len(cands)
    groups: List[List[int]] = []
    for i, d in enumerate(cands):
        if used[i]:
            continue
        centroid = embs[i]
        group = [i]
        used[i] = True
        for j in range(i + 1, len(cands)):
            if used[j]:
                continue
            if _cosine(centroid, embs[j]) >= sim_threshold:
                used[j] = True
                group.append(j)
        groups.append(group)
    return groups

def _normalize(arr: List[float]) -> List[float]:
    if not arr:
        return []
    lo, hi = min(arr), max(arr)
    if hi - lo < 1e-9:
        return [0.5 for _ in arr]
    return [(x - lo) / (hi - lo + 1e-9) for x in arr]

def _build_items(
    cands: List[Dict[str, Any]],
    groups: List[List[int]],
    alpha: float,
    beta: float,
    redundancy_scale: float,
) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:
    rel_raw = [float(d.get("fusion_score", d.get("retrieval_score", 0.0))) for d in cands]
    rel = _normalize(rel_raw)

    embs = [_get_emb(d) for d in cands]

    all_items: List[List[Dict[str, Any]]] = []
    all_vals: List[List[float]] = []

    for g in groups:
        items: List[Dict[str, Any]] = []
        vals: List[float] = []
        for idx in g:
            d = cands[idx]
            sims = []
            for j in g:
                if j == idx:
                    continue
                sims.append(_cosine(embs[idx], embs[j]))
            avg_sim = float(np.mean(sims)) if sims else 0.0
            diversity = 1.0 - avg_sim
            v = alpha * rel[idx] + beta * diversity

            token_len = simple_token_len(d.get("text") or d.get("content") or d.get("passage") or "")
            redundancy_cost = max(0.0, avg_sim) * redundancy_scale

            items.append(d)
            vals.append(v)
            d["_mmkp_token_len"] = int(token_len)
            d["_mmkp_redundancy"] = float(redundancy_cost)
        all_items.append(items)
        all_vals.append(vals)

    return all_items, all_vals

def _pareto_prune(states: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    items = sorted(states.items())
    pruned: Dict[Tuple[int, int], float] = {}
    for (c1, c2), val in items:
        dominated = False
        for (c1b, c2b), valb in pruned.items():
            if c1b <= c1 and c2b <= c2 and valb >= val:
                dominated = True
                break
        if not dominated:
            to_del = []
            for (c1b, c2b), valb in pruned.items():
                if c1 <= c1b and c2 <= c2b and val >= valb:
                    to_del.append((c1b, c2b))
            for k in to_del:
                pruned.pop(k, None)
            pruned[(c1, c2)] = val
    return pruned

def select_documents_mmkp(
    candidates: List[Dict[str, Any]],
    token_budget: int,
    redundancy_budget: int = 100,
    alpha: float = 0.7,
    beta: float = 0.3,
    sim_threshold: float = 0.82,
    redundancy_scale: float = 100.0,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    groups = _greedy_cluster(candidates, sim_threshold=sim_threshold)
    items_by_group, vals_by_group = _build_items(candidates, groups, alpha, beta, redundancy_scale)

    dp: Dict[Tuple[int, int], float] = {(0, 0): 0.0}
    back: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    for g_idx, (items, vals) in enumerate(zip(items_by_group, vals_by_group)):
        newdp: Dict[Tuple[int, int], float] = dict(dp)
        newback: Dict[Tuple[int, int], Tuple[int, int, int]] = dict(back)
        for (c1, c2), curv in dp.items():
            for j, (it, v) in enumerate(zip(items, vals)):
                w1 = int(it.get("_mmkp_token_len", 0))
                w2 = int(round(it.get("_mmkp_redundancy", 0.0)))
                nc1 = c1 + w1
                nc2 = c2 + w2
                if nc1 > token_budget or nc2 > redundancy_budget:
                    continue
                nv = curv + v
                key = (nc1, nc2)
                if nv > newdp.get(key, -1e18):
                    newdp[key] = nv
                    newback[key] = (c1, c2, j)

        newdp = _pareto_prune(newdp)
        back = {k: v for k, v in newback.items() if k in newdp}
        dp = newdp

    best_state = max(dp.items(), key=lambda kv: kv[1])[0] if dp else (0, 0)

    chosen_indices: List[Tuple[int, int]] = []
    cstate = best_state
    for g_idx in range(len(items_by_group) - 1, -1, -1):
        pass

    rem1, rem2 = token_budget, redundancy_budget
    selected: List[Dict[str, Any]] = []
    for items, vals in zip(items_by_group, vals_by_group):
        best_j, best_gain = -1, -1e18
        for j, (it, v) in enumerate(zip(items, vals)):
            w1 = int(it.get("_mmkp_token_len", 0))
            w2 = int(round(it.get("_mmkp_redundancy", 0.0)))
            if w1 <= rem1 and w2 <= rem2 and v > best_gain:
                best_gain = v
                best_j = j
        if best_j >= 0:
            it = items[best_j]
            selected.append(it)
            rem1 -= int(it.get("_mmkp_token_len", 0))
            rem2 -= int(round(it.get("_mmkp_redundancy", 0.0)))

    selected.sort(key=lambda d: -float(d.get("fusion_score", d.get("retrieval_score", 0.0))))
    return selected
