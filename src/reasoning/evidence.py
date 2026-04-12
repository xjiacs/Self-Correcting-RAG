
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def _split_sentences(text: str) -> List[str]:
    if not text: return []
    tmp = text.replace("!", ".").replace("?", ".").replace("！", "。").replace("？", "。")
    parts = []
    for seg in tmp.split("。"):
        seg = seg.strip()
        if not seg: continue
        parts.extend([s.strip() for s in seg.split(".") if s.strip()])
    return parts[:200]

def _build_tfidf(sents: List[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=100000)
    vec.fit(sents)
    return vec

def _score_sentences(query: str, sents: List[str], vec: TfidfVectorizer, emb_fn, alpha=0.6, beta=0.4) -> List[Tuple[int, float]]:
    qv = vec.transform([query])
    Sv = vec.transform(sents)
    tfidf = (qv @ Sv.T).toarray()[0]
    if emb_fn is None:
        sim = np.zeros_like(tfidf)
    else:
        qe = emb_fn([query])[0]
        se = emb_fn(sents)
        sim = (se @ qe).astype(float)
    scores = alpha * tfidf + beta * sim
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]

def extract_top_evidence(query: str, docs: List[Dict[str,Any]], emb_fn, per_doc=3, total=8):
    items = []
    for d in docs:
        sents = _split_sentences(d.get("text",""))
        if not sents: 
            continue
        vec = _build_tfidf(sents)
        ranked = _score_sentences(query, sents, vec, emb_fn)
        for i, sc in ranked[:per_doc]:
            items.append({
                "doc_id": d.get("doc_id"),
                "title": d.get("title",""),
                "sent": sents[i],
                "score": sc
            })
    items.sort(key=lambda x: -x["score"])
    return items[:total]
