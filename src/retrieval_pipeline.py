
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _rrf(ranks: List[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)

class TFIDFIndex:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.doc_ids: List[str] = []

    def build(self, docs: List[Dict[str, Any]]):
        texts = [d.get("text","") for d in docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=200000)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.doc_ids = [str(d.get("doc_id")) for d in docs]

    def search(self, query: str, topk: int = 200) -> List[Tuple[int, float]]:
        assert self.vectorizer is not None and self.matrix is not None
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(-scores)[:topk]
        return [(int(i), float(scores[i])) for i in idx]

def fused_retrieve(query: str, docs: List[Dict[str, Any]], emb_candidates: List[Dict[str, Any]], tfidf_index: Optional[TFIDFIndex] = None, topk: int = 50, rrf_k: int = 60, expand_centroid: bool = True) -> List[Dict[str, Any]]:
    cand = emb_candidates
    if expand_centroid and len(cand) >= 5 and "_emb" in cand[0]:
        top_vecs = np.stack([d["_emb"] for d in cand[:5]], axis=0)
        refined = (top_vecs.mean(axis=0, keepdims=True))
        c_vec = refined[0] / (np.linalg.norm(refined) + 1e-8)
        for d in cand:
            v = d["_emb"] / (np.linalg.norm(d["_emb"]) + 1e-8)
            d["centroid_score"] = float(v @ c_vec)
    else:
        for d in cand: d["centroid_score"] = 0.0

    tfidf_ranks = {}
    if tfidf_index is not None:
        tfidf = tfidf_index.search(query, topk=topk*4)
        for rank, (idx, sc) in enumerate(tfidf):
            did = str(docs[idx]["doc_id"])
            tfidf_ranks[did] = rank

    for rank, d in enumerate(cand):
        did = str(d.get("doc_id"))
        r_emb = rank
        r_tfidf = tfidf_ranks.get(did, 10_000)
        d["fusion_score"] = _rrf([r_emb, r_tfidf], k=rrf_k) + 0.3 * d.get("centroid_score", 0.0)

    cand.sort(key=lambda x: (-x["fusion_score"], -x.get("retrieval_score",0.0)))
    return cand[:topk]
