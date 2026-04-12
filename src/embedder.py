
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, embed_model_path: str, normalize: bool=True):
        self.model = SentenceTransformer(embed_model_path)
        self.normalize = normalize
        self.index: Optional[faiss.IndexFlatIP] = None
        self.doc_meta: List[Dict[str, Any]] = []
        self.doc_embs: Optional[np.ndarray] = None

    def encode(self, texts: List[str]) -> np.ndarray:
        X = self.model.encode(texts, batch_size=64, normalize_embeddings=self.normalize, convert_to_numpy=True)
        if not self.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            X = X / norms
        return X.astype(np.float32)

    def build_index(self, docs: List[Dict[str, Any]]):
        self.doc_meta = docs
        self.doc_embs = self.encode([d.get("text","") for d in docs])
        dim = self.doc_embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embs)

    def search(self, query: str, k: int=50) -> List[Dict[str, Any]]:
        assert self.index is not None, "Index not built"
        q = self.encode([query])
        D, I = self.index.search(q, min(k, len(self.doc_meta)))
        I = I[0]; D = D[0]
        out = []
        for score, idx in sorted(zip(D, I), key=lambda x: -x[0]):
            m = dict(self.doc_meta[idx])
            m["retrieval_score"] = float(score)
            m["_emb"] = self.doc_embs[idx]
            out.append(m)
        return out
