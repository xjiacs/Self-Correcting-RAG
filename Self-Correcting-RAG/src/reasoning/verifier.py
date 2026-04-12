
from __future__ import annotations
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class NLIVerifier:
    def __init__(self, model_name: str = "roberta-large-mnli", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.idx = {"contradict": 0, "neutral": 1, "entail": 2}

    @torch.inference_mode()
    def score(self, premise: str, hypothesis: str) -> Dict[str,float]:
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.model(**x).logits
        prob = torch.softmax(logits, dim=-1)[0]
        return {
            "entail": float(prob[self.idx["entail"]].item()),
            "neutral": float(prob[self.idx["neutral"]].item()),
            "contradict": float(prob[self.idx["contradict"]].item()),
        }
