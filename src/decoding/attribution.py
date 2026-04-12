from __future__ import annotations
from typing import List, Set
import torch
from transformers import LogitsProcessor


class EvidenceBiasProcessor(LogitsProcessor):
    def __init__(self, tokenizer, evidence_texts: List[str], boost: float = 1.5):
        self.tok = tokenizer
        vocab: Set[int] = set()
        for t in evidence_texts or []:
            ids = self.tok.encode(t, add_special_tokens=False)
            vocab.update(ids)
        self.bias_ids = torch.tensor(list(vocab), dtype=torch.long)
        self.boost = float(boost)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.bias_ids.numel() == 0:
            return scores

        ids = self.bias_ids.to(scores.device)
        vocab_size = scores.size(-1)
        if vocab_size <= 0:
            return scores
        ids = ids[(ids >= 0) & (ids < vocab_size)]
        if ids.numel() == 0:
            return scores

        ids = torch.unique(ids)

        batch = scores.size(0)
        update = torch.full(
            (batch, ids.numel()),
            self.boost,
            dtype=scores.dtype,
            device=scores.device,
        )

        scores.index_add_(dim=1, index=ids, source=update)
        return scores