
from __future__ import annotations
import os, json, yaml
from typing import List, Dict, Any

Record = Dict[str, Any]
Doc    = Dict[str, Any]

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path: return []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list): return data
            if isinstance(data, dict): return data.get("data", [])
            return []

class MultiHopRAGDataset:
    def __init__(self, yaml_config: Dict[str, Any]):
        self.cfg = yaml_config
        data_cfg = yaml_config.get("data", {})
        root = data_cfg.get("root", "")
        self.corpus_path  = data_cfg.get("corpus_file")
        self.records_path = data_cfg.get("records_file")
        if root:
            if self.corpus_path and not os.path.isabs(self.corpus_path):
                self.corpus_path = os.path.join(root, self.corpus_path)
            if self.records_path and not os.path.isabs(self.records_path):
                self.records_path = os.path.join(root, self.records_path)
        self.corpus: List[Doc] = []
        self.records: List[Record] = []

    def load(self) -> None:
        self.corpus  = self._build_corpus(_read_json_or_jsonl(self.corpus_path))
        self.records = self._build_records(_read_json_or_jsonl(self.records_path))

    def _build_corpus(self, raw: List[Dict[str, Any]]) -> List[Doc]:
        docs = []
        for i, d in enumerate(raw):
            doc_id = d.get("doc_id") or d.get("id") or d.get("url") or d.get("title") or f"doc_{i}"
            docs.append({
                "doc_id": str(doc_id),
                "title":  d.get("title", ""),
                "text":   d.get("text") or d.get("content") or d.get("fact") or "",
            })
        uniq = {d["doc_id"]: d for d in docs}
        return list(uniq.values())

    def _to_list(self, x):
        if x is None: return []
        return x if isinstance(x, list) else [x]

    def _extract_gold_doc_ids(self, record: Dict[str, Any]) -> List[str]:
        if "evidence_doc_ids" in record:
            return [str(x) for x in self._to_list(record["evidence_doc_ids"])]
        if "paragraphs" in record:
            ids = []
            for ev in record["paragraphs"]:
                ids.append(str(ev.get("doc_id") or ev.get("url") or ev.get("title") or ""))
            return [x for x in ids if x]
        if "supporting_facts" in record:
            return [str(t) for t in self._to_list(record["supporting_facts"])]
        return []

    def _build_records(self, raw: List[Dict[str, Any]]) -> List[Record]:
        out = []
        for r in raw:
            q = r.get("question") or r.get("query") or r.get("prompt") or ""
            answers = self._to_list(r.get("answers") or r.get("answer") or r.get("gold") or r.get("target") or "")
            gold_doc_ids = self._extract_gold_doc_ids(r)
            out.append({"question": q, "answers": [str(a) for a in answers if a != ""], "gold_doc_ids": gold_doc_ids})
        return out
