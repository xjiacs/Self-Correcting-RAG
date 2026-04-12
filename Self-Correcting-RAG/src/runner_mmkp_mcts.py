from __future__ import annotations
import os, argparse, yaml, json
from typing import List, Dict, Any
from tqdm import tqdm

from .data_loader import MultiHopRAGDataset
from .embedder import Retriever
from .retrieval_pipeline import TFIDFIndex, fused_retrieve
from .mmkp_selector import select_documents_mmkp
from .mcts_guided import MCTSGuidedGenerator
from .evaluation import evaluate_batch


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_pred_answer(raw_text: str) -> str:
    s = (raw_text or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, dict) and "answer" in obj:
                a = obj.get("answer")
                if isinstance(a, str):
                    return a.strip()
        except Exception:
            pass
    return s


def cmd_eval(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    retr_cfg = cfg.get("retriever", {})
    retv_cfg = cfg.get("retrieval", {})
    sel_cfg = cfg.get("selector", {})
    gen_cfg = cfg.get("generator", {})
    out_dir = cfg.get("output_dir", "outputs_full_mmkp_mcts")
    os.makedirs(out_dir, exist_ok=True)

    pred_txt = os.path.join(out_dir, "predictions.txt")
    inter_jsonl = os.path.join(out_dir, "intermediate.jsonl")
    pred_json_path = os.path.join(out_dir, "predictions.json")

    open(pred_txt, "w", encoding="utf-8").close()
    open(inter_jsonl, "w", encoding="utf-8").close()

    ds = MultiHopRAGDataset(cfg)
    ds.load()
    retr = Retriever(embed_model_path=retr_cfg.get("embed_model_path", "BAAI/bge-small-en-v1.5"))
    retr.build_index(ds.corpus)

    tfidf = TFIDFIndex() if retv_cfg.get("use_tfidf", True) else None
    if tfidf is not None:
        tfidf.build(ds.corpus)

    topk = int(retr_cfg.get("topk", 50))
    rrf_k = int(retv_cfg.get("rrf_k", 60))
    expand_centroid = bool(retv_cfg.get("expand_centroid", True))
    budget = int(sel_cfg.get("token_budget", 1500))
    alpha = float(sel_cfg.get("alpha", 0.7))
    beta = float(sel_cfg.get("beta", 0.3))
    redund_budget = int(sel_cfg.get("redundancy_budget", 100))
    sim_threshold = float(sel_cfg.get("sim_threshold", 0.82))
    redund_scale = float(sel_cfg.get("redundancy_scale", 100.0))

    mcts = MCTSGuidedGenerator(
        base_model_path=gen_cfg.get("model_path", "Qwen/Qwen2-1.5B-Instruct"),
        nli_model_name=gen_cfg.get("nli_model", "roberta-large-mnli"),
        simulations=int(gen_cfg.get("simulations", 24)),
        branching=int(gen_cfg.get("branching", 3)),
        max_depth=int(gen_cfg.get("max_depth", 3)),
        entail_w=float(gen_cfg.get("entail_weight", 1.0)),
        contra_w=float(gen_cfg.get("contradict_weight", -2.0)),
        neutral_w=float(gen_cfg.get("neutral_weight", -0.2)),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
    )

    preds: List[str] = []
    golds: List[List[str]] = []
    retrieved_ids_batch: List[List[str]] = []
    gold_ids_batch: List[List[str]] = []
    json_records: List[Dict[str, Any]] = []

    emb_fn = lambda texts: retr.encode(texts)

    for rec in tqdm(ds.records, desc="eval+MMKP+MCTS"):
        q = rec["question"]
        emb_cands = retr.search(q, k=topk * 4)
        cand = fused_retrieve(
            q,
            ds.corpus,
            emb_cands,
            tfidf if retv_cfg.get("use_tfidf", True) else None,
            topk=topk,
            rrf_k=rrf_k,
            expand_centroid=expand_centroid,
        )
        ctx_docs = select_documents_mmkp(
            cand,
            token_budget=budget,
            redundancy_budget=redund_budget,
            alpha=alpha,
            beta=beta,
            sim_threshold=sim_threshold,
            redundancy_scale=redund_scale,
        )

        ans = mcts.generate_mcts(q, ctx_docs, retriever=retr, emb_fn=emb_fn)

        try:
            evs_log = []
            from .reasoning.evidence import extract_top_evidence
            from .reasoning.verifier import NLIVerifier

            nli_tmp = NLIVerifier(model_name=gen_cfg.get("nli_model", "roberta-large-mnli"))
            evs = extract_top_evidence(q, ctx_docs, emb_fn, per_doc=3, total=8)
            for e in evs:
                sc = nli_tmp.score(e["sent"], ans)
                evs_log.append({"sent": e["sent"], "scores": sc, "doc_id": e.get("doc_id")})
        except Exception:
            evs_log = []

        sel_docs = []
        for d in ctx_docs:
            sel_docs.append(
                {
                    "doc_id": str(d.get("doc_id")),
                    "title": d.get("title"),
                    "fusion_score": float(d.get("fusion_score", d.get("retrieval_score", 0.0))),
                    "token_len": int(d.get("_mmkp_token_len", 0)),
                    "redundancy": float(d.get("_mmkp_redundancy", 0.0)),
                }
            )

        with open(pred_txt, "a", encoding="utf-8") as fpt:
            gold_joined = " ||| ".join([str(x) for x in rec.get("answers", [])])
            fpt.write(f"{q}\t{ans}\t{gold_joined}\n")

        inter_record = {
            "question": q,
            "prediction": ans,
            "gold_answers": rec.get("answers", []),
            "selected_docs": sel_docs,
            "evidence_nli": evs_log,
        }
        with open(inter_jsonl, "a", encoding="utf-8") as fj:
            fj.write(json.dumps(inter_record, ensure_ascii=False) + "\n")

        pred_answer = _extract_pred_answer(ans)
        retrieved_ids = [str(d.get("doc_id")) for d in ctx_docs]
        reasoning_brief = (
            f"Generated using MMKP-selected context and NLI-guided MCTS; uses {len(retrieved_ids)} evidence passages and includes NLI-based evidence scoring."
        )
        model_out_obj = {
            "answer": pred_answer,
            "evidence_doc_ids": retrieved_ids,
            "reasoning": reasoning_brief,
        }
        model_output_str = json.dumps(model_out_obj, ensure_ascii=False, indent=2)
        json_records.append(
            {
                "question": q,
                "model_output": model_output_str,
                "pred_answer": pred_answer,
                "answer": rec.get("answers", []),
            }
        )

        preds.append(ans)
        golds.append(rec.get("answers", []))
        retrieved_ids_batch.append(retrieved_ids)
        gold_ids_batch.append([str(x) for x in rec.get("gold_doc_ids", [])])

    metrics = evaluate_batch(
        preds,
        golds,
        retrieved_ids=retrieved_ids_batch,
        gold_relevant_ids=gold_ids_batch,
        k=int(retr_cfg.get("eval_k", 10)),
    )
    with open(os.path.join(out_dir, "metrics_mmkp_mcts.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    final_obj = {"title": "Self-Correcting RAG Predictions (MMKP+MCTS+NLI)", "data": json_records}
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["eval"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.command == "eval":
        cmd_eval(cfg)


if __name__ == "__main__":
    main()
