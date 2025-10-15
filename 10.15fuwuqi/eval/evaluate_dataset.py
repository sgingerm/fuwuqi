from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from tqdm import tqdm

# Import pipeline
from vector_graph_pipeline.code.util.pipeline import run_pipeline

def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _get_field(obj: Dict, keys: List[str], default=None):
    for k in keys:
        if k in obj:
            return obj[k]
    return default

def extract_question_and_id(obj: Dict) -> Tuple[str, str]:
    q = _get_field(obj, ["question", "query", "q", "text", "prompt"], None)
    if not q or not isinstance(q, str):
        raise ValueError("Cannot find 'question' text in item.")
    qid = _get_field(obj, ["id", "qid", "question_id", "uid"], None)
    if qid is None:
        qid = ""
    return q.strip(), str(qid)

def extract_gold_chunk_ids(obj: Dict) -> List[str]:
    gold: List[str] = []

    def push(cid):
        if cid and isinstance(cid, str):
            gold.append(cid)

    # Designated positives
    for key in ["positives", "positive_passages", "pos_passages", "gold_passages", "gold"]:
        if key in obj and isinstance(obj[key], list):
            for it in obj[key]:
                if isinstance(it, dict):
                    push(it.get("chunk_id") or it.get("id"))
                elif isinstance(it, str):
                    push(it)
            if gold:
                return sorted(set(gold))

    # Generic passages with label
    for key in ["passages", "docs", "documents"]:
        if key in obj and isinstance(obj[key], list):
            for it in obj[key]:
                if not isinstance(it, dict):
                    continue
                lbl = it.get("label") or it.get("gold") or it.get("is_positive")
                if lbl in (True, "positive", "pos", 1, "1", "true", "True"):
                    push(it.get("chunk_id") or it.get("id"))
            if gold:
                return sorted(set(gold))

    return sorted(set(gold))

def _to_base_id(cid: str) -> str:
    if not isinstance(cid, str):
        return ""
    return cid.split("#", 1)[0]

def precision_recall_f1_at_k(gold_ids: List[str], pred_ids: List[str], k: int) -> Tuple[float, float, float, int]:
    """Doc-level matching (@k). If a predicted chunk like '12345#p1' appears,
    it's treated as hitting the base-id '12345'. We also dedupe predicted base-ids within top-k.
    Precision is computed with denominator k (conservative when duplicates collapse)."""
    if k <= 0:
        return 0.0, 0.0, 0.0, 0
    # normalize gold to base-id (robust even if gold already base)
    G = set(_to_base_id(x) for x in gold_ids if x)
    if not G:
        return 0.0, 0.0, 0.0, 0
    # take top-k predictions, map to base-id, dedupe preserving order
    Pk = pred_ids[:k]
    seen = set()
    Pk_base_unique: List[str] = []
    for cid in Pk:
        b = _to_base_id(cid)
        if not b or b in seen:
            continue
        seen.add(b)
        Pk_base_unique.append(b)
    tp = sum(1 for b in Pk_base_unique if b in G)
    precision = tp / float(k)
    recall = tp / float(len(G))
    f1 = 0.0 if (precision + recall) == 0.0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, tp

def _aggregate_metrics(rows: List[Dict], k: int) -> Dict[str, float]:
    if not rows:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = sum(1 for r in rows if (r.get("tp", 0) or 0) > 0)
    n = len(rows)
    avg_p = sum(r.get("precision", 0.0) for r in rows) / n
    avg_r = sum(r.get("recall", 0.0) for r in rows) / n
    avg_f1 = sum(r.get("f1", 0.0) for r in rows) / n
    return {"hit_rate": hit / n, "precision": avg_p, "recall": avg_r, "f1": avg_f1}

def evaluate_dataset(
    dataset_path: Path,
    out_dir: Path,
    k: int,
    seed_top_k: int = 5,
    appnp_k_hop: int = 1,
    appnp_top_nodes: int = 300,
    bfs_depth: int = 2,
    bfs_max_chunks: int = 60,
    rerank_top_n: int = 64,
    device: Optional[str] = None,
    embed_model: Optional[Path] = None,
    reranker_model: Optional[str] = None,
    max_samples: Optional[int] = None,
    skip_errors: bool = True,
    early_report: int = 20,   # NEW: report after first N items
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_all: List[Dict] = []
    items = list(_read_jsonl(dataset_path))
    if max_samples is not None:
        items = items[:int(max_samples)]
    n = len(items)
    if n == 0:
        raise RuntimeError(f"No items found in {dataset_path}")

    early_report = int(early_report) if early_report is not None else 0
    early_report = max(0, early_report)

    pbar = tqdm(range(n), desc=f"Evaluating @k={k}", ncols=100)
    for idx in pbar:
        obj = items[idx]
        try:
            qtext, qid = extract_question_and_id(obj)
            if not qid: qid = str(idx)
            gold = extract_gold_chunk_ids(obj)

            res = run_pipeline(
                question=qtext,
                seed_top_k=seed_top_k,
                appnp_k_hop=appnp_k_hop,
                appnp_top_nodes=appnp_top_nodes,
                bfs_depth=bfs_depth,
                bfs_max_chunks=bfs_max_chunks,
                rerank_top_n=max(rerank_top_n, k),
                device=device,
                embed_model=embed_model,
                reranker_model=reranker_model,
                edge_policy="either",
            )
            preds = [x["chunk_id"] for x in (res.get("reranked") or []) if "chunk_id" in x]

            prec, rec, f1, tp = precision_recall_f1_at_k(gold, preds, k)

            row = {
                "idx": idx,
                "id": qid,
                "k": k,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "gold_count": len(gold),
                "pred_count": len(preds),
            }
            results_all.append(row)

        except Exception as e:
            if not skip_errors:
                raise
            # still compute gold_count for reporting
            try:
                gtmp = extract_gold_chunk_ids(obj)
            except Exception:
                gtmp = []
            row = {
                "idx": idx,
                "id": str(obj.get("id", idx)),
                "k": k,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "gold_count": len(gtmp),
                "pred_count": 0,
                "error": str(e),
            }
            results_all.append(row)

        # ---- live postfix for first N items ----
        if early_report and (idx + 1) <= early_report:
            early_rows = results_all[: idx + 1]
            agg = _aggregate_metrics(early_rows, k)
            pbar.set_postfix({
                "early": f"{idx+1}/{early_report}",
                f"hit@{k}": f"{agg['hit_rate']*100:.1f}%",
                f"F1@{k}": f"{agg['f1']:.3f}",
            })

        # ---- exact report when reaching N ----
        if early_report and (idx + 1) == early_report:
            early_rows = results_all[:early_report]
            agg = _aggregate_metrics(early_rows, k)
            print("\\n--- Early Report (first %d items) @k=%d ---" % (early_report, k), flush=True)
            print("Hit rate: %.2f%%  |  Avg P: %.4f  R: %.4f  F1: %.4f" % (
                agg["hit_rate"]*100, agg["precision"], agg["recall"], agg["f1"]
            ), flush=True)
            # dump early report
            (out_dir / f"eval_k{k}_early_{early_report}.json").write_text(
                json.dumps({"k": k, "early": early_report, "aggregate": agg, "rows": early_rows}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

    # sort by f1 desc, take top 30%
    import math, csv
    top_n = max(1, math.ceil(0.3 * len(results_all)))
    top_sorted = sorted(results_all, key=lambda r: r["f1"], reverse=True)[:top_n]

    # all
    all_csv = out_dir / f"eval_k{k}_all.csv"
    with all_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "idx", "id", f"precision@{k}", f"recall@{k}", f"f1@{k}", "tp", "gold_count", "pred_count"])
        for rank, r in enumerate(sorted(results_all, key=lambda r: r["f1"], reverse=True), start=1):
            w.writerow([rank, r["idx"], r["id"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", r["tp"], r["gold_count"], r["pred_count"]])

    # top30
    top_csv = out_dir / f"eval_k{k}_top30.csv"
    with top_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "idx", "id", f"precision@{k}", f"recall@{k}", f"f1@{k}"])
        for rank, r in enumerate(top_sorted, start=1):
            w.writerow([rank, r["idx"], r["id"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}"])

    # json dump
    with (out_dir / f"eval_k{k}_all.json").open("w", encoding="utf-8") as fh:
        json.dump(results_all, fh, ensure_ascii=False, indent=2)
    with (out_dir / f"eval_k{k}_top30.json").open("w", encoding="utf-8") as fh:
        json.dump(top_sorted, fh, ensure_ascii=False, indent=2)

    summary = {
        "k": k,
        "num_questions": len(results_all),
        "top_30_percent": top_n,
        "outputs": {
            "all_csv": str(all_csv),
            "top30_csv": str(top_csv),
            "all_json": str(out_dir / f"eval_k{k}_all.json"),
            "top30_json": str(out_dir / f"eval_k{k}_top30.json"),
            "early_json": str(out_dir / f"eval_k{k}_early_{early_report}.json") if early_report else None,
        }
    }
    with (out_dir / f"eval_k{k}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate retrieval metrics (precision/recall/F1) per question at @k (doc-level), with early report.")
    ap.add_argument("--dataset", type=Path, default=Path(r"D:\datanew\question-answer-passages_test.filtered.strict.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\kg_out\eval"))
    ap.add_argument("--k", type=int, required=True, help="cutoff @k (e.g., 2)")
    ap.add_argument("--seed-top-k", type=int, default=5)
    ap.add_argument("--appnp-k-hop", type=int, default=1)
    ap.add_argument("--appnp-top-nodes", type=int, default=300)
    ap.add_argument("--bfs-depth", type=int, default=2)
    ap.add_argument("--bfs-max-chunks", type=int, default=60)
    ap.add_argument("--rerank-top-n", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embed-model", type=Path, default=None)
    ap.add_argument("--reranker-model", type=str, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="Only evaluate first N samples.")
    ap.add_argument("--skip-errors", action="store_true", default=True)
    ap.add_argument("--early-report", type=int, default=20, help="Print and save an early summary after the first N items.")
    args = ap.parse_args()

    summary = evaluate_dataset(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        k=args.k,
        seed_top_k=args.seed_top_k,
        appnp_k_hop=args.appnp_k_hop,
        appnp_top_nodes=args.appnp_top_nodes,
        bfs_depth=args.bfs_depth,
        bfs_max_chunks=args.bfs_max_chunks,
        rerank_top_n=args.rerank_top_n,
        device=args.device,
        embed_model=args.embed_model,
        reranker_model=args.reranker_model,
        max_samples=args.max_samples,
        skip_errors=args.skip_errors,
        early_report=args.early_report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
