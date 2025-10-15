from __future__ import annotations

import json, math, csv, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from tqdm import tqdm
from vector_graph_pipeline.code.util.pipeline import run_pipeline

# -------------------------------
# IO helpers
# -------------------------------
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

# -------------------------------
# Gold extraction (robust, handles string-formatted lists)
# -------------------------------
_NUM_RE = re.compile(r"[A-Za-z0-9]+")

def _parse_list_like_string(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        out: List[str] = []
        if isinstance(obj, list):
            for x in obj:
                if x is None:
                    continue
                if isinstance(x, (int, float)):
                    out.append(str(int(x)))
                elif isinstance(x, str):
                    x = x.strip()
                    if x:
                        out.append(x)
            return out
    except Exception:
        pass
    # Regex fallback
    toks = _NUM_RE.findall(s)
    out = [t for t in toks if not (t.lower() in {"pmid", "id", "doc"})]
    return out

def extract_gold_doc_ids(obj: Dict) -> List[str]:
    """尽量覆盖常见键与嵌套结构，支持 list[str|int|dict] 与 str(JSON/list-like)"""
    gold: List[str] = []

    def push(x):
        if x is None:
            return
        if isinstance(x, str):
            x = x.strip()
            if x:
                gold.append(x)
        elif isinstance(x, (int, float)):
            gold.append(str(int(x)))

    # 常见键优先
    candidate_keys = [
        "relevant_passage_ids", "relevant_doc_ids", "relevant_base_ids",
        "relevant_documents", "gold_doc_ids", "gold_docs", "gold",
        "doc_ids", "docs_gold", "pmids", "pmid_list"
    ]
    for key in candidate_keys:
        if key in obj:
            v = obj[key]
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        for kk in ("id", "pmid", "doc_id", "base_id"):
                            if kk in it:
                                push(it[kk])
                    else:
                        push(it)
            elif isinstance(v, str):
                for it in _parse_list_like_string(v):
                    push(it)
            if gold:
                return sorted(set(gold))

    # 嵌套对象兜底
    for subk in ("gold", "relevant", "labels"):
        if subk in obj and isinstance(obj[subk], dict):
            for kk in ("ids", "doc_ids", "base_ids", "pmids"):
                v = obj[subk].get(kk)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            for k2 in ("id", "pmid", "doc_id", "base_id"):
                                if k2 in it:
                                    push(it[k2])
                        else:
                            push(it)
                elif isinstance(v, str):
                    for it in _parse_list_like_string(v):
                        push(it)
            if gold:
                return sorted(set(gold))

    # 单值兜底
    for key in ["pmid", "docid", "doc_id", "base_id", "id"]:
        if key in obj and obj.get(key) is not None:
            push(obj[key])
            break

    return sorted(set(gold))

# -------------------------------
# Normalization
# -------------------------------
def _norm_id(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'^(pmid:|doc:|id:)', '', s)
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s

# -------------------------------
# Build chunk_id -> base_id map from graph
# -------------------------------
def build_chunk2base_from_graph(graph_path: Path) -> Dict[str, str]:
    """同时从 nodes 与 edges 抽取映射；打印来源统计。"""
    mp: Dict[str, str] = {}
    if not graph_path or not graph_path.exists():
        print(f"\x1b[33m[warn]\x1b[0m Graph not found: {graph_path}")
        return mp
    with graph_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    added_from_nodes = 0
    added_from_edges = 0

    # nodes 侧
    for n in (data.get("nodes") or []):
        cid = n.get("chunk_id") or n.get("id") or n.get("chunkId")
        bid = n.get("base_id") or n.get("baseId") or n.get("doc_id") or n.get("pmid")
        if cid and bid and cid not in mp:
            mp[cid] = bid
            added_from_nodes += 1

    # edges 侧
    for e in (data.get("links") or data.get("edges") or []):
        cid = e.get("chunk_id") or e.get("source_chunk") or e.get("cid")
        bid = e.get("base_id")  or e.get("doc_id")       or e.get("pmid")
        if cid and bid and cid not in mp:
            mp[cid] = bid
            added_from_edges += 1

    print(f"[info] chunk2base built: {len(mp)} entries (from nodes={added_from_nodes}, edges={added_from_edges})")
    return mp

def pred_doc_ids_from_chunks(pred_chunk_ids: List[str], chunk2base: Dict[str, str], k: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for cid in pred_chunk_ids:
        bid = chunk2base.get(cid)
        # 若图里没有该 chunk 的 base，回退到 chunk 前缀（常见形式 base#chunk）
        bnorm = _norm_id(bid if bid is not None else cid.split("#", 1)[0])
        if not bnorm or bnorm in seen:
            continue
        seen.add(bnorm)
        out.append(bnorm)
        if len(out) >= k:
            break
    return out

# === Stage-diagnostics helpers ===

def doc_recall_curve_from_chunks(
    gold_doc_ids: List[str],
    pred_chunk_ids: List[str],
    chunk2base: Dict[str, str],
    ks: List[int]
) -> Dict[int, float]:
    """把一串 chunk 直接映到 base（去重后按顺序截断），计算各 @K 的 doc-Recall。"""
    G = set(_norm_id(x) for x in gold_doc_ids if x)
    out: Dict[int, float] = {}
    if not G:
        return {int(k): 0.0 for k in ks}

    seen_docs = set()
    doc_order: List[str] = []
    for cid in pred_chunk_ids:
        bid = chunk2base.get(cid)
        bnorm = _norm_id(bid if bid is not None else cid.split("#", 1)[0])
        if not bnorm or bnorm in seen_docs:
            continue
        seen_docs.add(bnorm)
        doc_order.append(bnorm)

    for k in ks:
        k = int(k)
        top = doc_order[:k]
        tp = sum(1 for b in top if b in G)
        out[k] = tp / float(len(G)) if G else 0.0
    return out


def doc_order_from_grouped_maxscore(
    reranked_items: List,  # list[dict] or list[(cid, score)]
    chunk2base: Dict[str, str]
) -> List[str]:
    """
    对重排结果按 base_id 分组取“最大分”作为该文档代表分，返回按代表分从高到低的文档序列。
    便于计算“重排后 + 文档聚合”的 doc-Recall@K。
    """
    best: Dict[str, float] = {}
    for it in (reranked_items or []):
        if isinstance(it, dict):
            cid = it.get("chunk_id") or it.get("cid") or it.get("id")
            score = it.get("score") or it.get("s") or 0.0
        else:
            try:
                cid, score = it
            except Exception:
                continue
        if not cid:
            continue
        bid = chunk2base.get(cid)
        bnorm = _norm_id(bid if bid is not None else cid.split("#", 1)[0])
        if not bnorm:
            continue
        if (bnorm not in best) or (float(score) > float(best[bnorm])):
            best[bnorm] = float(score)

    return [bid for bid, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]

# -------------------------------
# Metrics
# -------------------------------
def precision_recall_f1_at_k(gold_doc_ids: List[str], pred_chunk_ids: List[str], k: int, chunk2base: Dict[str, str]) -> Tuple[float, float, float, int]:
    if k <= 0:
        return 0.0, 0.0, 0.0, 0
    G = set(_norm_id(x) for x in gold_doc_ids if x)
    if not G:
        return 0.0, 0.0, 0.0, 0
    Pk = pred_doc_ids_from_chunks(pred_chunk_ids, chunk2base, k)
    tp = sum(1 for b in Pk if b in G)
    precision = tp / float(k)
    recall = tp / float(len(G))
    f1 = 0.0 if (precision + recall) == 0.0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, tp

def _agg_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = sum(1 for r in rows if (r.get("tp", 0) or 0) > 0)
    n = len(rows)
    avg_p = sum(r.get("precision", 0.0) for r in rows) / n
    avg_r = sum(r.get("recall", 0.0) for r in rows) / n
    avg_f1 = sum(r.get("f1", 0.0) for r in rows) / n
    return {"hit_rate": hit / n, "precision": avg_p, "recall": avg_r, "f1": avg_f1}

# -------------------------------
# Preflight 预检
# -------------------------------
def _preflight_scan(items: List[Dict], chunk2base: Dict[str, str], filter_gold_in_graph: bool):
    total = len(items)
    has_q = 0
    gold_raw = 0
    gold_after = 0

    graph_base_ids = set(_norm_id(b) for b in chunk2base.values())

    first_keys = []
    for i, obj in enumerate(items[:3]):
        first_keys.append(sorted(list(obj.keys())))

    for obj in items:
        try:
            q, _ = extract_question_and_id(obj)
            if isinstance(q, str) and q.strip():
                has_q += 1
        except Exception:
            pass
        graw = extract_gold_doc_ids(obj)
        if graw:
            gold_raw += 1
            if filter_gold_in_graph:
                gfil = [g for g in graw if _norm_id(g) in graph_base_ids]
                if gfil:
                    gold_after += 1
            else:
                gold_after += 1

    print("\n=== Preflight ===")
    print(f"items_total              : {total}")
    print(f"with_question_text       : {has_q}")
    print(f"with_gold_raw            : {gold_raw}")
    print(f"with_gold_after_filter   : {gold_after}  (filter_gold_in_graph={filter_gold_in_graph})")
    print(f"graph_base_ids_count     : {len(graph_base_ids)}")
    for i, ks in enumerate(first_keys):
        print(f"sample#{i} keys          : {ks}")
    print("===================\n")

    return {
        "total": total,
        "has_q": has_q,
        "gold_raw": gold_raw,
        "gold_after": gold_after,
        "graph_base_ids": len(graph_base_ids),
    }

# -------------------------------
# Main evaluation
# -------------------------------
def evaluate_dataset_multi(
    dataset_path: Path,
    out_dir: Path,
    graph_path: Path,
    ks: List[int],
    seed_top_k: int = 7,
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
    filter_gold_in_graph: bool = False,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(int(x) for x in ks if int(x) > 0))
    if not ks:
        raise ValueError("No valid Ks provided.")
    maxK = max(ks)
    rerank_top_n = max(rerank_top_n, maxK)

    items = list(_read_jsonl(dataset_path))
    if max_samples is not None:
        items = items[:int(max_samples)]
    n_total = len(items)
    if n_total == 0:
        raise RuntimeError(f"No items found in {dataset_path}")

    chunk2base = build_chunk2base_from_graph(graph_path)
    graph_base_ids = set(_norm_id(b) for b in chunk2base.values())

    # 预检
    stats = _preflight_scan(items, chunk2base, filter_gold_in_graph)

    # 如果过滤后 gold 全没了，自动关闭过滤避免“秒过”
    if filter_gold_in_graph and stats["gold_after"] == 0:
        print("\x1b[31m[warn]\x1b[0m all gold removed by --filter-gold-in-graph; auto-disable filter to avoid skipping all samples.")
        filter_gold_in_graph = False

    results_by_k: Dict[int, List[Dict]] = {k: [] for k in ks}
    effective = 0  # 有 gold（且未被过滤掉）的样本数
    pipelines_run = 0

    pbar = tqdm(range(n_total), desc=f"Evaluating @ks={ks}", ncols=100)
    for idx in pbar:
        obj = items[idx]
        try:
            # 问题文本
            qtext, qid = extract_question_and_id(obj)
            if not qid:
                qid = str(idx)

            # gold
            gold_docs_raw = extract_gold_doc_ids(obj)
            gold_docs = gold_docs_raw
            if filter_gold_in_graph:
                gold_docs = [g for g in gold_docs if _norm_id(g) in graph_base_ids]

            # 强制探针：若无 gold，仅对前 5 条样本做一次 pipeline 以验证是否“真的一跳而过”
            force_probe = False
            if not gold_docs and idx < 5:
                force_probe = True

            if not gold_docs and not force_probe:
                if gold_docs_raw:
                    print(f"[skip Q {idx}] all gold ({len(gold_docs_raw)}) filtered by graph; 检查 graph_base_ids/映射一致性", flush=True)
                else:
                    print(f"[skip Q {idx}] no gold fields matched; 检查 extract_gold_doc_ids() 支持的键", flush=True)
                continue  # 没有 gold，跳过评分

            # 跑 pipeline（评分或探针二选一都需要）


            res = run_pipeline(
                question=qtext,
                seed_top_k=seed_top_k,
                appnp_k_hop=appnp_k_hop,
                appnp_top_nodes=appnp_top_nodes,
                bfs_depth=bfs_depth,
                bfs_max_chunks=bfs_max_chunks,
                rerank_top_n=rerank_top_n,
                device=device,
                embed_model=embed_model,
                reranker_model=reranker_model,
                edge_policy="either",
            )
            pipelines_run += 1
            preds = [x["chunk_id"] for x in (res.get("reranked") or []) if "chunk_id" in x]

            # === Stage diagnostics: pre-rerank & post-rerank(grouped) ===
            try:
                cand_cids = list(res.get("candidate_chunks") or [])
                reranked_items = list(res.get("reranked") or [])

                # A) 候选阶段的 doc-Recall@K（不经重排，直接 candidate→doc 去重）
                pre_curve = doc_recall_curve_from_chunks(gold_docs, cand_cids, chunk2base, ks)
                msg_pre = " ".join([f"@{int(k)}={pre_curve[int(k)]:.3f}" for k in ks])

                # B) 重排后的两条对比：
                #    B1. 不聚合（即你原本的 preds 列表）：chunk→doc 去重
                post_curve_nogroup = doc_recall_curve_from_chunks(gold_docs, preds, chunk2base, ks)
                msg_post_ng = " ".join([f"@{int(k)}={post_curve_nogroup[int(k)]:.3f}" for k in ks])

                #    B2. 按文档分组取最大分，然后以“文档”为单位截断再算 Recall@K
                grouped_doc_order = doc_order_from_grouped_maxscore(reranked_items, chunk2base)
                Gset = set(_norm_id(x) for x in gold_docs if x)
                post_curve_group = {}
                for k in ks:
                    k = int(k)
                    top_docs = grouped_doc_order[:k]
                    tp = sum(1 for b in top_docs if b in Gset)
                    post_curve_group[k] = tp / float(len(Gset)) if Gset else 0.0
                msg_post_g = " ".join([f"@{int(k)}={post_curve_group[int(k)]:.3f}" for k in ks])

                # 打印（按你的风格带 flush）
                print(f"[stage] pre-rerank doc-Recall | {msg_pre}", flush=True)
                print(f"[stage] post-rerank doc-Recall | no-group {msg_post_ng} | grouped {msg_post_g}", flush=True)

            except Exception as _e:
                # 不要让诊断影响主流程
                print(f"[stage] diagnostics failed: {_e}", flush=True)

            seed_nodes = res.get("num_seed_nodes") or 0
            seed_edges = res.get("num_seed_edges") or 0
            exp_nodes  = res.get("num_expanded_nodes") or 0
            exp_edges  = res.get("num_expanded_edges") or 0
            candidates = len(res.get("candidate_chunks") or [])
            reranked   = len(res.get("reranked") or [])

            if force_probe and not gold_docs_raw:
                print(f"[probe Q {idx}] no-gold sample | seeds({seed_nodes},{seed_edges}) expanded({exp_nodes},{exp_edges}) "
                      f"candidates={candidates} reranked={reranked}", flush=True)
                # 探针不计入评分
                continue

            # 有 gold 的样本才计入评分
            effective += 1

            # Warmup for first effective
            if effective == 1:
                G = set(_norm_id(x) for x in gold_docs if x)
                P6 = pred_doc_ids_from_chunks(preds, chunk2base, 6)
                tp6 = sum(1 for b in P6 if b in G)
                print(
                    "\n[Warmup] "
                    f"gold_docs={len(G)} | "
                    f"seed_nodes={seed_nodes}, seed_edges={seed_edges} | "
                    f"expanded_nodes={exp_nodes}, expanded_edges={exp_edges} | "
                    f"candidates={candidates}, reranked={reranked} | "
                    f"hit@6={tp6}/{len(P6)} | "
                    f"graph_base_ids={len(graph_base_ids)}\n",
                    flush=True
                )

            # 每题命中打印
            Gset = set(_norm_id(x) for x in gold_docs if x)
            per_k_msgs = []
            for k in ks:
                Pk = pred_doc_ids_from_chunks(preds, chunk2base, k)
                tp = sum(1 for b in Pk if b in Gset)
                per_k_msgs.append(f"@{k} {tp}/{len(Pk)}")
            print(f"[Q {idx} | id={qid}] hits: " + "  ".join(per_k_msgs), flush=True)

            for k in ks:
                prec, rec, f1, tp = precision_recall_f1_at_k(gold_docs, preds, k, chunk2base)
                results_by_k[k].append({
                    "idx": idx,
                    "id": qid,
                    "k": k,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "tp": tp,
                    "gold_count": len(gold_docs),
                    "pred_count": len(preds),
                })

        except Exception as e:
            if not skip_errors:
                raise
            # 失败也尽可能记录一条 0 分样本（若该样本确有 gold）
            try:
                gtmp = extract_gold_doc_ids(obj)
                if filter_gold_in_graph:
                    gtmp = [g for g in gtmp if _norm_id(g) in graph_base_ids]
            except Exception:
                gtmp = []
            if gtmp:
                for k in ks:
                    results_by_k[k].append({
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
                    })

    # -------------------------------
    # 输出各 k 的文件 + 汇总
    # -------------------------------
    outputs = {}
    for k, rows in results_by_k.items():
        top_n = max(1, math.ceil(0.3 * len(rows))) if rows else 1
        all_sorted = sorted(rows, key=lambda r: r["f1"], reverse=True) if rows else []
        top_sorted = all_sorted[:top_n] if all_sorted else []

        agg_all = _agg_rows(rows)
        agg_top = _agg_rows(top_sorted)

        out_dir.mkdir(parents=True, exist_ok=True)
        all_csv = out_dir / f"eval_k{k}_all.csv"
        with all_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "idx", "id", f"precision@{k}", f"recall@{k}", f"f1@{k}", "tp", "gold_count", "pred_count"])
            for rank, r in enumerate(all_sorted, start=1):
                w.writerow([rank, r["idx"], r["id"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", r["tp"], r["gold_count"], r["pred_count"]])

        top_csv = out_dir / f"eval_k{k}_top30.csv"
        with top_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "idx", "id", f"precision@{k}", f"recall@{k}", f"f1@{k}"])
            for rank, r in enumerate(top_sorted, start=1):
                w.writerow([rank, r["idx"], r["id"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}"])

        (out_dir / f"eval_k{k}_all.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"eval_k{k}_top30.json").write_text(json.dumps(top_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            "k": k,
            "num_questions_scored": len(rows),
            "num_questions_effective": effective,  # 全局有效样本数
            "pipelines_run": pipelines_run,
            "aggregate_all": agg_all,
            "aggregate_top30": agg_top,
            "outputs": {
                "all_csv": str(all_csv),
                "top30_csv": str(top_csv),
                "all_json": str(out_dir / f"eval_k{k}_all.json"),
                "top30_json": str(out_dir / f"eval_k{k}_top30.json"),
            }
        }
        (out_dir / f"eval_k{k}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[k] = summary

    print(f"[Done] 有效样本(有gold) = {effective} / 总样本 = {n_total} | pipelines_run={pipelines_run}", flush=True)
    return {"ks": ks, "effective": effective, "total": n_total, "pipelines_run": pipelines_run, "summaries": outputs}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="One-pass multi-K (doc-level), graph-mapped; per-question hit printing + preflight.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=False)  # 会被固定到 diagnostics
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--ks", type=str, nargs="+", default=[
        "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2"
    ])

    ap.add_argument("--seed-top-k", type=int, default=32)
    ap.add_argument("--appnp-k-hop", type=int, default=1)
    ap.add_argument("--appnp-top-nodes", type=int, default=1500)
    ap.add_argument("--bfs-depth", type=int, default=1)
    ap.add_argument("--bfs-max-chunks", type=int, default=260)
    ap.add_argument("--rerank-top-n", type=int, default=256)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embed-model", type=Path, default=None)
    ap.add_argument("--reranker-model", type=str, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--skip-errors", action="store_true", default=True)
    ap.add_argument("--filter-gold-in-graph", action="store_true", default=False)
    args = ap.parse_args()

    # 解析 ks
    ks: List[int] = []
    for token in args.ks:
        if "," in token:
            ks.extend(int(t) for t in token.split(",") if t.strip())
        else:
            ks.append(int(token))
    ks = sorted(set(ks))

    # 固定输出目录：与 eval/ 平级的 diagnostics/
    project_root = Path(__file__).resolve().parent.parent
    if not args.out_dir:
        args.out_dir = project_root / "data1" / "eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_dataset_multi(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        graph_path=args.graph,
        ks=ks,
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
        filter_gold_in_graph=args.filter_gold_in_graph,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))