from __future__ import annotations

import json


from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Iterable, Set

import numpy as np
import networkx as nx

from .io_paths import (
    ensure_dirs,
    GLOBAL_GRAPH_JSON, CHUNKS_INDEX_JSON,
    EMBED_MODEL_DIR, RERANKER_DIR,
    SEED_SUBGRAPH_JSON, APPNP_SUBGRAPH_JSON,
    ALL_CHUNK_SIMS_NPY, ALL_CHUNK_SIMS_META,
    SEED_CHUNK_SCORES_JSON, APPNP_NODE_SCORES_JSON,
    RESULT_JSON
)
from .seed_retrieval import build_seed_subgraph
from .appnp_local import run_appnp_local
from .bfs_utils import bfs_collect_chunks
from .graph_io import load_graph, save_graph
from .text_index import TextIndex
from .reranker import create_reranker


# ============================================================
# Seed schema hardening & diversification helpers
# ============================================================

def _is_chunk_id_like(x: Any) -> bool:
    """粗判一个对象是否像 chunk_id：包含 #p?、长数字、或常见关键词。"""
    if isinstance(x, str):
        if "#" in x:  # e.g. '24067267#p1'
            return True
        if x.isdigit() and len(x) >= 6:
            return True
        lx = x.lower()
        for kw in ("chunk", "doc", "pmid", "cid", "pid", "base", "passage", "page"):
            if kw in lx:
                return True
    return False


def _coerce_score(x: Any) -> Optional[float]:
    """尽量把各种输入（float/str/dict/tuple/np.float）转成 float 分数；失败返回 None。"""
    # 数字/np 数字
    if isinstance(x, (int, float, np.floating)):
        try:
            return float(x)
        except Exception:
            return None
    # 字符串
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None
    # dict 常见字段
    if isinstance(x, dict):
        for k in ("score", "s", "logit", "prob", "p", "value"):
            if k in x:
                v = _coerce_score(x[k])
                if v is not None:
                    return v
        return None
    # list/tuple：找第一个能转成分数的
    if isinstance(x, (list, tuple)) and len(x) > 0:
        for e in x:
            v = _coerce_score(e)
            if v is not None:
                return v
    return None


def normalize_seed_lists(
    seed_chunks: List[Any],
    seed_scores: List[Any],
    strict: bool = False,
    max_error_preview: int = 5,
) -> Tuple[List[str], List[float]]:
    """
    把任意形态的 (seed_chunks, seed_scores) 规范化为两列：
      - [chunk_id:str], [score:float]，等长且一一对应，按分数降序。
    - 自动识别“列互换 / 混入 dict/tuple/str / 字段名不同”等情况；
    - strict=True 时若异常比例 >30% 则直接抛错（fail-fast）。
    - 会打印异常样例，便于追查上游问题。
    """
    chunks = list(seed_chunks or [])
    scores = list(seed_scores or [])
    assert len(chunks) == len(scores), f"seed_chunks/seed_scores 长度不一致: {len(chunks)} vs {len(scores)}"

    pairs: List[Tuple[str, float]] = []
    bad: List[Tuple[Any, Any]] = []

    for a, b in zip(chunks, scores):
        cid: Optional[str] = None
        sc: Optional[float] = None

        a_chunk, b_chunk = _is_chunk_id_like(a), _is_chunk_id_like(b)
        if a_chunk and not b_chunk:
            cid, sc = str(a), _coerce_score(b)
        elif b_chunk and not a_chunk:
            cid, sc = str(b), _coerce_score(a)
        else:
            # 两边都像/都不像：尝试把 a 当 chunk，再不行试 b，再不行就用“能转分数的那个”。
            if a_chunk:
                cid, sc = str(a), _coerce_score(b)
            elif b_chunk:
                cid, sc = str(b), _coerce_score(a)
            else:
                sa, sb = _coerce_score(a), _coerce_score(b)
                if sa is not None and not _is_chunk_id_like(b):
                    cid, sc = str(b), sa
                elif sb is not None and not _is_chunk_id_like(a):
                    cid, sc = str(a), sb

        if cid is None or sc is None:
            bad.append((a, b))
        else:
            pairs.append((cid, float(sc)))

    # 若全坏，兜底：按原顺序赋一个降序伪分，保证流程不被打断
    if not pairs:
        for rank, raw in enumerate(chunks):
            if raw is None:
                continue
            pairs.append((str(raw), float(len(chunks) - rank)))

    # 统计 & 严格策略
    bad_ratio = (len(bad) / max(1, len(chunks)))
    if strict and bad_ratio > 0.30:
        preview = "\n".join([f"  bad[{i}]: {x!r} | {y!r}" for i, (x, y) in enumerate(bad[:max_error_preview])])
        raise ValueError(f"[seed] 严格校验失败：异常比例过高({bad_ratio:.1%})，样例：\n{preview}")

    # 打印预览，便于你定位上游问题
    if bad:
        preview = "\n".join([f"  bad[{i}]: {x!r} | {y!r}" for i, (x, y) in enumerate(bad[:max_error_preview])])
        print(f"[seed] 发现 {len(bad)}/{len(chunks)} 条异常对，已自动纠正/跳过，示例：\n{preview}", flush=True)

    # 全局按分数降序
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    norm_chunks = [c for c, _ in pairs]
    norm_scores = [s for _, s in pairs]
    return norm_chunks, norm_scores


def _build_chunk2base_from_sources(
    graph_json_path: Path,
    chunks_index_json_path: Path,
) -> Dict[str, str]:
    """
    尽量从 index.json 或 graph.json 里构建 chunk->base 映射；都没有就回退 '#'
    兼容多种字段名（chunk_id/id/chunkId; base_id/baseId/doc_id/pmid)
    """
    mp: Dict[str, str] = {}

    # 1) 优先尝试 chunks_index（如果存在）
    try:
        if chunks_index_json_path and chunks_index_json_path.exists():
            data = json.loads(chunks_index_json_path.read_text(encoding="utf-8"))
            arr = data if isinstance(data, list) else (data.get("chunks") or [])
            for it in arr:
                cid = (it.get("chunk_id") or it.get("id") or it.get("chunkId"))
                bid = (it.get("base_id")  or it.get("baseId") or it.get("doc_id") or it.get("pmid"))
                if cid and bid and (cid not in mp):
                    mp[str(cid)] = str(bid)
    except Exception:
        pass

    # 2) 其次尝试 graph.json
    try:
        if graph_json_path and graph_json_path.exists():
            data = json.loads(graph_json_path.read_text(encoding="utf-8"))
            for n in (data.get("nodes") or []):
                cid = n.get("chunk_id") or n.get("id") or n.get("chunkId")
                bid = n.get("base_id")  or n.get("baseId") or n.get("doc_id") or n.get("pmid")
                if cid and bid and (cid not in mp):
                    mp[str(cid)] = str(bid)
            for e in (data.get("links") or data.get("edges") or []):
                cid = e.get("chunk_id") or e.get("source_chunk") or e.get("cid")
                bid = e.get("base_id")  or e.get("doc_id")       or e.get("pmid")
                if cid and bid and (cid not in mp):
                    mp[str(cid)] = str(bid)
    except Exception:
        pass

    return mp


def diversify_seeds_per_base(
    seed_chunks: List[str],
    seed_scores: List[float],
    chunk2base: Dict[str, str],
    per_base_cap: int = 2,
    final_top_k: Optional[int] = None,
) -> Tuple[List[str], List[float]]:
    """
    对种子做“每个 base_id 只保 N 个最高分”的二次筛选。
    输入的 seed_chunks/seed_scores 需同长度；内部会按分数降序处理。
    """
    assert len(seed_chunks) == len(seed_scores), "seed_chunks/seed_scores 长度不一致"

    # 先按 (cid, score) 配对，并保证整体降序
    pairs = list(zip(seed_chunks, seed_scores))
    pairs.sort(key=lambda x: float(x[1]), reverse=True)

    kept_by_base: Dict[str, int] = {}
    diversified: List[Tuple[str, float]] = []

    for cid, sc in pairs:
        bid = chunk2base.get(cid)
        if not bid:
            # 回退：形如 base#p1 → base
            bid = cid.split("#", 1)[0]
        if not bid:
            continue
        cnt = kept_by_base.get(bid, 0)
        if cnt >= per_base_cap:
            continue
        kept_by_base[bid] = cnt + 1
        diversified.append((cid, sc))
        if final_top_k is not None and len(diversified) >= int(final_top_k):
            break

    if final_top_k is not None and len(diversified) < int(final_top_k):
        # 若限流后不足 final_top_k，补齐：继续从原 pairs 里按分数找未入选的（放宽 base 限制）
        chosen = {c for c, _ in diversified}
        for cid, sc in pairs:
            if cid in chosen:
                continue
            diversified.append((cid, sc))
            if len(diversified) >= int(final_top_k):
                break

    new_chunks = [c for c, _ in diversified]
    new_scores = [s for _, s in diversified]
    return new_chunks, new_scores


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(
    question: str,
    seed_top_k: int = 5,
    appnp_k_hop: int = 1,
    appnp_top_nodes: int = 300,
    appnp_alpha: float = 0.1,
    appnp_iterations: int = 10,
    appnp_min_score: float = 0.0,
    bfs_depth: int = 2,
    bfs_max_chunks: int = 30,
    rerank_top_n: int = 10,
    embed_model: Optional[Path] = None,
    reranker_model: Optional[str] = None,
    device: Optional[str] = None,
    edge_policy: str = "either",
    # === 新增，可选：gold 的 base_id 集合（用于打印命中统计） ===
    gold_base_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """End-to-end: query-only encoding → one-shot similarities → G0 → local APPNP → BFS → rerank.

    - embed_model: local path to your retrieval encoder
    - reranker_model: local dir path or HF repo id (e.g., 'BAAI/bge-reranker-base')
    - gold_base_ids: 若提供，则会在三个阶段打印命中统计与 hit@k
    """

    ensure_dirs()
    embed_model = embed_model or EMBED_MODEL_DIR
    # Allow RERANKER_DIR to be Path OR plain string repo id
    if reranker_model is None:
        rr_src = RERANKER_DIR
        reranker_model = str(rr_src) if isinstance(rr_src, (str, Path)) else str(rr_src)
    else:
        reranker_model = str(reranker_model)

    # 1) Seeds (raw)
    G0, seed_chunks, seed_scores = build_seed_subgraph(
        question=question,
        model_path=embed_model,
        top_k=seed_top_k,
        device=device,
        save_all_scores=True
    )

    # --- FIX A: 若 seed_scores 是 dict，则按 seed_chunks 对齐成分数列表 ---
    if isinstance(seed_scores, dict):
        seed_scores = [float(seed_scores.get(c, 0.0)) for c in seed_chunks]
    # --------------------------------------------------------------------

    # 1.0) Seed schema hardening（强校验/规范化，避免把 chunk 当成分数）
    try:
        orig_pair_cnt = len(seed_chunks)
        seed_chunks, seed_scores = normalize_seed_lists(seed_chunks, seed_scores, strict=False)
        print(f"[seed] normalized: {len(seed_chunks)} pairs (from={orig_pair_cnt})", flush=True)
    except Exception as e:
        print(f"[seed] normalization failed: {e}", flush=True)
        # 兜底：不给后续抛异常
        seed_chunks = [str(x) for x in (seed_chunks or [])]
        seed_scores = [float(i) for i in range(len(seed_chunks), 0, -1)]

    # 1.1) Seed diversification：每个 base 仅保 1~2 个最高分种子
    try:
        c2b = _build_chunk2base_from_sources(
            graph_json_path=GLOBAL_GRAPH_JSON,
            chunks_index_json_path=CHUNKS_INDEX_JSON,
        )
        before = len(seed_chunks)
        seed_per_base_cap = 2  # 可调：1 更激进（更分散），2 稳健
        seed_chunks2, seed_scores2 = diversify_seeds_per_base(
            seed_chunks=seed_chunks,
            seed_scores=seed_scores,
            chunk2base=c2b,
            per_base_cap=seed_per_base_cap,
            final_top_k=seed_top_k,  # 保持和用户请求的 top_k 总量一致
        )

        # --- FIX B: 用多样化后的 chunk_id 按“边的 chunk_id”重建 G0（避免把 chunk_id 当节点裁剪） ---
        if set(seed_chunks2) != set(seed_chunks):
            keep = set(seed_chunks2)
            try:
                GG = load_graph(GLOBAL_GRAPH_JSON)
                seed_edges = [
                    (u, v, k)
                    for u, v, k, d in GG.edges(keys=True, data=True)
                    if d.get("chunk_id") in keep
                ]
                G0 = GG.edge_subgraph(seed_edges).copy() if seed_edges else nx.MultiGraph()
            except Exception:
                pass
            seed_chunks, seed_scores = seed_chunks2, seed_scores2
        # --------------------------------------------------------------------------------------

        # 把多样化后的种子分数写回（便于核对）
        try:
            payload = [{"chunk_id": c, "score": float(s)} for c, s in zip(seed_chunks, seed_scores)]
            SEED_CHUNK_SCORES_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        print(f"[seed] diversified: kept={len(seed_chunks)}/{before} (cap_per_base={seed_per_base_cap})", flush=True)
    except Exception as e:
        print(f"[seed] diversification skipped: {e}", flush=True)

    # ========== 打印第一次召回（种子阶段）基础信息 ==========
    print(f"[stage] first-recall seeds | num_seeds={len(seed_chunks)} | seed_nodes={G0.number_of_nodes()}, seed_edges={G0.number_of_edges()}", flush=True)

    # 2) APPNP on localized region
    G_full = load_graph(GLOBAL_GRAPH_JSON)
    seed_score_map: Dict[str, float] = {
        str(cid): float(score)
        for cid, score in zip(seed_chunks, seed_scores)
    }
    G_exp, appnp_node_scores, selected_nodes = run_appnp_local(
        G_full, G0, seed_score_map,
        k_hop_region=appnp_k_hop,
        top_nodes=appnp_top_nodes,
        alpha=appnp_alpha,
        iterations=appnp_iterations,
        min_score=appnp_min_score,
        edge_policy=edge_policy
    )

    print(f"[appnp] expanded_nodes={G_exp.number_of_nodes()}, expanded_edges={G_exp.number_of_edges()}", flush=True)

    # 3) BFS collect candidate chunks from expanded subgraph
    start_nodes = list(G0.nodes())
    candidate_chunks = bfs_collect_chunks(
        G_exp, start_nodes=start_nodes, depth=bfs_depth, max_chunks=bfs_max_chunks
    )
    print(f"[stage] second-recall bfs | candidates={len(candidate_chunks)}", flush=True)

    # 4) Rerank by cross-encoder (or fallback)
    # Prepare pairs
    idx = TextIndex(CHUNKS_INDEX_JSON)
    pairs: List[tuple[str, str]] = []
    mapping: List[str] = []
    q = str(question).strip()
    for cid in candidate_chunks:
        t = idx.get(cid).strip()
        if not t:
            continue
        pairs.append((q, t))
        mapping.append(cid)

    reranked: List[Dict[str, Any]] = []
    if pairs:
        rr = create_reranker(reranker_model, prefer=None, device=device)
        scores = rr.compute_score(pairs)
        # order by score desc, keep top N
        order = np.argsort(-np.asarray(scores))[:rerank_top_n]
        for i in order:
            i = int(i)
            if i < 0 or i >= len(mapping):
                continue
            reranked.append({"chunk_id": mapping[i], "score": float(scores[i]), "text": pairs[i][1]})

    print(f"[stage] post-rerank | reranked={len(reranked)} (top_n={rerank_top_n})", flush=True)

    # 5) 若给了 gold_base_ids，则打印三阶段的 gold 命中统计 + hit@k
    if gold_base_ids is not None:
        gold_set: Set[str] = {str(x) for x in gold_base_ids}

        # 准备映射函数：chunk → base
        c2b_all = _build_chunk2base_from_sources(GLOBAL_GRAPH_JSON, CHUNKS_INDEX_JSON)
        def _to_base(cid: str) -> str:
            b = c2b_all.get(cid)
            return b if b else cid.split("#", 1)[0]

        seed_bases = { _to_base(c) for c in seed_chunks }
        cand_bases = { _to_base(c) for c in candidate_chunks }
        rerank_bases = [ _to_base(d["chunk_id"]) for d in reranked ]

        # 阶段命中计数
        seed_hits = len(seed_bases & gold_set)
        cand_hits = len(cand_bases & gold_set)
        # rerank：命中去重后统计
        rerank_hits_total = len(set(rerank_bases) & gold_set)

        print(f"[gold] seeds: {seed_hits}/{len(gold_set)} | bfs: {cand_hits}/{len(gold_set)} | rerank(total): {rerank_hits_total}/{len(gold_set)}", flush=True)

        # 简单 hit@k（按 base 去重后计算 recall）
        ks = [k for k in (2, 3, 4, 5, 6) if k <= rerank_top_n]
        if ks and reranked:
            seen: List[str] = []
            recalls = []
            for k in ks:
                topk_bases = []
                seen.clear()
                # 保证按 base 去重后的前 k
                for b in rerank_bases:
                    if b in seen:
                        continue
                    seen.append(b)
                    topk_bases.append(b)
                    if len(topk_bases) >= k:
                        break
                num_hit = len(set(topk_bases) & gold_set)
                recall_k = (num_hit / max(1, len(gold_set)))
                recalls.append(f"@{k}={recall_k:.3f}")
            print("[stage] post-rerank doc-Recall | " + " ".join(recalls), flush=True)

    # 6) Dump final compact result
    result = {
        "question": question,
        "seed_chunk_ids": seed_chunks,
        "num_seed_edges": int(G0.number_of_edges()),
        "num_seed_nodes": int(G0.number_of_nodes()),
        "appnp_selected_nodes": selected_nodes,
        "num_expanded_edges": int(G_exp.number_of_edges()),
        "num_expanded_nodes": int(G_exp.number_of_nodes()),
        "candidate_chunks": candidate_chunks,
        "reranked": reranked,
        # references to split files:
        "files": {
            "seed_subgraph": str(SEED_SUBGRAPH_JSON),
            "expanded_subgraph": str(APPNP_SUBGRAPH_JSON),
            "all_chunk_similarities_npy": str(ALL_CHUNK_SIMS_NPY),
            "all_chunk_similarities_meta": str(ALL_CHUNK_SIMS_META),
            "seed_chunk_scores_json": str(SEED_CHUNK_SCORES_JSON),
            "appnp_node_scores_json": str(APPNP_NODE_SCORES_JSON),
        }
    }
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Vector graph pipeline: query-only encode → one-shot similarity → local APPNP → BFS → rerank.")
    ap.add_argument("question", type=str, help="Input question text.")
    ap.add_argument("--seed-top-k", type=int, default=5)
    ap.add_argument("--appnp-k-hop", type=int, default=1)
    ap.add_argument("--appnp-top-nodes", type=int, default=300)
    ap.add_argument("--appnp-alpha", type=float, default=0.1)
    ap.add_argument("--appnp-iterations", type=int, default=10)
    ap.add_argument("--appnp-min-score", type=float, default=0.0)
    ap.add_argument("--bfs-depth", type=int, default=2)
    ap.add_argument("--bfs-max-chunks", type=int, default=30)
    ap.add_argument("--rerank-top-n", type=int, default=10)
    ap.add_argument("--embed-model", type=Path, default=None)
    ap.add_argument("--reranker-model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--edge-policy", type=str, default="either", choices=["either", "both"])
    # CLI 下不要求 gold；如需本地测试，可额外加一个 --gold-list JSON 或文件路径的解析
    args = ap.parse_args()

    out = run_pipeline(
        question=args.question,
        seed_top_k=args.seed_top_k,
        appnp_k_hop=args.appnp_k_hop,
        appnp_top_nodes=args.appnp_top_nodes,
        appnp_alpha=args.appnp_alpha,
        appnp_iterations=args.appnp_iterations,
        appnp_min_score=args.appnp_min_score,
        bfs_depth=args.bfs_depth,
        bfs_max_chunks=args.bfs_max_chunks,
        rerank_top_n=args.rerank_top_n,
        embed_model=args.embed_model,
        reranker_model=args.reranker_model,
        device=args.device,
        edge_policy=args.edge_policy,
        gold_base_ids=None,  # 脚本模式默认不打印 gold；评测器可传入
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
