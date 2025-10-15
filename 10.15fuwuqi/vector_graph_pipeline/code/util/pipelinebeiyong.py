from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

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
) -> Dict[str, Any]:
    """End-to-end: query-only encoding → one-shot similarities → G0 → local APPNP → BFS → rerank.

    - embed_model: local path to your retrieval encoder

    - reranker_model: local dir path or HF repo id (e.g., 'BAAI/bge-reranker-base')

    """
    ensure_dirs()
    embed_model = embed_model or EMBED_MODEL_DIR
    # Allow RERANKER_DIR to be Path OR plain string repo id
    if reranker_model is None:
        rr_src = RERANKER_DIR
        reranker_model = str(rr_src) if isinstance(rr_src, (str, Path)) else str(rr_src)
    else:
        reranker_model = str(reranker_model)

    # 1) Seeds
    G0, seed_chunks, seed_scores = build_seed_subgraph(
        question=question,
        model_path=embed_model,
        top_k=seed_top_k,
        device=device,
        save_all_scores=True
    )

    # 2) APPNP on localized region
    G_full = load_graph(GLOBAL_GRAPH_JSON)
    G_exp, appnp_node_scores, selected_nodes = run_appnp_local(
        G_full, G0, seed_scores,
        k_hop_region=appnp_k_hop,
        top_nodes=appnp_top_nodes,
        alpha=appnp_alpha,
        iterations=appnp_iterations,
        min_score=appnp_min_score,
        edge_policy=edge_policy
    )

    # 3) BFS collect candidate chunks from expanded subgraph
    start_nodes = list(G0.nodes())
    candidate_chunks = bfs_collect_chunks(
        G_exp, start_nodes=start_nodes, depth=bfs_depth, max_chunks=bfs_max_chunks
    )

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

    # 5) Dump final compact result
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
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
