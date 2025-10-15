from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.readwrite import json_graph

from .io_paths import APPNP_SUBGRAPH_JSON, APPNP_NODE_SCORES_JSON
from .seed_retrieval import save_graph

def induce_khop_region(
    G_full: nx.MultiGraph,
    seed_nodes: Iterable[str],
    k_hop: int = 1,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> nx.MultiGraph:
    """Build a localized induced subgraph around seed_nodes within k hops.
    Apply optional caps for nodes/edges for safety.
    """
    # BFS on the full graph to get nodes within k hops
    visited = set(seed_nodes)
    frontier = list(seed_nodes)
    for _ in range(k_hop):
        new_frontier = []
        for u in frontier:
            for _, v, _key in G_full.edges(u, keys=True):
                if v not in visited:
                    visited.add(v)
                    new_frontier.append(v)
        frontier = new_frontier
        if not frontier:
            break

    H = G_full.subgraph(visited).copy()

    # Optional capping
    if max_nodes is not None and H.number_of_nodes() > max_nodes:
        # Downsample nodes by degree (keep high-degree first): simple heuristic
        deg_sorted = sorted(H.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = set(n for n, _ in deg_sorted)
        H = H.subgraph(keep).copy()

    if max_edges is not None and H.number_of_edges() > max_edges:
        # Downsample edges by weight if present, else arbitrary
        edges = list(H.edges(keys=True, data=True))
        edges.sort(key=lambda e: float(e[3].get("weight", 0.0)), reverse=True)
        keep_edges = edges[:max_edges]
        H = H.edge_subgraph([(u, v, k) for (u, v, k, _) in keep_edges]).copy()

    return H

def _build_sparse_row_stochastic_adj(G: nx.MultiGraph) -> Tuple[sp.csr_matrix, List[str]]:
    """Return row-stochastic adjacency with self-loops and node order."""
    nodes = list(G.nodes())
    n = len(nodes)
    index = {u: i for i, u in enumerate(nodes)}
    # accumulate weights (undirected -> symmetric entries)
    rows, cols, data = [], [], []
    for u, v, key, edata in G.edges(keys=True, data=True):
        w = float(edata.get("weight", 1.0))
        i, j = index[u], index[v]
        rows.append(i); cols.append(j); data.append(w)
        rows.append(j); cols.append(i); data.append(w)
    # self-loops
    for i in range(n):
        rows.append(i); cols.append(i); data.append(1.0)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    # row-normalize
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    Dinv = sp.diags(1.0 / deg, dtype=np.float32)
    P = Dinv @ A  # row-stochastic
    return P, nodes

def _aggregate_seed_node_scores(G_seed: nx.MultiGraph, seed_chunk_scores: Dict[str, float]) -> Dict[str, float]:
    """Aggregate chunk scores on seed edges to endpoint node scores."""
    node_scores: Dict[str, float] = {}
    for u, v, key, data in G_seed.edges(keys=True, data=True):
        cid = data.get("chunk_id")
        if cid is None: 
            continue
        s = float(seed_chunk_scores.get(cid, 0.0))
        node_scores[u] = node_scores.get(u, 0.0) + s
        node_scores[v] = node_scores.get(v, 0.0) + s
    return node_scores

def run_appnp_local(
    G_full: nx.MultiGraph,
    G_seed: nx.MultiGraph,
    seed_chunk_scores: Dict[str, float],
    k_hop_region: int = 1,
    top_nodes: int = 300,
    alpha: float = 0.1,
    iterations: int = 10,
    min_score: float = 0.0,
    edge_policy: str = "either"  # "either" or "both"
) -> Tuple[nx.MultiGraph, Dict[str, float], List[str]]:
    """APPNP on a localized k-hop induced subgraph of the full graph."""
    # 1) Region
    R = induce_khop_region(
        G_full, G_seed.nodes(), k_hop=k_hop_region,
        max_nodes=None, max_edges=None
    )

    # 2) Personalized vector from seed edges
    p_scores = _aggregate_seed_node_scores(G_seed, seed_chunk_scores)
    if not p_scores:
        raise ValueError("Empty personalization scores from seeds; check seed_chunk_scores.")

    # 3) Build P and node order
    P, nodes = _build_sparse_row_stochastic_adj(R)
    idx = {u: i for i, u in enumerate(nodes)}

    # 4) h0
    h0 = np.zeros((len(nodes),), dtype=np.float32)
    for u, s in p_scores.items():
        if u in idx:
            h0[idx[u]] = float(s)

    # 5) Iterate APPNP: h = (1-α)·P·h + α·h0
    h = h0.copy()
    for _ in range(int(iterations)):
        h = (1.0 - alpha) * (P @ h) + alpha * h0

    # 6) Select top nodes
    order = np.argsort(-h)
    selected: List[str] = []
    for i in order:
        if h[i] < min_score and selected:
            break
        selected.append(nodes[i])
        if len(selected) >= int(top_nodes):
            break
    if not selected and len(order) > 0:
        selected = [nodes[int(order[0])]]

    # 7) Build expanded subgraph based on edge policy
    if edge_policy == "both":
        edges = [(u, v, k) for u, v, k, d in R.edges(keys=True, data=True) if (u in selected and v in selected)]
    else:  # "either" (default, less brittle)
        sel = set(selected)
        edges = [(u, v, k) for u, v, k, d in R.edges(keys=True, data=True) if (u in sel or v in sel)]
    if not edges:
        raise ValueError("No edges selected after APPNP; try increasing top_nodes or lowering min_score.")

    G_exp = R.edge_subgraph(edges).copy()

    # 8) Save node scores split file
    node_scores = {nodes[i]: float(h[i]) for i in range(len(nodes))}
    with open(APPNP_NODE_SCORES_JSON, "w", encoding="utf-8") as fh:
        json.dump(node_scores, fh, ensure_ascii=False, indent=2)

    # 9) Persist subgraph
    save_graph(G_exp, APPNP_SUBGRAPH_JSON)

    return G_exp, node_scores, selected
