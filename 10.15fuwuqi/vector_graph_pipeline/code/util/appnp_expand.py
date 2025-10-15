
from __future__ import annotations
"""
APPNP-based expansion limited to a small k-hop neighborhood around the seed subgraph.
This matches the requirement "只考虑候选子图和比那个子图大一些范围的子图".
"""
from typing import Dict, Tuple, List, Set
import numpy as np
import networkx as nx

def _aggregate_seed_node_scores(seed_subgraph: nx.MultiGraph, seed_chunk_scores: Dict[str, float]) -> Dict[str, float]:
    node_scores: Dict[str, float] = {}
    for u, v, data in seed_subgraph.edges(data=True):
        cid = data.get("chunk_id")
        if not cid:
            continue
        s = float(seed_chunk_scores.get(cid, 0.0))
        node_scores[u] = node_scores.get(u, 0.0) + s
        node_scores[v] = node_scores.get(v, 0.0) + s
    return node_scores

def _khop_nodes(graph: nx.MultiGraph, start_nodes: List[str], k: int) -> Set[str]:
    if k <= 0:
        return set(start_nodes)
    visited = set(start_nodes)
    frontier = set(start_nodes)
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt.update(graph.neighbors(u))
        nxt.difference_update(visited)
        if not nxt:
            break
        visited.update(nxt)
        frontier = nxt
    return visited

def appnp_expand(
    global_graph: nx.MultiGraph,
    seed_subgraph: nx.MultiGraph,
    seed_chunk_scores: Dict[str, float],
    k_hop: int = 1,
    top_nodes: int = 300,
    alpha: float = 0.1,
    num_iterations: int = 10,
    min_score: float = 0.0,
) -> Tuple[nx.MultiGraph, Dict[str, float]]:
    """
    Run APPNP on an induced subgraph H that contains all nodes within k_hop of seed_subgraph nodes.
    Return (expanded_subgraph, node_scores_dict).
    """
    if top_nodes <= 0:
        raise ValueError("top_nodes must be positive")

    seed_nodes = list(seed_subgraph.nodes())
    if not seed_nodes:
        return nx.MultiGraph(), {}

    # 1) restrict to k-hop neighborhood
    region = _khop_nodes(global_graph, seed_nodes, k=k_hop)
    H = global_graph.subgraph(region).copy()
    if H.number_of_nodes() == 0:
        return nx.MultiGraph(), {}

    # 2) prepare APPNP matrices
    nodes = list(H.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for u, v, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        i, j = idx[u], idx[v]
        if i == j:
            A[i, j] += w
        else:
            A[i, j] += w
            A[j, i] += w
    A += np.eye(len(nodes), dtype=np.float32)  # self-loop to avoid zero-degree
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0.0] = 1.0
    P = A / deg  # row-normalized

    # 3) initial scores from seed edges aggregated to nodes
    seed_node_scores = _aggregate_seed_node_scores(seed_subgraph, seed_chunk_scores)
    h0 = np.array([float(seed_node_scores.get(n, 0.0)) for n in nodes], dtype=np.float32)
    h = h0.copy()
    for _ in range(int(num_iterations)):
        h = (1.0 - float(alpha)) * (P @ h) + float(alpha) * h0

    node_scores = {n: float(s) for n, s in zip(nodes, h)}

    # 4) select top_nodes by score (respect min_score if set)
    sorted_nodes = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[str] = []
    for n, s in sorted_nodes:
        if s < float(min_score) and selected:
            break
        selected.append(n)
        if len(selected) >= int(top_nodes):
            break
    if not selected and sorted_nodes:
        selected = [sorted_nodes[0][0]]

    # 5) build expanded subgraph: edges inside selected
    edges = [(u, v, k) for u, v, k in H.edges(keys=True) if (u in selected and v in selected)]
    Gexp = H.edge_subgraph(edges).copy() if edges else nx.MultiGraph()
    return Gexp, node_scores
