
from __future__ import annotations
"""
BFS sampling over the expanded subgraph to collect candidate chunk_ids.
Avoids duplicate chunks and respects max_chunks cap.
"""
from typing import List, Set, Tuple
from collections import deque
import networkx as nx

def bfs_collect(
    graph: nx.MultiGraph,
    start_nodes: List[str],
    depth: int = 2,
    max_chunks: int = 60,
) -> List[str]:
    if not start_nodes or graph.number_of_nodes() == 0:
        return []
    q = deque((s, 0) for s in start_nodes if s in graph)
    visited_nodes: Set[str] = set(start_nodes)
    collected: List[str] = []
    seen_chunks: Set[str] = set()

    while q and len(collected) < max_chunks:
        u, dist = q.popleft()
        # iterate incident edges
        for v, k, d in graph.edges(u, keys=True, data=True):
            cid = d.get("chunk_id")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                collected.append(cid)
                if len(collected) >= max_chunks:
                    break
            if dist < depth and v not in visited_nodes:
                visited_nodes.add(v)
                q.append((v, dist + 1))
        if len(collected) >= max_chunks:
            break
    return collected
