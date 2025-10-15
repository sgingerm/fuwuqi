from __future__ import annotations

from collections import deque
from typing import List, Set

import networkx as nx

def bfs_collect_chunks(
    graph: nx.MultiGraph,
    start_nodes: List[str],
    depth: int,
    max_chunks: int,
) -> List[str]:
    """Robust BFS that works with MultiGraph edges.
    Iterates edges as (u, v, key, data), derives neighbor correctly, and collects unique chunk_ids.
    """
    if depth < 0:
        return []
    queue = deque((n, 0) for n in start_nodes)
    seen_nodes: Set[str] = set(start_nodes)
    seen_chunks: Set[str] = set()
    collected: List[str] = []

    while queue and len(collected) < max_chunks:
        node, dist = queue.popleft()
        # Iterate all incident edges robustly
        for u, v, key, data in graph.edges(node, keys=True, data=True):
            # Determine neighbor robustly
            neighbor = v if u == node else u
            # Collect chunk_id
            cid = data.get("chunk_id")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                collected.append(cid)
                if len(collected) >= max_chunks:
                    break
            # Expand frontier if depth allows
            if dist < depth and neighbor not in seen_nodes:
                seen_nodes.add(neighbor)
                queue.append((neighbor, dist + 1))
        # Early stop if enough
        if len(collected) >= max_chunks:
            break

    return collected
