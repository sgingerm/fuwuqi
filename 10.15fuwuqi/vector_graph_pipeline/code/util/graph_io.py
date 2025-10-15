from __future__ import annotations
from pathlib import Path
import json
import networkx as nx
from networkx.readwrite import json_graph

def load_graph(path: Path) -> nx.MultiGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # Explicitly set edges="links" to silence NX 3.6 warning and preserve current schema
    return json_graph.node_link_graph(payload, multigraph=True, edges="links")

def save_graph(graph: nx.MultiGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_graph.node_link_data(graph, edges="links")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
