from __future__ import annotations
from pathlib import Path

# ===== Default paths =====
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../vector_graph_pipeline/code/util/io_paths.py → 回到仓库根
DATA_DIR  = PROJECT_ROOT / "data"      # 所有“输入”
DATA1_DIR = PROJECT_ROOT / "data1"     # 所有“输出”
(DATA1_DIR).mkdir(parents=True, exist_ok=True)

KG_OUT = DATA_DIR                       # 沿用原变量名，但指向 data/
VECTOR_DIR = DATA_DIR / "vector_graph"

# ---- 核心输入 ----
GLOBAL_GRAPH_JSON = DATA_DIR / "global_graph.json"
INDEX_CSV         = VECTOR_DIR / "index.csv"
EDGES_VEC_NPY     = VECTOR_DIR / "edges.vec.npy"
EDGES_NORM_NPY    = VECTOR_DIR / "edges.norm.npy"
NODES_VEC_NPY     = VECTOR_DIR / "nodes.vec.npy"
META_JSON         = VECTOR_DIR / "meta.json"
CHUNKS_INDEX_JSON = DATA_DIR / "chunks_index.json"

# ---- 缓存与输出（改这里）----
SCORES_DIR              = DATA1_DIR / "scores"          # 改成 data1
OUTPUT_DIR              = DATA1_DIR                     # 改成 data1
SEED_SUBGRAPH_JSON      = OUTPUT_DIR / "seed_subgraph.json"
APPNP_SUBGRAPH_JSON     = OUTPUT_DIR / "appnp_subgraph.json"
ALL_CHUNK_SIMS_NPY      = OUTPUT_DIR / "all_chunk_similarities.npy"
ALL_CHUNK_SIMS_META     = OUTPUT_DIR / "all_chunk_similarities.meta.json"
SEED_CHUNK_SCORES_JSON  = OUTPUT_DIR / "seed_chunk_scores.json"
APPNP_NODE_SCORES_JSON  = OUTPUT_DIR / "appnp_node_scores.json"
RESULT_JSON             = OUTPUT_DIR / "vector_pipeline_result.json"

# ---- 模型 ----
EMBED_MODEL_DIR = PROJECT_ROOT / "data" / "weitiaomoxing2"   # ✅ 这里没问题
RERANKER_DIR    = "BAAI/bge-reranker-large"

def ensure_dirs() -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
