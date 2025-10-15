from __future__ import annotations
import subprocess, sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]  # jiang_repo_bundle 根目录
    DATA_DIR  = repo_root / "data"
    OUT_DIR   = repo_root / "data1" / "eval"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DATASET = DATA_DIR / "question-answer-passages_test.filtered.strict——yuanshi.jsonl"
    GRAPH   = DATA_DIR / "global_graph.json"

    KS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    RERANK_TOP_N = max(KS)  # 保守：>= max(KS)
    BFS_MAX_CHUNKS = 30
    DEVICE = "cuda"
    RERANKER_MODEL = None
    MAX_SAMPLES = None
    FILTER_GOLD_IN_GRAPH = False

    ks_token = ",".join(str(k) for k in KS)
    cmd = [
        sys.executable, "-m", "eval.evaluate_dataset_multi",
        "--dataset", str(DATASET),
        "--out-dir", str(OUT_DIR),
        "--graph", str(GRAPH),
        "--ks", ks_token,
        "--rerank-top-n", str(RERANK_TOP_N),
        "--bfs-max-chunks", str(BFS_MAX_CHUNKS),
        "--device", DEVICE,
    ]
    if FILTER_GOLD_IN_GRAPH:
        cmd.append("--filter-gold-in-graph")
    if MAX_SAMPLES is not None:
        cmd += ["--max-samples", str(MAX_SAMPLES)]
    if RERANKER_MODEL:
        cmd += ["--reranker-model", str(RERANKER_MODEL)]

    print("\n=== Running evaluation ===")
    print("CWD:", repo_root)
    print("CMD:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True, cwd=str(repo_root))

if __name__ == "__main__":
    main()
