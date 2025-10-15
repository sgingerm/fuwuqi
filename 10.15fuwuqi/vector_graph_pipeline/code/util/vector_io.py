from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .io_paths import (
    META_JSON, EDGES_VEC_NPY, EDGES_NORM_NPY, INDEX_CSV, SCORES_DIR,
    CHUNKS_INDEX_JSON
)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_meta(meta_path: Path = META_JSON) -> Dict:
    """Load meta.json if present; otherwise return conservative defaults."""
    if not meta_path.exists():
        return {"dim": 1024, "normalized": True, "dtype": "float32", "metric": "cosine"}
    return json.loads(meta_path.read_text(encoding="utf-8"))

def load_edges_matrix(mmap: bool = True) -> np.ndarray:
    """Load edges.vec.npy, optionally as a memory-mapped array for large matrices."""
    mode = "r" if mmap else None
    return np.load(EDGES_VEC_NPY, mmap_mode=mode)

def load_edges_norms(optional: bool = True) -> Optional[np.ndarray]:
    """Load optional edges.norm.npy containing L2 norms per edge-row.
    If optional=False and file is missing, raise FileNotFoundError.
    """
    if EDGES_NORM_NPY.exists():
        return np.load(EDGES_NORM_NPY, mmap_mode="r")
    if optional:
        return None
    raise FileNotFoundError(str(EDGES_NORM_NPY))

def load_index_edges() -> Tuple[List[str], List[int]]:
    """Read index.csv and return (chunk_ids, rows) for rows marked kind=edge.
    rows is a list of integer row indices aligned with edges.vec.npy.
    """
    chunk_ids: List[str] = []
    rows: List[int] = []
    with INDEX_CSV.open("r", encoding="utf-8", newline="") as fh:
        import csv
        reader = csv.DictReader(fh)
        for r in reader:
            if (r.get("kind", "").lower() == "edge") and ("row" in r):
                try:
                    rows.append(int(r["row"]))
                    chunk_ids.append(r.get("chunk_id", ""))
                except Exception:
                    continue
    return chunk_ids, rows

def load_chunks_index() -> Dict[str, str]:
    """Load chunks_index.json if exists; otherwise return empty dict."""
    if not CHUNKS_INDEX_JSON.exists():
        return {}
    return json.loads(CHUNKS_INDEX_JSON.read_text(encoding="utf-8"))

def normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x, ord=2))
    if n < eps:
        return x.astype(np.float32, copy=False)
    return (x / n).astype(np.float32, copy=False)

def find_scores_cache_key(question: str, model_tag: str = "default") -> str:
    return f"{_sha1(model_tag + '|' + question)}.npy"

def compute_or_load_all_similarities(
    question: str,
    q_vec: np.ndarray,
    normalized_edges: bool,
    model_tag: str = "default",
    save_meta_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute (or load cached) similarities between ALL edge vectors and a single query vector.
    If edges are already L2-normalized (meta['normalized'] == True), we use scores = E @ q.
    Otherwise we divide by per-row norms from edges.norm.npy to avoid recomputing norms.

    Parameters
    ----------
    question : str
        Raw question string, used only for cache key metadata.
    q_vec : np.ndarray
        L2-normalized query embedding of shape [D].
    normalized_edges : bool
        Whether edges.vec.npy is already row-normalized.
    model_tag : str
        Part of the cache key to differentiate models.
    save_meta_path : Optional[Path]
        If provided, write a small JSON meta describing the cache file.

    Returns
    -------
    np.ndarray
        Scores aligned to the row order of edges.vec.npy (shape [M]).
    """
    cache_name = find_scores_cache_key(question, model_tag)
    cache_path = SCORES_DIR / cache_name
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    E = load_edges_matrix(mmap=True)  # [M, D]
    q_vec = q_vec.astype(E.dtype, copy=False)
    if normalized_edges:
        scores = E @ q_vec
    else:
        norms = load_edges_norms(optional=False).astype(np.float32)
        # q_vec should already be normalized; ensure numerical stability
        scores = (E @ q_vec) / np.clip(norms, 1e-12, None)

    # Persist cache
    np.save(cache_path, scores.astype(np.float32))

    if save_meta_path is not None:
        meta = {
            "question": question,
            "cache_file": str(cache_path),
            "length": int(scores.shape[0]),
            "normalized_edges": bool(normalized_edges),
        }
        save_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return scores
