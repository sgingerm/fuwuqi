from __future__ import annotations
from typing import List, Tuple, Optional, Protocol, Any, Iterable
from pathlib import Path

def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(map(_to_str, x))
    if isinstance(x, dict):
        # common fields first
        for k in ("text", "content", "passage", "body"):
            v = x.get(k)
            if isinstance(v, str):
                return v
        # fallback to JSON-ish repr
        try:
            import json as _json
            return _json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

class Reranker(Protocol):
    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]: ...

class FlagRerankerWrapper:
    """Preferred reranker: uses FlagEmbedding's cross-encoder.

    Accepts local dir path or HF repo id (str).
    """
    def __init__(self, model_name_or_path: str, use_fp16: Optional[bool] = None, device: Optional[str] = None):
        from FlagEmbedding import FlagReranker
        kwargs = {"model_name_or_path": model_name_or_path}
        if use_fp16 is not None:
            kwargs["use_fp16"] = use_fp16
        self._rr = FlagReranker(**kwargs)
        self._device = device  # kept for API parity; FlagReranker manages device internally

    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        # strong cleaning to str
        cleaned = [( _to_str(q).strip(), _to_str(p).strip() ) for q, p in pairs]
        cleaned = [ (q, p) for q, p in cleaned if q and p ]
        if not cleaned:
            return []
        return list(self._rr.compute_score(cleaned))

class STBiEncoderReranker:
    """Fallback reranker: SentenceTransformer bi-encoder (cosine diagonal).

    This is not as strong as cross-encoder, but avoids large downloads.

    """
    def __init__(self, model_path: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        from torch import device as _dev
        self._st = SentenceTransformer(model_path)
        if device:
            try:
                self._st = self._st.to(device)
            except Exception:
                pass

    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        from sentence_transformers import util
        qs = [ _to_str(q).strip() for q, _ in pairs ]
        ps = [ _to_str(p).strip() for _, p in pairs ]
        # filter empties while keeping mapping
        idx = [ i for i,(q,p) in enumerate(zip(qs, ps)) if q and p ]
        if not idx:
            return []
        qs2 = [qs[i] for i in idx]
        ps2 = [ps[i] for i in idx]
        qv = self._st.encode(qs2, convert_to_tensor=True, normalize_embeddings=True)
        pv = self._st.encode(ps2, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(qv, pv).diagonal().cpu().tolist()
        # return scores aligned to original order; missing -> 0.0
        out = [0.0] * len(pairs)
        for k, i in enumerate(idx):
            out[i] = float(scores[k])
        return out

def create_reranker(model_name_or_path: str, prefer: Optional[str] = None, device: Optional[str] = None) -> Reranker:
    """Factory that prefers FlagEmbedding if available; fallbacks to ST bi-encoder."""
    if prefer == "st":
        return STBiEncoderReranker(model_name_or_path, device=device)
    if prefer == "flag":
        return FlagRerankerWrapper(model_name_or_path, device=device)
    # auto
    try:
        return FlagRerankerWrapper(model_name_or_path, device=device)
    except Exception:
        return STBiEncoderReranker(model_name_or_path, device=device)
