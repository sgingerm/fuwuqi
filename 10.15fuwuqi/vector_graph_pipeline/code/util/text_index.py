from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json

class TextIndex:
    """Unified reader for chunks_index.json supporting two layouts:
    1) flat: { "chunk_id": "text", ... }
    2) bucketed: { "base_id": { "chunk_id": "text", ... }, ... }
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.raw: Dict[str, Any] = {}
        if self.path.exists():
            self.raw = json.loads(self.path.read_text(encoding="utf-8"))
        # detect flat vs bucket
        self.is_flat = False
        if self.raw:
            # if any value is a string -> flat
            self.is_flat = all(isinstance(v, str) for v in self.raw.values())

    def get(self, chunk_id: str) -> str:
        if not self.raw or not chunk_id:
            return ""
        if self.is_flat:
            v = self.raw.get(chunk_id, "")
            return v if isinstance(v, str) else str(v)
        # bucketed
        if "#" in chunk_id:
            base, _ = chunk_id.split("#", 1)
            bucket = self.raw.get(base)
            if isinstance(bucket, dict):
                v = bucket.get(chunk_id, "")
                return v if isinstance(v, str) else ("" if v is None else str(v))
        # sometimes index mixes both
        v = self.raw.get(chunk_id, "")
        return v if isinstance(v, str) else ("" if v is None else str(v))
