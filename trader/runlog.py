"""Run records: one JSON file per run, read back by the web order book (B timeline).

Layout: ``<root>/<account>/<session date>/<HHMMSS>-<step>.json``. The root is
``var/trader`` (gitignored) unless ``DONGPA_TRADER_RUNS`` points elsewhere;
a cloud store can replace :class:`LocalRunStore` behind the same two methods.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "var" / "trader"


class RunStore(Protocol):
    def save(self, account_id: str, record: dict) -> str: ...

    def recent(self, account_id: str, limit: int = 20) -> list[dict]: ...


class LocalRunStore:
    def __init__(self, root: Path | None = None):
        configured = os.environ.get("DONGPA_TRADER_RUNS", "").strip()
        self.root = Path(root or configured or DEFAULT_ROOT)

    def save(self, account_id: str, record: dict) -> str:
        started = datetime.fromisoformat(record["started_at"])
        folder = self.root / account_id / record["session"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{started:%H%M%S}-{record['step']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return str(path)

    def recent(self, account_id: str, limit: int = 20) -> list[dict]:
        base = self.root / account_id
        if not base.is_dir():
            return []
        files = sorted(base.glob("*/*.json"), key=lambda p: (p.parent.name, p.name), reverse=True)
        out = []
        for path in files[:limit]:
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return out
