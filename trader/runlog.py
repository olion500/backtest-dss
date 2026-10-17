"""Run records: one JSON file per run, read back by the web order book (B timeline).

Layout: ``<root>/<account>/<session date>/<HHMMSS>-<step>.json``. ``DONGPA_TRADER_RUNS``
picks the root: ``gs://<bucket>/<prefix>`` for Cloud Storage (so the deployed web
sees runs made anywhere), a directory path, or nothing for ``var/trader`` (gitignored).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "var" / "trader"
ENV_VAR = "DONGPA_TRADER_RUNS"
RECENT_DAYS = 90  # how far back GcsRunStore.recent lists


class RunStore(Protocol):
    def save(self, account_id: str, record: dict) -> str: ...

    def recent(self, account_id: str, limit: int = 20) -> list[dict]: ...


def record_name(account_id: str, record: dict) -> str:
    started = datetime.fromisoformat(record["started_at"])
    return f"{account_id}/{record['session']}/{started:%H%M%S}-{record['step']}.json"


def _dumps(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2, default=str)


class LocalRunStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or DEFAULT_ROOT)

    def save(self, account_id: str, record: dict) -> str:
        path = self.root / record_name(account_id, record)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_dumps(record), encoding="utf-8")
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


class GcsRunStore:
    """The same layout as objects under ``gs://<bucket>/<prefix>/``."""

    def __init__(self, bucket: str, prefix: str = "", client=None):
        if client is None:
            from google.cloud import storage  # only needed when records live in GCS

            client = storage.Client()
        self.bucket = client.bucket(bucket)
        self._client = client
        self.prefix = prefix.strip("/")

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def save(self, account_id: str, record: dict) -> str:
        key = self._key(record_name(account_id, record))
        self.bucket.blob(key).upload_from_string(_dumps(record), content_type="application/json; charset=utf-8")
        return f"gs://{self.bucket.name}/{key}"

    def recent(self, account_id: str, limit: int = 20, days: int = RECENT_DAYS) -> list[dict]:
        # Names sort by session date then time: list only the last ``days`` of names (bounded
        # however long the account runs), newest first, and download just ``limit`` of them.
        since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        names = sorted(
            (blob.name for blob in self._client.list_blobs(
                self.bucket, prefix=self._key(f"{account_id}/"), start_offset=self._key(f"{account_id}/{since}"))
             if blob.name.endswith(".json")),
            reverse=True,
        )
        out = []
        for name in names[:limit]:
            try:
                out.append(json.loads(self.bucket.blob(name).download_as_text(encoding="utf-8")))
            except Exception:  # a broken or vanished record must not hide the rest
                continue
        return out


def run_store(target: str | None = None) -> RunStore:
    target = (target if target is not None else os.environ.get(ENV_VAR, "")).strip()
    if target.startswith("gs://"):
        return _gcs_store(target)
    return LocalRunStore(target or None)


@lru_cache(maxsize=4)
def _gcs_store(target: str) -> "GcsRunStore":
    """One store (and one storage client: auth, HTTP session) per target, not one per request."""
    bucket, _, prefix = target[len("gs://"):].partition("/")
    if not bucket:
        raise ValueError(f"{ENV_VAR}에 버킷 이름이 없습니다: {target}")
    return GcsRunStore(bucket, prefix)
