"""Per-account trader settings: ``config/trader/<account id>.json`` (gitignored, never in the web image).

The id matches ``config/accounts/<id>.json``, which keeps the strategy side of the
account (start date, cash, steps). This file only says how to reach the broker::

    {
      "broker": "meritz",
      "credentials": "~/.config/dongpa/meritz-main.env",
      "auto_trade": false,
      "listings": {"SOXL": "AMEX"}
    }

``credentials`` is a KEY=VALUE file outside the repo; leave it out to read the
broker's keys from the environment instead (e.g. secrets mounted on Cloud Run).
Secrets never go in this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from engines.config_store import CONFIG_DIR, read_json

TRADER_DIR = CONFIG_DIR / "trader"

# Where each ticker trades, by neutral exchange name (see the broker adapters' EXCHANGES).
DEFAULT_LISTINGS = {"SOXL": "AMEX", "TQQQ": "NASDAQ", "QQQ": "NASDAQ", "SOXX": "NASDAQ"}


class TraderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TraderAccount:
    id: str
    broker: str
    credentials_path: Path | None
    auto_trade: bool = False
    listings: dict[str, str] = field(default_factory=dict)

    def credentials(self) -> dict[str, str]:
        if self.credentials_path is None:
            return dict(os.environ)
        try:
            return read_env_file(self.credentials_path)
        except OSError as exc:
            raise TraderConfigError(f"{self.id}: 자격증명 파일을 읽지 못했습니다 ({self.credentials_path}): {exc.strerror}") from None


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_trader_account(account_id: str, trader_dir: Path = TRADER_DIR) -> TraderAccount:
    path = Path(trader_dir) / f"{account_id}.json"
    raw = read_json(path)
    if not raw:
        raise TraderConfigError(f"{account_id}: 자동매매 설정이 없습니다 ({path})")
    broker = raw.get("broker")
    if not isinstance(broker, str) or not broker:
        raise TraderConfigError(f"{account_id}: broker가 없습니다 ({path})")
    credentials = raw.get("credentials")
    listings = {**DEFAULT_LISTINGS, **(raw.get("listings") or {})}
    return TraderAccount(
        id=account_id,
        broker=broker,
        credentials_path=Path(credentials).expanduser() if credentials else None,
        auto_trade=raw.get("auto_trade") is True,
        listings=listings,
    )


def trader_account_ids(trader_dir: Path = TRADER_DIR) -> list[str]:
    return sorted(path.stem for path in Path(trader_dir).glob("*.json")) if Path(trader_dir).is_dir() else []
