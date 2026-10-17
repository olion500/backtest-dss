"""Per-account trader settings: ``config/trader/<account id>.json`` (gitignored, never in the web image).

The id matches ``config/accounts/<id>.json``, which keeps the strategy side of the
account (start date, cash, steps). This file only says how to reach the broker::

    {
      "broker": "meritz",
      "auto_trade": false,
      "listings": {"SOXL": "AMEX"}
    }

The broker's keys come from the environment as ``<BROKER>_<ACCOUNT ID>_APP_KEY``,
``…_APP_SECRET`` (``MERITZ_MAIN_APP_KEY`` for ``config/trader/main.json``): the broker
name keeps two brokers' keys apart, the account id keeps two accounts apart (each
brokerage account has its own keys). ``python -m trader`` fills the environment from
the repo's ``.env`` (gitignored; on Cloud Run the same names come from Secret Manager).
An id that is not a valid variable name (e.g. Korean) sets ``"credentials_env": "MERITZ_YH"``.
A KEY=VALUE file outside the repo works too: ``"credentials": "~/path/meritz-main.env"``.
Secrets never go in this file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from engines.config_store import CONFIG_DIR, read_json

TRADER_DIR = CONFIG_DIR / "trader"
REPO_ENV = CONFIG_DIR.parent / ".env"

# Where each ticker trades, by neutral exchange name (see the broker adapters' EXCHANGES).
DEFAULT_LISTINGS = {"SOXL": "AMEX", "TQQQ": "NASDAQ", "QQQ": "NASDAQ", "SOXX": "NASDAQ"}


class TraderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TraderAccount:
    id: str
    broker: str
    credentials_path: Path | None = None
    auto_trade: bool = False
    listings: dict[str, str] = field(default_factory=dict)
    credentials_env: str | None = None

    @property
    def env_prefix(self) -> str:
        return (self.credentials_env or f"{self.broker}_{self.id}").upper().rstrip("_")

    def credentials(self) -> dict[str, str]:
        """The broker's keys by their short names (APP_KEY, APP_SECRET, …)."""
        if self.credentials_path is not None:
            try:
                return read_env_file(self.credentials_path)
            except OSError as exc:
                raise TraderConfigError(f"{self.id}: 자격증명 파일을 읽지 못했습니다 ({self.credentials_path}): {exc.strerror}") from None
        prefix = self.env_prefix + "_"
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", prefix):
            raise TraderConfigError(f"{self.id}: {prefix}… 는 환경변수 이름이 될 수 없습니다. 설정에 \"credentials_env\": \"{self.broker.upper()}_<영문>\"을 적으세요")
        found = {key[len(prefix):]: value for key, value in os.environ.items() if key.startswith(prefix) and value}
        if not found:
            raise TraderConfigError(f"{self.id}: 환경변수 {prefix}APP_KEY … 가 없습니다 (.env 확인)")
        return found


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_repo_env(path: Path = REPO_ENV) -> None:
    """Fill os.environ from the repo's .env, never overriding what the environment already set."""
    try:
        values = read_env_file(path)
    except OSError:
        return
    for key, value in values.items():
        os.environ.setdefault(key, value)


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
        credentials_env=raw.get("credentials_env") or None,
        auto_trade=raw.get("auto_trade") is True,
        listings=listings,
    )


def trader_account_ids(trader_dir: Path = TRADER_DIR) -> list[str]:
    return sorted(path.stem for path in Path(trader_dir).glob("*.json")) if Path(trader_dir).is_dir() else []
