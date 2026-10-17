"""Config files read by the web API (the web never writes them).

Layout::

    config/strategy.json         shared strategy (all accounts trade it)
    config/accounts/<id>.json    per-account: name, order + ACCOUNT_KEYS (gitignored)
    config/presets/<name>.json   saved strategy candidates
    config/vr/<id>.json          value rebalancing (VR) accounts (gitignored)

Strategy files never hold account keys and account files hold nothing else,
so editing one never touches the other.
"""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
STRATEGY_PATH = CONFIG_DIR / "strategy.json"
ACCOUNTS_DIR = CONFIG_DIR / "accounts"
PRESETS_DIR = CONFIG_DIR / "presets"
VR_DIR = CONFIG_DIR / "vr"

ACCOUNT_KEYS = ("start_date", "init_cash", "spread_buy_levels", "spread_buy_step", "spread_floor_pct", "fee_pct", "sec_fee_pct")
ACCOUNT_DEFAULTS = {
    "start_date": None, "init_cash": 10_000.0, "spread_buy_levels": 5, "spread_buy_step": 1,
    "spread_floor_pct": 0.0, "fee_pct": 0.0, "sec_fee_pct": 0.0,
}
DEFAULT_ACCOUNT_ID = "main"


def read_json(path: Path) -> dict:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def strategy_only(data: dict) -> dict:
    return {key: value for key, value in data.items() if key not in ACCOUNT_KEYS and key != "name"}


def load_strategy(path: Path = STRATEGY_PATH) -> dict:
    return strategy_only(read_json(path))


def _account_order(raw: dict) -> int | None:
    order = raw.get("order")
    return order if isinstance(order, int) and not isinstance(order, bool) else None


def load_accounts(accounts_dir: Path = ACCOUNTS_DIR) -> list[dict]:
    """Accounts sorted by ``order``, then file name; each has ``id``, ``name``, ``order`` and every ACCOUNT_KEY.

    Accounts without an ``order`` come after the ordered ones. With no account
    files there is a single default account, so callers never have to handle
    an empty list.
    """
    accounts = []
    paths = sorted(Path(accounts_dir).glob("*.json")) if Path(accounts_dir).is_dir() else []
    for path in paths:
        raw = read_json(path)
        account = {"id": path.stem, "name": raw.get("name") or path.stem, "order": _account_order(raw)}
        account.update({key: raw.get(key, default) for key, default in ACCOUNT_DEFAULTS.items()})
        accounts.append(account)
    accounts.sort(key=lambda account: (account["order"] is None, account["order"] or 0))
    if not accounts:
        accounts.append({"id": DEFAULT_ACCOUNT_ID, "name": "기본", "order": None, **ACCOUNT_DEFAULTS})
    return accounts


def strategy_files(presets_dir: Path = PRESETS_DIR, strategy_path: Path = STRATEGY_PATH) -> list[Path]:
    """strategy.json first, then presets newest first."""
    presets = sorted(Path(presets_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if Path(presets_dir).is_dir() else []
    return ([Path(strategy_path)] if Path(strategy_path).exists() else []) + presets


def load_vr_accounts(vr_dir: Path = VR_DIR) -> list[dict]:
    """VR account files as raw dicts with ``id`` added, sorted like :func:`load_accounts`."""
    accounts = []
    paths = sorted(Path(vr_dir).glob("*.json")) if Path(vr_dir).is_dir() else []
    for path in paths:
        raw = read_json(path)
        if raw:
            accounts.append({**raw, "id": path.stem, "name": raw.get("name") or path.stem, "order": _account_order(raw)})
    accounts.sort(key=lambda account: (account["order"] is None, account["order"] or 0))
    return accounts
