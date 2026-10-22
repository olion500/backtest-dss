"""``python -m trader data``: refresh data/ from Yahoo, with the broker's previous closes on top.

Yahoo has served a provisional close as final (SOXL 2026-10-21: $98.50, the broker's $103.24),
and the whole ledger follows that close. During the regular session the broker's previous close
is the last session's official close, so the job runs after the open (09:35 NY), before orders:
  1. each tracked ticker the broker lists → its previous close into data/broker_closes.csv
  2. scripts/update_market_data.py: Yahoo bars (7-day overlap), broker closes override
  3. ``--push``: the files read from and committed to GitHub (``DONGPA_DATA_REPO``, ``GITHUB_DATA_TOKEN``),
     where the trader jobs read data/ (``DONGPA_DATA_URL``). Without it, the repo's data/ in place.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable

import pandas as pd

from scripts.update_market_data import BROKER_CLOSES, DATA_DIR, main as update_bars, read_tickers
from trader.broker.base import Broker, BrokerError
from trader.calendar import last_closed_session, ny_now, us_session
from trader.config import TraderAccount, TraderConfigError, load_trader_account
from trader.github import GitHubError, GitHubRepo
from trader.runner import _default_broker

BrokerBuilder = Callable[[TraderAccount], Broker]


def repo_from_env() -> GitHubRepo:
    repo, token = os.environ.get("DONGPA_DATA_REPO", "").strip(), os.environ.get("GITHUB_DATA_TOKEN", "").strip()
    if not repo or not token:
        raise TraderConfigError("DONGPA_DATA_REPO(owner/name)와 GITHUB_DATA_TOKEN이 있어야 GitHub에 올립니다")
    return GitHubRepo(repo, token)


def record_closes(path: Path, session: date, closes: dict[str, Decimal]) -> None:
    """Add (or replace) one session's broker closes in broker_closes.csv."""
    day = pd.Timestamp(session)
    rows = pd.DataFrame({"Date": [day] * len(closes), "Ticker": list(closes), "Close": [float(c) for c in closes.values()]})
    if path.exists():
        table = pd.read_csv(path, parse_dates=["Date"])
        table = table[~((table["Date"] == day) & table["Ticker"].isin(list(closes)))]
        rows = pd.concat([table, rows])
    table = rows.sort_values(["Date", "Ticker"])
    table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%Y-%m-%d")
    path.write_text(table.to_csv(index=False))


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def run_data(
    account_id: str,
    now: datetime | None = None,
    force: bool = False,
    push: bool = False,
    broker_builder: BrokerBuilder = _default_broker,
    load_account: Callable[[str], TraderAccount] = load_trader_account,
    repo: GitHubRepo | None = None,
    data_dir: Path = DATA_DIR,
    update: Callable[[Path], int] = update_bars,
) -> dict:
    now = now or ny_now()
    result: dict = {"account": account_id, "step": "data", "started_at": now.isoformat(), "push": push}
    session = us_session(now.date())
    if not session.trading_at(now) and not force:
        # Outside the session the broker's "previous close" may still be the day before.
        return {**result, "status": "closed", "message": "정규장 시간에만 증권사 전일 종가를 믿을 수 있습니다"}
    closed = last_closed_session(now.date())
    result["close_of"] = closed.isoformat()
    try:
        account = load_account(account_id)
        broker = broker_builder(account)
        with tempfile.TemporaryDirectory() as scratch:
            if push:
                repo = repo or repo_from_env()
                parent = repo.head()
                workdir = Path(scratch)
                for name in ["tickers.txt", BROKER_CLOSES]:
                    content = repo.read(parent, f"data/{name}")
                    if content is not None:
                        (workdir / name).write_bytes(content)
                for ticker in read_tickers(workdir):
                    content = repo.read(parent, f"data/{ticker}.csv")
                    if content is not None:
                        (workdir / f"{ticker}.csv").write_bytes(content)
            else:
                workdir = data_dir
            before = _snapshot(workdir)

            closes = {}
            for ticker in read_tickers(workdir):
                if ticker in account.listings:
                    prev_close = broker.quote(ticker).prev_close
                    if prev_close:
                        closes[ticker] = prev_close
            result["broker_closes"] = {ticker: str(close) for ticker, close in closes.items()}
            if closes:
                record_closes(workdir / BROKER_CLOSES, closed, closes)
            code = update(workdir)

            after = _snapshot(workdir)
            changed = sorted(name for name, content in after.items() if before.get(name) != content)
            result["changed"] = changed
            if push and changed:
                message = f"chore: market data through {closed} (closes: {account.broker} {', '.join(closes) or 'none'})"
                result["commit"] = repo.commit(parent, {f"data/{name}": after[name] for name in changed}, message)
        status = "ok" if code == 0 else "partial"
        message = "모든 종목 갱신" if code == 0 else "일부 종목을 받지 못했습니다 (로그 참고)"
        return {**result, "status": status, "message": message}
    except (BrokerError, TraderConfigError, GitHubError) as exc:
        return {**result, "status": "error", "message": str(exc)}
