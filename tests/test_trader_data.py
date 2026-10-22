"""Trader data step: broker closes over Yahoo bars, the GitHub commit. Fakes only, no network."""
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from scripts.update_market_data import apply_broker_closes, load_broker_closes
from trader.broker.base import Quote
from trader.calendar import NY
from trader.config import TraderAccount
from trader.datasync import record_closes, run_data
from trader.github import GitHubRepo

SESSION = datetime(2026, 10, 22, 9, 35, tzinfo=NY)


def bars():
    index = pd.to_datetime(["2026-10-20", "2026-10-21"])
    return pd.DataFrame({
        "Open": [94.52, 104.18], "High": [102.62, 106.13], "Low": [91.35, 98.0],
        "Close": [97.55, 98.5], "Adj Close": [97.55, 98.5], "Volume": [100094100, 72708592],
    }, index=index)


def test_broker_close_overrides_yahoo_and_keeps_the_bar_whole():
    closes = pd.DataFrame({"Date": pd.to_datetime(["2026-10-21", "2026-10-21"]), "Ticker": ["SOXL", "QQQ"],
                           "Close": [103.24, 725.2]})
    frame = apply_broker_closes("SOXL", bars(), closes)
    row = frame.loc["2026-10-21"]
    assert row["Close"] == 103.24 and row["Adj Close"] == 103.24 and row["High"] == 106.13
    assert frame.loc["2026-10-20", "Close"] == 97.55  # rows without a broker close stay Yahoo's
    low = pd.DataFrame({"Date": pd.to_datetime(["2026-10-21"]), "Ticker": ["SOXL"], "Close": [97.0]})
    assert apply_broker_closes("SOXL", bars(), low).loc["2026-10-21", "Low"] == 97.0


def test_record_closes_replaces_the_same_session(tmp_path):
    path = tmp_path / "broker_closes.csv"
    record_closes(path, date(2026, 10, 21), {"SOXL": Decimal("98.5000")})
    record_closes(path, date(2026, 10, 21), {"SOXL": Decimal("103.2400"), "QQQ": Decimal("725.2")})
    record_closes(path, date(2026, 10, 22), {"SOXL": Decimal("85.40")})
    assert path.read_text().splitlines() == [
        "Date,Ticker,Close", "2026-10-21,QQQ,725.2", "2026-10-21,SOXL,103.24", "2026-10-22,SOXL,85.4",
    ]
    assert len(load_broker_closes(tmp_path)) == 3


class FakeBroker:
    def __init__(self):
        self.asked = []

    def quote(self, symbol):
        self.asked.append(symbol)
        return Quote(symbol, Decimal("85"), {"SOXL": Decimal("103.24"), "QQQ": Decimal("725.2")}[symbol], SESSION)


class FakeRepo:
    def __init__(self, files):
        self.files = files
        self.commits = []

    def head(self):
        return "parent"

    def read(self, commit, path):
        return self.files.get(path)

    def commit(self, parent, files, message):
        self.commits.append((parent, files, message))
        return "new"


def run(tmp_path, now=SESSION, push=False, repo=None, broker=None):
    account = TraderAccount(id="main", broker="meritz", credentials_path=None, listings={"SOXL": "AMEX", "QQQ": "NASDAQ"})

    def update(workdir):
        (workdir / "SOXL.csv").write_text("updated")
        return 0

    return run_data("main", now=now, push=push, repo=repo, data_dir=tmp_path, update=update,
                    broker_builder=lambda _: broker or FakeBroker(), load_account=lambda _: account)


def test_only_during_the_session(tmp_path):
    before_open = datetime(2026, 10, 22, 9, 0, tzinfo=NY)
    assert run(tmp_path, now=before_open)["status"] == "closed"
    assert run(tmp_path, now=datetime(2026, 10, 24, 10, 0, tzinfo=NY))["status"] == "closed"  # Saturday


def test_local_run_writes_the_previous_sessions_closes(tmp_path):
    (tmp_path / "tickers.txt").write_text("SOXL\nQQQ\n^VIX\n")
    broker = FakeBroker()
    result = run(tmp_path, broker=broker)
    assert result["status"] == "ok" and result["close_of"] == "2026-10-21"
    assert broker.asked == ["SOXL", "QQQ"]  # ^VIX is not listed at the broker
    assert result["changed"] == ["SOXL.csv", "broker_closes.csv"]
    assert "2026-10-21,SOXL,103.24" in (tmp_path / "broker_closes.csv").read_text()


def test_push_reads_from_github_and_commits_only_what_changed(tmp_path):
    repo = FakeRepo({"data/tickers.txt": b"SOXL\nQQQ\n", "data/SOXL.csv": b"old", "data/QQQ.csv": b"same"})
    result = run(tmp_path, push=True, repo=repo)
    assert result["commit"] == "new"
    [(parent, files, message)] = repo.commits
    assert parent == "parent" and sorted(files) == ["data/SOXL.csv", "data/broker_closes.csv"]
    assert files["data/SOXL.csv"] == b"updated" and "2026-10-21" in message
    assert not list(tmp_path.iterdir())  # the repo's own data/ is untouched


class FakeResponse:
    def __init__(self, status, body=None, content=b""):
        self.status_code, self._body, self.content, self.text = status, body, content, ""

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, json=None, timeout=None):
        path = url.split("/repos/o/r", 1)[1]
        self.calls.append((method, path, json))
        if path.startswith("/contents/"):
            return FakeResponse(404)
        answers = {
            "/git/ref/heads/main": {"object": {"sha": "c0"}},
            "/git/commits/c0": {"tree": {"sha": "t0"}},
            "/git/blobs": {"sha": "b1"},
            "/git/trees": {"sha": "t1"},
            "/git/commits": {"sha": "c1"},
            "/git/refs/heads/main": {},
        }
        return FakeResponse(200 if method in ("GET", "PATCH") else 201, answers[path])


def test_github_commit_is_one_fast_forward():
    http = FakeHttp()
    repo = GitHubRepo("o/r", "token", http=http)
    assert repo.read("c0", "data/missing.csv") is None
    assert repo.commit(repo.head(), {"data/SOXL.csv": b"x"}, "m") == "c1"
    methods = [(method, path) for method, path, _ in http.calls]
    assert methods[-1] == ("PATCH", "/git/refs/heads/main") and http.calls[-1][2] == {"sha": "c1", "force": False}
    tree = next(body for method, path, body in http.calls if path == "/git/trees")
    assert tree == {"base_tree": "t0", "tree": [{"path": "data/SOXL.csv", "mode": "100644", "type": "blob", "sha": "b1"}]}
