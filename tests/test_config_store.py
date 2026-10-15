"""Strategy / account config files stay separate."""
import os

from engines.config_store import load_accounts, load_strategy, load_vr_accounts, read_json, strategy_files


def test_load_strategy_drops_account_keys(tmp_path):
    path = tmp_path / "strategy.json"
    path.write_text('{"target": "SOXL", "name": "x", "start_date": "2026-06-01", "init_cash": 1, "spread_buy_step": 3}')
    assert load_strategy(path) == {"target": "SOXL"}


def test_read_json_tolerates_missing_and_non_dict_files(tmp_path):
    assert read_json(tmp_path / "missing.json") == {}
    (tmp_path / "list.json").write_text("[1, 2]")
    assert read_json(tmp_path / "list.json") == {}
    (tmp_path / "bad.json").write_text("{")
    assert read_json(tmp_path / "bad.json") == {}


def test_accounts_fill_defaults_and_fall_back_to_file_name(tmp_path):
    (tmp_path / "a.json").write_text('{"init_cash": 500}')
    [account] = load_accounts(tmp_path)
    assert account["id"] == "a" and account["name"] == "a"
    assert account["init_cash"] == 500
    assert account["spread_buy_step"] == 1 and account["spread_buy_levels"] == 5


def test_no_account_files_gives_one_default_account(tmp_path):
    [account] = load_accounts(tmp_path / "missing")
    assert account["id"] == "main" and account["init_cash"] == 10_000.0


def test_strategy_files_lists_strategy_then_presets_newest_first(tmp_path):
    strategy = tmp_path / "strategy.json"
    strategy.write_text("{}")
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "old.json").write_text("{}")
    (presets / "new.json").write_text("{}")
    os.utime(presets / "old.json", (1, 1))
    assert strategy_files(presets, strategy) == [strategy, presets / "new.json", presets / "old.json"]
    assert strategy_files(tmp_path / "none", tmp_path / "none.json") == []


def test_accounts_sort_by_order_then_file_name(tmp_path):
    (tmp_path / "a.json").write_text('{"name": "A"}')
    (tmp_path / "b.json").write_text('{"name": "B", "order": 2}')
    (tmp_path / "c.json").write_text('{"name": "C", "order": 1}')
    (tmp_path / "d.json").write_text('{"name": "D", "order": true}')
    assert [a["id"] for a in load_accounts(tmp_path)] == ["c", "b", "a", "d"]


def test_vr_accounts_skip_empty_files_and_sort_by_order(tmp_path):
    (tmp_path / "a.json").write_text('{"name": "A"}')
    (tmp_path / "b.json").write_text('{"name": "B", "order": 0}')
    (tmp_path / "empty.json").write_text("{}")
    assert [a["id"] for a in load_vr_accounts(tmp_path)] == ["b", "a"]
    assert load_vr_accounts(tmp_path / "missing") == []


def test_non_dict_account_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "c.json").write_text("[1, 2]")
    [account] = load_accounts(tmp_path)
    assert account["name"] == "c" and account["order"] is None and account["init_cash"] == 10_000.0
