"""Offense→defense transition rules: what to do with the open offense lots at the switch, and
earlier defense from the target's own fall or from a run of offense stop-losses.

The engine stays untouched. Lots keep the tp/sl/max_hold of the mode they were bought in
(``run_backtest``), and ``decide_mode`` runs at the end of every day before the next session's
orders, so ``run`` wraps it: it may force the next mode to defense and edit the open lots of the
calling ``run_backtest`` frame. A change made at the end of day d acts on the LOC/MOC orders of
d+1, like an order edited in the evening. The base mode (RSI or an injected series) keeps its own
state, so a forced defense never changes what the base signal would decide.
"""
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace

import pandas as pd

import engines.dongpa_engine as E
from engines.dongpa_engine import HUNDRED, ONE, CapitalParams, StrategyParams, money, to_decimal

from .data import load, scaled
from .engine import _injected


@dataclass(frozen=True)
class Policy:
    """on_switch, applied to the open offense lots when the next mode turns offense → defense:
      keep       as today: offense tp / sl / max hold
      liquidate  sell them all at the next close
      losers     sell only those below their fill
      retarget   defense tp from the fill, at most the defense hold from now, sl kept
      slcap      stop raised to fill × (1 − sl_cap%) (never lowered)
    Early defense (forced for the next sessions, the base signal untouched):
      dd_n/dd_pct/dd_hold  target close ≥ dd_pct% below its dd_n-session high → defense for
                           dd_hold sessions (0: only while the fall lasts)
      sl_k/sl_win/sl_hold  sl_k offense stop-losses within sl_win sessions → defense for sl_hold
    action='pause': the two triggers above stop new buys instead of forcing defense.
    No new buys around the RSI switch (open lots still sell on tp / sl / max hold):
      pause_pre   while the weekly RSI, with this week closed at today's close, would switch
                  offense → defense (the Friday close decides the real switch)
      pause_post  for that many sessions after the real switch
      pause_both  the same for defense → offense switches
    """
    on_switch: str = 'keep'
    sl_cap: float = 0.0
    dd_n: int = 0
    dd_pct: float = 0.0
    dd_hold: int = 0
    sl_k: int = 0
    sl_win: int = 0
    sl_hold: int = 0
    action: str = 'defense'
    pause_pre: bool = False
    pause_post: int = 0
    pause_both: bool = False

    @property
    def name(self) -> str:
        parts = [self.on_switch + (f'{self.sl_cap:g}' if self.on_switch == 'slcap' else '')]
        if self.dd_n: parts.append(f'dd{self.dd_n}/{self.dd_pct:g}%/{self.dd_hold}')
        if self.sl_k: parts.append(f'sl{self.sl_k}in{self.sl_win}/{self.sl_hold}')
        if self.action == 'pause' and (self.dd_n or self.sl_k): parts.append('->pause')
        if self.pause_pre: parts.append('pause_pre')
        if self.pause_post: parts.append(f'pause_post{self.pause_post}')
        if self.pause_both: parts.append('both')
        return ' '.join(parts)


KEEP = Policy()


def _apply_on_switch(policy, lots, trades, close, defense):
    for lot in lots:
        if trades[lot['trade_idx']]['매수모드'] != '공세':
            continue
        if policy.on_switch == 'liquidate' or (policy.on_switch == 'losers' and close < lot['fill']):
            lot['max_hold'] = 0                       # days + 1 >= 0: sold at the next close (MOC)
        elif policy.on_switch == 'retarget':
            lot['tp'] = money(lot['fill'] * (ONE + to_decimal(defense.tp_pct) / HUNDRED))
            lot['max_hold'] = min(lot['max_hold'], lot['days'] + int(defense.max_hold_days))
        elif policy.on_switch == 'slcap':
            cap = money(lot['fill'] * (ONE - to_decimal(policy.sl_cap) / HUNDRED))
            lot['sl'] = cap if lot['sl'] is None else max(lot['sl'], cap)


def provisional_mode(momo: pd.DataFrame, params) -> pd.Series:
    """Per day: the RSI decision if the week closed at that day's close (None = keep the mode).
    Same Wilder recursion as ``E.wilder_rsi``, one step from the last completed week."""
    c, w, a = momo['Close'], E.to_weekly_close(momo), 1 / params.rsi_period
    d = w.diff()
    up = d.clip(lower=0).ewm(alpha=a, adjust=False).mean().values
    dn = (-d.clip(upper=0)).ewm(alpha=a, adjust=False).mean().values
    rsi = E.wilder_rsi(w, params.rsi_period).values
    last = w.index.get_indexer(c.index.to_period('W-FRI').end_time.normalize()) - 1
    out = {}
    for day, k, close in zip(c.index, last, c.values):
        if k < 0:
            out[day] = None; continue
        ch = close - w.iloc[k]
        ru, rd = (1 - a) * up[k] + a * max(ch, 0.0), (1 - a) * dn[k] + a * max(-ch, 0.0)
        r = 50.0 if rd == 0 else 100 - 100 / (1 + ru / rd)
        out[day] = E.eval_rsi_conditions(r, rsi[k], r - rsi[k], params)
    return pd.Series(out, dtype=object)


@contextmanager
def _hooked(policy: Policy):
    decide, initial = E.decide_mode, E.determine_initial_mode
    st = {'base': None, 'force': 0, 'pause': 0, 'sl_days': [], 'hi': None, 'orig': None}

    def _fire(sessions):
        key = 'pause' if policy.action == 'pause' else 'force'
        st[key] = max(st[key], sessions)

    def _initial(start, indicators, params):
        st['base'] = initial(start, indicators, params)
        return st['base']

    def _decide(idx, prev_mode, indicators, params):
        prev_base = st['base'] or prev_mode
        base = decide(idx, prev_base, indicators, params)
        st['base'] = base
        f = sys._getframe(1).f_locals
        i, lots, trades = f['i'], f['lots'], f['trades']
        if st['orig'] is None:
            st['orig'] = (params.offense, params.defense)
            st['nobuy'] = tuple(replace(m, buy_cond_pct=-100.0) for m in st['orig'])
        if policy.pause_pre:
            if 'prov' not in st:
                st['prov'] = provisional_mode(f['daily_momo'], params)
            pred = st['prov'].get(idx)
            if pred is not None and pred != base and (policy.pause_both or base == 'offense'):
                st['pause'] = max(st['pause'], 1)
        if policy.pause_post and base != prev_base and (policy.pause_both or base == 'defense'):
            st['pause'] = max(st['pause'], policy.pause_post)
        if policy.dd_n:
            if st['hi'] is None:
                c = f['df']['Close']
                st['c'], st['hi'] = c.values, c.rolling(policy.dd_n, min_periods=1).max().values
            if st['c'][i] / st['hi'][i] - 1 <= -policy.dd_pct / 100:
                _fire(policy.dd_hold or 1)
        if policy.sl_k:
            for lot, _ in f['sold']:
                t = trades[lot['trade_idx']]
                if t['청산사유'] == 'SL' and t['매수모드'] == '공세':
                    st['sl_days'].append(i)
            st['sl_days'] = [x for x in st['sl_days'] if x > i - policy.sl_win]
            if len(st['sl_days']) >= policy.sl_k:
                _fire(policy.sl_hold)
                st['sl_days'] = []
        nxt = 'defense' if st['force'] > 0 else base
        st['force'] = max(0, st['force'] - 1)
        if prev_mode == 'offense' and nxt == 'defense' and policy.on_switch != 'keep':
            _apply_on_switch(policy, lots, trades, f['close'], st['orig'][1])
        offense, defense = st['nobuy'] if st['pause'] > 0 else st['orig']
        object.__setattr__(params, 'offense', offense)       # frozen dataclass, this run's copy
        object.__setattr__(params, 'defense', defense)
        st['pause'] = max(0, st['pause'] - 1)
        return nxt

    E.decide_mode, E.determine_initial_mode = _decide, _initial
    try:
        yield
    finally:
        E.decide_mode, E.determine_initial_mode = decide, initial


def run(target, modes, defense, offense, start, end, policy: Policy | None = KEEP, step=None, cash=1e9,
        momo='QQQ', scale=True):
    """Like ``engine.run``; ``policy=None`` runs without the hook (the plain engine)."""
    t = (scaled(target) if scale else load(target)).loc[start:end]
    m = load(momo).loc[:end]
    p = StrategyParams(target_ticker=target, momentum_ticker=momo, defense=defense, offense=offense,
                       spread_buy_step=step, **(modes if isinstance(modes, dict) else {}))
    inject = _injected(modes.reindex(t.index)) if isinstance(modes, pd.Series) else nullcontext()
    with inject, (_hooked(policy) if policy is not None else nullcontext()):
        return E.run_backtest(t, m, p, CapitalParams(initial_cash=cash))
