"""Named strategies used by the comparison scripts. mode_fn(target) -> daily mode series."""
import json, os
from evalx import cur_params
from opt import build, Fixed
from harness import load
import signals as S
HERE = os.path.dirname(os.path.abspath(__file__))

def _rob(tag, rank):
    rows = json.load(open(os.path.join(HERE, 'studies', f'{tag}.robust.json')))
    return [r for r in rows if r['rank_in_study'] == rank][0]['params']

def current():
    c, D, O = cur_params()
    fn = lambda t: S.rsi_mode(load(t).index, c['rsi_high_threshold'], c['rsi_mid_high'], c['rsi_neutral'],
                              c['rsi_mid_low'], c['rsi_low_threshold'])
    return fn, D, O

def from_study(fam, tag, rank):
    prm = _rob(tag, rank)
    _, D, O = build(Fixed(prm), fam, load('SOXL').index)
    return (lambda t: build(Fixed(prm), fam, load(t).index)[0]), D, O

CANDIDATES = {
    'current': current,
    'A': lambda: from_study('sma_vix', 'full_SOXL_X_sma_vix', 1),
    'B': lambda: from_study('sma_vix', 'full_SOXL_X_sma_vix', 0),
    'C': lambda: from_study('ddq', 'full_SOXL_X_ddq', 1),
}
