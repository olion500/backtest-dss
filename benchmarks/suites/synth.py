"""Build synthetic pre-listing histories SOXL_X, TQQQ_X, USD_X (2001~) from SOXX/QQQ.

Run after `fetch`. Prints how well each matches the real fund where both exist.
"""
from benchmarks.lib.data import synth


def main(argv):
    synth('SOXX', 'SOXL', er=0.0095 + 0.014, out='SOXL_X')
    synth('QQQ', 'TQQQ', er=0.0095 + 0.0104, out='TQQQ_X')
    synth('SOXX', 'USD', lev=2, er=0.0095, out='USD_X')
