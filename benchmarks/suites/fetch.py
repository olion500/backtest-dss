"""Download daily OHLCV from Yahoo into benchmarks/data/.

usage: fetch [TICKER ...] [--end YYYY-MM-DD]   (default: every ticker the suites use)
"""
import argparse

from benchmarks.lib.data import TICKERS, fetch


def main(argv):
    ap = argparse.ArgumentParser(prog='fetch')
    ap.add_argument('tickers', nargs='*', default=TICKERS)
    ap.add_argument('--end', default=None)
    a = ap.parse_args(argv)
    fetch(a.tickers, end=a.end)
