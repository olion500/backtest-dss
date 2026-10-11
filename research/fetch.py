import yfinance as yf, sys
out=sys.argv[1]
for t in ['SOXL','QQQ','SOXX','TQQQ','^VIX','TECL','UPRO','SPY','SMH','^VIX3M','FNGU','USD']:
    df=yf.download(t,start='1999-01-01',end='2026-10-11',progress=False,auto_adjust=False,repair=True)
    if hasattr(df.columns,'levels'): df.columns=df.columns.get_level_values(0)
    df.index.name='Date'
    df.to_csv(f"{out}/{t.replace('^','')}.csv")
    print(t, len(df), df.index[0].date() if len(df) else None, df.index[-1].date() if len(df) else None)
