import json, os
from harness import *
END = '2026-10-09'
def cur_params():
    c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'strategy.json')))
    D = mp(c['defense_slices'], c['defense_buy'], c['defense_tp'], c['defense_hold'], c['defense_sl'])
    O = mp(c['offense_slices'], c['offense_buy'], c['offense_tp'], c['offense_hold'], c['offense_sl'])
    return c, D, O

def evaluate(target, modes, D, O, start='2011-01-01'):
    r = run(target, modes, D, O, start, END)
    eq = r.equity
    out = {}
    out['cagr'], out['mdd'] = metrics(eq)
    e21 = eq.loc['2021-01-01':]; e21 = e21 / e21.iloc[0]
    out['cagr21'], out['mdd21'] = metrics(e21)
    e22 = eq.loc['2022-01-01':'2022-12-31']
    out['ret22'] = e22.iloc[-1] / eq.loc[:'2021-12-31'].iloc[-1] - 1
    out['mdd22'] = metrics(e22)[1]
    pre = eq.loc[:'2019-12-31']; out['cagr_pre20'] = metrics(pre)[0]
    out['mdd_pre20'] = metrics(pre)[1]
    if isinstance(modes, pd.Series):
        mm = modes.loc[start:END]; out['switches'] = int((mm != mm.shift()).sum() - 1)
        out['off_pct'] = float((mm == 'offense').mean())
    return out

def fmt(name, o):
    return (f"{name:38s} full {o['cagr']*100:6.1f}% / {o['mdd']*100:6.1f}% | 21~ {o['cagr21']*100:6.1f}% / {o['mdd21']*100:6.1f}% "
            f"| 22 {o['ret22']*100:6.1f}% dd{o['mdd22']*100:6.1f}% | pre20 {o['cagr_pre20']*100:5.1f}%/{o['mdd_pre20']*100:6.1f}% "
            f"| sw {o.get('switches','-')} off {o.get('off_pct',0)*100:.0f}%")
