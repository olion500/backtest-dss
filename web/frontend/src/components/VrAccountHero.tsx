import type { VRResult } from "../types";

function usd(value: number | null | undefined, digits = 0) {
  if (value == null) return "—";
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

function signedUsd(value: number) {
  return `${value < 0 ? "−" : "+"}${usd(Math.abs(value))}`;
}

function pct(value: number, digits = 1) {
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

const RING_R = 46;
const RING_C = 2 * Math.PI * RING_R;

/** 계좌 현황: 주식 비중 링 + 계좌총액, 오른쪽에 보유 명세 */
export function VrAccountHero({ result }: { result: VRResult }) {
  const { current, holding } = result;
  const stock = current.stock_value ?? 0;
  const cash = current.pool;
  const total = current.total ?? stock + cash;
  const stockPct = total ? (stock / total) * 100 : 0;
  const profit = total - current.principal;
  const unrealized = holding.unrealized ?? 0;
  const unrealPct = holding.cost_basis ? (unrealized / holding.cost_basis) * 100 : 0;

  return (
    <div className="vrh">
      <div className="vrh-summary">
        <svg viewBox="0 0 120 120" className="vrh-ring" role="img" aria-label={`주식 비중 ${stockPct.toFixed(0)}%`}>
          <circle cx="60" cy="60" r={RING_R} className="cash" />
          <circle cx="60" cy="60" r={RING_R} className="stock" strokeDasharray={`${(RING_C * stockPct) / 100} ${RING_C}`} transform="rotate(-90 60 60)" />
          <text x="60" y="58" textAnchor="middle" className="big">{stockPct.toFixed(0)}%</text>
          <text x="60" y="76" textAnchor="middle" className="small">주식 비중</text>
        </svg>
        <div className="vrh-total">
          <span className="eyebrow">계좌총액 // TOTAL</span>
          <strong>{usd(total)}</strong>
          <small className={profit < 0 ? "negative" : "lime"}>
            {current.return_pct == null ? "—" : pct(current.return_pct)} ({signedUsd(profit)})
          </small>
          <small>원금 {usd(current.principal)}</small>
        </div>
      </div>
      <dl className="vrh-list">
        <div><dt><i className="stock" />주식 평가금</dt><dd>{usd(stock)}</dd></div>
        <div><dt><i className="cash" />현금 (Pool)</dt><dd>{usd(cash)}</dd></div>
        <div><dt>보유 수량</dt><dd>{current.shares.toLocaleString()}주</dd></div>
        <div><dt>평단가 / 현재가</dt><dd>{usd(holding.avg_cost, 2)} / {usd(current.close, 2)}</dd></div>
        <div><dt>평가손익</dt><dd className={unrealized < 0 ? "negative" : "lime"}>{signedUsd(unrealized)} ({pct(unrealPct)})</dd></div>
        <div><dt>매입금액</dt><dd>{usd(holding.cost_basis)}</dd></div>
      </dl>
    </div>
  );
}
