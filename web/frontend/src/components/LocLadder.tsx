import { useMemo, useState, type ReactNode } from "react";

import { copyText } from "../clipboard";
import { isMocKind } from "../pip/steps";
import type { CellValue, OrderBookPayload, TablePayload } from "../types";

type Side = "sell" | "buy";

interface LadderRow {
  key: string;
  side: Side;
  /** 스프레드 행에서 바깥쪽으로 센 순번 (0이 종가에 가장 가까움) */
  rank: number;
  kind: string;
  tag: string;
  price: number;
  qty: number;
  total: number;
  distancePct: number | null;
  note: string;
  /** 만료·손절 매도: 지정가 없이 MOC로 넣는다 (주문가는 참고용 직전 종가) */
  moc: boolean;
}

function tagOf(kind: string, side: Side) {
  const inner = kind.match(/\((.+)\)/)?.[1];
  if (inner) return inner;
  return side === "sell" ? "매도" : "매수";
}

function num(value: CellValue) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function buildRows(table: TablePayload, prevClose: number | null) {
  const parsed = table.rows.flatMap((row, index) => {
    const kind = String(row["구분"] ?? "");
    const price = num(row["주문가"]);
    if (price == null) return [];
    const side: Side = kind.startsWith("매도") ? "sell" : "buy";
    return [{
      key: `${side}-${index}`,
      side,
      rank: 0,
      kind,
      tag: tagOf(kind, side),
      price,
      qty: num(row["수량"]) ?? 0,
      total: 0,
      distancePct: prevClose ? ((price - prevClose) / prevClose) * 100 : null,
      note: String(row["비고"] ?? ""),
      moc: isMocKind(kind),
    } satisfies LadderRow];
  });
  // 매도는 종가에서 위로, 매수는 종가에서 아래로 누적한다
  const sells = parsed.filter((r) => r.side === "sell").sort((a, b) => a.price - b.price);
  const buys = parsed.filter((r) => r.side === "buy").sort((a, b) => b.price - a.price);
  let total = 0;
  sells.forEach((r, rank) => { total += r.qty; r.total = total; r.rank = rank; });
  total = 0;
  buys.forEach((r, rank) => { total += r.qty; r.total = total; r.rank = rank; });
  return { sells: sells.reverse(), buys };
}

function qtyText(value: number) {
  return value.toLocaleString("ko-KR", { maximumFractionDigits: 2 });
}

function pctText(value: number | null) {
  if (value == null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function Row({
  row,
  maxDistance,
  highlighted,
  flashNonce,
  onEnter,
  onCopy,
}: {
  row: LadderRow;
  maxDistance: number;
  highlighted: boolean;
  flashNonce: number | null;
  onEnter: () => void;
  onCopy: () => void;
}) {
  // 막대 폭: 종가(0%)를 칸의 절반으로 두고, 종가보다 높으면 그만큼 길게, 낮으면 짧게 그린다
  const depth = row.distancePct == null ? 0.5 : 0.5 + (0.5 * row.distancePct) / maxDistance;
  return (
    <li
      className={`ladder-row ${row.side}${highlighted ? " highlighted" : ""}`}
      title={row.moc ? `MOC 주문 — 지정가 없이 장마감 시장가로 넣으세요${row.note ? `\n${row.note}` : ""}` : `클릭하면 주문가 ${row.price.toFixed(2)} 복사${row.note ? `\n${row.note}` : ""}`}
      onPointerEnter={onEnter}
      onClick={onCopy}
    >
      {flashNonce != null && <i key={flashNonce} className="ladder-flash" aria-hidden />}
      <span className="ladder-tag"><b>{row.tag}</b></span>
      <span className="ladder-price">{row.moc ? "MOC" : row.price.toFixed(2)}</span>
      <span className="ladder-dist">{row.moc ? "시장가" : pctText(row.distancePct)}</span>
      <span className="ladder-qty">{qtyText(row.qty)}</span>
      <span className="ladder-note">
        <i className="ladder-depth" style={{ width: `${(depth * 100).toFixed(2)}%` }} aria-hidden />
        <em>{row.note || "—"}</em>
      </span>
    </li>
  );
}

export function LocLadder({ orderBook, footer, action }: { orderBook: OrderBookPayload; footer: string; action?: ReactNode }) {
  const { state } = orderBook;
  const { sells, buys } = useMemo(() => buildRows(orderBook.orders, state.prev_close), [orderBook.orders, state.prev_close]);
  const [hovered, setHovered] = useState<{ side: Side; rank: number } | null>(null);
  const [flash, setFlash] = useState<{ key: string; nonce: number } | null>(null);

  const maxDistance = Math.max(1e-9, ...[...sells, ...buys].map((r) => Math.abs(r.distancePct ?? 0)));
  const sellQty = sells[0]?.total ?? 0;
  const buyQty = buys[buys.length - 1]?.total ?? 0;

  const copy = (row: LadderRow) => {
    if (row.moc) return;
    copyText(row.price.toFixed(2));
    setFlash({ key: row.key, nonce: Date.now() });
  };
  const isHighlighted = (row: LadderRow) => hovered != null && hovered.side === row.side && row.rank <= hovered.rank;

  return (
    <section className="order-sheet-panel ladder-panel" onPointerLeave={() => setHovered(null)}>
      <header className="panel-header">
        <span className="eyebrow lime">// 다음 거래일 주문 시트</span>
        <small>장 마감 30분 전 제출 · {[...sells, ...buys].some((r) => r.moc) ? "만료·손절은 MOC, 나머지 LOC" : "전부 LOC"}</small>
        <div className="ladder-summary">
          <span className="sell">매도 {sells.length}건 · {qtyText(sellQty)}주</span>
          <span className="buy">매수 {buys.length}건 · {qtyText(buyQty)}주</span>
        </div>
        {action}
      </header>

      {sells.length + buys.length === 0 ? (
        <div className="compact-empty">예정된 주문이 없습니다.</div>
      ) : (
        <div className="ladder">
          <div className="ladder-head" aria-hidden>
            <span className="ladder-tag">구분</span>
            <span className="ladder-price">주문가</span>
            <span className="ladder-dist">종가대비</span>
            <span className="ladder-qty">수량</span>
            <span className="ladder-note">근거</span>
          </div>

          <ul className="ladder-side sell" aria-label="매도 주문">
            {sells.map((row) => (
              <Row key={row.key} row={row} maxDistance={maxDistance} highlighted={isHighlighted(row)} flashNonce={flash?.key === row.key ? flash.nonce : null} onEnter={() => setHovered({ side: "sell", rank: row.rank })} onCopy={() => copy(row)} />
            ))}
            {!sells.length && <li className="ladder-empty">매도 주문 없음</li>}
          </ul>

          <div className="ladder-spread">
            <span className="spread-label">최근 종가</span>
            <strong>{state.prev_close != null ? `$${state.prev_close.toFixed(2)}` : "—"}</strong>
            <span className="spread-date">{state.last_date}</span>
          </div>

          <ul className="ladder-side buy" aria-label="매수 주문">
            {buys.map((row) => (
              <Row key={row.key} row={row} maxDistance={maxDistance} highlighted={isHighlighted(row)} flashNonce={flash?.key === row.key ? flash.nonce : null} onEnter={() => setHovered({ side: "buy", rank: row.rank })} onCopy={() => copy(row)} />
            ))}
            {!buys.length && <li className="ladder-empty">매수 주문 없음</li>}
          </ul>
        </div>
      )}
      <footer>{footer}</footer>
    </section>
  );
}
