import type { CellValue, OrderBookPayload } from "../types";

export type Side = "sell" | "buy";

export interface PipOrder {
  side: Side;
  label: string;
  price: number;
  qty: number;
  note: string;
  /** 만료·손절 매도: 지정가 없이 MOC(장마감 시장가)로 넣는다 */
  moc: boolean;
}

/** 한 번에 보여줄 단위. 1주짜리 스프레드 매수는 한 장으로 묶는다. */
export type PipStep = { kind: "single"; order: PipOrder } | { kind: "ladder"; orders: PipOrder[] };

/** engines/order_book_engine.py의 MOC_SELL·SL_MOC_SELL과 같은 값 */
const MOC_SELL = "매도 (MOC)";
const SL_MOC_SELL = "매도 (손절)";
export const isMocKind = (kind: string) => kind === MOC_SELL || kind === SL_MOC_SELL;

function num(value: CellValue) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function buildSteps(orders: OrderBookPayload["orders"]): PipStep[] {
  const parsed = orders.rows.flatMap((row) => {
    const kind = String(row["구분"] ?? "");
    const price = num(row["주문가"]);
    const qty = num(row["수량"]);
    if (price == null || qty == null || qty <= 0) return [];
    const side: Side = kind.startsWith("매도") ? "sell" : "buy";
    // "매도 (익절)" → "매도 익절", "매수 (+3주)" → "매수 +3주"
    const label = kind.replace(/\s*\((.+)\)/, " $1");
    // 매수 근거의 "→ 익절: $x, 손절: $y"는 입력과 무관해서 빼고, 현금 부족·퉁치기 같은 나머지 문구만 남긴다
    const note = String(row["비고"] ?? "").split(" | ").filter((part) => !part.startsWith("→ 익절")).join(" | ");
    return [{ side, label, price, qty, note, moc: isMocKind(kind), spread: /^매수 \(\+/.test(kind) }];
  });
  // MTS 입력 순서: 매도 → 매수, 각각 높은 가격부터
  const byPrice = (a: PipOrder, b: PipOrder) => b.price - a.price;
  const sells = parsed.filter((o) => o.side === "sell").sort(byPrice);
  const buys = parsed.filter((o) => o.side === "buy" && !o.spread).sort(byPrice);
  const spreads = parsed.filter((o) => o.spread).sort(byPrice);
  const steps: PipStep[] = [...sells, ...buys].map((order) => ({ kind: "single", order }));
  if (spreads.length) steps.push({ kind: "ladder", orders: spreads });
  return steps;
}

export function stepSide(step: PipStep): Side {
  return step.kind === "single" ? step.order.side : "buy";
}

export function orderCount(steps: PipStep[]) {
  return steps.reduce((sum, step) => sum + (step.kind === "single" ? 1 : step.orders.length), 0);
}

export const priceText = (value: number) => `$${value.toFixed(2)}`;
/** MOC 주문은 가격 칸에 지정가 대신 "MOC"를 보여준다. */
export const orderPriceText = (order: PipOrder) => (order.moc ? "MOC" : priceText(order.price));
export const qtyText = (value: number) => value.toLocaleString("ko-KR", { maximumFractionDigits: 4 });
/** 스프레드 묶음 안에서는 "매수"를 떼고 "+3주"만 보여준다. */
export const ladderLabel = (order: PipOrder) => order.label.replace(/^매수\s*/, "");
