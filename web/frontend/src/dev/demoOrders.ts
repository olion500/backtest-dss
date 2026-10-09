// 개발용: URL에 ?demo 가 있으면 오더북 주문 시트를 샘플 데이터로 바꿔 래더 렌더링을 확인한다.
import type { ViewerResult } from "../types";

export const isDemoMode = () => new URLSearchParams(window.location.search).has("demo");

const PREV_CLOSE = 141.93;

const ORDERS = [
  { 구분: "매도 (익절)", 주문가: 153.08, 수량: 100, 비고: "매수일: 2026-09-10, 매수가: $148.62" },
  { 구분: "매도 (익절)", 주문가: 149.15, 수량: 40, 비고: "매수일: 2026-09-12, 매수가: $144.81" },
  { 구분: "매도 (퉁치기)", 주문가: 148.5, 수량: 12, 비고: "종가 > $148.03 시 매수미체결 → 매도" },
  { 구분: "매도 (익절)", 주문가: 146.2, 수량: 25, 비고: "매수일: 2026-09-15, 매수가: $141.94" },
  { 구분: "매도 (MOC)", 주문가: PREV_CLOSE, 수량: 30, 비고: "보유기간 만료 → 가격 무관 종가 매도 (잔여일: 1일)" },
  { 구분: "매도 (손절)", 주문가: PREV_CLOSE, 수량: 25, 비고: "2026-10-08 종가 $141.93 ≤ 손절가 $142.10 → 백테스트는 매도 완료, 계좌는 MOC 매도 | 매수일: 2026-09-15" },
  { 구분: "매수", 주문가: 148.03, 수량: 100, 비고: "→ 익절: $153.08, 손절: $121.24 | 퉁치기 후 순매수 (30주 상쇄)" },
  { 구분: "매수 (+3주)", 주문가: 144.23, 수량: 3, 비고: "익절: $149.15, 손절: $118.12" },
  { 구분: "매수 (+6주)", 주문가: 140.15, 수량: 3, 비고: "익절: $144.93, 손절: $114.78" },
  { 구분: "매수 (+9주)", 주문가: 136.29, 수량: 3, 비고: "익절: $140.94, 손절: $111.62" },
  { 구분: "매수 (+12주)", 주문가: 132.64, 수량: 3, 비고: "익절: $137.16, 손절: $108.63" },
];

export function applyDemoOrders(result: ViewerResult): ViewerResult {
  if (!result.order_book) return result;
  return {
    ...result,
    order_book: {
      ...result.order_book,
      state: { ...result.order_book.state, prev_close: PREV_CLOSE, current_position_qty: 195 },
      orders: { columns: ["구분", "주문가", "수량", "비고"], rows: ORDERS },
      netting_message: "DEMO DATA · 매도 (MOC) 30주와 매수 100주가 퉁치기 대상입니다.",
    },
  };
}
