import { ladderLabel, orderCount, orderPriceText, priceText, qtyText, stepSide, type PipOrder, type PipStep } from "./steps";

// 영상 PiP용 주문 카드. 영상 PiP 창에는 HTML을 넣을 수 없어서 canvas에 그린 뒤
// captureStream()으로 영상을 만든다. 넘기기 버튼은 PiP 창의 시스템 이전·다음 트랙 버튼이 맡는다.
const C = {
  bg: "#0a0a0a",
  fg: "#f5f5f5",
  muted: "#9a9a9a",
  border: "#222",
  lime: "#d4ff00",
  sell: "#4a8fff",
  buy: "#ff4747",
  sellSoft: "rgba(74, 143, 255, .18)",
  buySoft: "rgba(255, 71, 71, .18)",
};
const FONT = "Juache, Helvetica, Arial, sans-serif";

export const VIDEO_CARD_SIZE = { width: 720, height: 540 };

const font = (px: number) => `${Math.round(px)}px ${FONT}`;

/** target 크기로 그렸을 때 maxWidth를 넘으면 그만큼 줄인 글자 크기 */
function fitSize(ctx: CanvasRenderingContext2D, text: string, target: number, maxWidth: number) {
  ctx.font = font(target);
  const width = ctx.measureText(text).width;
  return width > maxWidth ? target * (maxWidth / width) : target;
}

function drawQty(ctx: CanvasRenderingContext2D, qty: string, size: number, x: number, y: number) {
  ctx.fillStyle = C.fg;
  ctx.font = font(size);
  ctx.fillText(qty, x, y);
  const width = ctx.measureText(qty).width;
  ctx.fillStyle = C.muted;
  ctx.font = font(size * 0.45);
  ctx.fillText("주", x + width + size * 0.05, y);
}

function drawSingle(ctx: CanvasRenderingContext2D, W: number, top: number, bottom: number, pad: number, order: PipOrder) {
  const price = orderPriceText(order);
  const qty = qtyText(order.qty);
  const H = bottom - top;
  const noteSize = Math.max(10, Math.min(H * 0.1, W * 0.045));
  // 설명 줄 자리는 설명이 없어도 비워 둬서 매수·매도의 숫자 위치를 맞춘다
  const area = H - noteSize * 1.6;
  const avail = W - pad * 2;
  const colGap = avail * 0.06;
  const priceSize = fitSize(ctx, price, area * 0.8, avail * 0.64);
  const qtySize = Math.min(priceSize, fitSize(ctx, `${qty}주`, priceSize, avail * 0.36 - colGap));
  const baseline = top + (area + priceSize * 0.72) / 2;
  ctx.textAlign = "left";
  ctx.fillStyle = C[order.side];
  ctx.font = font(priceSize);
  ctx.fillText(price, pad, baseline);
  drawQty(ctx, qty, qtySize, pad + avail * 0.64 + colGap, baseline);
  if (order.note) {
    ctx.fillStyle = C.muted;
    ctx.font = font(noteSize);
    ctx.fillText(order.note, pad, bottom - noteSize * 0.2, avail);
  }
}

function drawLadder(ctx: CanvasRenderingContext2D, W: number, top: number, bottom: number, pad: number, orders: PipOrder[]) {
  const rowH = (bottom - top) / orders.length;
  const size = Math.min(rowH * 0.62, W * 0.075);
  orders.forEach((order, k) => {
    const y = top + rowH * (k + 0.5) + size * 0.36;
    ctx.textAlign = "left";
    ctx.fillStyle = C.muted;
    ctx.font = font(size * 0.72);
    ctx.fillText(ladderLabel(order), pad, y);
    ctx.textAlign = "right";
    ctx.fillStyle = C.buy;
    ctx.font = font(size);
    ctx.fillText(priceText(order.price), W - pad - size * 3.3, y);
    ctx.fillStyle = C.muted;
    ctx.font = font(size * 0.8);
    ctx.fillText(`× ${qtyText(order.qty)}주`, W - pad, y);
    if (k < orders.length - 1) {
      ctx.fillStyle = C.border;
      ctx.fillRect(pad, top + rowH * (k + 1), W - pad * 2, 1);
    }
  });
  ctx.textAlign = "left";
}

function drawMessage(ctx: CanvasRenderingContext2D, W: number, H: number, title: string, sub: string) {
  ctx.textAlign = "center";
  ctx.fillStyle = C.lime;
  ctx.font = font(fitSize(ctx, title, H * 0.16, W * 0.88));
  ctx.fillText(title, W / 2, H * 0.52);
  ctx.fillStyle = C.muted;
  ctx.font = font(fitSize(ctx, sub, H * 0.07, W * 0.88));
  ctx.fillText(sub, W / 2, H * 0.68);
  ctx.textAlign = "left";
}

export function drawCard(ctx: CanvasRenderingContext2D, steps: PipStep[], index: number) {
  const { width: W, height: H } = ctx.canvas;
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.textBaseline = "alphabetic";
  if (!steps.length) return drawMessage(ctx, W, H, "주문 없음", "오늘 넣을 LOC 주문이 없습니다");
  if (index >= steps.length) return drawMessage(ctx, W, H, "오늘 주문 끝", `${orderCount(steps)}건 · 이전 버튼으로 돌아갈 수 있습니다`);

  const pad = Math.min(W, H) * 0.06;
  // 진행 막대
  const segH = Math.max(3, H * 0.022);
  const gap = Math.max(2, W * 0.006);
  const segW = (W - pad * 2 - gap * (steps.length - 1)) / steps.length;
  steps.forEach((step, k) => {
    ctx.globalAlpha = k < index ? 0.55 : 1;
    ctx.fillStyle = k === index ? C.lime : k < index ? C[stepSide(step)] : C.border;
    ctx.fillRect(pad + k * (segW + gap), pad, segW, segH);
  });
  ctx.globalAlpha = 1;

  // 구분 태그 + 순번
  const step = steps[index];
  const side = stepSide(step);
  const tag = step.kind === "single" ? step.order.label : `매수 스프레드 ${step.orders.length}건`;
  const metaSize = Math.min(H * 0.085, W * 0.06);
  const metaY = pad + segH + metaSize * 1.45;
  ctx.font = font(metaSize);
  ctx.fillStyle = side === "sell" ? C.sellSoft : C.buySoft;
  ctx.fillRect(pad, metaY - metaSize * 1.02, ctx.measureText(tag).width + metaSize * 0.8, metaSize * 1.35);
  ctx.fillStyle = C[side];
  ctx.fillText(tag, pad + metaSize * 0.4, metaY);
  ctx.fillStyle = C.muted;
  ctx.textAlign = "right";
  ctx.fillText(`${index + 1} / ${steps.length}`, W - pad, metaY);
  ctx.textAlign = "left";

  const top = metaY + metaSize * 0.7;
  if (step.kind === "single") drawSingle(ctx, W, top, H - pad, pad, step.order);
  else drawLadder(ctx, W, top, H - pad, pad, step.orders);
}
