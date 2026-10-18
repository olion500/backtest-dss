import type { ReactNode } from "react";

import type { CellValue, OrderBookPayload, TablePayload, TraderCheck, TraderStatus } from "../types";

/* 오더북 맨 위 자동매매 영역 (시안 B): 오늘 걸린 주문의 가격 지도 + 하루 타임라인.
   상태는 trader 실행 기록(var/trader)에서 오고, 주문 전송·자동 손절 단계는 아직 없다. */

const STATUS_TEXT: Record<string, string> = { ok: "✓ 일치", halted: "✗ 멈춤", closed: "· 휴장", error: "! 오류" };
const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

function num(value: CellValue | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function offsetOf(iso: string | null) {
  const match = iso?.match(/([+-]\d\d:\d\d)$/);
  return match ? match[1] : "-05:00";
}

/** "09:40" NY on the session date → Date (the session's own UTC offset, so DST is right). */
function nyAt(date: string, hhmm: string, offset: string) {
  return new Date(`${date}T${hhmm}:00${offset}`);
}

function clock(at: Date, timeZone: string) {
  return new Intl.DateTimeFormat("ko-KR", { timeZone, hour: "2-digit", minute: "2-digit", hour12: false }).format(at);
}

function sessionTitle(date: string) {
  const [y, m, d] = date.split("-").map(Number);
  const weekday = WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
  return `${m}월 ${d}일 (${weekday})`;
}

interface Mark {
  price: number;
  label: string;
  kind: "sell" | "buy" | "stop";
}

function priceMap(orders: TablePayload, tradeLog: TablePayload, prevClose: number | null) {
  const marks: Mark[] = [];
  const ladder: number[] = [];
  let mocQty = 0;
  for (const row of orders.rows) {
    const kind = String(row["구분"] ?? "");
    const price = num(row["주문가"]);
    const qty = num(row["수량"]) ?? 0;
    if (price == null) continue;
    if (kind === "매도 (MOC)" || kind === "매도 (손절)") {
      mocQty += qty;
    } else if (kind.startsWith("매도")) {
      marks.push({ price, label: `${kind.match(/\((.+)\)/)?.[1] ?? "매도"} ${qty}`, kind: "sell" });
    } else if (kind.startsWith("매수 (+")) {
      ladder.push(price);
    } else {
      marks.push({ price, label: `${kind.includes("퉁치기") ? "퉁치기" : "매수"} ${qty}`, kind: "buy" });
    }
  }
  for (const row of tradeLog.rows) {
    const stop = num(row["SL목표가"]);
    if (row["상태"] !== "완료" && stop != null) marks.push({ price: stop, label: "", kind: "stop" });
  }
  const prices = [...marks.map((m) => m.price), ...ladder, ...(prevClose != null ? [prevClose] : [])];
  if (!prices.length) return null;
  const lo = Math.min(...prices) * 0.97;
  const hi = Math.max(...prices) * 1.03;
  const at = (price: number) => ((price - lo) / (hi - lo)) * 100;
  // 매도 라벨은 축 위, 매수 라벨은 축 아래. 같은 쪽 두 줄 중 비어 있는 줄에 놓는다
  const LABEL_GAP = 9; // % of the track a label needs
  const lastLeft: Record<string, number[]> = { sell: [-100, -100], buy: [-100, -100] };
  const placed = marks
    .filter((m) => m.kind !== "stop")
    .sort((a, b) => a.price - b.price)
    .map((m) => {
      const left = at(m.price);
      const lanes = lastLeft[m.kind];
      const free = lanes.findIndex((prev) => left - prev >= LABEL_GAP);
      const lane = free >= 0 ? free : lanes[0] <= lanes[1] ? 0 : 1;
      lanes[lane] = left;
      return { ...m, left, lane };
    });
  const stops = marks.filter((m) => m.kind === "stop").map((m) => ({ ...m, left: at(m.price) }));
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  return {
    placed,
    stops,
    ladder: ladder.length ? { from: at(Math.min(...ladder)), to: at(Math.max(...ladder)), count: ladder.length, low: Math.min(...ladder), high: Math.max(...ladder) } : null,
    close: prevClose != null ? at(prevClose) : null,
    ticks,
    mocQty,
  };
}

interface StopLot {
  date: string;
  qty: number;
  stop: number;
  gap: number | null;
}

function stopLots(tradeLog: TablePayload, prevClose: number | null): StopLot[] {
  return tradeLog.rows.flatMap((row) => {
    const stop = num(row["SL목표가"]);
    if (row["상태"] === "완료" || stop == null) return [];
    return [{
      date: String(row["매수일자"] ?? "").slice(5),
      qty: num(row["매수수량"]) ?? 0,
      stop,
      gap: prevClose ? (prevClose / stop - 1) * 100 : null,
    }];
  }).sort((a, b) => (a.gap ?? 0) - (b.gap ?? 0));
}

function CheckTable({ checks }: { checks: TraderCheck[] }) {
  return (
    <div className="at-checks" role="table" aria-label="장부 대조">
      <div className="at-check head" role="row"><span role="columnheader">항목</span><span role="columnheader">엔진</span><span role="columnheader">증권사 실제</span><span role="columnheader">차이</span><span role="columnheader">판정</span></div>
      {checks.map((c) => (
        <div key={c.key} role="row" className={`at-check ${c.ok === false ? "bad" : c.ok ? "good" : "info"}`}>
          <span role="cell">{c.label}<small>{c.note}</small></span>
          <span role="cell">{c.expected}</span>
          <span role="cell">{c.actual}</span>
          <span role="cell">{c.diff || "—"}</span>
          <span role="cell">{c.ok === false ? "✗" : c.ok ? "✓" : "—"}</span>
        </div>
      ))}
    </div>
  );
}

type StepState = "done" | "failed" | "waiting" | "off";

function Step({ at, kst, state, title, badge, children }: { at: string; kst: string; state: StepState; title: string; badge?: string; children?: ReactNode }) {
  return (
    <li className={`at-step ${state}`}>
      <div className="at-time"><strong>{at}</strong><small>한국 {kst}</small></div>
      <div className="at-rail" aria-hidden="true"><i /><b /></div>
      <article className="at-card">
        <header><strong>{title}</strong>{badge ? <span className="at-badge">{badge}</span> : null}</header>
        {children}
      </article>
    </li>
  );
}

export function AutoTradePanel({
  status,
  error,
  orderBook,
  tradeLog,
  ticker,
}: {
  status: TraderStatus | null;
  error: string | null;
  orderBook: OrderBookPayload;
  tradeLog: TablePayload;
  ticker: string;
}) {
  const prevClose = orderBook.state.prev_close;
  const map = priceMap(orderBook.orders, tradeLog, prevClose);
  const lots = stopLots(tradeLog, prevClose);
  const session = status?.session;
  const offset = offsetOf(session?.close ?? null);
  const time = (iso: string | null, fallback: string) => {
    const at = iso ? new Date(iso) : session ? nyAt(session.date, fallback, offset) : null;
    return at ? { ny: clock(at, "America/New_York"), kst: clock(at, "Asia/Seoul") } : { ny: fallback, kst: "—" };
  };
  const checkAt = time(null, "09:40");
  const ordersAt = time(null, "09:45");
  const stopAt = time(session?.stop_check_at ?? null, "15:45");
  const closeAt = time(session?.close ?? null, "16:00");
  const check = status?.check ?? null;
  // Only this session's check may claim that today's orders are held back; an older one is history.
  const halted = check != null && check.current && (check.status === "halted" || check.status === "error");
  const mode = !status?.configured ? "수동" : status.auto_trade ? "자동 켜짐" : "자동 꺼짐";

  const checkState: StepState = !check ? "off" : !check.current ? "waiting" : check.status === "ok" || check.status === "closed" ? "done" : "failed";
  const sells = orderBook.orders.rows.filter((r) => String(r["구분"] ?? "").startsWith("매도")).length;
  const buys = orderBook.orders.rows.length - sells;

  return (
    <section className="at-panel" aria-label="자동매매">
      <header className="at-head">
        <div>
          <span className="eyebrow">자동매매 // AUTO TRADE</span>
          <h2>{session ? sessionTitle(session.date) : "—"} <small>{session && !session.is_today ? "다음 거래일" : "오늘"}{session?.early_close ? " · 조기 마감 13:00" : ""}</small></h2>
        </div>
        <span className={`at-mode ${status?.auto_trade ? "on" : ""}`}>{mode}{status?.broker ? ` · ${status.broker}` : ""}</span>
      </header>

      {error ? <p className="at-note bad">자동매매 상태를 불러오지 못했습니다: {error}</p> : null}

      {halted && check ? (
        <div className="at-alert" role="alert">
          <span className="eyebrow">{check.status === "error" ? "오류 // ERROR" : "멈춤 // HALTED"} · {check.session}{check.forced ? " 시험 실행" : ""}</span>
          <strong>{check.status === "error" ? "대조를 끝내지 못해 주문을 내지 않습니다" : "장부와 실제 잔고가 달라서 주문을 내지 않습니다"}</strong>
          <p>{check.message}</p>
        </div>
      ) : null}

      {map ? (
        <div className="at-map" aria-label="오늘 걸린 주문 가격 지도">
          <div className="at-map-legend">
            <span className="eyebrow">오늘 걸린 주문 // PRICE MAP</span>
            <span className="sell">■ 매도</span><span className="buy">■ 매수 · 사다리</span><span className="stop">▲ 손절가</span>
            {map.ladder ? <span className="buy">사다리 {map.ladder.count}건 ${map.ladder.high.toFixed(2)} → ${map.ladder.low.toFixed(2)}</span> : null}
            {map.mocQty ? <span className="moc">MOC 매도 {map.mocQty}주</span> : null}
            <span className="at-map-close">전일 종가 <b>${prevClose?.toFixed(2) ?? "—"}</b></span>
          </div>
          <div className="at-track">
            <i className="at-axis" aria-hidden="true" />
            {map.ladder ? <i className="at-ladder" aria-hidden="true" style={{ left: `${map.ladder.from}%`, width: `${Math.max(0.6, map.ladder.to - map.ladder.from)}%` }} /> : null}
            {map.placed.map((m, i) => (
              <span key={`m${i}`} className={`at-mark ${m.kind} lane${m.lane}`} style={{ left: `${m.left}%` }}>
                <i aria-hidden="true" /><em>{m.label}</em>
              </span>
            ))}
            {map.stops.map((m, i) => <i key={`s${i}`} className="at-stop" aria-hidden="true" style={{ left: `${m.left}%` }} />)}
            {map.close != null ? <span className="at-close" style={{ left: `${map.close}%` }}><i aria-hidden="true" /><em>${prevClose?.toFixed(2)}</em></span> : null}
          </div>
          <div className="at-ticks" aria-hidden="true">{map.ticks.map((t) => <span key={t}>${t.toFixed(0)}</span>)}</div>
        </div>
      ) : null}

      <ol className="at-timeline">
        <Step at={checkAt.ny} kst={checkAt.kst} state={checkState} title="전날 체결 대조" badge={check ? STATUS_TEXT[check.status] : "기록 없음"}>
          {check ? (
            <>
              {/* The table already says what differs; the message line only stands in when there is no table. */}
              {!check.checks.length || !check.current ? (
                <p className="at-note">
                  {!check.checks.length ? check.message : null}
                  {!check.checks.length && !check.current ? " · " : null}
                  {!check.current ? <b>최근 대조 {check.session}{check.forced ? " (시험 실행)" : ""}</b> : null}
                </p>
              ) : null}
              {check.checks.length ? <CheckTable checks={check.checks} /> : null}
            </>
          ) : (
            <p className="at-note">{status?.configured === false ? "이 계좌는 자동매매 설정이 없습니다 (config/trader)." : "아직 대조 기록이 없습니다."} <code>python -m trader check --account {status?.account ?? "<id>"}</code></p>
          )}
        </Step>
        <Step at={ordersAt.ny} kst={ordersAt.kst} state="off" title="LOC 주문 전송" badge="준비 중">
          <p className="at-note">지금은 주문표대로 직접 주문 · 매도 {sells}건 · 매수 {buys}건</p>
        </Step>
        <Step at={stopAt.ny} kst={stopAt.kst} state="off" title={session?.early_close ? "손절 확인 · 조기 마감" : "손절 확인"} badge="준비 중">
          {lots.length ? (
            <ul className="at-stops">
              {lots.map((lot) => (
                <li key={`${lot.date}-${lot.stop}`}>
                  <span>{lot.date} · {lot.qty}주</span>
                  <span className="stop">▲ ${lot.stop.toFixed(2)}</span>
                  <span className="at-gauge" aria-hidden="true"><i style={{ width: `${Math.min(100, Math.max(2, (lot.gap ?? 0) * 4))}%` }} className={lot.gap != null && lot.gap < 5 ? "near" : ""} /></span>
                  <span className={lot.gap != null && lot.gap < 5 ? "near" : ""}>{lot.gap == null ? "—" : `${lot.gap >= 0 ? "+" : ""}${lot.gap.toFixed(1)}%`}</span>
                </li>
              ))}
            </ul>
          ) : <p className="at-note">손절가가 있는 보유 물량이 없습니다.</p>}
          <p className="at-note">걸리면: 그 묶음이 든 익절·퉁치기 LOC를 정정 → 그 수량만 MOC 매도 (마감 {time(session?.order_cutoff ?? null, "15:50").ny}까지)</p>
        </Step>
        <Step at={closeAt.ny} kst={closeAt.kst} state="waiting" title="종가 · 체결">
          <p className="at-note">다음 거래일 대조에서 {ticker} 체결을 백테스트와 맞춰 봅니다.</p>
        </Step>
      </ol>
    </section>
  );
}
