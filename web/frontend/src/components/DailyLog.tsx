import { Fragment, useMemo, useState } from "react";

import { cellText, downloadCsv } from "../csv";
import type { CellValue, TablePayload } from "../types";

// Spectrum UI의 audit-log 테이블 블록(필터 칩 · 정렬 · 페이지네이션 · 행 상세)을
// 참고해 일일 저널 전용으로 다시 만들었다.

type Row = Record<string, CellValue>;
type ModeFilter = "공세" | "안전" | null;
type DirectionFilter = "상승" | "하락" | null;
type SortDir = "asc" | "desc";

const PAGE_SIZE = 10;

interface Column {
  id: string;
  label: string;
  align?: "right";
  /** 정렬·비교에 쓰는 값 */
  value: (row: Row) => number | string | null;
  render: (row: Row) => React.ReactNode;
  /** 이 폭 미만에서는 숨긴다 (CSS 클래스) */
  hide?: "md" | "lg";
  /** 고정 열 폭(px). table-layout: fixed와 함께 페이지를 넘겨도 폭이 흔들리지 않게 한다 */
  width: number;
}

const num = (value: CellValue) => (typeof value === "number" && Number.isFinite(value) ? value : null);
const money = (value: number | null, digits = 0) => (value == null ? "—" : `$${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`);
const signedMoney = (value: number | null) => (value == null || value === 0 ? "—" : `${value > 0 ? "+" : "−"}$${Math.abs(value).toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
const signedPct = (value: number | null) => (value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(2)}%`);
const qty = (value: number | null) => (value == null ? "—" : value.toLocaleString("ko-KR", { maximumFractionDigits: 2 }));
const toneOf = (value: number | null) => (value == null || value === 0 ? "" : value > 0 ? "up" : "down");

function fill(qtyValue: CellValue, priceValue: CellValue) {
  const q = num(qtyValue);
  const p = num(priceValue);
  if (!q) return <span className="dim">—</span>;
  return <>{qty(q)}<i>@</i>{p == null ? "—" : p.toFixed(2)}</>;
}

const COLUMNS: Column[] = [
  { id: "date", label: "거래일자", width: 132, value: (r) => String(r["거래일자"] ?? ""), render: (r) => String(r["거래일자"] ?? "").slice(0, 10) },
  { id: "mode", label: "모드", width: 88, value: (r) => String(r["모드"] ?? ""), render: (r) => <span className={`mode-pill ${r["모드"] === "공세" ? "offense" : "defense"}`}><i />{String(r["모드"] ?? "—")}</span> },
  { id: "close", label: "종가", align: "right", width: 90, value: (r) => num(r["종가"]), render: (r) => num(r["종가"])?.toFixed(2) ?? "—" },
  { id: "chg", label: "등락률", align: "right", width: 96, value: (r) => num(r["등락률(%)"]), render: (r) => <b className={toneOf(num(r["등락률(%)"]))}>{signedPct(num(r["등락률(%)"]))}</b> },
  { id: "buy", label: "매수", align: "right", width: 124, value: (r) => num(r["매수수량"]), render: (r) => fill(r["매수수량"], r["매수체결가"]) },
  { id: "sell", label: "매도", align: "right", width: 124, value: (r) => num(r["매도수량"]), render: (r) => fill(r["매도수량"], r["매도평균"]) },
  { id: "pnl", label: "실현손익", align: "right", width: 104, value: (r) => num(r["실현손익"]), render: (r) => <b className={toneOf(num(r["실현손익"]))}>{signedMoney(num(r["실현손익"]))}</b> },
  { id: "pos", label: "보유", align: "right", hide: "md", width: 80, value: (r) => num(r["보유수량"]), render: (r) => qty(num(r["보유수량"])) },
  { id: "cash", label: "현금", align: "right", hide: "md", width: 104, value: (r) => num(r["현금"]), render: (r) => money(num(r["현금"])) },
  { id: "equity", label: "EQUITY", align: "right", width: 104, value: (r) => num(r["Equity"]), render: (r) => money(num(r["Equity"])) },
  { id: "dd", label: "낙폭", align: "right", hide: "lg", width: 90, value: (r) => num(r["낙폭(DD%)"]), render: (r) => { const v = num(r["낙폭(DD%)"]); return <span className={v != null && v < -5 ? "down" : ""}>{v == null ? "—" : `${v.toFixed(2)}%`}</span>; } },
];

// 행 상세에 보여줄 나머지 필드 (표 컬럼에 없는 것들)
const DETAIL_FIELDS: Array<[string, string]> = [
  ["매수조건(%)", "매수조건"], ["매수주문가", "매수주문가"], ["매수금액", "매수금액"], ["매수거래ID", "매수 ID"],
  ["매도금액", "매도금액"], ["매도거래ID목록", "매도 ID"], ["수수료", "수수료"], ["평가금액", "평가금액"], ["누적손익", "누적손익"],
  ["누적수익률(%)", "누적수익률"], ["일일트렌치예산", "트랜치 예산"], ["트렌치기반현금", "트랜치 기반현금"], ["TP평균(보유)", "TP 평균"],
];
const NETTING_FIELDS: Array<[string, string]> = [
  ["원매수수량", "원매수수량"], ["원매수금액", "원매수금액"], ["원매도수량", "원매도수량"], ["원매도금액", "원매도금액"], ["원매도평균", "원매도평균"],
];

function compare(a: number | string | null, b: number | string | null) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

function Chip({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button type="button" className={`log-chip${active ? " active" : ""}`} aria-pressed={active} onClick={onClick}>
      {label}<b>{count}</b>
    </button>
  );
}

export function DailyLog({ table, filename }: { table: TablePayload; filename: string }) {
  const [mode, setMode] = useState<ModeFilter>(null);
  const [direction, setDirection] = useState<DirectionFilter>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ id: string; dir: SortDir }>({ id: "date", dir: "desc" });
  const [page, setPage] = useState(0);
  const [openKey, setOpenKey] = useState<string | null>(null);

  const rows = table.rows;
  const counts = useMemo(() => ({
    공세: rows.filter((r) => r["모드"] === "공세").length,
    안전: rows.filter((r) => r["모드"] === "안전").length,
    상승: rows.filter((r) => (num(r["등락률(%)"]) ?? 0) > 0).length,
    하락: rows.filter((r) => (num(r["등락률(%)"]) ?? 0) < 0).length,
  }), [rows]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter((r) => {
      if (mode && r["모드"] !== mode) return false;
      const chg = num(r["등락률(%)"]) ?? 0;
      if (direction === "상승" && chg <= 0) return false;
      if (direction === "하락" && chg >= 0) return false;
      if (q && !`${r["거래일자"]} ${r["예약요약"] ?? ""} ${r["퉁치기상세"] ?? ""}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [rows, mode, direction, query]);

  const sorted = useMemo(() => {
    const column = COLUMNS.find((c) => c.id === sort.id) ?? COLUMNS[0];
    const out = [...filtered].sort((a, b) => compare(column.value(a), column.value(b)));
    return sort.dir === "desc" ? out.reverse() : out;
  }, [filtered, sort]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = sorted.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);
  const rangeStart = sorted.length ? safePage * PAGE_SIZE + 1 : 0;
  const rangeEnd = Math.min(sorted.length, (safePage + 1) * PAGE_SIZE);
  const filtering = mode != null || direction != null || query.trim() !== "";

  const toggleSort = (id: string) => {
    setPage(0);
    setSort((current) => (current.id === id ? { id, dir: current.dir === "asc" ? "desc" : "asc" } : { id, dir: id === "date" ? "desc" : "desc" }));
  };
  const pick = <T,>(setter: (fn: (current: T | null) => T | null) => void, value: T) => {
    setPage(0);
    setter((current) => (current === value ? null : value));
  };

  return (
    <section className="table-card daily-log">
      <header className="section-heading table-heading">
        <div>
          <span className="eyebrow">DAILY LOG</span>
          <h2>일일 거래 요약</h2>
        </div>
        <div className="log-toolbar">
          <div className="log-chips" role="group" aria-label="모드 필터">
            <Chip label="공세" count={counts.공세} active={mode === "공세"} onClick={() => pick(setMode, "공세")} />
            <Chip label="안전" count={counts.안전} active={mode === "안전"} onClick={() => pick(setMode, "안전")} />
          </div>
          <div className="log-chips" role="group" aria-label="등락 필터">
            <Chip label="상승" count={counts.상승} active={direction === "상승"} onClick={() => pick(setDirection, "상승")} />
            <Chip label="하락" count={counts.하락} active={direction === "하락"} onClick={() => pick(setDirection, "하락")} />
          </div>
          <input
            type="search"
            className="log-search"
            placeholder="날짜 · 예약 검색"
            value={query}
            onChange={(event) => { setQuery(event.target.value); setPage(0); }}
            aria-label="일일 거래 요약 검색"
          />
          <button type="button" className="log-download" onClick={() => downloadCsv(table, filename)} disabled={!rows.length}>CSV</button>
        </div>
      </header>

      <div className="table-scroll log-scroll">
        {pageRows.length ? (
          <table className="log-table">
            <thead>
              <tr>
                {COLUMNS.map((column) => {
                  const active = sort.id === column.id;
                  return (
                    <th
                      key={column.id}
                      className={`${column.align ?? ""} ${column.hide ? `hide-${column.hide}` : ""}${active ? " sorted" : ""}`}
                      aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                      style={{ width: column.width }}
                    >
                      <button type="button" onClick={() => toggleSort(column.id)}>
                        {column.label}<i>{active ? (sort.dir === "asc" ? "▲" : "▼") : "▾"}</i>
                      </button>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row) => {
                const key = String(row["거래일자"]);
                const open = openKey === key;
                const netted = row["퉁치기적용"] === true || row["퉁치기적용"] === "True";
                return (
                  <Fragment key={key}>
                    <tr className={`log-row${open ? " open" : ""}${netted ? " netted" : ""}`} onClick={() => setOpenKey(open ? null : key)} title="클릭하면 상세 보기">
                      {COLUMNS.map((column) => (
                        <td key={column.id} className={`${column.align ?? ""} ${column.hide ? `hide-${column.hide}` : ""}`}>{column.render(row)}</td>
                      ))}
                    </tr>
                    {open && (
                      <tr className="log-detail">
                        <td colSpan={COLUMNS.length}>
                          <div className="log-detail-grid">
                            {DETAIL_FIELDS.map(([field, label]) => (
                              <div key={field}><span>{label}</span><strong>{cellText(row[field])}</strong></div>
                            ))}
                            {netted && NETTING_FIELDS.map(([field, label]) => (
                              <div key={field} className="netting"><span>{label}</span><strong>{cellText(row[field])}</strong></div>
                            ))}
                          </div>
                          {netted && row["퉁치기상세"] ? <p className="log-detail-note"><span>퉁치기</span>{String(row["퉁치기상세"])}</p> : null}
                          {row["예약요약"] ? <p className="log-detail-note"><span>다음날 예약</span>{String(row["예약요약"])}</p> : null}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        ) : <div className="empty-table">{rows.length ? "조건에 맞는 날이 없습니다." : "NO DATA"}</div>}
      </div>

      <footer className="log-pager">
        <span>{rangeStart}–{rangeEnd} / {sorted.length}{filtering ? ` (전체 ${rows.length})` : ""}</span>
        <div>
          <button type="button" onClick={() => setPage(0)} disabled={safePage === 0} aria-label="첫 페이지">«</button>
          <button type="button" onClick={() => setPage(safePage - 1)} disabled={safePage === 0} aria-label="이전 페이지">‹</button>
          <b>{safePage + 1} / {pageCount}</b>
          <button type="button" onClick={() => setPage(safePage + 1)} disabled={safePage >= pageCount - 1} aria-label="다음 페이지">›</button>
          <button type="button" onClick={() => setPage(pageCount - 1)} disabled={safePage >= pageCount - 1} aria-label="마지막 페이지">»</button>
        </div>
      </footer>
    </section>
  );
}
