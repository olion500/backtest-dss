import type { CellValue, TablePayload } from "../types";

interface DataTableProps {
  title: string;
  eyebrow: string;
  table: TablePayload;
  filename: string;
  limit?: number;
  embedded?: boolean;
}

function cellText(value: CellValue) {
  if (value == null || value === "") return "—";
  if (typeof value === "number") return value.toLocaleString("ko-KR", { maximumFractionDigits: 4 });
  if (typeof value === "boolean") return value ? "ON" : "OFF";
  return String(value);
}

function downloadCsv(table: TablePayload, filename: string) {
  const escape = (value: CellValue) => `"${cellText(value).replaceAll('"', '""')}"`;
  const lines = [
    table.columns.map((column) => escape(column)).join(","),
    ...table.rows.map((row) => table.columns.map((column) => escape(row[column])).join(",")),
  ];
  const blob = new Blob(["\ufeff", lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function valueTone(value: CellValue) {
  const text = String(value ?? "");
  if (text.includes("공세") || text.includes("익절") || text.startsWith("매도")) return "tone-positive";
  if (text.includes("손절") || text.startsWith("매수")) return "tone-negative";
  if (text.includes("안전") || text.includes("보유중")) return "tone-muted";
  return "";
}

export function DataTable({ title, eyebrow, table, filename, limit = 100, embedded = false }: DataTableProps) {
  const visibleRows = limit > 0 ? table.rows.slice(-limit).reverse() : table.rows;

  return (
    <section className={`table-card${embedded ? " embedded" : ""}`}>
      <header className="section-heading table-heading">
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
        <div className="table-actions">
          <span>최근 {visibleRows.length.toLocaleString()}행 · 전체 {table.rows.length.toLocaleString()}행</span>
          <button type="button" onClick={() => downloadCsv(table, filename)} disabled={!table.rows.length}>DOWNLOAD CSV</button>
        </div>
      </header>
      <div className="table-scroll">
        {visibleRows.length ? (
          <table>
            <thead>
              <tr>{table.columns.map((column) => <th key={column}>{column}</th>)}</tr>
            </thead>
            <tbody>
              {visibleRows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {table.columns.map((column) => <td key={column} className={valueTone(row[column])}>{cellText(row[column])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div className="empty-table">NO DATA</div>}
      </div>
    </section>
  );
}
