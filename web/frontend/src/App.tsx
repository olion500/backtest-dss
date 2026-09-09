import { useEffect, useState } from "react";

import { getDefaults, runViewer } from "./api/client";
import { DataTable } from "./components/DataTable";
import { EquityChart } from "./components/EquityChart";
import { MetricGrid } from "./components/MetricGrid";
import { OrderBookView } from "./components/OrderBookView";
import { RunToolbar } from "./components/RunToolbar";
import { SettingsPanel } from "./components/SettingsPanel";
import { StrategyStrip } from "./components/StrategyStrip";
import type { BacktestRequest, RunView, ViewerResult, ViewName } from "./types";

// v2: server defaults now merge config/personal_settings.json; the version
// bump discards v1 blobs saved with the old defaults (2022 start, 10k cash).
const STORAGE_KEY = "dongpa-viewer-settings-v2";

function loadLocalSettings(defaults: BacktestRequest) {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return defaults;
    const parsed = JSON.parse(saved) as BacktestRequest;
    return {
      ...defaults,
      ...parsed,
      // A stored end_date freezes the backtest at the day the settings were
      // saved; always follow the server's "today" instead.
      end_date: defaults.end_date,
      strategy: {
        ...defaults.strategy,
        ...parsed.strategy,
        defense: { ...defaults.strategy.defense, ...parsed.strategy?.defense },
        offense: { ...defaults.strategy.offense, ...parsed.strategy?.offense },
      },
    };
  } catch {
    return defaults;
  }
}

function latestMode(result?: ViewerResult) {
  if (result?.order_book) return result.order_book.state.current_mode === "offense" ? "공세" : "안전";
  const value = result?.journal.rows.at(-1)?.["모드"];
  return typeof value === "string" ? value : "—";
}

function EmptyResult({ label, onSettings }: { label: string; onSettings: () => void }) {
  return (
    <section className="result-empty">
      <span className="eyebrow">READY // NO RESULT</span>
      <strong>{label}</strong>
      <p>상단 실행 버튼을 누르면 현재 브라우저 설정으로 결과를 계산합니다.</p>
      <button className="button ghost" type="button" onClick={onSettings}>전략 설정 확인 →</button>
    </section>
  );
}

function LoadingResult() {
  return <div className="loading-state"><div className="loading-bar"><i /></div><span>YAHOO FINANCE DATA / ENGINE CALCULATION</span></div>;
}

function BacktestResult({ result }: { result: ViewerResult }) {
  const period = `${result.meta.start_date} – ${result.meta.end_date}`;
  return (
    <div className="result-stack">
      <MetricGrid summary={result.summary} realized={result.realized_metrics} target={result.meta.target_ticker} momentum={result.meta.momentum_ticker} period={period} />
      <EquityChart points={result.equity} modeBands={result.mode_bands} logScale={result.meta.log_scale} target={result.meta.target_ticker} />
      <DataTable title="일일 거래 요약" eyebrow="DAILY LOG" table={result.journal} filename={`dongpa_daily_${result.meta.target_ticker}.csv`} limit={8} />
      <DataTable title="트랜치별 매수·매도 기록" eyebrow="TRANCHES" table={result.trade_log} filename={`dongpa_trades_${result.meta.target_ticker}.csv`} limit={9} />
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<ViewName>(() => {
    const requested = window.location.hash.replace("#", "");
    return requested === "settings" || requested === "backtest" ? requested : "order-book";
  });
  const [defaults, setDefaults] = useState<BacktestRequest | null>(null);
  const [request, setRequest] = useState<BacktestRequest | null>(null);
  const [results, setResults] = useState<Partial<Record<RunView, ViewerResult>>>({});
  const [lastRuns, setLastRuns] = useState<Partial<Record<RunView, Date>>>({});
  const [running, setRunning] = useState<RunView | null>(null);
  const [errors, setErrors] = useState<Partial<Record<RunView, string>>>({});

  useEffect(() => {
    getDefaults()
      .then((value) => {
        setDefaults(value);
        setRequest(loadLocalSettings(value));
      })
      .catch((reason: Error) => setErrors({ backtest: reason.message }));
  }, []);

  useEffect(() => {
    if (request) localStorage.setItem(STORAGE_KEY, JSON.stringify(request));
  }, [request]);

  const run = async (targetView: RunView) => {
    if (!request) return;
    setRunning(targetView);
    setErrors((current) => ({ ...current, [targetView]: undefined }));
    try {
      const result = await runViewer(targetView, request);
      setResults((current) => ({ ...current, [targetView]: result }));
      setLastRuns((current) => ({ ...current, [targetView]: new Date() }));
    } catch (reason) {
      setErrors((current) => ({ ...current, [targetView]: reason instanceof Error ? reason.message : "알 수 없는 오류가 발생했습니다." }));
    } finally {
      setRunning(null);
    }
  };

  const ready = request != null;
  useEffect(() => {
    if (view === "order-book" && ready) void run("order-book");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, ready]);

  const navItems: Array<[ViewName, string]> = [
    ["order-book", "ORDER BOOK"],
    ["backtest", "BACKTEST"],
    ["settings", "SETTINGS"],
  ];

  if (!request) return <div className="boot-state">BOOTING VIEWER...</div>;

  const activeRunView = view === "settings" ? null : view;
  const activeResult = activeRunView ? results[activeRunView] : undefined;
  const activeError = activeRunView ? errors[activeRunView] : undefined;
  const isLoading = activeRunView != null && running === activeRunView;

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" type="button" onClick={() => setView("order-book")}>DONGPA<span>//</span></button>
        <nav aria-label="주요 화면">
          {navItems.slice(0, 2).map(([name, label]) => <button key={name} className={view === name ? "active" : ""} type="button" onClick={() => setView(name)}>{label}</button>)}
          <button type="button" className="disabled" disabled>OPTIMIZER</button>
          <button className={view === "settings" ? "active" : ""} type="button" onClick={() => setView("settings")}>SETTINGS</button>
        </nav>
        <div className="top-status"><span><i />API READY // VIEWER</span><b>browser_local.json</b><em>v0.4</em></div>
      </header>

      {view === "settings" ? (
        <SettingsPanel request={request} onChange={setRequest} onReset={() => defaults && setRequest(defaults)} onApply={() => setView("backtest")} />
      ) : view === "backtest" ? (
        <main className="screen">
          <RunToolbar request={request} currentMode={latestMode(results.backtest)} lastRun={lastRuns.backtest} loading={running === "backtest"} onChange={setRequest} onRun={() => run("backtest")} />
          <div className="page-body">
            {activeError && <div className="error-banner"><strong>REQUEST FAILED</strong><span>{activeError}</span></div>}
            {isLoading ? <LoadingResult /> : activeResult ? <BacktestResult result={activeResult} /> : <EmptyResult label="백테스트 결과가 없습니다." onSettings={() => setView("settings")} />}
            <footer className="footer"><span>DONGPA VIEWER / READ ONLY</span><span>모의 계산 결과 · 실제 주문은 사용자 책임</span></footer>
          </div>
        </main>
      ) : (
        <main className="screen">
          <StrategyStrip request={request} orderBook={results["order-book"]?.order_book} onSettings={() => setView("settings")} />
          <div className="page-body">
            {activeError && <div className="error-banner"><strong>REQUEST FAILED</strong><span>{activeError}</span></div>}
            {isLoading ? <LoadingResult /> : activeResult?.order_book ? <OrderBookView orderBook={activeResult.order_book} result={activeResult} request={request} /> : <EmptyResult label="오더북 결과가 없습니다." onSettings={() => setView("settings")} />}
            <footer className="footer"><span>DONGPA VIEWER / READ ONLY</span><span>모의 계산 결과 · 실제 주문은 사용자 책임</span></footer>
          </div>
        </main>
      )}
    </div>
  );
}
