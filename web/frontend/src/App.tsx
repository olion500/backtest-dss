import { useEffect, useMemo, useState } from "react";

import { getAccounts, getDefaults, getPresets, getVrAccounts, runViewer, runVr } from "./api/client";
import { matchPreset, sameStrategy } from "./presets";
import { DailyLog } from "./components/DailyLog";
import { DataTable } from "./components/DataTable";
import { EquityChart } from "./components/EquityChart";
import { MetricGrid } from "./components/MetricGrid";
import { OrderBookView } from "./components/OrderBookView";
import { RunToolbar } from "./components/RunToolbar";
import { SettingsPanel } from "./components/SettingsPanel";
import { AccountSwitcher } from "./components/AccountSwitcher";
import { StrategyStrip } from "./components/StrategyStrip";
import { VrView } from "./components/VrView";
import { applyDemoOrders, isDemoMode } from "./dev/demoOrders";
import type { Account, BacktestRequest, Preset, RunView, ViewerResult, ViewName, VRAccount, VRResult } from "./types";

// v2: server defaults merge config/strategy.json with the first account; the
// version bump discarded v1 blobs saved with the old defaults (2022 start, 10k cash).
const STORAGE_KEY = "dongpa-viewer-settings-v2";

/** The order book runs the shared strategy with the account's own start, cash and steps. */
function withAccount(request: BacktestRequest, account?: Account): BacktestRequest {
  if (!account) return request;
  return {
    ...request,
    start_date: account.start_date,
    // The server's "today" is UTC, so an account starting today (KST) can be
    // ahead of it; the order book then shows that account's first orders.
    end_date: request.end_date < account.start_date ? account.start_date : request.end_date,
    initial_cash: account.initial_cash,
    spread_buy_levels: account.spread_buy_levels,
    spread_buy_step: account.spread_buy_step,
  };
}

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
      <MetricGrid result={result} period={period} />
      <EquityChart points={result.equity} modeBands={result.mode_bands} logScale={result.meta.log_scale} target={result.meta.target_ticker} />
      <DailyLog table={result.journal} filename={`dongpa_daily_${result.meta.target_ticker}.csv`} />
      <DataTable title="트랜치별 매수·매도 기록" eyebrow="TRANCHES" table={result.trade_log} filename={`dongpa_trades_${result.meta.target_ticker}.csv`} limit={9} />
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<ViewName>(() => {
    const requested = window.location.hash.replace("#", "");
    return requested === "settings" || requested === "backtest" || requested === "vr" ? requested : "order-book";
  });
  const [defaults, setDefaults] = useState<BacktestRequest | null>(null);
  const [request, setRequest] = useState<BacktestRequest | null>(null);
  const [results, setResults] = useState<Partial<Record<RunView, ViewerResult>>>({});
  const [lastRuns, setLastRuns] = useState<Partial<Record<RunView, Date>>>({});
  const [running, setRunning] = useState<RunView | null>(null);
  const [errors, setErrors] = useState<Partial<Record<RunView, string>>>({});
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [accountsLoaded, setAccountsLoaded] = useState(false);
  // The page always opens on the first account (by order); a switch lasts until reload.
  const [accountId, setAccountId] = useState<string | null>(null);
  const [vrAccounts, setVrAccounts] = useState<VRAccount[]>([]);
  const [vrAccountId, setVrAccountId] = useState<string | null>(null);
  const [vrResult, setVrResult] = useState<VRResult | null>(null);
  const [vrError, setVrError] = useState<string | undefined>();
  const [vrLoading, setVrLoading] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => setPresets([]));
  }, []);

  useEffect(() => {
    // Without accounts the order book falls back to the browser settings.
    getAccounts()
      .then(setAccounts)
      .catch(() => setAccounts([]))
      .finally(() => setAccountsLoaded(true));
  }, []);

  useEffect(() => {
    getVrAccounts().then(setVrAccounts).catch(() => setVrAccounts([]));
  }, []);

  const vrAccount = vrAccounts.find((item) => item.id === vrAccountId) ?? vrAccounts[0];
  useEffect(() => {
    if (view !== "vr" || !vrAccount) return;
    let stale = false;
    setVrLoading(true);
    setVrError(undefined);
    runVr(vrAccount.id)
      .then((value) => { if (!stale) setVrResult(value); })
      .catch((reason: Error) => { if (!stale) setVrError(reason.message); })
      .finally(() => { if (!stale) setVrLoading(false); });
    return () => { stale = true; };
  }, [view, vrAccount?.id]);


  const account = accounts.find((item) => item.id === accountId) ?? accounts[0];
  const orderBookRequest = useMemo(() => (request ? withAccount(request, account) : null), [request, account]);


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
    const payload = targetView === "order-book" ? orderBookRequest : request;
    if (!payload) return;
    setRunning(targetView);
    setErrors((current) => ({ ...current, [targetView]: undefined }));
    try {
      const fetched = await runViewer(targetView, payload);
      const result = isDemoMode() ? applyDemoOrders(fetched) : fetched;
      setResults((current) => ({ ...current, [targetView]: result }));
      setLastRuns((current) => ({ ...current, [targetView]: new Date() }));
    } catch (reason) {
      setErrors((current) => ({ ...current, [targetView]: reason instanceof Error ? reason.message : "알 수 없는 오류가 발생했습니다." }));
    } finally {
      setRunning(null);
    }
  };

  // Wait for the accounts too, or the first order book run would use the browser settings.
  const ready = request != null && accountsLoaded;
  // Re-run when the strategy changes too (e.g. "운용 전략으로 되돌리기"), so the sheet never shows stale orders.
  const strategyKey = request ? JSON.stringify(request.strategy) : "";
  useEffect(() => {
    if (view === "order-book" && ready) void run("order-book");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, ready, account?.id, strategyKey]);

  const navItems: Array<[ViewName, string]> = [
    ["order-book", "ORDER BOOK"],
    ["backtest", "BACKTEST"],
    ["settings", "SETTINGS"],
  ];

  if (!request) return <div className="boot-state">BOOTING VIEWER...</div>;

  const livePreset = presets.find((preset) => preset.live);
  const offLive = livePreset != null && !sameStrategy(livePreset.strategy, request.strategy);
  const presetLabel = matchPreset(presets, request.strategy)?.name ?? "사용자 설정";
  const backToLive = () => livePreset && setRequest({ ...request, strategy: livePreset.strategy });

  const activeRunView = view === "settings" || view === "vr" ? null : view;
  const activeResult = activeRunView ? results[activeRunView] : undefined;
  const activeError = activeRunView ? errors[activeRunView] : undefined;
  const isLoading = activeRunView != null && running === activeRunView;

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" type="button" onClick={() => setView("order-book")}>DONGPA<span>//</span></button>
        <nav aria-label="주요 화면">
          {navItems.slice(0, 2).map(([name, label]) => <button key={name} className={view === name ? "active" : ""} type="button" onClick={() => setView(name)}>{label}</button>)}
          <button className={view === "vr" ? "active" : ""} type="button" onClick={() => setView("vr")}>VR</button>
          <button type="button" className="disabled" disabled>OPTIMIZER</button>
          <button className={view === "settings" ? "active" : ""} type="button" onClick={() => setView("settings")}>SETTINGS</button>
        </nav>
        <div className="top-status"><span><i />API READY // VIEWER</span><b>{presetLabel}</b><em>v0.4</em></div>
      </header>

      {view === "settings" ? (
        <SettingsPanel request={request} presets={presets} vrAccounts={vrAccounts} onChange={setRequest} onReset={() => defaults && setRequest(defaults)} onApply={() => setView("backtest")} />
      ) : view === "vr" ? (
        <main className="screen">
          {vrAccounts.length > 1 && vrAccount && <AccountSwitcher accounts={vrAccounts} selected={vrAccount.id} disabled={vrLoading} onSelect={setVrAccountId} />}
          <div className="page-body">
            {vrError && <div className="error-banner"><strong>REQUEST FAILED</strong><span>{vrError}</span></div>}
            {!vrAccount ? (
              <section className="result-empty">
                <span className="eyebrow">VR // NO ACCOUNT</span>
                <strong>VR 계좌가 없습니다.</strong>
                <p>config/vr/example.json.sample을 config/vr/&lt;id&gt;.json으로 복사해 시작 상태를 적으세요.</p>
              </section>
            ) : vrLoading ? <LoadingResult /> : vrResult && vrResult.account.id === vrAccount.id ? <VrView result={vrResult} /> : null}
          </div>
        </main>
      ) : view === "backtest" ? (
        <main className="screen">
          <RunToolbar request={request} presets={presets} currentMode={latestMode(results.backtest)} lastRun={lastRuns.backtest} loading={running === "backtest"} onChange={setRequest} onRun={() => run("backtest")} />
          <div className="page-body">
            {activeError && <div className="error-banner"><strong>REQUEST FAILED</strong><span>{activeError}</span></div>}
            {isLoading ? <LoadingResult /> : activeResult ? <BacktestResult result={activeResult} /> : <EmptyResult label="백테스트 결과가 없습니다." onSettings={() => setView("settings")} />}
            <footer className="footer"><span>DONGPA VIEWER / READ ONLY</span><span>모의 계산 결과 · 실제 주문은 사용자 책임</span></footer>
          </div>
        </main>
      ) : (
        <main className="screen">
          {accounts.length > 1 && account && <AccountSwitcher accounts={accounts} selected={account.id} disabled={running === "order-book"} onSelect={setAccountId} />}
          <StrategyStrip request={orderBookRequest ?? request} orderBook={results["order-book"]?.order_book} onSettings={() => setView("settings")} />
          <div className="page-body">
            {offLive && (
              <div className="error-banner preset-warning">
                <strong>운용 전략과 다름</strong>
                <span>이 주문표는 현재 운용 중인 전략(strategy.json)이 아니라 "{presetLabel}" 설정으로 계산했습니다.</span>
                <button className="button ghost" type="button" onClick={backToLive}>운용 전략으로 되돌리기</button>
              </div>
            )}
            {activeError && <div className="error-banner"><strong>REQUEST FAILED</strong><span>{activeError}</span></div>}
            {isLoading ? <LoadingResult /> : activeResult?.order_book ? <OrderBookView orderBook={activeResult.order_book} result={activeResult} request={orderBookRequest ?? request} accountId={account?.id} /> : <EmptyResult label="오더북 결과가 없습니다." onSettings={() => setView("settings")} />}
            <footer className="footer"><span>DONGPA VIEWER / READ ONLY</span><span>모의 계산 결과 · 실제 주문은 사용자 책임</span></footer>
          </div>
        </main>
      )}
    </div>
  );
}
