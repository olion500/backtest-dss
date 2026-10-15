import type { Account, BacktestRequest, BenchmarkSummary, Preset, RunView, ViewerResult, VRAccount, VRResult } from "../types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message = data?.detail ?? `요청에 실패했습니다. (${response.status})`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return data as T;
}

export function getDefaults(): Promise<BacktestRequest> {
  return requestJson<BacktestRequest>("/api/v1/config/defaults");
}

export async function getAccounts(): Promise<Account[]> {
  const data = await requestJson<{ accounts: Account[] }>("/api/v1/accounts");
  return data.accounts;
}

export async function getPresets(): Promise<Preset[]> {
  const data = await requestJson<{ presets: Preset[] }>("/api/v1/presets");
  return data.presets;
}

export function runViewer(view: RunView, payload: BacktestRequest): Promise<ViewerResult> {
  const endpoint = view === "backtest"
    ? "/api/v1/backtests/run"
    : "/api/v1/order-book/preview";
  return requestJson<ViewerResult>(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getVrAccounts(): Promise<VRAccount[]> {
  const data = await requestJson<{ accounts: VRAccount[] }>("/api/v1/vr/accounts");
  return data.accounts;
}

export function runVr(accountId: string): Promise<VRResult> {
  return requestJson<VRResult>("/api/v1/vr/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account_id: accountId }),
  });
}

export function getBenchmarkSummary(): Promise<BenchmarkSummary> {
  return requestJson<BenchmarkSummary>("/api/v1/benchmarks/summary");
}
