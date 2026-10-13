import { useState } from "react";

import type { VRAccount } from "../types";
import { AccountSwitcher } from "./AccountSwitcher";

const CYCLE_DAYS = 14;

/** G for the cycle running today: steps count from the VR start date, like the engine. */
function currentG(account: VRAccount, today = new Date()) {
  const origin = new Date(`${account.origin_date ?? account.start_date}T00:00:00`);
  const cycles = Math.max(0, Math.floor((today.getTime() - origin.getTime()) / 86_400_000 / CYCLE_DAYS));
  const steps = account.g_step_cycles > 0 ? Math.floor(cycles / account.g_step_cycles) : 0;
  return account.g + steps * account.g_step;
}

const usd = (value: number) => `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

function Row({ label, value }: { label: React.ReactNode; value: React.ReactNode }) {
  return (
    <div className="setting-row readonly-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Card({ title, children }: { title: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="setting-card">
      <div className="setting-card-title">{title}</div>
      <div className="setting-card-body">{children}</div>
    </section>
  );
}

/** Read-only view of config/vr/<id>.json: the server file is the source of truth. */
export function VrSettings({ accounts }: { accounts: VRAccount[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const account = accounts.find((item) => item.id === selectedId) ?? accounts[0];
  if (!account) {
    return <p className="setting-help">VR 계좌가 없습니다. config/vr/&lt;id&gt;.json을 추가하세요.</p>;
  }
  return (
    <>
      {accounts.length > 1 && <AccountSwitcher accounts={accounts} selected={account.id} disabled={false} onSelect={setSelectedId} label="VR 계좌 // ACCOUNT" />}
      <section className="local-settings-panel single">
        <div className="local-settings-copy">
          <span className="eyebrow">VR 설정 // config/vr/{account.id}.json · 보기 전용</span>
          <p>VR 계좌 설정은 서버 파일에서 읽습니다. 값을 바꾸려면 파일을 고친 뒤 다시 배포해야 합니다.</p>
        </div>
      </section>
      <section className="settings-card-grid">
        <Card title="계좌 // ACCOUNT">
          <Row label="이름" value={account.name} />
          <Row label="종목" value={account.ticker} />
          <Row label={<>VR 시작일 <small>주차·G 기준</small></>} value={account.origin_date ?? account.start_date} />
          <Row label={<>첫 사이클 <small>이어받기</small></>} value={account.start_date} />
          <Row label="시작 보유" value={`${account.start_shares.toLocaleString()}주`} />
          <Row label="시작 Pool" value={usd(account.start_pool)} />
          <Row label="시작 V" value={account.start_v != null ? usd(account.start_v) : "평가금"} />
          <Row label="투자원금" value={account.principal != null ? usd(account.principal) : "평가금 + Pool"} />
        </Card>
        <Card title="V 공식 // VALUE">
          <Row label="공식" value={account.advanced ? "실력공식 (VR 5.0)" : "기본공식"} />
          <Row label={<>시작 G <small>VR 시작일</small></>} value={account.g} />
          <Row label="지금 G" value={currentG(account)} />
          <Row label="G 증가" value={account.g_step_cycles > 0 ? `${account.g_step_cycles}사이클마다 +${account.g_step}` : "없음"} />
          <Row label="밴드" value={`±${account.band_pct}%`} />
        </Card>
        <Card title="적립 // DEPOSIT">
          <Row label="사이클당" value={`${account.deposit >= 0 ? "+" : "−"}${usd(Math.abs(account.deposit))}`} />
          {account.deposit_changes.length === 0 ? <Row label="변경" value="없음" /> : account.deposit_changes.map((change) => (
            <Row key={change.from} label={<>{change.from} <small>부터</small></>} value={usd(change.deposit)} />
          ))}
        </Card>
        <Card title="주문 // ORDERS">
          <Row label="주문 방식" value={account.order_type === "loc" ? "LOC (종가)" : "지정가 (저가·고가)"} />
          <Row label="매수 한도" value={`사이클 시작 Pool의 ${account.pool_limit_pct}%`} />
          <Row label="매수 묶음" value={`${account.buy_unit}주`} />
          <Row label="매도 단위" value={`${account.sell_unit}주`} />
          <Row label="매도 상한" value={`현재가 +${account.sell_cap_pct}%`} />
        </Card>
        <Card title="보정 // ADJUSTMENTS">
          {account.adjustments.length === 0 ? <Row label="기록" value="없음" /> : account.adjustments.map((item) => (
            <Row
              key={item.date}
              label={item.date}
              value={[
                item.shares != null ? `${item.shares}주` : null,
                item.pool != null ? `Pool ${usd(item.pool)}` : null,
                item.v != null ? `V ${usd(item.v)}` : null,
              ].filter(Boolean).join(" · ")}
            />
          ))}
        </Card>
      </section>
    </>
  );
}
