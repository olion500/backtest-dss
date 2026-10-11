import type { Account } from "../types";

export function AccountSwitcher({
  accounts,
  selected,
  disabled,
  onSelect,
}: {
  accounts: Array<Pick<Account, "id" | "name">>;
  selected: string;
  disabled: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="account-switcher" role="tablist" aria-label="계좌 선택">
      <span className="account-switcher-label">계좌 // ACCOUNT</span>
      {accounts.map((account) => (
        <button
          key={account.id}
          type="button"
          role="tab"
          aria-selected={account.id === selected}
          className={account.id === selected ? "active" : ""}
          disabled={disabled && account.id !== selected}
          onClick={() => onSelect(account.id)}
        >
          {account.name}
        </button>
      ))}
    </div>
  );
}
