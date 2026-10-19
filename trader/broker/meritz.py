"""Meritz Securities (메리츠증권) Open API adapter.

Rules taken from the official examples (github.com/meritz-securities/open-api, MIT):

- One app key = one account; account numbers are never sent (the gateway fills them).
- Errors come back as HTTP 500 too, so success is judged by ``rsp_cd``, never the status.
- A list response is empty when ``data`` is empty, not by ``rsp_cd`` (``5820`` can carry rows).
- ``5762`` means "more pages": resend with ``tr_cont=1`` and the ``tr_cont_key`` verbatim.
  Only some APIs really page (``ovs_holdings`` does, ``orders/today`` does not).
- Every call needs a 12-char ``mac_address`` header; each API allows 10 calls per second.
- Tokens last 12 hours and are reused; asking for a new one per call is wrong.

Orders: :meth:`submit` sends LOC/MOC/limit/market. Success is ``rsp_cd`` ``0001`` with
``warn_cls_code`` ``0``; any other warning means the order was not taken. A dropped
connection on an order is never retried (it may have been taken). :meth:`cancel` refuses.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Callable, Mapping
from zoneinfo import ZoneInfo

import requests

from trader.broker.base import (
    BrokerError,
    Cash,
    Order,
    OrderRequest,
    OrderState,
    OrderType,
    Position,
    Quote,
    Side,
)

PROD_URL = "https://openapi.imeritz.com:9443"
OK_CODES = frozenset({"0000", "0001", "5620", "5762", "5766", "5820", "5822"})
PAGE_MORE = "5762"
RATE_LIMITED = "EGW00200"
MIN_INTERVAL = 0.11  # seconds between calls to the same API (limit: 10 per second)
MAX_PAGES = 50

# Neutral exchange name -> (vendor id used by accounts/orders, market code used by quotes)
EXCHANGES = {
    "NASDAQ": ("0537", "OQ"),
    "NYSE": ("0321", "NY"),
    "AMEX": ("0066", "AX"),  # NYSE American and NYSE Arca (SOXL, most leveraged ETFs)
}

US_MARKET = "01"  # ovrs_stck_mrkt_tr_cls_code for the US in market-hours

ORDER_ACCEPTED = "0001"
# oder_cls_code, oder_cond_cls_code for new orders
_ORDER_CODES = {
    OrderType.LIMIT: ("01", "0"),
    OrderType.MARKET: ("05", "0"),
    OrderType.LOC: ("32", "7"),
    OrderType.MOC: ("34", "7"),
}

# aspr_type_code in order listings
_ORDER_TYPES = {"1": OrderType.MARKET, "2": OrderType.LIMIT, "5": OrderType.LOC, "7": OrderType.MOC}
_SIDES = {"1": Side.SELL, "2": Side.BUY}


def mac_address() -> str:
    node = uuid.getnode()
    return "".join(f"{(node >> shift) & 0xFF:02X}" for shift in range(40, -1, -8))


def _dec(value: object, default: Decimal | None = None) -> Decimal | None:
    if value in (None, ""):
        return default
    try:
        return Decimal(str(value).strip())
    except InvalidOperation:
        return default


class MeritzBroker:
    name = "meritz"
    required_credentials = ("APP_KEY", "APP_SECRET")  # read as <credentials_env>_APP_KEY …

    def __init__(
        self,
        credentials: Mapping[str, str],
        listings: Mapping[str, str],
        base_url: str = PROD_URL,
        http: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 15,
    ):
        missing = [key for key in self.required_credentials if not credentials.get(key)]
        if missing:
            raise BrokerError(f"메리츠 자격증명이 없습니다: {', '.join(missing)}", code="NO_CREDENTIALS")
        self._key = credentials["APP_KEY"]
        self._secret = credentials["APP_SECRET"]
        self._listings = dict(listings)
        self._base = base_url.rstrip("/")
        self._http = http or requests.Session()
        self._clock = clock
        self._sleep = sleep
        self._timeout = timeout
        self._token: str | None = None
        self._token_expires = 0.0
        self._last_call: dict[str, float] = {}

    # --- transport ---

    def _access_token(self) -> str:
        if self._token and self._clock() < self._token_expires:
            return self._token
        try:
            response = self._http.post(
                f"{self._base}/oauth2/token",
                data={"grant_type": "client_credentials", "client_id": self._key,
                      "client_secret": self._secret, "scope": "login"},
                headers={"content-type": "application/x-www-form-urlencoded"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise BrokerError(f"메리츠 토큰 서버에 연결하지 못했습니다: {exc}", code="UNREACHABLE", retryable=True) from None
        body = _json(response)
        if response.status_code != 200 or "access_token" not in body:
            code = body.get("rsp_cd") or body.get("error")
            message = body.get("rsp_msg") or body.get("error_description") or ""
            # Never echo the body: it may hold the key the gateway rejected.
            raise BrokerError(f"메리츠 토큰 발급 실패 (HTTP {response.status_code}) {code or ''} {message}".strip(), code=code)
        self._token = body["access_token"]
        self._token_expires = self._clock() + int(body.get("expires_in", 43200)) - 60
        return self._token

    def _throttle(self, path: str) -> None:
        last = self._last_call.get(path)
        if last is not None:
            wait = MIN_INTERVAL - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_call[path] = self._clock()

    def _get(self, path: str, params: Mapping[str, str] | None = None) -> dict:
        self._throttle(path)
        headers = {"authorization": f"Bearer {self._access_token()}", "mac_address": mac_address()}
        try:
            response = self._http.get(f"{self._base}{path}", params=dict(params or {}), headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BrokerError(f"메리츠 {path} 연결 실패: {exc}", code="UNREACHABLE", retryable=True) from None
        body = _json(response)
        code = body.get("rsp_cd")
        if code not in OK_CODES:
            detail = " / ".join(str(x) for x in (body.get("rsp_msg"), body.get("rsp_sub_msg")) if x)
            raise BrokerError(
                f"메리츠 {path} 실패 (HTTP {response.status_code}, {code}) {detail}".strip(),
                code=code,
                retryable=code == RATE_LIMITED,
            )
        return body

    def _post_order(self, path: str, body: Mapping[str, object]) -> dict:
        self._throttle(path)
        headers = {
            "authorization": f"Bearer {self._access_token()}",
            "mac_address": mac_address(),
            "content-type": "application/json; charset=utf-8",
        }
        try:
            response = self._http.post(f"{self._base}{path}", json=dict(body), headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            # The order may have been taken: never resend blind, look at today's orders first.
            raise BrokerError(f"메리츠 주문 응답을 못 받았습니다 (접수됐을 수 있음, 당일 주문 확인): {exc}", code="UNKNOWN_OUTCOME") from None
        payload = _json(response)
        code = payload.get("rsp_cd")
        if code != ORDER_ACCEPTED:
            detail = " / ".join(str(x) for x in (payload.get("rsp_msg"), payload.get("rsp_sub_msg")) if x)
            raise BrokerError(f"메리츠 주문 거절 (HTTP {response.status_code}, {code}) {detail}".strip(), code=code)
        return payload

    def _get_pages(self, path: str, params: Mapping[str, str] | None = None) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        request = dict(params or {})
        for _ in range(MAX_PAGES):
            body = self._get(path, request)
            rows.extend(_rows(body))
            if body.get("rsp_cd") != PAGE_MORE:
                return rows
            key = body.get("tr_cont_key")  # sent back verbatim: trailing spaces matter
            if not key or key in seen:
                break
            seen.add(key)
            request = dict(params or {}, tr_cont="1", tr_cont_key=key)
        raise BrokerError(f"메리츠 {path} 연속조회가 끝나지 않았습니다 ({len(rows)}건에서 중단)", code="TRUNCATED")

    def _exchange(self, symbol: str) -> tuple[str, str]:
        name = self._listings.get(symbol)
        if name not in EXCHANGES:
            raise BrokerError(f"{symbol}의 거래소를 모릅니다 (설정 listings에 NASDAQ/NYSE/AMEX 중 하나를 적으세요)", code="NO_LISTING")
        return EXCHANGES[name]

    # --- reads ---

    def positions(self) -> list[Position]:
        out = []
        for row in self._get_pages("/accounts/v1/me/overseas/holdings"):
            qty = _dec(row.get("rmnd_dqty"), Decimal(0))
            if not qty:
                continue
            out.append(Position(
                symbol=str(row.get("ovrs_stck_symb_iscd") or "").strip(),
                qty=qty,
                sellable_qty=_dec(row.get("seln_able_dqty"), Decimal(0)),
                avg_price=_dec(row.get("avrg_unpr"), Decimal(0)),
                last_price=_dec(row.get("psnt_prc")),
            ))
        return out

    def cash(self, currency: str = "USD") -> Cash:
        for row in _rows(self._get("/accounts/v1/me/overseas/currencies")):
            if str(row.get("crcd") or "").strip() == currency:
                return Cash(
                    currency=currency,
                    settled=_dec(row.get("frcr_amt"), Decimal(0)),
                    estimated=_dec(row.get("estm_frcr_amt"), Decimal(0)),
                )
        return Cash(currency=currency, settled=Decimal(0), estimated=Decimal(0))

    def quote(self, symbol: str) -> Quote:
        _, market = self._exchange(symbol)
        rows = _rows(self._get("/market/v1/overseas/prices", {
            "mrkt_div_code": "OV", "mrkt_cls_code": market, "iscd": symbol, "dely_rltm_cls_code": "0",
        }))
        last = _dec(rows[0].get("prpr")) if rows else None
        if not last:
            raise BrokerError(f"메리츠 {symbol} 현재가가 비어 있습니다", code="NO_QUOTE")
        return Quote(symbol=symbol, last=last, prev_close=_dec(rows[0].get("prdy_clpr")), at=datetime.now(ZoneInfo("America/New_York")))

    def market_open(self) -> bool:
        # Not market-days: its bsop_date already reads the next day during the US session
        # (Mon 11:55 NY gave Tuesday), so it cannot say whether today trades.
        rows = _rows(self._get("/reference/v1/overseas/market-hours", {"ovrs_stck_mrkt_tr_cls_code": US_MARKET}))
        if not rows:
            raise BrokerError("메리츠 장운영 응답이 비어 있습니다", code="BAD_RESPONSE")
        # Times are "0000" while the market is closed (KST HHMM otherwise).
        return str(rows[0].get("mrkt_begn_time") or "0000") != "0000"

    def orders_today(self) -> list[Order]:
        body = self._get("/trading/v1/overseas/orders/today", {"cntg_cls_code": "0", "seln_byov_clcd": "0"})
        if body.get("rsp_cd") == PAGE_MORE:
            # This API does not page; a partial list must not pass for the whole day.
            raise BrokerError("메리츠 당일 주문 목록이 한 번에 다 오지 않았습니다", code="TRUNCATED")
        return [_order(row) for row in _rows(body) if str(row.get("oder_rctf_cncl_cls_code", "0")) == "0"]

    # --- orders ---

    def submit(self, order: OrderRequest) -> Order:
        vendor, _ = self._exchange(order.symbol)
        if order.qty <= 0:
            raise BrokerError(f"주문 수량이 0 이하입니다: {order.qty}", code="BAD_ORDER")
        if order.type is OrderType.MOC and order.side is not Side.SELL:
            raise BrokerError("메리츠 MOC는 매도만 됩니다", code="BAD_ORDER")
        market = order.type in (OrderType.MOC, OrderType.MARKET)
        if not market and (order.price is None or order.price <= 0):
            raise BrokerError(f"{order.type.value} 주문에 가격이 없습니다", code="BAD_ORDER")
        cls_code, cond_code = _ORDER_CODES[order.type]
        side = "sell" if order.side is Side.SELL else "buy"
        payload = self._post_order(f"/trading/v1/overseas/orders/{side}", {
            "ovrs_stck_symb_iscd": order.symbol,
            "ovrs_exch_vndr_idno": vendor,
            "oder_cls_code": cls_code,
            "oder_cond_cls_code": cond_code,
            "odqt": order.qty,
            "ovrs_stck_oder_prc": 0 if market else float(order.price),
        })
        rows = _rows(payload)
        data = rows[0] if rows else {}
        warn = str(data.get("warn_cls_code") or "0").strip()
        if warn != "0":
            # 1 warning / 3 PTP tax notice need warn_cnfr_yn=Y; 2 is on hold. Either way: not taken.
            raise BrokerError(f"메리츠 주문 미접수 (경고 {warn}) {str(data.get('warn_msg') or '').strip()}".strip(), code=f"WARN_{warn}")
        number = _dec(data.get("oder_no"))
        if number is None:
            raise BrokerError("메리츠 주문번호가 없습니다 (접수됐을 수 있음, 당일 주문 확인)", code="UNKNOWN_OUTCOME")
        return Order(
            order_id=str(int(number)),
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            qty=Decimal(order.qty),
            price=None if market else order.price,
            filled_qty=Decimal(0),
            fill_price=None,
            state=OrderState.OPEN,
        )

    def cancel(self, order_id: str) -> None:
        raise BrokerError("주문 취소는 아직 켜지 않았습니다", code="NOT_ENABLED")


def _json(response: requests.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _rows(body: dict) -> list[dict]:
    data = body.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return [data] if isinstance(data, dict) and data else []


def _order(row: dict) -> Order:
    qty = _dec(row.get("odqt"), Decimal(0))
    filled = _dec(row.get("cntg_qty"), Decimal(0))
    cancelled = _dec(row.get("cncl_cnfr_qty"), Decimal(0))
    status = str(row.get("ovrs_stck_oder_stat_cls_code") or "0")
    if status in ("1", "2", "4"):
        state = OrderState.REJECTED
    elif filled >= qty > 0:
        state = OrderState.FILLED
    elif cancelled > 0 or status == "3":
        state = OrderState.CANCELLED
    elif filled > 0:
        state = OrderState.PARTIAL
    else:
        state = OrderState.OPEN
    order_type = _ORDER_TYPES.get(str(row.get("aspr_type_code") or ""), OrderType.LIMIT)
    return Order(
        order_id=str(row.get("oder_no") or ""),
        symbol=str(row.get("ovrs_stck_symb_iscd") or "").strip(),
        side=_SIDES.get(str(row.get("seln_byov_clcd") or ""), Side.BUY),
        type=order_type,
        qty=qty,
        price=None if order_type in (OrderType.MOC, OrderType.MARKET) else _dec(row.get("ovrs_stck_oder_prc")),
        filled_qty=filled,
        fill_price=_dec(row.get("cntg_crnc_prc")) if filled else None,
        state=state,
        reason=str(row.get("ovrs_stck_rfus_reas") or "").strip(),
    )
