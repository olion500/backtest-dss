"""Meritz adapter against a fake HTTP session: response judging, paging, parsing. No network."""
from decimal import Decimal

import pytest

from trader.broker import BROKERS, make_broker
from trader.broker.base import Broker, BrokerError, OrderRequest, OrderState, OrderType, Side
from trader.broker.meritz import MeritzBroker

CREDS = {"APP_KEY": "k", "APP_SECRET": "s"}
LISTINGS = {"SOXL": "AMEX"}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHttp:
    """Answers GETs from a {path: [responses...]} table, in order; counts token requests."""

    def __init__(self, routes, token_status=200):
        self.routes = {path: list(items) for path, items in routes.items()}
        self.gets = []
        self.posts = []
        self.token_calls = 0
        self.token_status = token_status

    def post(self, url, data=None, json=None, headers=None, timeout=None):
        if not url.endswith("/oauth2/token"):
            path = url.split(":9443", 1)[1]
            assert headers["authorization"] == "Bearer tok" and len(headers["mac_address"]) == 12
            self.posts.append((path, json))
            answer = self.routes[path].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.token_calls += 1
        if self.token_status != 200:
            return FakeResponse(self.token_status, {"rsp_cd": "EGW00103", "rsp_msg": "유효하지 않은 client_id"})
        return FakeResponse(200, {"access_token": "tok", "expires_in": 43200})

    def get(self, url, params=None, headers=None, timeout=None):
        path = url.split(":9443", 1)[1]
        assert headers["authorization"] == "Bearer tok"
        assert len(headers["mac_address"]) == 12
        self.gets.append((path, dict(params or {})))
        return self.routes[path].pop(0)


def broker(routes, **kwargs):
    http = FakeHttp(routes, **kwargs)
    clock = iter(range(0, 10_000)).__next__  # 1 s per tick: never throttles
    return MeritzBroker(CREDS, LISTINGS, http=http, clock=lambda: float(clock()), sleep=lambda s: None), http


def ok(data, code="5766", **extra):
    return FakeResponse(200, {"rsp_cd": code, "rsp_msg": "조회가 완료되었습니다.", "data": data, **extra})


def test_registry_builds_meritz_and_it_implements_the_protocol():
    assert "meritz" in BROKERS
    built = make_broker("meritz", CREDS, LISTINGS)
    assert isinstance(built, Broker)
    with pytest.raises(BrokerError) as err:
        make_broker("nope", CREDS, LISTINGS)
    assert err.value.code == "UNKNOWN_BROKER"


def test_missing_credentials_are_refused():
    with pytest.raises(BrokerError) as err:
        MeritzBroker({"APP_KEY": "k"}, LISTINGS)
    assert err.value.code == "NO_CREDENTIALS"


def test_positions_parse_decimal_strings_and_reuse_the_token():
    b, http = broker({"/accounts/v1/me/overseas/holdings": [ok([
        {"ovrs_stck_symb_iscd": "SOXL", "rmnd_dqty": "300.0000000000", "seln_able_dqty": "300.0000000000",
         "avrg_unpr": "131.000000", "psnt_prc": "93.670000"},
        {"ovrs_stck_symb_iscd": "OLD", "rmnd_dqty": "0.0000000000"},
    ])], "/accounts/v1/me/overseas/currencies": [ok([
        {"crcd": "JPY", "frcr_amt": "1", "estm_frcr_amt": "1"},
        {"crcd": "USD", "frcr_amt": "3500.000", "estm_frcr_amt": "12000.000"},
    ])]})
    [soxl] = b.positions()
    assert soxl.symbol == "SOXL" and soxl.qty == Decimal("300") and soxl.avg_price == Decimal("131.0000")
    cash = b.cash("USD")
    assert cash.settled == Decimal("3500.00") and cash.estimated == Decimal("12000.00")
    assert http.token_calls == 1


def test_error_comes_as_http_500_and_is_judged_by_rsp_cd():
    b, _ = broker({"/accounts/v1/me/overseas/currencies": [
        FakeResponse(500, {"rsp_cd": "0760", "rsp_msg": "권한이 없습니다.", "rsp_sub_msg": "포털에서 신청"}),
    ]})
    with pytest.raises(BrokerError) as err:
        b.cash()
    assert err.value.code == "0760" and "포털에서 신청" in str(err.value) and not err.value.retryable


def test_rate_limit_is_marked_retryable():
    b, _ = broker({"/accounts/v1/me/overseas/currencies": [FakeResponse(500, {"rsp_cd": "EGW00200"})]})
    with pytest.raises(BrokerError) as err:
        b.cash()
    assert err.value.retryable


def test_non_json_body_is_an_error_not_an_empty_success():
    b, _ = broker({"/accounts/v1/me/overseas/currencies": [FakeResponse(502, ValueError("html"))]})
    with pytest.raises(BrokerError):
        b.cash()


def test_token_failure_does_not_leak_the_body():
    b, _ = broker({}, token_status=403)
    with pytest.raises(BrokerError) as err:
        b.positions()
    assert err.value.code == "EGW00103" and "403" in str(err.value)


def test_holdings_follow_pages_with_the_key_verbatim():
    row = lambda sym: {"ovrs_stck_symb_iscd": sym, "rmnd_dqty": "1", "seln_able_dqty": "1", "avrg_unpr": "1"}
    b, http = broker({"/accounts/v1/me/overseas/holdings": [
        ok([row("A")], code="5762", tr_cont_key="K1  "),
        ok([row("B")]),
    ]})
    assert [p.symbol for p in b.positions()] == ["A", "B"]
    assert http.gets[1][1] == {"tr_cont": "1", "tr_cont_key": "K1  "}


def test_paging_that_repeats_its_key_is_truncated_not_silently_partial():
    row = {"ovrs_stck_symb_iscd": "A", "rmnd_dqty": "1", "seln_able_dqty": "1", "avrg_unpr": "1"}
    b, _ = broker({"/accounts/v1/me/overseas/holdings": [
        ok([row], code="5762", tr_cont_key="K"), ok([row], code="5762", tr_cont_key="K"),
    ]})
    with pytest.raises(BrokerError) as err:
        b.positions()
    assert err.value.code == "TRUNCATED"


def test_5820_with_rows_still_returns_the_rows():
    b, _ = broker({"/accounts/v1/me/overseas/holdings": [
        ok([{"ovrs_stck_symb_iscd": "SOXL", "rmnd_dqty": "3", "seln_able_dqty": "3", "avrg_unpr": "1"}], code="5820"),
    ]})
    assert b.positions()[0].qty == 3


def test_quote_uses_the_quote_market_code_not_the_vendor_id():
    b, http = broker({"/market/v1/overseas/prices": [ok({"prpr": "93.6700", "prdy_clpr": "93.6700"}, code="0000")]})
    quote = b.quote("SOXL")
    assert quote.last == Decimal("93.67")
    assert http.gets[0][1]["mrkt_cls_code"] == "AX"
    with pytest.raises(BrokerError) as err:
        b.quote("UNKNOWN")
    assert err.value.code == "NO_LISTING"


def test_market_open_reads_us_market_hours():
    hours = "/reference/v1/overseas/market-hours"
    b, http = broker({hours: [
        ok({"ovrs_stck_mrkt_tr_cls_code": "01", "ovrs_mkop_cls_code": "004", "mrkt_begn_time": "2230", "mrkt_fin_time": "0500"}),
        ok({"ovrs_stck_mrkt_tr_cls_code": "01", "ovrs_mkop_cls_code": "012", "mrkt_begn_time": "0000", "mrkt_fin_time": "0000"}),
        ok([], code="5820"),
    ]})
    assert b.market_open() is True
    assert http.gets[0][1]["ovrs_stck_mrkt_tr_cls_code"] == "01"
    assert b.market_open() is False
    with pytest.raises(BrokerError) as err:
        b.market_open()
    assert err.value.code == "BAD_RESPONSE"


def test_orders_today_map_states_and_skip_change_rows():
    base = {"ovrs_stck_symb_iscd": "SOXL", "seln_byov_clcd": "2", "aspr_type_code": "5", "ovrs_stck_oder_prc": "97.70",
            "odqt": "57", "cntg_qty": "0", "cncl_cnfr_qty": "0", "ovrs_stck_oder_stat_cls_code": "0", "oder_rctf_cncl_cls_code": "0"}
    b, _ = broker({"/trading/v1/overseas/orders/today": [ok([
        {**base, "oder_no": 1},
        {**base, "oder_no": 2, "cntg_qty": "57", "cntg_crnc_prc": "93.67"},
        {**base, "oder_no": 3, "seln_byov_clcd": "1", "aspr_type_code": "7", "ovrs_stck_oder_stat_cls_code": "1", "ovrs_stck_rfus_reas": "잔고 부족"},
        {**base, "oder_no": 4, "oder_rctf_cncl_cls_code": "2"},
    ])]})
    first, filled, rejected = b.orders_today()
    assert first.state == OrderState.OPEN and first.type == OrderType.LOC and first.side == Side.BUY and first.price == Decimal("97.70")
    assert filled.state == OrderState.FILLED and filled.fill_price == Decimal("93.67")
    assert rejected.state == OrderState.REJECTED and rejected.type == OrderType.MOC and rejected.price is None and rejected.reason == "잔고 부족"


def test_orders_today_that_claims_more_pages_is_refused():
    b, _ = broker({"/trading/v1/overseas/orders/today": [ok([], code="5762")]})
    with pytest.raises(BrokerError) as err:
        b.orders_today()
    assert err.value.code == "TRUNCATED"


BUY = "/trading/v1/overseas/orders/buy"
SELL = "/trading/v1/overseas/orders/sell"


def accepted(number=545361, warn="0", msg=""):
    return FakeResponse(200, {"rsp_cd": "0001", "rsp_msg": "주문이 완료되었습니다.",
                              "data": {"oder_no": number, "warn_cls_code": warn, "warn_msg": msg}})


def test_loc_buy_and_moc_sell_use_the_close_codes():
    b, http = broker({BUY: [accepted(545361)], SELL: [accepted(545362)]})
    buy = b.submit(OrderRequest("SOXL", Side.BUY, OrderType.LOC, 3, Decimal("93.10")))
    assert buy.order_id == "545361" and buy.state is OrderState.OPEN and buy.price == Decimal("93.10")
    assert http.posts[0] == (BUY, {"ovrs_stck_symb_iscd": "SOXL", "ovrs_exch_vndr_idno": "0066", "oder_cls_code": "32",
                                   "oder_cond_cls_code": "7", "odqt": 3, "ovrs_stck_oder_prc": 93.1})
    sell = b.submit(OrderRequest("SOXL", Side.SELL, OrderType.MOC, 71))
    assert sell.order_id == "545362" and sell.price is None
    assert http.posts[1][0] == SELL and http.posts[1][1]["oder_cls_code"] == "34" and http.posts[1][1]["ovrs_stck_oder_prc"] == 0


def test_order_warning_or_rejection_is_not_taken():
    b, _ = broker({BUY: [
        accepted(warn="1", msg="가격 확인"),
        FakeResponse(500, {"rsp_cd": "IGW50002", "rsp_msg": "주문가격 오류"}),
        FakeResponse(200, {"rsp_cd": "0000", "rsp_msg": "?", "data": {}}),
    ]})
    request = OrderRequest("SOXL", Side.BUY, OrderType.LOC, 1, Decimal("93.10"))
    for code in ("WARN_1", "IGW50002", "0000"):
        with pytest.raises(BrokerError) as err:
            b.submit(request)
        assert err.value.code == code and not err.value.retryable


def test_order_with_no_answer_is_unknown_not_retried():
    import requests
    b, http = broker({BUY: [requests.ConnectionError("reset")]})
    with pytest.raises(BrokerError) as err:
        b.submit(OrderRequest("SOXL", Side.BUY, OrderType.LOC, 1, Decimal("93.10")))
    assert err.value.code == "UNKNOWN_OUTCOME" and not err.value.retryable and len(http.posts) == 1


def test_bad_orders_never_reach_the_broker():
    b, http = broker({})
    for request in (
        OrderRequest("SOXL", Side.BUY, OrderType.MOC, 1),                       # Meritz MOC: sell only
        OrderRequest("SOXL", Side.BUY, OrderType.LOC, 1),                       # LOC without a price
        OrderRequest("SOXL", Side.SELL, OrderType.LOC, 0, Decimal("90")),
    ):
        with pytest.raises(BrokerError) as err:
            b.submit(request)
        assert err.value.code == "BAD_ORDER"
    assert http.token_calls == 0 and http.posts == []


def test_calls_to_the_same_api_are_spaced():
    waits = []
    now = [0.0]
    http = FakeHttp({"/accounts/v1/me/overseas/currencies": [ok([]), ok([])]})
    b = MeritzBroker(CREDS, LISTINGS, http=http, clock=lambda: now[0], sleep=waits.append)
    b.cash()
    b.cash()
    assert waits and waits[0] == pytest.approx(0.11)


def test_buy_and_sell_posts_share_one_spacing():
    waits = []
    now = [0.0]
    http = FakeHttp({BUY: [accepted(1)], SELL: [accepted(2)]})
    b = MeritzBroker(CREDS, LISTINGS, http=http, clock=lambda: now[0], sleep=waits.append)
    b.submit(OrderRequest("SOXL", Side.BUY, OrderType.LOC, 1, Decimal("93.10")))
    b.submit(OrderRequest("SOXL", Side.SELL, OrderType.LOC, 1, Decimal("99.10")))
    assert waits == [pytest.approx(0.35)]


def test_rate_limited_order_is_resent_then_given_up():
    limited = lambda: FakeResponse(500, {"rsp_cd": "EGW00200", "rsp_msg": "초당 거래건수를 초과하였습니다."})
    b, http = broker({BUY: [limited(), limited(), accepted(7)]})
    order = b.submit(OrderRequest("SOXL", Side.BUY, OrderType.LOC, 1, Decimal("93.10")))
    assert order.order_id == "7" and len(http.posts) == 3
    b, http = broker({BUY: [limited() for _ in range(4)]})
    with pytest.raises(BrokerError) as err:
        b.submit(OrderRequest("SOXL", Side.BUY, OrderType.LOC, 1, Decimal("93.10")))
    assert err.value.code == "EGW00200" and len(http.posts) == 4


TODAY = "/trading/v1/overseas/orders/today"
CANCEL = "/trading/v1/overseas/orders/cancel"
ROW = {"ovrs_stck_symb_iscd": "SOXL", "seln_byov_clcd": "1", "aspr_type_code": "5", "ovrs_stck_oder_prc": "114.31",
       "odqt": "82", "cntg_qty": "0", "cncl_cnfr_qty": "0", "ovrs_stck_oder_stat_cls_code": "0",
       "oder_rctf_cncl_cls_code": "0", "oder_date": "20261020", "oder_no": 1213426}


def test_cancel_sends_the_open_quantity_with_the_original_codes():
    b, http = broker({TODAY: [ok([{**ROW, "cntg_qty": "2"}, {**ROW, "oder_no": 9, "oder_rctf_cncl_cls_code": "2"}])],
                      CANCEL: [accepted(1300000)]})
    b.cancel("1213426")
    assert http.posts == [(CANCEL, {"oder_date": "20261020", "orgl_oder_no": 1213426, "oder_cls_code": "32", "odqt": 80})]


@pytest.mark.parametrize("rows, code", [
    ([], "NO_ORDER"),
    ([{**ROW, "cntg_qty": "82"}], "NOT_OPEN"),              # already filled
    ([{**ROW, "cncl_cnfr_qty": "82"}], "NOT_OPEN"),         # already cancelled
    ([{**ROW, "oder_date": ""}], "BAD_RESPONSE"),
])
def test_cancel_refuses_what_it_cannot_cancel(rows, code):
    b, http = broker({TODAY: [ok(rows)]})
    with pytest.raises(BrokerError) as err:
        b.cancel("1213426")
    assert err.value.code == code and http.posts == []


def test_cancel_rejection_is_raised():
    b, _ = broker({TODAY: [ok([ROW])], CANCEL: [FakeResponse(500, {"rsp_cd": "IGW00001", "rsp_msg": "취소가능수량 없음"})]})
    with pytest.raises(BrokerError) as err:
        b.cancel("1213426")
    assert err.value.code == "IGW00001"


def test_cancel_row_marks_the_original_cancelled():
    # As seen live: the original keeps cncl_cnfr_qty 0, the cancel row (its own oder_no) carries it.
    cancel_row = {**ROW, "oder_no": 1223071, "orgl_oder_no": 1213426, "oder_rctf_cncl_cls_code": "2", "cncl_cnfr_qty": "82"}
    b, _ = broker({TODAY: [ok([cancel_row, {**ROW, "orgl_oder_no": 0}, {**ROW, "oder_no": 1213427, "orgl_oder_no": 0}])]})
    cancelled, still_open = b.orders_today()
    assert cancelled.order_id == "1213426" and cancelled.state is OrderState.CANCELLED
    assert still_open.state is OrderState.OPEN
