"""Write tools enforce configured trading limits before approval (#30).

Max order value and a daily order cap are required to trade; max quantity
and a symbol allowlist are optional. A refusal happens before the reviewer is
asked and leaves the preview usable. The daily count survives a restart.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from conftest import make_ctx, run, working_order
from mcp.server.mcpserver import Context as MCPContext, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from schwab.client import AsyncClient

from schwab_mcp import tools as tools_module
from schwab_mcp.approvals import ApprovalDecision, ApprovalManager, ApprovalRequest
from schwab_mcp.context import SchwabContext, SchwabServerContext
from schwab_mcp.limits import (
    OrderLedger,
    TradingLimits,
    check_order,
    leg_symbols,
    order_value,
    parse_symbol_allowlist,
    quote_symbols,
)
from schwab_mcp.previews import PreviewOperation
from schwab_mcp.tools import _registration, orders

ACCOUNT = "A" * 60 + "1234"
TARGET_ID = "1000123"
EASTERN = datetime.timezone(datetime.timedelta(hours=-5))


def equity_spec(quantity: int = 10, price: str | None = "150.00", **extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "orderType": "LIMIT" if price is not None else "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {"instruction": "BUY", "quantity": quantity, "instrument": {"symbol": "AAPL", "assetType": "EQUITY"}}
        ],
        **extra,
    }
    if price is not None:
        spec["price"] = price
    return spec


def option_spec(quantity: int = 2, price: str = "3.50", symbol: str = "SPY   251219C00500000") -> dict[str, Any]:
    return {
        "orderType": "LIMIT",
        "price": price,
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": "BUY_TO_OPEN",
                "quantity": quantity,
                "instrument": {"symbol": symbol, "assetType": "OPTION"},
            }
        ],
    }


LIMITS = TradingLimits(max_order_value=5_000.0, max_daily_orders=3)


# --- order value -----------------------------------------------------------------
#
# Every leg is valued against a market quote as well as the order's own price,
# so a far-from-market limit or stop price cannot understate the order.

SPY_500C = "SPY   251219C00500000"
SPY_510C = "SPY   251219C00510000"
SPY_550C = "SPY   251219C00550000"
SPY_500C_JAN = "SPY   260116C00500000"
SPY_480P = "SPY   251219P00480000"
SPY_470P = "SPY   251219P00470000"
SPY_520C = "SPY   251219C00520000"
SPY_530C = "SPY   251219C00530000"

QUOTES: dict[str, float] = {
    "AAPL": 1.0,
    "NVDA": 100.0,
    "SPY": 450.0,
    "SPY251219C00500000": 5.0,
    SPY_510C: 2.0,
    SPY_550C: 1.0,
    SPY_500C_JAN: 7.0,
    SPY_480P: 3.0,
    SPY_470P: 1.5,
    SPY_520C: 2.5,
    SPY_530C: 1.25,
}


def value(spec: dict[str, Any], preview: float | None = None, quotes: dict[str, float] = QUOTES) -> float:
    return order_value(spec, preview, quotes)


def leg(instruction: str, quantity: int, symbol: str) -> dict[str, Any]:
    asset = "OPTION" if len(symbol) == 21 else "EQUITY"
    return {"instruction": instruction, "quantity": quantity, "instrument": {"symbol": symbol, "assetType": asset}}


def combo(order_type: str, price: str | None, *legs: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {"orderType": order_type, "orderStrategyType": "SINGLE", "orderLegCollection": list(legs)}
    if price is not None:
        spec["price"] = price
    return spec


def nvda(instruction: str, quantity: int, order_type: str, **prices: str) -> dict[str, Any]:
    return {
        "orderType": order_type,
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [leg(instruction, quantity, "NVDA")],
        **prices,
    }


def test_limit_buy_above_the_market_uses_its_price() -> None:
    assert value(equity_spec(quantity=10, price="150.00")) == 1_500.0


def test_limit_buy_below_the_market_uses_the_quote() -> None:
    assert value(nvda("BUY", 10, "LIMIT", price="90.00")) == 1_000.0


def test_a_sell_far_below_the_market_is_valued_at_the_market() -> None:
    dump = nvda("SELL", 10_000, "LIMIT", price="0.01")

    assert value(dump) == 1_000_000.0
    with pytest.raises(PermissionError, match=r"\$1,000,000.00 exceeds the \$1,000.00 max order value"):
        check_order(TradingLimits(max_order_value=1_000.0, max_daily_orders=3), dump, None, QUOTES)


def test_a_buy_stop_below_the_market_is_valued_at_the_market() -> None:
    assert value(nvda("BUY", 10_000, "STOP", stopPrice="0.05")) == 1_000_000.0


@pytest.mark.parametrize("order_type", ["MARKET", "TRAILING_STOP"])
def test_unpriced_orders_are_valued_from_the_quote(order_type: str) -> None:
    assert value(nvda("BUY", 10, order_type)) == 1_000.0


def test_long_option_is_valued_at_its_premium() -> None:
    assert value(option_spec(quantity=2, price="3.50")) == 5.0 * 100 * 2


def test_short_option_is_valued_at_strike_plus_premium() -> None:
    short_call = combo("LIMIT", "0.01", leg("SELL_TO_OPEN", 50, SPY_500C))

    assert value(short_call) == (500.0 + 5.0) * 100 * 50


def test_short_call_in_the_money_is_valued_at_the_underlying() -> None:
    # A stale one-cent option quote cannot hide a deep in-the-money call.
    quotes = {**QUOTES, "SPY": 600.0, "SPY251219C00500000": 0.01}
    short_call = combo("LIMIT", "0.01", leg("SELL_TO_OPEN", 10, SPY_500C))

    assert value(short_call, quotes=quotes) == (600.0 + 0.01) * 100 * 10


def test_short_call_without_an_underlying_quote_is_refused() -> None:
    quotes = {key: price for key, price in QUOTES.items() if key != "SPY"}

    with pytest.raises(PermissionError, match="no market quote for 'SPY'"):
        value(combo("LIMIT", "1.00", leg("SELL_TO_OPEN", 1, SPY_500C)), quotes=quotes)


def test_short_puts_and_verticals_do_not_need_the_underlying() -> None:
    quotes = {key: price for key, price in QUOTES.items() if key != "SPY"}
    credit = combo("NET_CREDIT", "0.05", leg("SELL_TO_OPEN", 1, SPY_500C), leg("BUY_TO_OPEN", 1, SPY_550C))

    assert value(combo("LIMIT", "1.00", leg("SELL_TO_OPEN", 1, SPY_480P)), quotes=quotes) == (480.0 + 3.0) * 100
    assert value(credit, quotes=quotes) == (50.0 + 1.0) * 100


def test_quote_symbols_add_the_underlying_of_short_calls() -> None:
    spec = combo(
        "NET_CREDIT",
        "1.00",
        leg("SELL_TO_OPEN", 1, SPY_520C),
        leg("SELL_TO_OPEN", 1, SPY_480P),
        leg("BUY_TO_OPEN", 1, SPY_530C),
    )

    assert quote_symbols(spec) == [SPY_520C, SPY_480P, SPY_530C, "SPY"]
    assert quote_symbols(option_spec(quantity=1)) == leg_symbols(option_spec(quantity=1))


def test_credit_vertical_is_valued_at_its_width() -> None:
    credit = combo("NET_CREDIT", "0.05", leg("SELL_TO_OPEN", 100, SPY_500C), leg("BUY_TO_OPEN", 100, SPY_550C))

    assert value(credit) == (50.0 + 1.0) * 100 * 100


def test_debit_vertical_is_valued_at_its_width_and_premium() -> None:
    debit = combo("NET_DEBIT", "3.00", leg("BUY_TO_OPEN", 1, SPY_500C), leg("SELL_TO_OPEN", 1, SPY_510C))

    assert value(debit) == (10.0 + 5.0) * 100


def test_iron_condor_adds_both_spreads() -> None:
    condor = combo(
        "NET_CREDIT",
        "1.00",
        leg("SELL_TO_OPEN", 1, SPY_480P),
        leg("BUY_TO_OPEN", 1, SPY_470P),
        leg("SELL_TO_OPEN", 1, SPY_520C),
        leg("BUY_TO_OPEN", 1, SPY_530C),
    )

    assert value(condor) == (10.0 + 1.5) * 100 + (10.0 + 1.25) * 100


@pytest.mark.parametrize(
    "legs",
    [
        pytest.param((leg("SELL_TO_OPEN", 1, SPY_500C), leg("BUY_TO_OPEN", 1, SPY_500C_JAN)), id="calendar"),
        pytest.param((leg("SELL_TO_OPEN", 2, SPY_500C), leg("BUY_TO_OPEN", 1, SPY_510C)), id="ratio"),
    ],
)
def test_unmatched_short_legs_count_at_full_notional(legs: tuple[dict[str, Any], ...]) -> None:
    short, long = legs
    expected = (500.0 + 5.0) * 100 * short["quantity"] + QUOTES[long["instrument"]["symbol"]] * 100 * long["quantity"]

    assert value(combo("NET_CREDIT", "1.00", *legs)) == expected


def test_a_leg_without_a_quote_is_refused() -> None:
    with pytest.raises(PermissionError, match="no market quote for 'MSFT'"):
        value(combo("LIMIT", "10.00", leg("BUY", 1, "MSFT")))


def test_quotes_match_option_symbols_with_or_without_padding() -> None:
    assert value(combo("LIMIT", "1.00", leg("BUY_TO_OPEN", 1, SPY_500C))) == 500.0


def test_preview_value_raises_the_bound() -> None:
    assert value(equity_spec(quantity=10, price="150.00"), 2_000.0) == 2_000.0


@pytest.mark.parametrize("preview_value", [0.0, -0.0, float("nan"), float("inf"), -500.0])
def test_preview_value_never_lowers_the_bound(preview_value: float) -> None:
    assert value(nvda("BUY", 10_000, "MARKET"), preview_value) == 1_000_000.0


@pytest.mark.parametrize("quantity", [0, -5])
def test_a_leg_without_a_positive_quantity_is_refused(quantity: int) -> None:
    with pytest.raises(PermissionError, match="value cannot be determined"):
        value(nvda("BUY", quantity, "LIMIT", price="1.00"))


def test_a_spec_without_legs_is_refused() -> None:
    with pytest.raises(PermissionError, match="value cannot be determined"):
        value({"orderStrategyType": "SINGLE"})


# --- composite strategies --------------------------------------------------------


def sell_spec(quantity: int = 100, price: str = "120.00") -> dict[str, Any]:
    spec = equity_spec(quantity=quantity, price=price)
    spec["orderLegCollection"][0]["instruction"] = "SELL"
    return spec


def trigger(parent: dict[str, Any], *children: dict[str, Any]) -> dict[str, Any]:
    return {**parent, "orderStrategyType": "TRIGGER", "childOrderStrategies": list(children)}


def oco(*children: dict[str, Any]) -> dict[str, Any]:
    return {"orderStrategyType": "OCO", "childOrderStrategies": list(children)}


def test_oco_value_is_its_largest_branch() -> None:
    assert value(oco(equity_spec(quantity=10, price="150.00"), equity_spec(quantity=10, price="90.00"))) == 1_500.0


def test_trigger_value_adds_the_parent_and_its_child() -> None:
    buy_then_buy = trigger(equity_spec(quantity=100, price="100.00"), equity_spec(quantity=100, price="100.00"))

    assert value(buy_then_buy) == 20_000.0
    with pytest.raises(PermissionError, match=r"\$20,000.00 exceeds the \$15,000.00 max order value"):
        check_order(TradingLimits(max_order_value=15_000.0, max_daily_orders=3), buy_then_buy, None, QUOTES)


def test_bracket_adds_the_entry_and_its_largest_exit() -> None:
    bracket = trigger(
        equity_spec(quantity=100, price="100.00"), oco(sell_spec(price="120.00"), sell_spec(price="90.00"))
    )

    assert value(bracket) == 10_000.0 + 12_000.0


def test_nested_strategies_combine_by_kind() -> None:
    nested = oco(
        trigger(equity_spec(quantity=10, price="100.00"), equity_spec(quantity=10, price="50.00")),
        equity_spec(quantity=10, price="120.00"),
    )

    assert value(nested) == 1_500.0


def test_an_unpriced_trigger_stage_is_valued_from_its_quote() -> None:
    staged = trigger(equity_spec(quantity=10, price="150.00"), nvda("BUY", 10, "MARKET"))

    assert value(staged) == 1_500.0 + 1_000.0


# --- check_order ---------------------------------------------------------------------


def test_order_over_the_value_cap_is_refused() -> None:
    with pytest.raises(PermissionError, match=r"\$6,000.00 exceeds the \$5,000.00 max order value"):
        check_order(LIMITS, equity_spec(quantity=40, price="150.00"), None, QUOTES)


def test_order_at_the_value_cap_passes() -> None:
    check_order(LIMITS, equity_spec(quantity=10, price="500.00"), None, QUOTES)


def test_quantity_cap_applies_to_every_leg() -> None:
    limits = TradingLimits(max_order_value=1e9, max_daily_orders=3, max_order_quantity=5)

    check_order(limits, option_spec(quantity=5), None, QUOTES)
    with pytest.raises(PermissionError, match="quantity 6 exceeds the max order quantity of 5"):
        check_order(limits, option_spec(quantity=6), None, QUOTES)


def test_allowlist_matches_equities_and_option_underlyings() -> None:
    limits = TradingLimits(max_order_value=1e9, max_daily_orders=3, allowed_symbols=frozenset({"SPY", "AAPL"}))
    quotes = {**QUOTES, "QQQ   251219C00500000": 1.0}

    check_order(limits, equity_spec(), None, quotes)
    check_order(limits, option_spec(symbol=SPY_500C), None, quotes)
    with pytest.raises(PermissionError, match="'QQQ' is not in the symbol allowlist"):
        check_order(limits, option_spec(symbol="QQQ   251219C00500000"), None, quotes)


def test_allowlist_parsing() -> None:
    assert parse_symbol_allowlist(["spy, aapl", " brk.b "]) == frozenset({"SPY", "AAPL", "BRK.B"})
    assert parse_symbol_allowlist([" , "]) is None
    assert parse_symbol_allowlist([]) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_order_value": 0, "max_daily_orders": 1},
        {"max_order_value": float("nan"), "max_daily_orders": 1},
        {"max_order_value": float("inf"), "max_daily_orders": 1},
        {"max_order_value": 100.0, "max_daily_orders": 0},
        {"max_order_value": 100.0, "max_daily_orders": 1, "max_order_quantity": 0},
        {"max_order_value": 100.0, "max_daily_orders": 1, "allowed_symbols": frozenset()},
    ],
)
def test_limits_must_be_positive(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TradingLimits(**kwargs)


# --- ledger ----------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=EASTERN)

    def __call__(self) -> datetime.datetime:
        return self.now


def test_ledger_counts_survive_a_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "daily-orders.json")
    clock = Clock()
    ledger = OrderLedger(path, clock=clock)
    for _ in range(2):
        ledger.reserve(3).commit()

    restarted = OrderLedger(path, clock=clock)
    restarted.reserve(3).commit()
    with pytest.raises(PermissionError, match="daily cap of 3 orders"):
        restarted.reserve(3)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_ledger_resets_on_a_new_market_day(tmp_path: Path) -> None:
    clock = Clock()
    ledger = OrderLedger(str(tmp_path / "daily-orders.json"), clock=clock)
    ledger.reserve(1).commit()
    with pytest.raises(PermissionError):
        ledger.reserve(1)

    clock.now += datetime.timedelta(days=1)

    ledger.reserve(1).commit()


def test_pending_reservations_count_and_release_frees_them() -> None:
    ledger = OrderLedger(None)
    first = ledger.reserve(2)
    ledger.reserve(2)
    with pytest.raises(PermissionError, match="1 awaiting|2 awaiting"):
        ledger.reserve(2)

    first.release()
    ledger.reserve(2)


def test_commit_rechecks_orders_sent_by_another_process(tmp_path: Path) -> None:
    path = str(tmp_path / "daily-orders.json")
    clock = Clock()
    one, two = OrderLedger(path, clock=clock), OrderLedger(path, clock=clock)
    slot = one.reserve(1)
    two.reserve(1).commit()

    with pytest.raises(PermissionError, match="daily cap of 1 orders"):
        slot.commit()


def test_unreadable_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "daily-orders.json"
    path.write_text("not json")

    with pytest.raises(ValueError, match="daily-orders.json"):
        OrderLedger(str(path))


@pytest.mark.parametrize("payload", [[], {"date": "2026-11-12"}, {"date": "2026-11-12", "count": -1}])
def test_malformed_ledger_fails_closed(tmp_path: Path, payload: Any) -> None:
    path = tmp_path / "daily-orders.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="daily-orders.json"):
        OrderLedger(str(path))


# --- tools ------------------------------------------------------------------------------


class Approvals(ApprovalManager):
    def __init__(self, decision: ApprovalDecision = ApprovalDecision.APPROVED) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    async def require(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class Client:
    class Instrument:
        Projection = Enum("Projection", {"SYMBOL_SEARCH": "symbol-search"})

    def __init__(self, preview: Any = None, quotes: dict[str, float] | None = None, quote_error: bool = False) -> None:
        self.preview = preview if preview is not None else {}
        self.quotes = {"AAPL": 1.0} if quotes is None else quotes
        self.quote_error = quote_error
        self.quote_calls = 0
        self.quotes_after_first: dict[str, float] | None = None
        self.calls: list[str] = []

    async def get_instruments(self, symbol: str, **kwargs: Any) -> Any:
        return {"instruments": [{"symbol": symbol, "assetType": "EQUITY"}]}

    async def get_quotes(self, symbols: list[str], **kwargs: Any) -> Any:
        self.quote_calls += 1
        if self.quote_error:
            raise httpx.ConnectError("quotes down")
        quotes = self.quotes if self.quotes_after_first is None or self.quote_calls == 1 else self.quotes_after_first
        return {symbol: {"quote": {"lastPrice": quotes[symbol]}} for symbol in symbols if symbol in quotes}

    async def preview_order(self, *args: Any, **kwargs: Any) -> Any:
        return self.preview

    async def get_order(self, *args: Any, **kwargs: Any) -> Any:
        return working_order(TARGET_ID)

    async def place_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("place_order")
        return {"orderId": 7, "accountHash": ACCOUNT}

    async def replace_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("replace_order")
        return {"orderId": 8, "accountHash": ACCOUNT}

    async def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("cancel_order")


@pytest.fixture(autouse=True)
def passthrough_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def call(func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("response_handler", None)
        return await func(*args, **kwargs)

    monkeypatch.setattr(orders, "call", call)


def _ctx(
    client: Client,
    approvals: ApprovalManager,
    limits: TradingLimits | None = LIMITS,
    ledger: OrderLedger | None = None,
) -> SchwabContext:
    ctx = make_ctx(client)
    ctx.schwab.approval_manager = approvals
    ctx.schwab.trading_limits = limits
    ctx.schwab.order_ledger = ledger if ledger is not None else OrderLedger(None)
    return ctx


def _preview(ctx: SchwabContext, quantity: int = 10, price: float | None = 150.0) -> str:
    order_type = "LIMIT" if price is not None else "MARKET"
    result = run(orders.preview_equity_order(ctx, ACCOUNT, "AAPL", quantity, "BUY", order_type, price=price))
    return result["preview_id"]


def test_order_over_the_cap_is_refused_before_approval_and_the_preview_survives() -> None:
    client, approvals = Client(), Approvals()
    ctx = _ctx(client, approvals)
    preview_id = _preview(ctx, quantity=40)

    with pytest.raises(PermissionError, match="max order value"):
        run(orders.place_previewed_order(ctx, ACCOUNT, preview_id))

    assert approvals.requests == []
    assert client.calls == []
    ctx.previews.peek(preview_id, ACCOUNT, operation=PreviewOperation.PLACE_ORDER)


def test_market_order_is_valued_from_the_quote() -> None:
    client, approvals = Client(quotes={"AAPL": 900.0}), Approvals()
    ctx = _ctx(client, approvals)

    with pytest.raises(PermissionError, match=r"\$9,000.00 exceeds"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx, price=None)))

    assert approvals.requests == []


def test_preview_value_still_raises_the_bound() -> None:
    preview = {"orderStrategy": {"orderBalance": {"orderValue": 9_000.0}}}
    client, approvals = Client(preview), Approvals()
    ctx = _ctx(client, approvals)

    with pytest.raises(PermissionError, match=r"\$9,000.00 exceeds"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx, price=None)))

    assert approvals.requests == []


@pytest.mark.parametrize(("quotes", "quote_error"), [({}, False), (None, True)], ids=["no-quote", "quotes-down"])
def test_order_without_a_market_quote_is_refused_and_the_preview_survives(
    quotes: dict[str, float] | None, quote_error: bool
) -> None:
    client, approvals = Client(quotes=quotes, quote_error=quote_error), Approvals()
    ctx = _ctx(client, approvals)
    preview_id = _preview(ctx)

    with pytest.raises(PermissionError, match="market quote"):
        run(orders.place_previewed_order(ctx, ACCOUNT, preview_id))

    assert approvals.requests == []
    assert client.calls == []
    ctx.previews.peek(preview_id, ACCOUNT, operation=PreviewOperation.PLACE_ORDER)


def test_a_sell_far_below_the_market_is_refused_before_approval() -> None:
    client, approvals = Client(quotes={"AAPL": 100.0}), Approvals()
    ctx = _ctx(client, approvals, limits=TradingLimits(max_order_value=1_000.0, max_daily_orders=3))
    result = run(orders.preview_equity_order(ctx, ACCOUNT, "AAPL", 10_000, "SELL", "LIMIT", price=0.01))

    with pytest.raises(PermissionError, match=r"\$1,000,000.00 exceeds"):
        run(orders.place_previewed_order(ctx, ACCOUNT, result["preview_id"]))

    assert approvals.requests == []
    assert client.calls == []


def test_daily_cap_counts_placed_orders_and_survives_a_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "daily-orders.json")
    client, approvals = Client(), Approvals()
    ctx = _ctx(client, approvals, ledger=OrderLedger(path))
    for _ in range(3):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx)))

    restarted = _ctx(client, approvals, ledger=OrderLedger(path))
    with pytest.raises(PermissionError, match="daily cap of 3 orders"):
        run(orders.place_previewed_order(restarted, ACCOUNT, _preview(restarted)))

    assert client.calls == ["place_order"] * 3
    assert len(approvals.requests) == 3


@pytest.mark.parametrize("decision", [ApprovalDecision.DENIED, ApprovalDecision.EXPIRED])
def test_denied_or_expired_orders_do_not_count(decision: ApprovalDecision) -> None:
    ledger = OrderLedger(None)
    ctx = _ctx(Client(), Approvals(decision), ledger=ledger)
    for _ in range(4):
        with pytest.raises((PermissionError, TimeoutError)):
            run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx)))

    approved = _ctx(Client(), Approvals(), ledger=ledger)
    for _ in range(3):
        run(orders.place_previewed_order(approved, ACCOUNT, _preview(approved)))


def test_replace_checks_value_and_counts_toward_the_cap() -> None:
    client, approvals = Client(), Approvals()
    ledger = OrderLedger(None)
    ctx = _ctx(client, approvals, limits=TradingLimits(max_order_value=5_000.0, max_daily_orders=1), ledger=ledger)

    big = {"symbol": "AAPL", "quantity": 100, "instruction": "BUY", "order_type": "LIMIT", "price": 150.0}
    preview = run(orders.preview_replacement_order(ctx, ACCOUNT, TARGET_ID, cast(orders._OrderDescInput, big)))
    with pytest.raises(PermissionError, match="max order value"):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview["preview_id"]))
    assert approvals.requests == []

    small = {**big, "quantity": 1}
    preview = run(orders.preview_replacement_order(ctx, ACCOUNT, TARGET_ID, cast(orders._OrderDescInput, small)))
    run(orders.replace_previewed_order(ctx, ACCOUNT, preview["preview_id"]))

    preview = run(orders.preview_replacement_order(ctx, ACCOUNT, TARGET_ID, cast(orders._OrderDescInput, small)))
    with pytest.raises(PermissionError, match="daily cap of 1 orders"):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview["preview_id"]))
    assert client.calls == ["replace_order"]


def test_cancel_is_not_counted() -> None:
    ledger = OrderLedger(None)
    client = Client()
    ctx = _ctx(client, Approvals(), limits=TradingLimits(max_order_value=1.0, max_daily_orders=1), ledger=ledger)
    for _ in range(3):
        run(orders.cancel_order(ctx, ACCOUNT, TARGET_ID))

    assert client.calls == ["cancel_order"] * 3
    ledger.reserve(1)


def test_order_writes_refuse_without_limits() -> None:
    client, approvals = Client(), Approvals()
    ctx = _ctx(client, approvals, limits=None)

    with pytest.raises(PermissionError, match="Trading limits are not configured"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx)))

    assert approvals.requests == []


def test_a_ledger_error_after_approval_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "daily-orders.json"
    client = Client()
    outcomes: list[tuple[str, bool]] = []

    class CorruptsLedger(Approvals):
        async def require(self, request: ApprovalRequest) -> ApprovalDecision:
            path.write_text("not json")
            return await super().require(request)

        async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
            outcomes.append((outcome, succeeded))

    ctx = _ctx(client, CorruptsLedger(), ledger=OrderLedger(str(path)))

    with pytest.raises(ValueError, match="daily-orders.json"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx)))

    assert client.calls == []
    [(outcome, succeeded)] = outcomes
    assert outcome.startswith("not sent: ValueError: Daily order ledger")
    assert succeeded is False


def test_cap_filled_by_another_process_after_approval_is_reported(tmp_path: Path) -> None:
    path = str(tmp_path / "daily-orders.json")
    client = Client()
    outcomes: list[tuple[str, bool]] = []

    class OtherProcessApproves(Approvals):
        async def require(self, request: ApprovalRequest) -> ApprovalDecision:
            # Another server process sends the last order of the day meanwhile.
            OrderLedger(path).reserve(1).commit()
            return await super().require(request)

        async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
            outcomes.append((outcome, succeeded))

    limits = TradingLimits(max_order_value=5_000.0, max_daily_orders=1)
    ctx = _ctx(client, OtherProcessApproves(), limits=limits, ledger=OrderLedger(path))

    with pytest.raises(PermissionError, match="daily cap of 1 orders"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx)))

    assert client.calls == []
    [(outcome, succeeded)] = outcomes
    assert outcome.startswith("not sent: Refusing order: the daily cap of 1 orders is reached")
    assert succeeded is False


# --- through the registered MCP tool runner ---------------------------------------


class RegisteredHarness:
    """Real tool registration and argument validation, fake Schwab client."""

    def __init__(self, limits: TradingLimits, preview: Any = None, quotes: dict[str, float] | None = None) -> None:
        self.server = MCPServer(name="limits")
        self.client = Client(preview, quotes=quotes)
        self.approvals = Approvals()
        tools_module.register_tools(
            self.server, cast(AsyncClient, self.client), allow_write=True, enable_technical=False
        )
        lifespan_context = SchwabServerContext(
            client=cast(AsyncClient, self.client),
            approval_manager=self.approvals,
            trading_limits=limits,
            order_ledger=OrderLedger(None),
        )
        request_context = SimpleNamespace(lifespan_context=lifespan_context, request_id="request-1", meta=None)
        self.context = MCPContext.model_construct(_request_context=cast(Any, request_context), _mcp_server=self.server)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        return run(self.server.call_tool(name, arguments, self.context))

    def refused(self, name: str, arguments: dict[str, Any], pattern: str) -> None:
        """The registered tool fails with a PermissionError matching ``pattern``."""
        with pytest.raises(ToolError) as excinfo:
            self.call(name, arguments)
        cause = excinfo.value.__cause__
        assert isinstance(cause, PermissionError), repr(cause)
        assert re.search(pattern, str(cause)), str(cause)

    def preview_id(self, name: str, arguments: dict[str, Any]) -> str:
        result = self.call(name, arguments)
        assert not result.is_error, result
        return json.loads(result.content[0].text)["preview_id"]


_ENTRY = {"symbol": "AAPL", "quantity": 100, "instruction": "BUY", "order_type": "LIMIT", "price": 100.0}


def test_registered_same_direction_trigger_is_refused_over_the_cap() -> None:
    harness = RegisteredHarness(TradingLimits(max_order_value=15_000.0, max_daily_orders=3))
    preview_id = harness.preview_id(
        "preview_trigger_order", {"account_hash": ACCOUNT, "entry_order": _ENTRY, "exit_orders": [_ENTRY]}
    )

    harness.refused(
        "place_previewed_order", {"account_hash": ACCOUNT, "preview_id": preview_id}, r"\$20,000.00 exceeds"
    )

    assert harness.approvals.requests == []
    assert harness.client.calls == []


def test_registered_sell_far_below_the_market_is_refused() -> None:
    harness = RegisteredHarness(TradingLimits(max_order_value=1_000.0, max_daily_orders=3), quotes={"AAPL": 100.0})
    preview_id = harness.preview_id(
        "preview_equity_order",
        {
            "account_hash": ACCOUNT,
            "symbol": "AAPL",
            "quantity": 10_000,
            "instruction": "SELL",
            "order_type": "LIMIT",
            "price": 0.01,
        },
    )

    harness.refused(
        "place_previewed_order", {"account_hash": ACCOUNT, "preview_id": preview_id}, r"\$1,000,000.00 exceeds"
    )
    assert harness.approvals.requests == []
    assert harness.client.calls == []


@pytest.mark.parametrize("preview_value", [0.0, None])
def test_registered_market_order_is_bounded_by_its_quote(preview_value: float | None) -> None:
    preview = {"orderStrategy": {"orderValue": preview_value}} if preview_value is not None else {}
    harness = RegisteredHarness(
        TradingLimits(max_order_value=1.0, max_daily_orders=3), preview=preview, quotes={"AAPL": 100.0}
    )
    preview_id = harness.preview_id(
        "preview_equity_order",
        {"account_hash": ACCOUNT, "symbol": "AAPL", "quantity": 100_000, "instruction": "BUY", "order_type": "MARKET"},
    )

    harness.refused(
        "place_previewed_order", {"account_hash": ACCOUNT, "preview_id": preview_id}, r"\$10,000,000.00 exceeds"
    )
    assert harness.approvals.requests == []
    assert harness.client.calls == []


# --- send-time re-check and early capacity check ---------------------------------


class OutcomeRecorder(Approvals):
    def __init__(self) -> None:
        super().__init__()
        self.outcomes: list[tuple[str, bool]] = []

    async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
        self.outcomes.append((outcome, succeeded))


def test_a_price_move_during_approval_is_caught_before_sending() -> None:
    client, approvals = Client(quotes={"AAPL": 1.0}), OutcomeRecorder()
    ledger = OrderLedger(None)
    ctx = _ctx(client, approvals, limits=TradingLimits(max_order_value=5_000.0, max_daily_orders=1), ledger=ledger)
    preview_id = _preview(ctx, quantity=10, price=150.0)
    client.quotes_after_first = {"AAPL": 600.0}

    with pytest.raises(PermissionError, match=r"\$6,000.00 exceeds"):
        run(orders.place_previewed_order(ctx, ACCOUNT, preview_id))

    assert len(approvals.requests) == 1
    assert client.calls == []
    [(outcome, succeeded)] = approvals.outcomes
    assert outcome.startswith("not sent: Refusing order: estimated value $6,000.00 exceeds")
    assert succeeded is False
    # The refused order did not use the day's only slot.
    ledger.reserve(1)


def test_quotes_unavailable_at_send_time_block_the_order() -> None:
    client, approvals = Client(quotes={"AAPL": 1.0}), OutcomeRecorder()
    ctx = _ctx(client, approvals)
    preview_id = _preview(ctx)
    client.quotes_after_first = {}

    with pytest.raises(PermissionError, match="no market quote for 'AAPL'"):
        run(orders.place_previewed_order(ctx, ACCOUNT, preview_id))

    assert client.calls == []
    assert approvals.outcomes[0][0].startswith("not sent: Refusing order: no market quote")


def test_replace_rechecks_the_value_before_sending() -> None:
    client, approvals = Client(quotes={"AAPL": 1.0}), OutcomeRecorder()
    ctx = _ctx(client, approvals)
    small = {"symbol": "AAPL", "quantity": 10, "instruction": "BUY", "order_type": "LIMIT", "price": 150.0}
    preview = run(orders.preview_replacement_order(ctx, ACCOUNT, TARGET_ID, cast(orders._OrderDescInput, small)))
    client.quotes_after_first = {"AAPL": 600.0}

    with pytest.raises(PermissionError, match=r"\$6,000.00 exceeds"):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview["preview_id"]))

    assert client.calls == []


@pytest.mark.parametrize("tool", ["place_previewed_order", "replace_previewed_order"])
def test_full_approval_queue_refuses_before_fetching_quotes(tool: str) -> None:
    client, approvals = Client(), Approvals()
    ctx = _ctx(client, approvals)
    if tool == "place_previewed_order":
        preview_id = _preview(ctx)
    else:
        small = {"symbol": "AAPL", "quantity": 1, "instruction": "BUY", "order_type": "LIMIT", "price": 1.0}
        preview_id = run(
            orders.preview_replacement_order(ctx, ACCOUNT, TARGET_ID, cast(orders._OrderDescInput, small))
        )["preview_id"]
    ctx.schwab.pending_approvals = _registration.MAX_PENDING_APPROVALS

    with pytest.raises(PermissionError, match="already pending"):
        run(getattr(orders, tool)(ctx, ACCOUNT, preview_id))

    assert client.quote_calls == 0


def test_a_cancelled_send_time_check_reports_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    client, approvals = Client(), OutcomeRecorder()
    ledger = OrderLedger(None)
    ctx = _ctx(client, approvals, limits=TradingLimits(max_order_value=5_000.0, max_daily_orders=1), ledger=ledger)
    preview_id = _preview(ctx)
    real_quotes = orders._market_quotes
    seen = 0

    async def cancel_second(ctx: Any, order_spec: dict[str, Any]) -> dict[str, float]:
        nonlocal seen
        seen += 1
        if seen == 2:
            raise asyncio.CancelledError
        return await real_quotes(ctx, order_spec)

    monkeypatch.setattr(orders, "_market_quotes", cancel_second)

    async def place() -> None:
        with pytest.raises(asyncio.CancelledError):
            await orders.place_previewed_order(ctx, ACCOUNT, preview_id)
        await asyncio.sleep(0)

    run(place())

    assert client.calls == []
    assert approvals.outcomes == [("not sent (the call was cancelled)", False)]
    ledger.reserve(1)


def test_a_not_sent_report_survives_cancellation() -> None:
    reporting = asyncio.Event()
    release = asyncio.Event()

    class SlowRecorder(OutcomeRecorder):
        async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
            reporting.set()
            await release.wait()
            await super().report_outcome(request, outcome, succeeded=succeeded)

    client, approvals = Client(quotes={"AAPL": 1.0}), SlowRecorder()
    ledger = OrderLedger(None)
    ctx = _ctx(client, approvals, limits=TradingLimits(max_order_value=5_000.0, max_daily_orders=1), ledger=ledger)
    preview_id = _preview(ctx, quantity=10, price=150.0)
    client.quotes_after_first = {"AAPL": 600.0}

    async def scenario() -> None:
        task = asyncio.create_task(orders.place_previewed_order(ctx, ACCOUNT, preview_id))
        await reporting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(10):
            await asyncio.sleep(0)

    run(scenario())

    assert client.calls == []
    [(outcome, succeeded)] = approvals.outcomes
    assert outcome.startswith("not sent: Refusing order: estimated value $6,000.00 exceeds")
    assert succeeded is False
    ledger.reserve(1)
