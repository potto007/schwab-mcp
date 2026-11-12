"""Write tools enforce configured trading limits before approval (#30).

Max order value and a daily order cap are required to trade; max quantity
and a symbol allowlist are optional. A refusal happens before the reviewer is
asked and leaves the preview usable. The daily count survives a restart.
"""

from __future__ import annotations

import datetime
import json
import os
import stat
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import make_ctx, run, working_order

from schwab_mcp.approvals import ApprovalDecision, ApprovalManager, ApprovalRequest
from schwab_mcp.context import SchwabContext
from schwab_mcp.limits import OrderLedger, TradingLimits, check_order, order_value, parse_symbol_allowlist
from schwab_mcp.previews import PreviewOperation
from schwab_mcp.tools import orders

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


def test_value_is_price_times_quantity() -> None:
    assert order_value(equity_spec(quantity=10, price="150.00"), None) == 1_500.0


def test_option_value_uses_the_contract_multiplier() -> None:
    assert order_value(option_spec(quantity=2, price="3.50"), None) == 700.0


def test_stop_limit_value_uses_the_higher_price() -> None:
    spec = equity_spec(quantity=10, price="100.00", stopPrice="110.00", orderType="STOP_LIMIT")

    assert order_value(spec, None) == 1_100.0


def test_market_order_value_comes_from_the_preview() -> None:
    assert order_value(equity_spec(price=None), 1_234.5) == 1_234.5


def test_preview_value_raises_a_price_based_value() -> None:
    assert order_value(equity_spec(quantity=10, price="150.00"), 2_000.0) == 2_000.0


def test_composite_value_is_the_largest_node() -> None:
    oco = {
        "orderStrategyType": "OCO",
        "childOrderStrategies": [equity_spec(quantity=10, price="150.00"), equity_spec(quantity=10, price="90.00")],
    }

    assert order_value(oco, None) == 1_500.0


@pytest.mark.parametrize("preview_value", [None])
def test_unknown_value_is_refused(preview_value: float | None) -> None:
    with pytest.raises(PermissionError, match="value cannot be determined"):
        order_value(equity_spec(price=None), preview_value)


def test_one_unpriced_node_needs_the_preview_value() -> None:
    trigger = {**equity_spec(quantity=10, price="150.00"), "childOrderStrategies": [equity_spec(price=None)]}

    with pytest.raises(PermissionError, match="value cannot be determined"):
        order_value(trigger, None)
    assert order_value(trigger, 1_600.0) == 1_600.0


# --- check_order ---------------------------------------------------------------------


def test_order_over_the_value_cap_is_refused() -> None:
    with pytest.raises(PermissionError, match=r"\$6,000.00 exceeds the \$5,000.00 max order value"):
        check_order(LIMITS, equity_spec(quantity=40, price="150.00"), None)


def test_order_at_the_value_cap_passes() -> None:
    check_order(LIMITS, equity_spec(quantity=10, price="500.00"), None)


def test_quantity_cap_applies_to_every_leg() -> None:
    limits = TradingLimits(max_order_value=1e9, max_daily_orders=3, max_order_quantity=5)

    check_order(limits, option_spec(quantity=5), None)
    with pytest.raises(PermissionError, match="quantity 6 exceeds the max order quantity of 5"):
        check_order(limits, option_spec(quantity=6), None)


def test_allowlist_matches_equities_and_option_underlyings() -> None:
    limits = TradingLimits(max_order_value=1e9, max_daily_orders=3, allowed_symbols=frozenset({"SPY", "AAPL"}))

    check_order(limits, equity_spec(), None)
    check_order(limits, option_spec(symbol="SPY   251219C00500000"), None)
    with pytest.raises(PermissionError, match="'QQQ' is not in the symbol allowlist"):
        check_order(limits, option_spec(symbol="QQQ   251219C00500000"), None)


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

    def __init__(self, preview: Any = None) -> None:
        self.preview = preview if preview is not None else {}
        self.calls: list[str] = []

    async def get_instruments(self, symbol: str, **kwargs: Any) -> Any:
        return {"instruments": [{"symbol": symbol, "assetType": "EQUITY"}]}

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


def test_market_order_uses_the_preview_value() -> None:
    preview = {"orderStrategy": {"orderBalance": {"orderValue": 9_000.0}}}
    client, approvals = Client(preview), Approvals()
    ctx = _ctx(client, approvals)

    with pytest.raises(PermissionError, match=r"\$9,000.00 exceeds"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx, price=None)))

    assert approvals.requests == []


def test_market_order_without_a_preview_value_is_refused() -> None:
    client, approvals = Client(), Approvals()
    ctx = _ctx(client, approvals)

    with pytest.raises(PermissionError, match="value cannot be determined"):
        run(orders.place_previewed_order(ctx, ACCOUNT, _preview(ctx, price=None)))

    assert approvals.requests == []


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
