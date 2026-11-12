"""Approval prompts carry everything the approver needs: the current target
order for cancel and replace (#26) and Schwab's preview result (#28)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, cast

import pytest
from conftest import make_ctx, run

from schwab_mcp.approvals import ApprovalDecision, ApprovalManager, ApprovalRequest, signal as signal_backend
from schwab_mcp.context import SchwabContext
from schwab_mcp.previews import PreviewOperation
from schwab_mcp.tools import orders
from schwab_mcp.tools.utils import SchwabAPIError

ACCOUNT = "A" * 60 + "1234"
ORDER_ID = "1000123"

WORKING_ORDER: dict[str, Any] = {
    "session": "NORMAL",
    "duration": "DAY",
    "orderType": "LIMIT",
    "complexOrderStrategyType": "NONE",
    "quantity": 10,
    "filledQuantity": 0,
    "remainingQuantity": 10,
    "requestedDestination": "AUTO",
    "price": 150.0,
    "orderLegCollection": [
        {
            "orderLegType": "EQUITY",
            "legId": 1,
            "instrument": {"assetType": "EQUITY", "cusip": "037833100", "symbol": "AAPL", "instrumentId": 1},
            "instruction": "BUY",
            "positionEffect": "OPENING",
            "quantity": 10,
        }
    ],
    "orderStrategyType": "SINGLE",
    "orderId": int(ORDER_ID),
    "cancelable": True,
    "editable": True,
    "status": "WORKING",
    "enteredTime": "2026-11-12T15:00:00+0000",
    "accountNumber": 12341234,
    "orderActivityCollection": [{"activityType": "EXECUTION", "executionLegs": [{"price": 1.0}]}],
}
WORKING_SUMMARY = (
    "SINGLE BUY 10 AAPL EQUITY positionEffect=OPENING LIMIT @ $150.0 "
    "session=NORMAL duration=DAY status=WORKING filledQuantity=0 remainingQuantity=10"
)

PREVIEW: dict[str, Any] = {
    "orderId": 0,
    "orderStrategy": {
        "orderType": "LIMIT",
        "orderValue": 1500.0,
        "orderBalance": {
            "orderValue": 1500.0,
            "projectedAvailableFund": 8500.0,
            "projectedBuyingPower": 8500.0,
            "projectedCommission": 0.0,
        },
    },
    "orderValidationResult": {
        "accepts": [{"validationRuleName": "ACCEPT_ALL", "message": "Accepted"}],
        "warns": [{"validationRuleName": "PRICE_FAR", "message": "Limit price is far from the market."}],
        "alerts": [{"validationRuleName": "HOURS", "message": "Market is closed."}],
        "reviews": [],
        "rejects": [],
    },
    "commissionAndFee": {
        "commission": {"commissionLegs": [{"commissionValues": [{"value": 0.0, "type": "COMMISSION"}]}]},
        "fee": {"feeLegs": [{"feeValues": [{"value": 0.02, "type": "SEC_FEE"}, {"value": 0.01, "type": "TAF_FEE"}]}]},
    },
}
PREVIEW_LINE = (
    "est. value $1,500.00; commission and fees $0.03; buying power after $8,500.00; "
    "warnings: Limit price is far from the market. | Market is closed."
)


class RecordingApprovals(ApprovalManager):
    def __init__(self, decision: ApprovalDecision = ApprovalDecision.DENIED) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    async def require(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class FakeClient:
    """Fake Schwab client: get_order and preview_order return configured payloads."""

    class Instrument:
        Projection = Enum("Projection", {"SYMBOL_SEARCH": "symbol-search"})

    def __init__(self, order: Any = WORKING_ORDER, preview: Any = PREVIEW, order_error: bool = False) -> None:
        self.order = order
        self.preview = preview
        self.order_error = order_error
        self.calls: list[str] = []

    async def get_instruments(self, *args: Any, **kwargs: Any) -> Any:
        return {}

    async def get_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("get_order")
        if self.order_error:
            raise SchwabAPIError(status_code=500, url="https://api.schwabapi.com/trader", body="boom")
        return self.order

    async def preview_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("preview_order")
        return self.preview

    async def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("cancel_order")

    async def replace_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("replace_order")

    async def place_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("place_order")


@pytest.fixture(autouse=True)
def passthrough_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def call(func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("response_handler", None)
        return await func(*args, **kwargs)

    monkeypatch.setattr(orders, "call", call)


def _ctx(client: FakeClient, approvals: RecordingApprovals) -> SchwabContext:
    ctx = make_ctx(client)
    ctx.schwab.approval_manager = approvals
    return ctx


def _equity_preview(ctx: SchwabContext) -> str:
    result = run(orders.preview_equity_order(ctx, ACCOUNT, "AAPL", 10, "BUY", "LIMIT", price=150.0))
    return result["preview_id"]


def _replacement_preview(ctx: SchwabContext) -> str:
    replacement = {"symbol": "AAPL", "quantity": 5, "instruction": "BUY", "order_type": "LIMIT", "price": 149.0}
    result = run(orders.preview_replacement_order(ctx, ACCOUNT, ORDER_ID, cast(orders._OrderDescInput, replacement)))
    return result["preview_id"]


# --- #26: cancel shows the current order -------------------------------------


def test_cancel_prompt_shows_the_current_order() -> None:
    client, approvals = FakeClient(), RecordingApprovals()

    with pytest.raises(PermissionError):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    [request] = approvals.requests
    assert request.arguments["current_order"] == WORKING_SUMMARY
    assert client.calls == ["get_order"]


@pytest.mark.parametrize("status", ["FILLED", "CANCELED", "REJECTED", "EXPIRED", "REPLACED"])
def test_cancel_of_a_finished_order_is_refused_before_approval(status: str) -> None:
    client, approvals = FakeClient(order={**WORKING_ORDER, "status": status}), RecordingApprovals()

    with pytest.raises(ValueError, match=status):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    assert approvals.requests == []
    assert "cancel_order" not in client.calls


def test_cancel_of_a_non_cancelable_order_is_refused() -> None:
    client, approvals = FakeClient(order={**WORKING_ORDER, "cancelable": False}), RecordingApprovals()

    with pytest.raises(ValueError, match="cancel"):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    assert approvals.requests == []


@pytest.mark.parametrize("order", [None, [], "nope"])
def test_cancel_refused_when_the_order_cannot_be_read(order: Any) -> None:
    client, approvals = FakeClient(order=order), RecordingApprovals()

    with pytest.raises(ValueError, match=ORDER_ID):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    assert approvals.requests == []


@pytest.mark.parametrize("strip", [("orderLegCollection",), ("orderLegCollection", "orderType", "price")])
def test_cancel_refused_when_the_order_has_no_legs(strip: tuple[str, ...]) -> None:
    order = {key: value for key, value in WORKING_ORDER.items() if key not in strip}
    client, approvals = FakeClient(order=order), RecordingApprovals()

    with pytest.raises(ValueError, match="no legs"):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    assert approvals.requests == []


def test_cancel_refused_when_the_order_fetch_fails() -> None:
    client, approvals = FakeClient(order_error=True), RecordingApprovals()

    with pytest.raises(SchwabAPIError):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    assert approvals.requests == []


def test_current_order_summary_covers_child_orders() -> None:
    child = {**WORKING_ORDER, "orderId": 2, "orderType": "STOP", "stopPrice": 140.0}
    child.pop("price")
    parent = {**WORKING_ORDER, "orderStrategyType": "OCO", "childOrderStrategies": [WORKING_ORDER, child]}
    for key in ("orderLegCollection", "orderType", "price"):
        parent.pop(key)

    lines = orders._current_order_summary(parent).splitlines()

    assert lines[0].startswith("OCO ")
    assert "status=WORKING" in lines[0]
    assert lines[1].strip().startswith("SINGLE BUY 10 AAPL")
    assert "STOP" in lines[2] and "stop $140.0" in lines[2] and "orderId=2" in lines[2]


# --- #26: replace shows the current order next to the new one --------------


def test_replace_prompt_shows_current_and_new_order() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    preview_id = _replacement_preview(ctx)

    with pytest.raises(PermissionError):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview_id))

    [request] = approvals.requests
    assert request.arguments["current_order"] == WORKING_SUMMARY
    assert request.arguments["order_summary"].startswith("SINGLE BUY 5 AAPL EQUITY LIMIT @ $149.00")
    assert "replace_order" not in client.calls


@pytest.mark.parametrize("change", [{"status": "FILLED"}, {"editable": False}])
def test_replace_of_a_finished_or_locked_order_is_refused(change: dict[str, Any]) -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    preview_id = _replacement_preview(ctx)
    client.order = {**WORKING_ORDER, **change}

    with pytest.raises(ValueError, match="replace"):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview_id))

    assert approvals.requests == []
    assert "replace_order" not in client.calls


def test_replace_refusal_leaves_the_preview_usable() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    preview_id = _replacement_preview(ctx)
    client.order = {**WORKING_ORDER, "status": "FILLED"}

    with pytest.raises(ValueError):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview_id))

    assert ctx.previews.pop(preview_id, ACCOUNT, operation=PreviewOperation.REPLACE_ORDER).target_order_id == ORDER_ID


# --- #28: the prompt carries Schwab's preview result ------------------------


def test_preview_line_renders_schwab_preview() -> None:
    assert orders._preview_line(PREVIEW) == PREVIEW_LINE


@pytest.mark.parametrize("preview", [{}, None, "ok", {"orderValidationResult": {"type": "ACCEPTED"}}])
def test_preview_line_marks_missing_fields(preview: Any) -> None:
    assert orders._preview_line(preview) == (
        "est. value not provided; commission and fees not provided; buying power after not provided; no warnings"
    )


def test_preview_line_falls_back_to_projected_commission() -> None:
    preview = {"orderStrategy": {"orderValue": 10, "orderBalance": {"projectedCommission": 0.65}}}
    assert orders._preview_line(preview).startswith("est. value $10.00; commission and fees $0.65;")


def test_schwab_rejected_preview_is_refused_and_not_stored() -> None:
    rejected = {
        **PREVIEW,
        "orderValidationResult": {"rejects": [{"validationRuleName": "BP", "message": "Insufficient buying power."}]},
    }
    client, approvals = FakeClient(preview=rejected), RecordingApprovals()
    ctx = _ctx(client, approvals)

    with pytest.raises(ValueError, match="Insufficient buying power"):
        _equity_preview(ctx)

    assert ctx.previews._entries == {}


def test_place_prompt_carries_the_preview_line() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    preview_id = _equity_preview(ctx)

    with pytest.raises(PermissionError):
        run(orders.place_previewed_order(ctx, ACCOUNT, preview_id))

    [request] = approvals.requests
    assert request.arguments["schwab_preview"] == PREVIEW_LINE
    assert "place_order" not in client.calls


def test_replace_prompt_carries_the_preview_line() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    preview_id = _replacement_preview(ctx)

    with pytest.raises(PermissionError):
        run(orders.replace_previewed_order(ctx, ACCOUNT, preview_id))

    assert approvals.requests[0].arguments["schwab_preview"] == PREVIEW_LINE


# --- Signal renders the new fields -------------------------------------------


def _signal_body(request: ApprovalRequest) -> str:
    renderer = signal_backend._TOOL_RENDERERS[request.tool_name]
    return renderer(signal_backend._decode_arguments(request.arguments), {"1234": "Brokerage"})


def test_signal_cancel_shows_current_order() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    with pytest.raises(PermissionError):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, ORDER_ID))

    assert _signal_body(approvals.requests[0]) == (
        f"cancel order {ORDER_ID} in the Brokerage account:\n{WORKING_SUMMARY}"
    )


def test_signal_place_shows_preview_line() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    with pytest.raises(PermissionError):
        run(orders.place_previewed_order(ctx, ACCOUNT, _equity_preview(ctx)))

    body = _signal_body(approvals.requests[0])
    assert body.endswith(f"\nSchwab preview: {PREVIEW_LINE}")


def test_signal_replace_shows_current_new_and_preview() -> None:
    client, approvals = FakeClient(), RecordingApprovals()
    ctx = _ctx(client, approvals)
    with pytest.raises(PermissionError):
        run(orders.replace_previewed_order(ctx, ACCOUNT, _replacement_preview(ctx)))

    body = _signal_body(approvals.requests[0])
    assert body.startswith(
        f"replace order {ORDER_ID} in the Brokerage account (previewed via preview_replacement_order):"
    )
    assert f"\ncurrent:\n{WORKING_SUMMARY}\nnew:\nSINGLE BUY 5 AAPL" in body
    assert body.endswith(f"\nSchwab preview: {PREVIEW_LINE}")
