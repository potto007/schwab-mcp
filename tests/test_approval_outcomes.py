"""After an approved write, the reviewer hears what happened (#27).

The approval channel gets one outcome line per approved write: the order id
and Schwab status on success, the error on failure. Denied and expired
requests send nothing extra, and a failing or stalled report never changes
the tool's result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, cast

import pytest
from conftest import make_ctx, run, working_order

from schwab_mcp.approvals import ApprovalDecision, ApprovalManager, ApprovalRequest, NoOpApprovalManager
from schwab_mcp.approvals.base import format_outcome
from schwab_mcp.context import SchwabContext
from schwab_mcp.tools import _registration, orders
from schwab_mcp.tools.utils import SchwabAPIError

ACCOUNT = "A" * 60 + "1234"
TARGET_ID = "1000123"
NEW_ID = 42


class OutcomeApprovals(ApprovalManager):
    """Approves or denies, and records every outcome report."""

    def __init__(self, decision: ApprovalDecision = ApprovalDecision.APPROVED) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []
        self.outcomes: list[tuple[str, str, bool]] = []

    async def require(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision

    async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
        self.outcomes.append((request.id, outcome, succeeded))


class BrokenApprovals(OutcomeApprovals):
    async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
        raise RuntimeError("approval channel is down")


class StalledApprovals(OutcomeApprovals):
    async def report_outcome(self, request: ApprovalRequest, outcome: str, *, succeeded: bool) -> None:
        await asyncio.sleep(3600)


class Client:
    """Fake Schwab client: get_order serves queued payloads, writes return a write result."""

    class Instrument:
        Projection = Enum("Projection", {"SYMBOL_SEARCH": "symbol-search"})

    def __init__(
        self,
        orders_served: list[Any] | None = None,
        write_result: Any = None,
        write_error: Exception | None = None,
    ) -> None:
        self.orders_served = list(orders_served or [])
        self.write_result = {"orderId": NEW_ID, "accountHash": ACCOUNT} if write_result is None else write_result
        self.write_error = write_error
        self.calls: list[str] = []

    async def get_instruments(self, symbol: str, **kwargs: Any) -> Any:
        return {"instruments": [{"symbol": symbol, "assetType": "EQUITY"}]}

    async def preview_order(self, *args: Any, **kwargs: Any) -> Any:
        return {}

    async def get_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("get_order")
        if not self.orders_served:
            raise SchwabAPIError(status_code=503, url="https://api.schwabapi.com/trader", body="unavailable")
        return self.orders_served.pop(0)

    async def _write(self, name: str) -> Any:
        self.calls.append(name)
        if self.write_error is not None:
            raise self.write_error
        return self.write_result

    async def place_order(self, *args: Any, **kwargs: Any) -> Any:
        return await self._write("place_order")

    async def replace_order(self, *args: Any, **kwargs: Any) -> Any:
        return await self._write("replace_order")

    async def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("cancel_order")
        if self.write_error is not None:
            raise self.write_error
        return None


@pytest.fixture(autouse=True)
def passthrough_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def call(func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("response_handler", None)
        return await func(*args, **kwargs)

    monkeypatch.setattr(orders, "call", call)


def _ctx(client: Client, approvals: ApprovalManager) -> SchwabContext:
    ctx = make_ctx(client)
    ctx.schwab.approval_manager = approvals
    return ctx


def _place(ctx: SchwabContext) -> Any:
    preview = run(orders.preview_equity_order(ctx, ACCOUNT, "AAPL", 1, "BUY", "LIMIT", price=1.0))
    return run(orders.place_previewed_order(ctx, ACCOUNT, preview["preview_id"]))


def _replace(ctx: SchwabContext) -> Any:
    replacement = {"symbol": "AAPL", "quantity": 1, "instruction": "BUY", "order_type": "LIMIT", "price": 2.0}
    preview = run(orders.preview_replacement_order(ctx, ACCOUNT, TARGET_ID, cast(orders._OrderDescInput, replacement)))
    return run(orders.replace_previewed_order(ctx, ACCOUNT, preview["preview_id"]))


# --- place ---------------------------------------------------------------------


def test_approved_place_reports_the_new_order_and_its_status() -> None:
    client, approvals = Client([working_order(NEW_ID)]), OutcomeApprovals()

    result = _place(_ctx(client, approvals))

    [request] = approvals.requests
    assert approvals.outcomes == [(request.id, f"placed: order {NEW_ID}, status WORKING", True)]
    assert result["orderId"] == NEW_ID


def test_rejected_order_reports_schwabs_reason() -> None:
    rejected = {**working_order(NEW_ID), "status": "REJECTED", "statusDescription": "Buying power exceeded"}
    client, approvals = Client([rejected]), OutcomeApprovals()

    result = _place(_ctx(client, approvals))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == f"placed: order {NEW_ID}, status REJECTED (Buying power exceeded)"
    assert succeeded is True
    # The agent sees the reason as well.
    assert result["statusDescription"] == "Buying power exceeded"


def test_place_reports_an_unconfirmed_status_when_the_fetch_fails() -> None:
    client, approvals = Client([]), OutcomeApprovals()

    _place(_ctx(client, approvals))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == f"placed: order {NEW_ID}, status not confirmed; Order placed; status fetch failed"
    assert succeeded is True


def test_place_without_an_order_id_says_so() -> None:
    client, approvals = Client([], write_result={"location": "somewhere"}), OutcomeApprovals()

    _place(_ctx(client, approvals))

    [(_, outcome, _)] = approvals.outcomes
    assert outcome == "placed: Schwab returned no order id"


def test_failed_place_reports_the_schwab_error_without_the_account_hash() -> None:
    error = SchwabAPIError(
        status_code=400,
        url=f"https://api.schwabapi.com/trader/v1/accounts/{ACCOUNT}/orders",
        body='{"message": "Order rejected: market closed"}',
    )
    client, approvals = Client([], write_error=error), OutcomeApprovals()

    with pytest.raises(SchwabAPIError):
        _place(_ctx(client, approvals))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == 'place failed: Schwab API error (HTTP 400): {"message": "Order rejected: market closed"}'
    assert succeeded is False
    assert ACCOUNT not in outcome


def test_failed_place_with_an_unexpected_error_reports_its_type() -> None:
    client, approvals = Client([], write_error=ConnectionError(f"reset talking to {ACCOUNT}")), OutcomeApprovals()

    with pytest.raises(ConnectionError):
        _place(_ctx(client, approvals))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == "place failed: ConnectionError: reset talking to …1234"
    assert succeeded is False


@pytest.mark.parametrize(
    ("decision", "error"),
    [(ApprovalDecision.DENIED, PermissionError), (ApprovalDecision.EXPIRED, TimeoutError)],
)
def test_denied_or_expired_place_reports_no_outcome(decision: ApprovalDecision, error: type[Exception]) -> None:
    client, approvals = Client([working_order(NEW_ID)]), OutcomeApprovals(decision)

    with pytest.raises(error):
        _place(_ctx(client, approvals))

    assert approvals.outcomes == []
    assert "place_order" not in client.calls


@pytest.mark.parametrize("approvals_type", [BrokenApprovals, StalledApprovals])
def test_a_failing_report_never_changes_the_tool_result(
    approvals_type: type[OutcomeApprovals], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_registration, "_OUTCOME_REPORT_TIMEOUT_SECONDS", 0.05)
    client, approvals = Client([working_order(NEW_ID)]), approvals_type()

    result = _place(_ctx(client, approvals))

    assert result["orderId"] == NEW_ID
    assert result["status"] == "WORKING"


def test_a_failing_report_never_masks_the_write_error() -> None:
    error = SchwabAPIError(status_code=400, url="https://api.schwabapi.com/trader", body="nope")
    client, approvals = Client([], write_error=error), BrokenApprovals()

    with pytest.raises(SchwabAPIError):
        _place(_ctx(client, approvals))


# --- replace and cancel --------------------------------------------------------


def test_approved_replace_reports_the_target_and_the_new_order() -> None:
    client = Client([working_order(TARGET_ID), working_order(NEW_ID)])
    approvals = OutcomeApprovals()

    _replace(_ctx(client, approvals))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == f"replaced order {TARGET_ID}: order {NEW_ID}, status WORKING"
    assert succeeded is True


def test_failed_replace_reports_the_error() -> None:
    error = SchwabAPIError(status_code=400, url="https://api.schwabapi.com/trader", body="cannot replace")
    client, approvals = Client([working_order(TARGET_ID)], write_error=error), OutcomeApprovals()

    with pytest.raises(SchwabAPIError):
        _replace(_ctx(client, approvals))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == "replace failed: Schwab API error (HTTP 400): cannot replace"
    assert succeeded is False


def test_approved_cancel_reports_the_order_status() -> None:
    canceled = {**working_order(TARGET_ID), "status": "CANCELED"}
    client, approvals = Client([working_order(TARGET_ID), canceled]), OutcomeApprovals()

    run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, TARGET_ID))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == f"cancel submitted for order {TARGET_ID}: order {TARGET_ID}, status CANCELED"
    assert succeeded is True


def test_cancel_reports_pending_when_the_status_fetch_fails() -> None:
    client, approvals = Client([working_order(TARGET_ID)]), OutcomeApprovals()

    run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, TARGET_ID))

    [(_, outcome, _)] = approvals.outcomes
    assert outcome == (
        f"cancel submitted for order {TARGET_ID}: order {TARGET_ID}, status PENDING_CANCEL; "
        "Cancel submitted; status fetch failed"
    )


def test_failed_cancel_reports_the_error() -> None:
    error = SchwabAPIError(status_code=404, url="https://api.schwabapi.com/trader", body="order not found")
    client, approvals = Client([working_order(TARGET_ID)], write_error=error), OutcomeApprovals()

    with pytest.raises(SchwabAPIError):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, TARGET_ID))

    [(_, outcome, succeeded)] = approvals.outcomes
    assert outcome == "cancel failed: Schwab API error (HTTP 404): order not found"
    assert succeeded is False


def test_denied_cancel_reports_no_outcome() -> None:
    client, approvals = Client([working_order(TARGET_ID)]), OutcomeApprovals(ApprovalDecision.DENIED)

    with pytest.raises(PermissionError):
        run(orders.cancel_order(_ctx(client, approvals), ACCOUNT, TARGET_ID))

    assert approvals.outcomes == []


# --- automatically wrapped write tools ----------------------------------------


def test_wrapped_write_tool_reports_completion() -> None:
    approvals = OutcomeApprovals()
    ctx = _ctx(Client(), approvals)

    async def write_tool(ctx: SchwabContext, value: str) -> str:
        return f"wrote {value}"

    wrapped = _registration._wrap_with_approval(write_tool)

    assert run(wrapped(ctx, "x")) == "wrote x"
    [request] = approvals.requests
    assert approvals.outcomes == [(request.id, "completed", True)]


def test_wrapped_write_tool_reports_failure() -> None:
    approvals = OutcomeApprovals()
    ctx = _ctx(Client(), approvals)

    async def write_tool(ctx: SchwabContext, value: str) -> str:
        raise ValueError(f"bad {value}")

    wrapped = _registration._wrap_with_approval(write_tool)

    with pytest.raises(ValueError, match="bad x"):
        run(wrapped(ctx, "x"))
    assert [(outcome, succeeded) for _, outcome, succeeded in approvals.outcomes] == [
        ("failed: ValueError: bad x", False)
    ]


# --- base -----------------------------------------------------------------------


def test_managers_without_a_channel_ignore_outcomes() -> None:
    request = ApprovalRequest(id="a", tool_name="t", request_id="r", client_id=None, arguments={})

    assert run(NoOpApprovalManager().report_outcome(request, "placed", succeeded=True)) is None


def test_format_outcome_keeps_one_inert_line() -> None:
    text = "line one\nline two‮`code`\x00 end"

    assert format_outcome(text, limit=200) == "line one line twoˋcodeˋ  end"


def test_format_outcome_truncates() -> None:
    assert format_outcome("x" * 50, limit=10) == "x" * 9 + "…"
