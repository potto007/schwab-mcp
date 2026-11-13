"""Registered tools refuse unknown top-level arguments before running.

Every call here goes through ``MCPServer.call_tool``, the same runner an MCP
client reaches, so argument validation happens exactly as in production.
"""

from __future__ import annotations

import datetime
import json
import types
from collections.abc import Awaitable, Callable
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import run, working_order
from mcp.server.mcpserver import Context as MCPContext, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
from pydantic import ConfigDict
from schwab.client import AsyncClient

from schwab_mcp import tools as tools_module
from schwab_mcp.approvals import ApprovalDecision, ApprovalManager, ApprovalRequest
from schwab_mcp.context import SchwabServerContext
from schwab_mcp.server import SchwabMCPServer
from schwab_mcp.tools import _registration, orders

ACCOUNT = "A" * 64
UNKNOWN = "zz_unknown_argument"


class RecordingApprovals(ApprovalManager):
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def require(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision.APPROVED


class RecordingClient:
    """Fake Schwab client: records every method a tool reaches."""

    class Instrument:
        Projection = Enum("Projection", {"SYMBOL_SEARCH": "symbol-search"})

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        async def method(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            if name != "get_order":
                return {}
            if any(called == "cancel_order" for called, _ in self.calls):
                return {"orderId": "1", "status": "CANCELED"}
            return working_order("1")

        return method


class Harness:
    def __init__(self) -> None:
        self.server = MCPServer(name="registered-arguments")
        self.client = RecordingClient()
        self.approvals = RecordingApprovals()
        tools_module.register_tools(
            self.server, cast(AsyncClient, self.client), allow_write=True, enable_technical=True
        )
        lifespan_context = SchwabServerContext(client=cast(AsyncClient, self.client), approval_manager=self.approvals)
        request_context = SimpleNamespace(
            lifespan_context=lifespan_context, request_id="request-1", meta={"client_id": "client-1"}
        )
        self.context = MCPContext.model_construct(_request_context=cast(Any, request_context), _mcp_server=self.server)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        return run(self.server.call_tool(name, arguments, self.context))

    def refused(self, name: str, arguments: dict[str, Any], key: str) -> None:
        with pytest.raises(ToolError, match=rf"(?s){key}.*Extra inputs are not permitted"):
            self.call(name, arguments)
        assert self.client.calls == []
        assert self.approvals.requests == []


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    async def passthrough(func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    monkeypatch.setattr(orders, "call", passthrough)
    return Harness()


def _production_server(client: Any = None) -> MCPServer:
    """The MCPServer inside a real SchwabMCPServer with every tool enabled."""
    schwab_server = SchwabMCPServer(
        "registered-arguments",
        cast(AsyncClient, client if client is not None else RecordingClient()),
        RecordingApprovals(),
        allow_write=True,
        enable_technical_tools=True,
        use_json=True,
    )
    return cast(MCPServer, schwab_server._server)


_ALL_TOOLS = sorted(tool.name for tool in run(_production_server().list_tools()))


def test_sweep_covers_read_write_and_technical_tools() -> None:
    assert {"get_order", "cancel_order", "place_previewed_order", "preview_bracket_order"} <= set(_ALL_TOOLS)
    assert "get_datetime" in _ALL_TOOLS
    assert len(_ALL_TOOLS) >= 30


@pytest.mark.parametrize("name", _ALL_TOOLS)
def test_every_production_tool_refuses_an_unknown_argument(name: str) -> None:
    client = RecordingClient()
    server = _production_server(client)

    with pytest.raises(ToolError, match=rf"(?s){UNKNOWN}.*Extra inputs are not permitted"):
        run(server.call_tool(name, {UNKNOWN: 1}))

    assert client.calls == []


@pytest.mark.parametrize("name", _ALL_TOOLS)
def test_every_production_tool_publishes_a_closed_schema(name: str) -> None:
    tool = next(tool for tool in run(_production_server().list_tools()) if tool.name == name)
    assert tool.input_schema["additionalProperties"] is False


def test_client_supplied_ctx_is_refused() -> None:
    with pytest.raises(ToolError, match=r"(?s)ctx.*Extra inputs are not permitted"):
        run(_production_server().call_tool("get_order", {"account_hash": ACCOUNT, "order_id": "1", "ctx": {}}))


def test_server_refuses_to_start_with_an_open_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """An SDK change that silently reopens a tool fails startup, not trading."""
    monkeypatch.setattr(_registration, "_forbid_unknown_arguments", lambda server, name: None)

    with pytest.raises(RuntimeError, match="accept unknown arguments"):
        _production_server()


@pytest.mark.parametrize("name", _ALL_TOOLS)
def test_every_production_tool_validates_arguments_strictly(name: str) -> None:
    tool = _production_server()._tool_manager.get_tool(name)
    assert tool is not None
    assert tool.fn_metadata.arg_model.model_config.get("strict") is True


def test_server_refuses_to_start_with_a_closed_but_lax_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that forbids unknown keys but coerces loose types fails startup."""

    def close_without_strict(server: MCPServer, name: str) -> None:
        tool = server._tool_manager.get_tool(name)
        assert tool is not None
        base: type[ArgModelBase] = tool.fn_metadata.arg_model
        closed = cast(
            type[ArgModelBase],
            types.new_class(
                base.__name__,
                (base,),
                exec_body=lambda ns: ns.update(__module__=base.__module__, model_config=ConfigDict(extra="forbid")),
            ),
        )
        tool.fn_metadata.arg_model = closed
        tool.parameters = closed.model_json_schema(by_alias=True)

    monkeypatch.setattr(_registration, "_forbid_unknown_arguments", close_without_strict)

    with pytest.raises(RuntimeError, match=r"not strict"):
        _production_server()


def test_tool_registered_around_register_tool_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    original = tools_module.register_tools

    def register_with_stray_tool(server: MCPServer, *args: Any, **kwargs: Any) -> None:
        original(server, *args, **kwargs)

        @server.tool(name="stray_tool")
        async def stray_tool(value: int) -> int:
            return value

    monkeypatch.setattr("schwab_mcp.server.register_tools", register_with_stray_tool)

    with pytest.raises(RuntimeError, match="stray_tool"):
        _production_server()


_BRACKET = {
    "account_hash": ACCOUNT,
    "symbol": "AAPL",
    "quantity": 1,
    "entry_instruction": "BUY",
    "entry_type": "LIMIT",
    "entry_price": 100.0,
    "profit_price": 120.0,
}
_EQUITY = {
    "account_hash": ACCOUNT,
    "symbol": "AAPL",
    "quantity": 1,
    "instruction": "SELL",
    "order_type": "MARKET",
}


def test_bracket_with_misnamed_stop_loss_is_refused(harness: Harness) -> None:
    harness.refused("preview_bracket_order", {**_BRACKET, "stop_loss": 90.0}, "stop_loss")


def test_bracket_with_loss_price_still_builds_the_stop_leg(harness: Harness) -> None:
    result = harness.call("preview_bracket_order", {**_BRACKET, "loss_price": 90.0})

    assert not result.is_error
    [(name, kwargs)] = [call for call in harness.client.calls if call[0] == "preview_order"]
    exits = kwargs["order_spec"]["childOrderStrategies"][0]["childOrderStrategies"]
    assert sorted(child["orderType"] for child in exits) == ["LIMIT", "STOP"]


def test_equity_order_with_camelcase_stop_price_is_refused(harness: Harness) -> None:
    harness.refused("preview_equity_order", {**_EQUITY, "stopPrice": 90.0}, "stopPrice")


def test_equity_order_with_known_arguments_still_previews(harness: Harness) -> None:
    result = harness.call("preview_equity_order", _EQUITY)

    assert not result.is_error
    assert [name for name, _ in harness.client.calls if name == "preview_order"] == ["preview_order"]


def test_read_tool_refuses_unknown_argument_and_still_reads(harness: Harness) -> None:
    harness.refused("get_order", {"account_hash": ACCOUNT, "order_id": "1", "include_fills": True}, "include_fills")

    result = harness.call("get_order", {"account_hash": ACCOUNT, "order_id": "1"})

    assert not result.is_error
    assert [name for name, _ in harness.client.calls] == ["get_order"]


def test_write_tool_refuses_unknown_argument_before_approval(harness: Harness) -> None:
    harness.refused("cancel_order", {"account_hash": ACCOUNT, "order_id": "1", "force": True}, "force")

    result = harness.call("cancel_order", {"account_hash": ACCOUNT, "order_id": "1"})

    assert not result.is_error
    assert len(harness.approvals.requests) == 1
    assert [name for name, _ in harness.client.calls] == ["get_order", "cancel_order", "get_order"]


_LIMIT = {**_EQUITY, "order_type": "LIMIT", "price": 150.0}
_OCO_LEG = {"symbol": "AAPL", "quantity": 1, "instruction": "SELL", "order_type": "LIMIT", "price": 150.0}
_OCO_STOP = {"symbol": "AAPL", "quantity": 1, "instruction": "SELL", "order_type": "STOP", "stop_price": 90.0}


def _refused_loose(harness: Harness, name: str, arguments: dict[str, Any], pattern: str) -> None:
    with pytest.raises(ToolError, match=rf"(?s){pattern}"):
        harness.call(name, arguments)
    assert harness.client.calls == []
    assert harness.approvals.requests == []


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        pytest.param({"quantity": "5"}, r"quantity.*valid integer", id="string-quantity"),
        pytest.param({"quantity": True}, r"quantity.*valid integer", id="bool-quantity"),
        pytest.param({"price": "1.15"}, r"price.*valid number", id="string-price"),
    ],
)
def test_equity_order_refuses_loosely_typed_arguments(
    harness: Harness, overrides: dict[str, Any], pattern: str
) -> None:
    _refused_loose(harness, "preview_equity_order", {**_LIMIT, **overrides}, pattern)


def test_nested_oco_leg_refuses_a_string_quantity(harness: Harness) -> None:
    _refused_loose(
        harness,
        "preview_oco_order",
        {"account_hash": ACCOUNT, "first_order": {**_OCO_LEG, "quantity": "5"}, "second_order": _OCO_STOP},
        r"first_order.*quantity.*valid integer",
    )


def test_read_tool_refuses_an_int_where_a_string_is_expected(harness: Harness) -> None:
    _refused_loose(harness, "get_order", {"account_hash": ACCOUNT, "order_id": 123}, r"order_id.*valid string")


def test_integer_price_is_still_accepted_for_a_float_field(harness: Harness) -> None:
    result = harness.call("preview_equity_order", {**_LIMIT, "price": 150})

    assert not result.is_error
    [(name, kwargs)] = [call for call in harness.client.calls if call[0] == "preview_order"]
    assert kwargs["order_spec"]["price"] == "150.00"


def test_nested_order_sent_as_a_json_string_is_still_accepted(harness: Harness) -> None:
    arguments = {
        "account_hash": ACCOUNT,
        "first_order": json.dumps(_OCO_LEG),
        "second_order": json.dumps(_OCO_STOP),
    }

    result = harness.call("preview_oco_order", arguments)

    assert not result.is_error
    assert [name for name, _ in harness.client.calls if name == "preview_order"] == ["preview_order"]


def test_nested_order_sent_as_a_json_string_still_refuses_loose_leg_types(harness: Harness) -> None:
    arguments = {
        "account_hash": ACCOUNT,
        "first_order": json.dumps({**_OCO_LEG, "quantity": "5"}),
        "second_order": _OCO_STOP,
    }

    _refused_loose(harness, "preview_oco_order", arguments, r"first_order.*quantity.*valid integer")


@pytest.mark.parametrize("name", ["get_option_chain", "get_advanced_option_chain"])
def test_option_chain_expiration_dates_still_accept_iso_strings(name: str) -> None:
    tool = _production_server()._tool_manager.get_tool(name)
    assert tool is not None

    validated = tool.fn_metadata.validate_arguments(
        {"symbol": "AAPL", "from_date": "2024-03-01", "to_date": "2024-04-01"}
    )

    assert validated["from_date"] == datetime.date(2024, 3, 1)
    assert validated["to_date"] == datetime.date(2024, 4, 1)
