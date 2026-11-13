"""Equity and option order symbols each have one strict grammar.

Equity: ``[A-Za-z0-9.$/-]{1,15}`` with no whitespace. Option: Schwab's
21-character form, a 1-6 character root of ``[A-Z0-9.]`` right-padded with
spaces to 6, ``YYMMDD``, ``C`` or ``P``, then 8 strike digits.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import run
from mcp.server.mcpserver import Context as MCPContext, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from schwab.client import AsyncClient

from schwab_mcp import tools as tools_module
from schwab_mcp.approvals import ApprovalDecision, ApprovalManager, ApprovalRequest
from schwab_mcp.context import SchwabServerContext
from schwab_mcp.tools import orders

ACCOUNT = "A" * 64
OPTION = "SPY   251219C00500000"
OPTION_2 = "SPY   251219C00510000"

GOOD_EQUITY = ["AAPL", "BRK.B", "BRK/B", "$SPX", "/ES", "BF-B", "X", "A" * 15]
BAD_EQUITY = [
    "AA PL",
    "AAPL ",
    " AAPL",
    "AA\tPL",
    "\tAAPL",
    "AAPL\n",
    "A" * 16,
    "",
    " ",
    "AAPL;",
    OPTION,
]

GOOD_OPTION = [
    OPTION,
    "SPY   251219P00500000",
    "A     251219C00001000",
    "BRK.B 251219C00500000",
    "ABCDEF251219C00500000",
    "BF.B  251219P00045000",
    "X1    251219C00500000",
]
BAD_OPTION = [
    "SPY 251219C500",  # short form
    "SPY 251219C00500000",  # length 19
    "SPY   251219C0050000",  # length 20
    "SPY   251219C005000000",  # length 22
    "spy   251219C00500000",  # lowercase root
    "SPY   251219c00500000",  # lowercase type letter
    "SPY   251219X00500000",  # bad type letter
    "SPY   251219 00500000",  # missing type letter
    "SPY251219C00500000",  # unpadded root
    "SPY    251219C00500000",  # over-padded root
    " SPY  251219C00500000",  # leading space
    "SPY  251219C00500000 ",  # trailing space in place of padding
    "      251219C00500000",  # blank root
    "SPYABCD251219C00500000",  # 7-char root
    "SPY   25121XC00500000",  # non-digit date
    "SPY   251219C0050000X",  # non-digit strike
    "SPY\t  251219C00500000",  # tab in padding
    "SPY   251219C00500000\n",
    "",
    "AAPL",
]


@pytest.mark.parametrize("symbol", GOOD_EQUITY)
def test_equity_helper_accepts(symbol: str) -> None:
    assert orders._validate_equity_symbol(symbol) == symbol


@pytest.mark.parametrize("symbol", BAD_EQUITY)
def test_equity_helper_refuses(symbol: str) -> None:
    with pytest.raises(ValueError, match="1 to 15"):
        orders._validate_equity_symbol(symbol)


@pytest.mark.parametrize("symbol", [None, 1, ["AAPL"], b"AAPL"])
def test_equity_helper_refuses_non_strings(symbol: Any) -> None:
    with pytest.raises(ValueError, match="symbol"):
        orders._validate_equity_symbol(symbol)


@pytest.mark.parametrize("symbol", GOOD_OPTION)
def test_option_helper_accepts(symbol: str) -> None:
    assert len(symbol) == 21
    assert orders._validate_option_symbol(symbol) == symbol


@pytest.mark.parametrize("symbol", BAD_OPTION)
def test_option_helper_refuses(symbol: str) -> None:
    with pytest.raises(ValueError, match="21-character"):
        orders._validate_option_symbol(symbol)


@pytest.mark.parametrize("symbol", [None, 1, [OPTION], OPTION.encode()])
def test_option_helper_refuses_non_strings(symbol: Any) -> None:
    with pytest.raises(ValueError, match="symbol"):
        orders._validate_option_symbol(symbol)


def test_option_error_names_the_field_and_format() -> None:
    with pytest.raises(ValueError, match=r"legs\[1\]\.symbol.*21-character.*SPY   251219C00500000"):
        orders._validate_option_symbol("SPY 251219C500", "legs[1].symbol")


def test_create_option_symbol_output_is_a_valid_option_symbol() -> None:
    created = run(orders.create_option_symbol("SPY", "251219", "C", "500"))
    assert created == OPTION
    assert orders._validate_option_symbol(created) == created
    created = run(orders.create_option_symbol("BRK.B", "251219", "PUT", "45.5"))
    assert orders._validate_option_symbol(created) == created


# ---------------------------------------------------------------------------
# Through the registered tools (MCPServer.call_tool)
# ---------------------------------------------------------------------------


class _Approvals(ApprovalManager):
    async def require(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVED


class _Client:
    """Fake Schwab client: records every method a tool reaches."""

    class Instrument:
        Projection = Enum("Projection", {"SYMBOL_SEARCH": "symbol-search"})

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        async def method(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            return {}

        return method


class _Harness:
    def __init__(self) -> None:
        self.server = MCPServer(name="symbol-grammar")
        self.client = _Client()
        tools_module.register_tools(self.server, cast(AsyncClient, self.client), allow_write=True)
        lifespan_context = SchwabServerContext(client=cast(AsyncClient, self.client), approval_manager=_Approvals())
        request_context = SimpleNamespace(lifespan_context=lifespan_context, request_id="r1", meta={"client_id": "c1"})
        self.context = MCPContext.model_construct(_request_context=cast(Any, request_context), _mcp_server=self.server)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        return run(self.server.call_tool(name, {"account_hash": ACCOUNT, **arguments}, self.context))

    def accepted(self, name: str, arguments: dict[str, Any]) -> None:
        self.call(name, arguments)
        assert any(call[0] == "preview_order" for call in self.client.calls)

    def refused(self, name: str, arguments: dict[str, Any], match: str) -> None:
        with pytest.raises(ToolError) as excinfo:
            self.call(name, arguments)
        # The tool's ValueError is the cause; the wrapper carries only the tool name.
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert re.search(match, str(excinfo.value.__cause__))
        assert not any(call[0] == "preview_order" for call in self.client.calls)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    async def passthrough(func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    monkeypatch.setattr(orders, "call", passthrough)
    return _Harness()


def _equity_leg(symbol: str, instruction: str, order_type: str, **extra: Any) -> dict[str, Any]:
    return {"symbol": symbol, "quantity": 1, "instruction": instruction, "order_type": order_type, **extra}


def _equity_calls(symbol: str) -> dict[str, dict[str, Any]]:
    return {
        "preview_equity_order": {"symbol": symbol, "quantity": 1, "instruction": "BUY", "order_type": "MARKET"},
        "preview_equity_trailing_stop_order": {
            "symbol": symbol,
            "quantity": 1,
            "instruction": "SELL",
            "trail_offset": 1.0,
        },
        "preview_bracket_order": {
            "symbol": symbol,
            "quantity": 1,
            "entry_instruction": "BUY",
            "entry_type": "MARKET",
            "profit_price": 110.0,
            "loss_price": 90.0,
        },
        "preview_oco_order": {
            "first_order": _equity_leg(symbol, "SELL", "LIMIT", price=110.0),
            "second_order": _equity_leg(symbol, "SELL", "STOP", stop_price=90.0),
        },
        "preview_trigger_order": {
            "entry_order": _equity_leg(symbol, "BUY", "MARKET"),
            "exit_orders": [_equity_leg(symbol, "SELL", "LIMIT", price=110.0)],
        },
        "preview_replacement_order": {
            "order_id": "123",
            "replacement_order": _equity_leg(symbol, "BUY", "MARKET"),
        },
    }


_EQUITY_TOOL_NAMES = list(_equity_calls("AAPL"))


@pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "BRK/B"])
@pytest.mark.parametrize("name", _EQUITY_TOOL_NAMES)
def test_equity_tools_accept_valid_symbols(harness: _Harness, name: str, symbol: str) -> None:
    harness.accepted(name, _equity_calls(symbol)[name])


@pytest.mark.parametrize("symbol", ["AA PL", " AAPL", "AAPL ", "AA\tPL", "A" * 16, OPTION])
@pytest.mark.parametrize("name", _EQUITY_TOOL_NAMES)
def test_equity_tools_refuse_invalid_symbols(harness: _Harness, name: str, symbol: str) -> None:
    harness.refused(name, _equity_calls(symbol)[name], "1 to 15")


def _option_leg(symbol: str, instruction: str, **extra: Any) -> dict[str, Any]:
    return _equity_leg(symbol, instruction, "LIMIT", asset_type="OPTION", **extra)


def _option_calls(symbol: str) -> dict[str, dict[str, Any]]:
    return {
        "preview_option_order": {
            "symbol": symbol,
            "quantity": 1,
            "instruction": "BUY_TO_OPEN",
            "order_type": "LIMIT",
            "price": 2.5,
        },
        "preview_option_combo_order": {
            "legs": [
                {"symbol": symbol, "quantity": 1, "instruction": "BUY_TO_OPEN"},
                {"symbol": OPTION_2, "quantity": 1, "instruction": "SELL_TO_OPEN"},
            ],
            "order_type": "NET_DEBIT",
            "price": 1.0,
        },
        "preview_oco_order": {
            "first_order": _option_leg(symbol, "SELL_TO_CLOSE", price=2.0),
            "second_order": _option_leg(symbol, "SELL_TO_CLOSE", price=1.0),
        },
        "preview_trigger_order": {
            "entry_order": _option_leg(symbol, "BUY_TO_OPEN", price=2.0),
            "exit_orders": [_option_leg(symbol, "SELL_TO_CLOSE", price=3.0)],
        },
        "preview_replacement_order": {
            "order_id": "123",
            "replacement_order": _option_leg(symbol, "BUY_TO_OPEN", price=2.0),
        },
    }


_OPTION_TOOL_NAMES = list(_option_calls(OPTION))


@pytest.mark.parametrize("name", _OPTION_TOOL_NAMES)
def test_option_tools_accept_a_valid_symbol(harness: _Harness, name: str) -> None:
    harness.accepted(name, _option_calls(OPTION)[name])


@pytest.mark.parametrize(
    "symbol",
    [
        "SPY 251219C500",
        "spy   251219C00500000",
        "SPY    251219C00500000",
        "SPY   251219C0050000",
        "SPY   251219C005000000",
        "SPY   251219X00500000",
        "AAPL",
    ],
)
@pytest.mark.parametrize("name", _OPTION_TOOL_NAMES)
def test_option_tools_refuse_invalid_symbols(harness: _Harness, name: str, symbol: str) -> None:
    harness.refused(name, _option_calls(symbol)[name], "21-character")


def test_combo_leg_error_names_the_leg(harness: _Harness) -> None:
    arguments = {
        "legs": [
            {"symbol": OPTION, "quantity": 1, "instruction": "BUY_TO_OPEN"},
            {"symbol": "SPY 251219C510", "quantity": 1, "instruction": "SELL_TO_OPEN"},
        ],
        "order_type": "NET_DEBIT",
        "price": 1.0,
    }
    harness.refused("preview_option_combo_order", arguments, r"legs\[1\]\.symbol")


def test_desc_asset_type_chooses_the_grammar(harness: _Harness) -> None:
    option_as_equity = _equity_leg(OPTION, "BUY", "MARKET")
    equity_as_option = _equity_leg("AAPL", "BUY_TO_OPEN", "MARKET", asset_type="OPTION")
    harness.refused("preview_replacement_order", {"order_id": "1", "replacement_order": option_as_equity}, "1 to 15")
    harness.refused(
        "preview_replacement_order", {"order_id": "1", "replacement_order": equity_as_option}, "21-character"
    )


def test_published_option_symbol_examples_are_valid() -> None:
    from schwab_mcp.resources import OPTION_SYMBOLS

    for example in OPTION_SYMBOLS["examples"]:
        assert orders._validate_option_symbol(example["symbol"]) == example["symbol"]
