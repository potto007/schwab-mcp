"""The server enables write tools only with trading limits set (#30)."""

from __future__ import annotations

from typing import Any

import pytest
from schwab.client import AsyncClient

from schwab_mcp import cli
from schwab_mcp.approvals import NoOpApprovalManager
from schwab_mcp.limits import OrderLedger, TradingLimits
from schwab_mcp.server import SchwabMCPServer

BASE = ["server", "--client-id", "client", "--client-secret", "secret"]
WRITE = [*BASE, "--jesus-take-the-wheel"]


def _invoke(cli_runner, args: list[str]) -> Any:
    return cli_runner.invoke(cli.cli, args, catch_exceptions=False)


@pytest.fixture
def no_limit_env(monkeypatch) -> None:
    for name in (
        "SCHWAB_MCP_MAX_ORDER_VALUE",
        "SCHWAB_MCP_MAX_DAILY_ORDERS",
        "SCHWAB_MCP_MAX_ORDER_QUANTITY",
        "SCHWAB_MCP_ALLOWED_SYMBOLS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "limit_args",
    [[], ["--max-order-value", "1000"], ["--max-daily-orders", "5"]],
)
def test_write_mode_without_required_limits_refuses_to_start(
    cli_server_capture, cli_runner, no_limit_env, limit_args: list[str]
) -> None:
    result = _invoke(cli_runner, [*WRITE, *limit_args])

    assert result.exit_code == 1
    assert "--max-order-value" in result.output
    assert "--max-daily-orders" in result.output
    assert "server_name" not in cli_server_capture


def test_limits_from_flags_reach_the_server(cli_server_capture, cli_runner, no_limit_env) -> None:
    result = _invoke(
        cli_runner,
        [
            *WRITE,
            "--max-order-value",
            "2500.50",
            "--max-daily-orders",
            "4",
            "--max-order-quantity",
            "10",
            "--allowed-symbol",
            "spy",
            "--allowed-symbol",
            "aapl,msft",
        ],
    )

    assert result.exit_code == 0
    assert cli_server_capture["allow_write"] is True
    assert cli_server_capture["trading_limits"] == TradingLimits(
        max_order_value=2500.5,
        max_daily_orders=4,
        max_order_quantity=10,
        allowed_symbols=frozenset({"SPY", "AAPL", "MSFT"}),
    )
    assert isinstance(cli_server_capture["order_ledger"], OrderLedger)


def test_limits_from_env_reach_the_server(cli_server_capture, cli_runner, no_limit_env, monkeypatch) -> None:
    monkeypatch.setenv("SCHWAB_MCP_MAX_ORDER_VALUE", "1000")
    monkeypatch.setenv("SCHWAB_MCP_MAX_DAILY_ORDERS", "2")
    monkeypatch.setenv("SCHWAB_MCP_MAX_ORDER_QUANTITY", "3")
    monkeypatch.setenv("SCHWAB_MCP_ALLOWED_SYMBOLS", "spy, qqq")

    result = _invoke(cli_runner, WRITE)

    assert result.exit_code == 0
    assert cli_server_capture["trading_limits"] == TradingLimits(
        max_order_value=1000.0,
        max_daily_orders=2,
        max_order_quantity=3,
        allowed_symbols=frozenset({"SPY", "QQQ"}),
    )


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-5"])
def test_invalid_order_value_refuses_to_start(cli_server_capture, cli_runner, no_limit_env, value: str) -> None:
    result = _invoke(cli_runner, [*WRITE, "--max-order-value", value, "--max-daily-orders", "2"])

    assert result.exit_code != 0
    assert "server_name" not in cli_server_capture


def test_unreadable_ledger_refuses_to_start(cli_server_capture, cli_runner, no_limit_env) -> None:
    with open(cli_server_capture["ledger_path"], "w") as ledger:
        ledger.write("not json")

    result = _invoke(cli_runner, [*WRITE, "--max-order-value", "1000", "--max-daily-orders", "2"])

    assert result.exit_code == 1
    assert "daily-orders" in result.output
    assert "server_name" not in cli_server_capture


def test_limits_without_write_mode_warn_and_stay_read_only(cli_server_capture, cli_runner, no_limit_env) -> None:
    result = _invoke(cli_runner, [*BASE, "--max-order-value", "1000", "--max-daily-orders", "2"])

    assert result.exit_code == 0
    assert cli_server_capture["allow_write"] is False
    assert cli_server_capture["trading_limits"] is None
    assert "trading limits are ignored" in (result.output + (result.stderr or ""))


def test_server_refuses_write_tools_without_limits() -> None:
    with pytest.raises(ValueError, match="trading limits"):
        SchwabMCPServer(
            "schwab-mcp",
            AsyncClient.__new__(AsyncClient),
            approval_manager=NoOpApprovalManager(),
            allow_write=True,
        )
