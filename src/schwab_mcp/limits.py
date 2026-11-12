"""Trading limits that must be configured before write tools are enabled.

``TradingLimits`` holds the operator's caps: max order value and max orders
per day are required, max quantity and a symbol allowlist are optional.
``check_order`` applies them to an order spec before the reviewer is asked.
``OrderLedger`` counts the orders sent to Schwab per US/Eastern market day in
an owner-only file, so the daily cap survives a restart and holds across
every server process that shares the data directory.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from schwab_mcp import tokens

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None

_MARKET_TZ = ZoneInfo("America/New_York")

# Option prices are quoted per share; one contract covers 100 shares.
_OPTION_MULTIPLIER = 100


@dataclass(frozen=True, slots=True)
class TradingLimits:
    """Operator caps applied to every order write before approval."""

    max_order_value: float
    """Largest estimated order value, in dollars, a write may send."""
    max_daily_orders: int
    """Most orders (placements and replacements) sent per US/Eastern day."""
    max_order_quantity: int | None = None
    """Largest quantity on any leg (shares or contracts)."""
    allowed_symbols: frozenset[str] | None = None
    """Uppercase equity symbols and option underlyings that may be traded."""

    def __post_init__(self) -> None:
        value = self.max_order_value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("max order value must be a positive, finite number of dollars")
        if self.max_daily_orders < 1:
            raise ValueError("max daily orders must be at least 1")
        if self.max_order_quantity is not None and self.max_order_quantity < 1:
            raise ValueError("max order quantity must be at least 1")
        if self.allowed_symbols is not None and not self.allowed_symbols:
            raise ValueError("the symbol allowlist cannot be empty")


def parse_symbol_allowlist(values: Iterable[str]) -> frozenset[str] | None:
    """Parse comma-separated symbol entries into an uppercase set, or None if empty."""
    symbols = {chunk.strip().upper() for value in values for chunk in value.split(",") if chunk.strip()}
    return frozenset(symbols) or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _nodes(order_spec: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    stack: list[Any] = [order_spec]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        nodes.append(node)
        children = node.get("childOrderStrategies")
        if isinstance(children, list):
            stack.extend(children)
    return nodes


def _legs(node: dict[str, Any]) -> list[dict[str, Any]]:
    legs = node.get("orderLegCollection")
    return [leg for leg in legs if isinstance(leg, dict)] if isinstance(legs, list) else []


def _instrument(leg: dict[str, Any]) -> dict[str, Any]:
    instrument = leg.get("instrument")
    return instrument if isinstance(instrument, dict) else {}


def _is_option(leg: dict[str, Any]) -> bool:
    return str(_instrument(leg).get("assetType", "")).upper() == "OPTION"


def _node_value(node: dict[str, Any], legs: list[dict[str, Any]]) -> float | None:
    prices = [price for price in (_number(node.get("price")), _number(node.get("stopPrice"))) if price is not None]
    quantities = [_number(leg.get("quantity")) for leg in legs]
    if not prices or any(quantity is None for quantity in quantities):
        return None
    multiplier = _OPTION_MULTIPLIER if any(_is_option(leg) for leg in legs) else 1
    return max(abs(price) for price in prices) * max(abs(q) for q in quantities if q is not None) * multiplier


def order_value(order_spec: dict[str, Any], preview_value: float | None) -> float:
    """Estimate an order's value in dollars, conservatively.

    Each node with legs is valued at its highest price times its largest leg
    quantity (times 100 for options); Schwab's preview order value counts
    too, and the largest figure wins. A node without a price (market, trailing
    stop) needs the preview value. Raises ``PermissionError`` when the value
    cannot be determined, since the cap cannot be enforced.
    """
    values: list[float] = []
    unknown = False
    for node in _nodes(order_spec):
        legs = _legs(node)
        if not legs:
            continue
        value = _node_value(node, legs)
        if value is None:
            unknown = True
        else:
            values.append(value)
    if preview_value is not None and math.isfinite(preview_value):
        values.append(abs(preview_value))
    elif unknown:
        values = []
    if not values:
        raise PermissionError(
            "Refusing order: its value cannot be determined from its prices or Schwab's preview, "
            "so the max order value cannot be enforced."
        )
    return max(values)


def _allowlist_symbol(leg: dict[str, Any]) -> str:
    symbol = str(_instrument(leg).get("symbol", "")).upper()
    # Option symbols carry the underlying root in their first six characters.
    return symbol[:6].strip() if _is_option(leg) else symbol.strip()


def check_order(limits: TradingLimits, order_spec: dict[str, Any], preview_value: float | None) -> None:
    """Raise ``PermissionError`` if the order breaks a configured limit."""
    value = order_value(order_spec, preview_value)
    if value > limits.max_order_value:
        raise PermissionError(
            f"Refusing order: estimated value ${value:,.2f} exceeds the ${limits.max_order_value:,.2f} max order value."
        )
    legs = [leg for node in _nodes(order_spec) for leg in _legs(node)]
    if limits.max_order_quantity is not None:
        for leg in legs:
            quantity = _number(leg.get("quantity"))
            if quantity is None or quantity > limits.max_order_quantity:
                shown = leg.get("quantity") if quantity is None else f"{quantity:g}"
                raise PermissionError(
                    f"Refusing order: quantity {shown} exceeds the max order quantity of {limits.max_order_quantity}."
                )
    if limits.allowed_symbols is not None:
        for leg in legs:
            symbol = _allowlist_symbol(leg)
            if symbol not in limits.allowed_symbols:
                raise PermissionError(f"Refusing order: {symbol!r} is not in the symbol allowlist.")


def _market_now() -> datetime.datetime:
    return datetime.datetime.now(_MARKET_TZ)


class OrderSlot:
    """One reserved place under the daily cap, held while the reviewer decides."""

    def __init__(self, ledger: OrderLedger, cap: int) -> None:
        self._ledger = ledger
        self._cap = cap
        self._open = True

    def commit(self) -> None:
        """Count the order as sent. Call right before sending it to Schwab."""
        if not self._open:
            raise RuntimeError("Order slot already settled.")
        self._open = False
        self._ledger._commit(self._cap)

    def release(self) -> None:
        """Give the slot back (the order was not sent). No-op once settled."""
        if self._open:
            self._open = False
            self._ledger._reserved -= 1


class OrderLedger:
    """Orders sent to Schwab today (US/Eastern), persisted owner-only.

    ``path=None`` keeps the count in memory (tests). With a path, every check
    re-reads the file under an exclusive lock, so several server processes
    sharing the data directory share one cap. An unreadable or malformed
    file fails closed: fix or remove it.
    """

    def __init__(self, path: str | None, *, clock: Callable[[], datetime.datetime] = _market_now) -> None:
        self._path = path
        self._clock = clock
        self._reserved = 0
        self._memory: tuple[str, int] = ("", 0)
        with self._locked():
            self._read()

    def reserve(self, cap: int) -> OrderSlot:
        """Hold a slot under ``cap`` or raise ``PermissionError``."""
        with self._locked():
            sent = self._read()
        if sent + self._reserved >= cap:
            raise PermissionError(
                f"Refusing order: the daily cap of {cap} orders is reached "
                f"({sent} sent today, {self._reserved} awaiting approval)."
            )
        self._reserved += 1
        return OrderSlot(self, cap)

    def _commit(self, cap: int) -> None:
        self._reserved -= 1
        with self._locked():
            sent = self._read()
            if sent >= cap:
                raise PermissionError(f"Refusing order: the daily cap of {cap} orders is reached ({sent} sent today).")
            self._write(sent + 1)

    def _today(self) -> str:
        return self._clock().astimezone(_MARKET_TZ).date().isoformat()

    def _read(self) -> int:
        if self._path is None:
            day, count = self._memory
            return count if day == self._today() else 0
        try:
            with open(self._path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            raise ValueError(f"Daily order ledger {self._path} is unreadable ({exc}); fix or remove it.") from exc
        day = data.get("date") if isinstance(data, dict) else None
        count = data.get("count") if isinstance(data, dict) else None
        if not isinstance(day, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Daily order ledger {self._path} is malformed; fix or remove it.")
        return count if day == self._today() else 0

    def _write(self, count: int) -> None:
        day = self._today()
        if self._path is None:
            self._memory = (day, count)
            return
        tokens.write_private_text(self._path, json.dumps({"date": day, "count": count}))

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        if self._path is None or fcntl is None:
            yield
            return
        fd = os.open(f"{self._path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


__all__ = ["OrderLedger", "OrderSlot", "TradingLimits", "check_order", "order_value", "parse_symbol_allowlist"]
