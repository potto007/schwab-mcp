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
from collections.abc import Callable, Iterable, Iterator, Mapping
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


def _quote_key(symbol: str) -> str:
    """Match quotes and legs whether or not option symbols carry their padding."""
    return symbol.upper().replace(" ", "")


def leg_symbols(order_spec: dict[str, Any]) -> list[str]:
    """Every distinct leg symbol in an order spec, in order of appearance."""
    symbols: list[str] = []
    for node in _nodes(order_spec):
        for leg in _legs(node):
            symbol = _instrument(leg).get("symbol")
            if isinstance(symbol, str) and symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _option_terms(symbol: str) -> tuple[str, str, str, float] | None:
    """(root, expiry, C/P, strike) from a 21-character option symbol, or None."""
    text = symbol.upper()
    if len(text) != 21:
        return None
    root, expiry, kind, strike = text[:6].strip(), text[6:12], text[12], text[13:]
    if not root or not expiry.isdigit() or kind not in ("C", "P") or not strike.isdigit():
        return None
    return root, expiry, kind, int(strike) / 1000


def _undeterminable() -> PermissionError:
    return PermissionError("Refusing order: its value cannot be determined, so the max order value cannot be enforced.")


@dataclass(slots=True, eq=False)
class _LegValue:
    leg: dict[str, Any]
    symbol: str
    quantity: float
    unit: float
    """Per-share (equity) or per-share premium (option) price used for this leg."""


def _leg_values(node: dict[str, Any], legs: list[dict[str, Any]], references: dict[str, float]) -> list[_LegValue]:
    order_prices = [
        price
        for price in (_number(node.get("price")), _number(node.get("stopPrice")))
        if price is not None and price > 0
    ]
    values: list[_LegValue] = []
    for leg in legs:
        quantity = _number(leg.get("quantity"))
        if quantity is None or quantity <= 0:
            raise _undeterminable()
        symbol = str(_instrument(leg).get("symbol", ""))
        reference = references.get(_quote_key(symbol))
        if reference is None:
            raise PermissionError(f"Refusing order: no market quote for {symbol!r}, so its value cannot be bounded.")
        # A single-leg order's own price is per share; a combo's is a net price.
        unit = max(reference, *order_prices) if len(legs) == 1 and order_prices else reference
        values.append(_LegValue(leg, symbol, quantity, unit))
    return values


def _instruction(leg: dict[str, Any]) -> str:
    return str(leg.get("instruction", "")).upper()


def _pair_spreads(values: list[_LegValue]) -> tuple[float, list[_LegValue]]:
    """Value matched verticals at their width; return that total and the unmatched legs.

    A short option (SELL_TO_OPEN) is matched with a long option (BUY_TO_OPEN)
    on the same root, expiry and type with the same quantity. The pair is
    valued at (strike width + long premium) x 100 x quantity, which covers
    the spread's worst case.
    """
    total = 0.0
    longs = [value for value in values if _is_option(value.leg) and _instruction(value.leg) == "BUY_TO_OPEN"]
    unmatched = list(values)
    for short in [value for value in values if _is_option(value.leg) and _instruction(value.leg) == "SELL_TO_OPEN"]:
        short_terms = _option_terms(short.symbol)
        if short_terms is None:
            continue
        for long in longs:
            long_terms = _option_terms(long.symbol)
            if (
                long in unmatched
                and long_terms is not None
                and long_terms[:3] == short_terms[:3]
                and long.quantity == short.quantity
            ):
                total += (abs(short_terms[3] - long_terms[3]) + long.unit) * _OPTION_MULTIPLIER * short.quantity
                unmatched.remove(short)
                unmatched.remove(long)
                break
    return total, unmatched


def _leg_exposure(value: _LegValue) -> float:
    if not _is_option(value.leg):
        return value.unit * value.quantity
    if _instruction(value.leg) == "SELL_TO_OPEN":
        terms = _option_terms(value.symbol)
        if terms is None:
            raise _undeterminable()
        # Assignment: the strike plus the premium covers a put's obligation
        # and an in-the-money call's underlying value.
        return (terms[3] + value.unit) * _OPTION_MULTIPLIER * value.quantity
    return value.unit * _OPTION_MULTIPLIER * value.quantity


def _node_value(node: dict[str, Any], legs: list[dict[str, Any]], references: dict[str, float]) -> float:
    """Exposure of one node with legs, every leg valued against its market quote.

    Equity legs count shares x max(order price, quote); long options count
    premium x 100; short options count (strike + premium) x 100 unless matched
    in a vertical, which counts its width. A combo's own net price is one
    more bound.
    """
    values = _leg_values(node, legs, references)
    total, unmatched = _pair_spreads(values)
    total += sum(_leg_exposure(value) for value in unmatched)
    net = _number(node.get("price"))
    if len(legs) > 1 and net is not None and net > 0:
        multiplier = _OPTION_MULTIPLIER if any(_is_option(leg) for leg in legs) else 1
        total = max(total, net * max(value.quantity for value in values) * multiplier)
    return total


def _strategy_bound(node: dict[str, Any], references: dict[str, float]) -> float:
    """Dollar bound of a strategy node.

    An OCO executes at most one child, so it takes the largest child. Any other
    strategy with children (TRIGGER) can execute the node and then each child,
    so their values add.
    """
    legs = _legs(node)
    bound = _node_value(node, legs, references) if legs else 0.0
    children_raw = node.get("childOrderStrategies")
    children = [child for child in children_raw if isinstance(child, dict)] if isinstance(children_raw, list) else []
    if children:
        values = [_strategy_bound(child, references) for child in children]
        is_oco = str(node.get("orderStrategyType", "")).upper() == "OCO"
        bound += max(values) if is_oco else sum(values)
    return bound


def order_value(order_spec: dict[str, Any], preview_value: float | None, quotes: Mapping[str, float]) -> float:
    """Estimate the most an order can move, in dollars, conservatively.

    Every leg is valued against its market quote (``quotes``, keyed by symbol)
    as well as the order's own price, so a far-from-market sell or stop is
    valued at the market; see ``_node_value`` for options and spreads.
    Composites follow their strategy: an OCO takes its largest child, a
    TRIGGER adds its stages. Schwab's preview order value can only raise the
    figure. Raises ``PermissionError`` when a leg has no quote or the value
    cannot be determined, since the cap cannot be enforced.
    """
    references = {_quote_key(symbol): price for symbol, price in quotes.items() if price > 0 and math.isfinite(price)}
    bound = _strategy_bound(order_spec, references)
    preview = abs(preview_value) if preview_value is not None and math.isfinite(preview_value) else 0.0
    value = max(bound, preview)
    if value <= 0:
        raise _undeterminable()
    return value


def _allowlist_symbol(leg: dict[str, Any]) -> str:
    symbol = str(_instrument(leg).get("symbol", "")).upper()
    # Option symbols carry the underlying root in their first six characters.
    return symbol[:6].strip() if _is_option(leg) else symbol.strip()


def check_order(
    limits: TradingLimits, order_spec: dict[str, Any], preview_value: float | None, quotes: Mapping[str, float]
) -> None:
    """Raise ``PermissionError`` if the order breaks a configured limit."""
    value = order_value(order_spec, preview_value, quotes)
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


__all__ = [
    "OrderLedger",
    "OrderSlot",
    "TradingLimits",
    "check_order",
    "leg_symbols",
    "order_value",
    "parse_symbol_allowlist",
]
