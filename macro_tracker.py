from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .database import DataStore

CANONICAL_MACRO_BENCHMARKS: Dict[str, List[Tuple[str, str, str]]] = {
    "yields": [
        ("^IRX", "US 13W T-Bill", "pct"),
        ("^FVX", "US 5Y Treasury", "pct"),
        ("^TNX", "US 10Y Treasury", "pct"),
        ("^TYX", "US 30Y Treasury", "pct"),
    ],
    "indices": [
        ("^GSPC", "S&P 500", "usd"),
        ("^NDX", "Nasdaq 100", "usd"),
        ("^DJI", "Dow 30", "usd"),
        ("^RUT", "Russell 2000", "usd"),
    ],
    "commodities": [
        ("GC=F", "Gold", "usd"),
        ("SI=F", "Silver", "usd"),
        ("HG=F", "Copper", "usd"),
        ("CL=F", "WTI Crude", "usd"),
        ("NG=F", "Natural Gas", "usd"),
    ],
    "fx": [
        ("EURUSD=X", "EUR/USD", "fx"),
        ("USDJPY=X", "USD/JPY", "fx"),
        ("GBPUSD=X", "GBP/USD", "fx"),
    ],
    "crypto": [
        ("BTC-USD", "Bitcoin", "usd"),
        ("ETH-USD", "Ethereum", "usd"),
        ("SOL-USD", "Solana", "usd"),
    ],
}

_DEFAULT_MACRO_PRICES: Dict[str, Tuple[float, float]] = {
    "^IRX": (4.45, -0.02),
    "^FVX": (4.10, 0.01),
    "^TNX": (4.25, 0.03),
    "^TYX": (4.48, 0.04),
    "^GSPC": (5850.0, 0.42),
    "^NDX": (20450.0, 0.65),
    "^DJI": (42800.0, 0.25),
    "^RUT": (2240.0, -0.15),
    "GC=F": (2740.0, 0.55),
    "SI=F": (33.20, 0.80),
    "HG=F": (4.38, -0.30),
    "CL=F": (71.50, -0.95),
    "NG=F": (2.65, 1.10),
    "EURUSD=X": (1.084, 0.12),
    "USDJPY=X": (152.30, -0.20),
    "GBPUSD=X": (1.298, 0.18),
    "BTC-USD": (72400.0, 2.15),
    "ETH-USD": (2680.0, 1.45),
    "SOL-USD": (178.50, 3.10),
}


def _normalize_rate_decimal(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f / 100.0 if abs(f) > 0.5 else f


def _normalize_rate_pct_points(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f * 100.0 if abs(f) <= 0.5 else f


def compute_equity_risk_premium(
    fcf_yield: Optional[float], us10y_yield: Optional[float]
) -> Optional[float]:
    """
    Compute Equity Risk Premium (ERP) = FCF Yield - US 10Y Treasury Yield as a decimal float.
    Normalizes both inputs so either decimal (0.045) or percentage (4.5) form is accepted.
    """
    fy = _normalize_rate_decimal(fcf_yield)
    rf = _normalize_rate_decimal(us10y_yield)
    if fy is None or rf is None:
        return None
    return float(fy - rf)


def compute_yield_curve_spread(quotes: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute US10Y (^TNX or US10Y) minus US13W/2Y (^IRX or US2Y) in percentage points.
    """
    if not quotes:
        return None
    by_sym: Dict[str, Optional[float]] = {}
    for q in quotes:
        if not isinstance(q, dict):
            continue
        sym = str(q.get("symbol") or "").strip().upper()
        if sym:
            by_sym[sym] = _normalize_rate_pct_points(q.get("price"))

    y10 = by_sym.get("^TNX") if by_sym.get("^TNX") is not None else by_sym.get("US10Y")
    y_short = (
        by_sym.get("US2Y")
        if by_sym.get("US2Y") is not None
        else (by_sym.get("^IRX") if by_sym.get("^IRX") is not None else by_sym.get("US13W"))
    )
    if y10 is None or y_short is None:
        return None
    return float(y10 - y_short)


def get_cached_or_default_macro_quotes(store: DataStore) -> List[Dict[str, Any]]:
    rows = store.get_macro_quotes()
    if rows:
        return rows
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seed_rows: List[Dict[str, Any]] = []
    for cat, items in CANONICAL_MACRO_BENCHMARKS.items():
        for sym, label, unit in items:
            px, chg = _DEFAULT_MACRO_PRICES.get(sym, (100.0, 0.0))
            seed_rows.append(
                {
                    "symbol": sym,
                    "label": label,
                    "category": cat,
                    "price": px,
                    "change_pct": chg,
                    "unit": unit,
                    "updated_at": now_iso,
                }
            )
    store.upsert_macro_quotes(seed_rows)
    return store.get_macro_quotes()
