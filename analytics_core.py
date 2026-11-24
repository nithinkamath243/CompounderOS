import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analysis import clean_numeric_series

__all__ = [
    "ReverseDCFSolver",
    "TailRiskScenarioTester",
    "RoicMomentumAnalyzer",
    "HeadlessDataReporter",
]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Shared column-hunting helpers (tolerant of yfinance vs SEC naming)
# ----------------------------------------------------------------------------

def _norm(name: Any) -> str:
    return "".join(str(name).split()).lower()


def _find_col(df: Optional[pd.DataFrame], keys: Iterable[str]) -> Optional[Any]:
    """Return the first column whose normalized name matches a key exactly, 
    falling back to containment if no exact match is found."""
    if df is None or df.empty:
        return None
    normalized = {_norm(c): c for c in df.columns}
    
    # Pass 1: Exact match priority (prevents 'Net Income Common' masking 'Net Income')
    for key in keys:
        nk = _norm(key)
        if nk in normalized:
            return normalized[nk]
            
    # Pass 2: Containment match fallback
    for key in keys:
        nk = _norm(key)
        for nc, original in normalized.items():
            if nk in nc:
                return original
    return None

def _series(df: Optional[pd.DataFrame], keys: Iterable[str]) -> Optional[pd.Series]:
    col = _find_col(df, keys)
    if col is None:
        return None
    return clean_numeric_series(df[col])


def _is_quarterly(index: pd.Index) -> bool:
    """Median spacing under ~130 days => quarterly statements."""
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
        if len(idx) < 2:
            return False
        deltas = np.diff(idx.values).astype("timedelta64[D]").astype(float)
        return float(np.median(deltas)) <= 130.0
    except Exception:
        return False


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 1. Reverse DCF Solver
# ----------------------------------------------------------------------------

class ReverseDCFSolver:
    """Solves for the FCF growth rate the market price currently implies.

    Model: EquityValue(g) = sum_{t=1..N} FCF0*(1+g)^t / (1+r)^t
                          + [FCF0*(1+g)^N * (1+g_term) / (r - g_term)] / (1+r)^N
                          - net_debt
    Find g such that EquityValue(g) == current_price * shares_outstanding.
    """

    def __init__(self,
                 discount_rate: float = 0.09,
                 terminal_growth: float = 0.025,
                 years: int = 10,
                 max_growth: float = 3.0,
                 tolerance: float = 1e-7,
                 max_iterations: int = 256):
        self.discount_rate = float(discount_rate)
        self.terminal_growth = float(terminal_growth)
        self.years = int(years)
        self.max_growth = float(max_growth)      # 300%/yr hard cap on the bracket
        self.tolerance = float(tolerance)        # convergence in g-space
        self.max_iterations = int(max_iterations)

    # -- baseline derivation ---------------------------------------------
    def derive_baseline_fcf(self, fcf_series: Optional[pd.Series]) -> Tuple[Optional[float], str]:
        """Return (baseline_fcf, note). Quarterly series -> TTM sum of the last
        four points; short quarterly series -> last quarter annualized (x4,
        flagged); annual series -> latest value."""
        s = clean_numeric_series(fcf_series)
        if s is None or s.empty:
            return None, "no usable FCF observations"
        s = s.sort_index()
        if _is_quarterly(s.index):
            if len(s) >= 4:
                base = float(s.iloc[-4:].sum())
                return base, "TTM sum of last 4 quarters"
            base = float(s.iloc[-1]) * 4.0
            return base, f"only {len(s)} quarters available; last quarter annualized x4"
        return float(s.iloc[-1]), "latest annual value"

    # -- forward valuation --------------------------------------------------
    def intrinsic_equity_value(self,
                               baseline_fcf: float,
                               growth: float,
                               net_debt: float,
                               discount_rate: Optional[float] = None) -> float:
        r = self.discount_rate if discount_rate is None else float(discount_rate)
        gt = self.terminal_growth
        pv = 0.0
        cash = float(baseline_fcf)
        for t in range(1, self.years + 1):
            cash *= (1.0 + growth)
            pv += cash / ((1.0 + r) ** t)
        terminal = cash * (1.0 + gt) / (r - gt)
        pv += terminal / ((1.0 + r) ** self.years)
        return pv - float(net_debt)

    # -- the inverse problem -------------------------------------------------
    def solve(self,
              current_price: float,
              shares_outstanding: float,
              net_debt: float,
              fcf_series: Optional[pd.Series] = None,
              baseline_fcf: Optional[float] = None,
              discount_rate: Optional[float] = None) -> Dict[str, Any]:
        r = self.discount_rate if discount_rate is None else float(discount_rate)
        result: Dict[str, Any] = {
            "implied_growth": None,
            "converged": False,
            "iterations": 0,
            "error": None,
            "note": None,
            "discount_rate": r,
            "terminal_growth": self.terminal_growth,
            "years": self.years,
        }

        price = _finite(current_price)
        shares = _finite(shares_outstanding)
        nd = _finite(net_debt)
        nd = 0.0 if nd is None else nd
        if price is None or price <= 0 or shares is None or shares <= 0:
            result["error"] = "current_price and shares_outstanding must be positive finite numbers"
            return result
        if r <= self.terminal_growth:
            result["error"] = (f"discount_rate ({r:.4f}) must exceed terminal_growth "
                               f"({self.terminal_growth:.4f}) for a finite terminal value")
            return result

        note = None
        if baseline_fcf is None:
            baseline_fcf, note = self.derive_baseline_fcf(fcf_series)
        base = _finite(baseline_fcf)
        if base is None:
            result["error"] = f"could not derive baseline FCF ({note or 'no input provided'})"
            return result
        if base <= 0:
            result["error"] = ("baseline FCF is non-positive; implied growth is undefined "
                               "under this model — supply a normalized positive FCF")
            result["baseline_fcf"] = base
            return result

        target = price * shares
        result["market_cap"] = target
        result["baseline_fcf"] = base
        result["baseline_note"] = note

        f = lambda g: self.intrinsic_equity_value(base, g, nd, discount_rate=r)

        lo, hi = -0.95, 0.50
        f_lo = f(lo)
        if f_lo > target:
            result["implied_growth"] = lo
            result["error"] = ("market cap sits below the PV of even a -95%/yr FCF path "
                               "(net debt likely dominates); implied growth < -95%")
            return result
        f_hi = f(hi)
        while f_hi < target and hi < self.max_growth:
            hi = min(self.max_growth, hi + 0.25)
            f_hi = f(hi)
        if f_hi < target:
            result["implied_growth"] = self.max_growth
            result["note"] = (f"price implies growth above the {self.max_growth:.0%} cap; "
                              "result clamped, treat as 'absurdly optimistic'")
            result["converged"] = False
            return result

        iterations = 0
        g_mid = 0.0
        while iterations < self.max_iterations:
            iterations += 1
            g_mid = 0.5 * (lo + hi)
            v = f(g_mid)
            if abs(hi - lo) < self.tolerance:
                break
            if v < target:
                lo = g_mid
            else:
                hi = g_mid
        result["implied_growth"] = g_mid
        result["iterations"] = iterations
        result["converged"] = abs(hi - lo) < self.tolerance * 10
        result["equity_value_at_solution"] = f(g_mid)
        return result

    def sensitivity(self,
                    current_price: float,
                    shares_outstanding: float,
                    net_debt: float,
                    baseline_fcf: float,
                    discount_rates: Iterable[float]) -> Dict[float, Optional[float]]:
        """Implied growth across alternative discount rates; None where invalid."""
        out: Dict[float, Optional[float]] = {}
        for r in discount_rates:
            res = self.solve(current_price, shares_outstanding, net_debt,
                             baseline_fcf=baseline_fcf, discount_rate=float(r))
            out[float(r)] = res["implied_growth"] if res.get("error") is None else None
        return out


# ----------------------------------------------------------------------------
# 2. Tail-Risk Scenario Tester
# ----------------------------------------------------------------------------

class TailRiskScenarioTester:
    """Structural shock transformations over income / cash-flow dataframes.

    Non-mutating: every method returns (shocked_copy, report_dict). Columns
    absent from the input are skipped and listed in report['skipped']."""

    REV_KEYS = ["TotalRevenue", "Total Revenue", "OperatingRevenue", "Revenue"]
    GP_KEYS = ["GrossProfit", "Gross Profit"]
    OPINC_KEYS = ["OperatingIncome", "Operating Income", "OperatingIncomeLoss", "EBIT"]
    NI_KEYS = ["NetIncome", "Net Income", "NetIncomeLoss"]
    EBITDA_KEYS = ["EBITDA", "NormalizedEBITDA", "Normalized EBITDA"]
    OCF_KEYS = ["OperatingCashFlow", "Operating Cash Flow",
                "NetCashProvidedByUsedInOperatingActivities",
                "Total Cash From Operating Activities"]
    CAPEX_KEYS = ["CapitalExpenditure", "Capital Expenditure", "CapEx",
                  "PaymentsToAcquirePropertyPlantAndEquipment"]
    FCF_KEYS = ["FreeCashFlow", "Free Cash Flow"]

    def apply_obsolescence_shock(self,
                                 fin_df: Optional[pd.DataFrame],
                                 margin_contraction: float = 0.20,
                                 mode: str = "relative",
                                 flatten_revenue: bool = True) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Obsolescence scenario.

        Mechanics, in order:
          1. Revenue growth -> 0%: every period's revenue is set to the latest
             observed revenue level (company frozen at current scale), so any
             downstream YoY-growth metric computes to exactly 0.
          2. Gross margin contraction: each period's OWN margin ratio is kept
             (structure preserved) then contracted —
               mode="relative":     GM' = GM * (1 - margin_contraction)
               mode="absolute_pp":  GM' = max(0, GM - margin_contraction)
          3. Propagation with opex dollars HELD CONSTANT:
               OpInc'  = GP' - (GP - OpInc)
               EBITDA' = OpInc' + (EBITDA - OpInc)        [D&A wedge held]
               NI'     = OpInc' - (OpInc - NI)            [below-the-line wedge held]
        """
        report: Dict[str, Any] = {
            "shock": "obsolescence",
            "params": {"margin_contraction": margin_contraction, "mode": mode,
                       "flatten_revenue": flatten_revenue},
            "applied": [], "skipped": [], "error": None,
        }
        if fin_df is None or fin_df.empty:
            report["error"] = "empty or missing income dataframe"
            return fin_df, report
        if mode not in ("relative", "absolute_pp"):
            report["error"] = f"unknown mode '{mode}' (use 'relative' or 'absolute_pp')"
            return fin_df, report

        out = fin_df.copy()

        rev_col = _find_col(out, self.REV_KEYS)
        gp_col = _find_col(out, self.GP_KEYS)
        op_col = _find_col(out, self.OPINC_KEYS)
        ni_col = _find_col(out, self.NI_KEYS)
        eb_col = _find_col(out, self.EBITDA_KEYS)

        if rev_col is None:
            report["error"] = "no revenue column found; shock not applied"
            return fin_df, report

        rev = pd.to_numeric(out[rev_col], errors="coerce")
        finite_rev = rev.dropna()
        if finite_rev.empty:
            report["error"] = "revenue column contains no finite values"
            return fin_df, report
        latest_rev = float(finite_rev.sort_index().iloc[-1])

        rev_shocked = pd.Series(latest_rev, index=out.index) if flatten_revenue else rev
        if flatten_revenue:
            out[rev_col] = rev_shocked
            report["applied"].append(f"{rev_col}: flattened at {latest_rev:,.0f} (0% growth)")

        if gp_col is not None:
            gp = pd.to_numeric(out[gp_col], errors="coerce")
            gm = gp / rev.replace(0, np.nan)
            if mode == "relative":
                gm_shocked = gm * (1.0 - float(margin_contraction))
            else:
                gm_shocked = (gm - float(margin_contraction)).clip(lower=0.0)
            gp_shocked = rev_shocked * gm_shocked
            out[gp_col] = gp_shocked
            report["applied"].append(f"{gp_col}: margin contracted ({mode} {margin_contraction:.0%})")

            if op_col is not None:
                op = pd.to_numeric(fin_df[op_col], errors="coerce")
                opex = gp - op                       # historical opex dollars, held
                op_shocked = gp_shocked - opex
                out[op_col] = op_shocked
                report["applied"].append(f"{op_col}: rebuilt from shocked GP minus held opex")

                if eb_col is not None:
                    eb = pd.to_numeric(fin_df[eb_col], errors="coerce")
                    out[eb_col] = op_shocked + (eb - op)
                    report["applied"].append(f"{eb_col}: D&A wedge held")
                else:
                    report["skipped"].append("EBITDA (column absent)")

                if ni_col is not None:
                    ni = pd.to_numeric(fin_df[ni_col], errors="coerce")
                    out[ni_col] = op_shocked - (op - ni)
                    report["applied"].append(f"{ni_col}: below-the-line wedge held")
                else:
                    report["skipped"].append("NetIncome (column absent)")
            else:
                report["skipped"].append("OperatingIncome/EBITDA/NetIncome (no OpInc column)")
        else:
            report["skipped"].append("GrossProfit chain (no GP column; only revenue flattened)")

        return out, report

    def apply_capex_intensification_shock(self,
                                          cf_df: Optional[pd.DataFrame],
                                          multiplier: float = 2.0) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Doubles (by default) CapEx-to-revenue intensity at unchanged revenue,
        i.e. CapEx' = multiplier * CapEx, sign-preserving (yfinance stores CapEx
        negative, SEC-built frames store positive payments — x2 respects both).
        FCF is rebuilt as OCF - |CapEx'| when OCF exists; otherwise the
        incremental capex is subtracted from the existing FCF column."""
        report: Dict[str, Any] = {
            "shock": "capex_intensification",
            "params": {"multiplier": multiplier},
            "applied": [], "skipped": [], "error": None,
        }
        if cf_df is None or cf_df.empty:
            report["error"] = "empty or missing cash-flow dataframe"
            return cf_df, report

        out = cf_df.copy()
        capex_col = _find_col(out, self.CAPEX_KEYS)
        ocf_col = _find_col(out, self.OCF_KEYS)
        fcf_col = _find_col(out, self.FCF_KEYS)

        if capex_col is None:
            report["error"] = "no CapEx column found; shock not applied"
            return cf_df, report

        capex = pd.to_numeric(cf_df[capex_col], errors="coerce")
        capex_shocked = capex * float(multiplier)
        out[capex_col] = capex_shocked
        report["applied"].append(f"{capex_col}: x{multiplier:g} (sign preserved)")

        if fcf_col is not None or ocf_col is not None:
            if ocf_col is not None:
                ocf = pd.to_numeric(cf_df[ocf_col], errors="coerce")
                fcf_shocked = ocf - capex_shocked.abs()
                target_col = fcf_col if fcf_col is not None else "FreeCashFlow"
                out[target_col] = fcf_shocked
                report["applied"].append(f"{target_col}: rebuilt as OCF - |CapEx'|")
            else:
                fcf = pd.to_numeric(cf_df[fcf_col], errors="coerce")
                out[fcf_col] = fcf - (capex_shocked.abs() - capex.abs())
                report["applied"].append(f"{fcf_col}: reduced by incremental capex")
        else:
            report["skipped"].append("FCF rebuild (no OCF or FCF column present)")

        return out, report


# ----------------------------------------------------------------------------
# 3. ROIC Momentum Analyzer
# ----------------------------------------------------------------------------

class RoicMomentumAnalyzer:
    """ROIC series plus a rolling 3-year OLS slope to catch moat decay.

    NOPAT is annualized before dividing by invested capital (a stock):
    quarterly frames use a TTM rolling-4 sum; annual frames pass through.
    Invested capital uses the spec formula when the columns exist, with a
    flagged financing-side fallback otherwise (see module docstring #3)."""

    TA_KEYS = ["TotalAssets", "Total Assets"]
    CASH_KEYS = ["CashAndCashEquivalents", "Cash And Cash Equivalents", "Cash"]
    CL_KEYS = ["CurrentLiabilities", "Current Liabilities",
               "TotalCurrentLiabilities", "Total Current Liabilities"]
    DEBT_KEYS = ["TotalDebt", "Total Debt", "LongTermDebt", "Long Term Debt"]
    EQ_KEYS = ["TotalStockholderEquity", "Total Stockholder Equity",
               "StockholdersEquity", "Stockholders Equity",
               "Total Equity Gross Minority Interest"]
    OPINC_KEYS = TailRiskScenarioTester.OPINC_KEYS
    TAX_KEYS = ["TaxProvision", "Tax Provision", "IncomeTaxExpense", "Income Tax Expense"]
    PRETAX_KEYS = ["PretaxIncome", "Pretax Income", "IncomeBeforeTax", "Income Before Tax"]

    def __init__(self,
                 default_tax_rate: float = 0.21,
                 window_years: float = 3.0,
                 decay_threshold: float = 0.015):
        self.default_tax_rate = float(default_tax_rate)
        self.window_years = float(window_years)
        self.decay_threshold = float(decay_threshold)   # ROIC units per year (0.015 = 1.5pp/yr)

    # -- components -----------------------------------------------------------
    def _effective_tax_rate(self, fin_df: Optional[pd.DataFrame], quarterly: bool) -> Tuple[float, str]:
        tax = _series(fin_df, self.TAX_KEYS)
        pre = _series(fin_df, self.PRETAX_KEYS)
        if tax is not None and pre is not None:
            df = pd.concat({"t": tax, "p": pre}, axis=1).dropna()
            if not df.empty:
                if quarterly and len(df) >= 4:
                    t_sum = float(df["t"].iloc[-4:].sum())
                    p_sum = float(df["p"].iloc[-4:].sum())
                else:
                    t_sum = float(df["t"].iloc[-1])
                    p_sum = float(df["p"].iloc[-1])
                if p_sum > 0:
                    rate = max(0.0, min(0.50, t_sum / p_sum))
                    return rate, "effective (from statements, clipped 0-50%)"
        return self.default_tax_rate, "default"

    def compute_invested_capital(self, bal_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.Series], str]:
        ta = _series(bal_df, self.TA_KEYS)
        cl = _series(bal_df, self.CL_KEYS)
        cash = _series(bal_df, self.CASH_KEYS)
        if ta is not None and cl is not None:
            frame = pd.concat({"ta": ta, "cl": cl}, axis=1).dropna()
            if not frame.empty:
                cash_aligned = cash.reindex(frame.index).fillna(0.0) if cash is not None else 0.0
                ic = frame["ta"] - cash_aligned - frame["cl"]
                return ic.sort_index(), "TotalAssets - Cash - CurrentLiabilities"
        debt = _series(bal_df, self.DEBT_KEYS)
        eq = _series(bal_df, self.EQ_KEYS)
        if debt is not None and eq is not None:
            frame = pd.concat({"d": debt, "e": eq}, axis=1).dropna()
            if not frame.empty:
                cash_aligned = cash.reindex(frame.index).fillna(0.0) if cash is not None else 0.0
                ic = frame["d"] + frame["e"] - cash_aligned
                return ic.sort_index(), "FALLBACK: TotalDebt + Equity - Cash (TA/CL columns absent)"
        return None, "unavailable (insufficient balance-sheet columns)"

    def compute_nopat(self, fin_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.Series], float, str, bool]:
        op = _series(fin_df, self.OPINC_KEYS)
        if op is None or op.empty:
            return None, self.default_tax_rate, "default", False
        op = op.sort_index()
        quarterly = _is_quarterly(op.index)
        tax_rate, tax_src = self._effective_tax_rate(fin_df, quarterly)
        if quarterly:
            op_annual = op.rolling(4, min_periods=4).sum().dropna()
        else:
            op_annual = op
        nopat = op_annual * (1.0 - tax_rate)
        return nopat, tax_rate, tax_src, quarterly

    def compute_roic(self,
                     fin_df: Optional[pd.DataFrame],
                     bal_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"roic": None, "invested_capital_method": None,
                               "tax_rate": None, "tax_rate_source": None,
                               "periodicity": None, "error": None, "notes": []}
        nopat, tax_rate, tax_src, quarterly = self.compute_nopat(fin_df)
        out["tax_rate"] = tax_rate
        out["tax_rate_source"] = tax_src
        out["periodicity"] = "quarterly" if quarterly else "annual"
        if nopat is None or nopat.empty:
            out["error"] = "no operating income series available"
            return out
        ic, ic_method = self.compute_invested_capital(bal_df)
        out["invested_capital_method"] = ic_method
        if ic is None or ic.empty:
            out["error"] = "invested capital could not be computed"
            return out

        # Balance sheet is point-in-time -> forward-fill onto NOPAT dates.
        union = nopat.index.union(ic.index).sort_values()
        ic_ff = ic.reindex(union).ffill()
        avg_ic = ic_ff.rolling(2, min_periods=1).mean()
        aligned = pd.concat({"nopat": nopat.reindex(union), "ic": avg_ic}, axis=1).dropna()
        aligned = aligned[aligned["ic"] > 0]
        if aligned.empty:
            out["error"] = "no overlapping periods with positive invested capital"
            return out
        out["roic"] = (aligned["nopat"] / aligned["ic"]).sort_index()
        return out

    def rolling_slope(self, roic: Optional[pd.Series],
                      min_points: Optional[int] = None) -> Optional[pd.Series]:
        """OLS slope of ROIC vs time (in YEARS) over a trailing window, per
        window-end date. Units: ROIC change per year (0.01 = +1pp/yr)."""
        if roic is None or roic.empty:
            return None
        s = roic.dropna().sort_index()
        if len(s) < 2:
            return None
        quarterly = _is_quarterly(s.index)
        if min_points is None:
            min_points = 6 if quarterly else 3
        window = timedelta(days=self.window_years * 365.25)
        idx = pd.DatetimeIndex(pd.to_datetime(s.index))
        slopes: Dict[pd.Timestamp, float] = {}
        for end in idx:
            mask = (idx > (end - window)) & (idx <= end)
            pts = s[mask]
            if len(pts) < min_points:
                continue
            x = (pd.DatetimeIndex(pts.index) - pd.DatetimeIndex(pts.index)[0]).days / 365.25
            y = pts.values.astype(float)
            if np.allclose(x, x[0]):
                continue
            slope = float(np.polyfit(np.asarray(x, dtype=float), y, 1)[0])
            slopes[end] = slope
        if not slopes:
            return None
        return pd.Series(slopes).sort_index()

    def analyze(self,
                fin_df: Optional[pd.DataFrame],
                bal_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        base = self.compute_roic(fin_df, bal_df)
        result: Dict[str, Any] = dict(base)
        result.update({"slope": None, "latest_roic": None,
                       "latest_slope": None, "verdict": "INSUFFICIENT DATA"})
        roic = base.get("roic")
        if base.get("error") or roic is None or roic.empty:
            return result
        result["latest_roic"] = float(roic.iloc[-1])
        slope = self.rolling_slope(roic)
        result["slope"] = slope
        if slope is None or slope.empty:
            result["notes"].append("not enough history inside the rolling window for a slope")
            return result
        latest_slope = float(slope.iloc[-1])
        result["latest_slope"] = latest_slope
        if latest_slope <= -self.decay_threshold:
            result["verdict"] = "STRUCTURAL DECAY (moat erosion signal)"
        elif latest_slope >= self.decay_threshold:
            result["verdict"] = "COMPOUNDING IMPROVEMENT"
        else:
            result["verdict"] = "STABLE"
        return result


# ----------------------------------------------------------------------------
# 4. Headless Data Reporter
# ----------------------------------------------------------------------------

class HeadlessDataReporter:
    """Server-side compiler over a DataStore (duck-typed: needs
    list_sp500_scores(); optionally .conn/.lock for the history upgrade).

    Insider clusters are fetched live via yfinance (lazy import) because the
    local database has no insider table yet — flagged in report notes."""

    BUY_TOKENS = ("buy", "purchase", "acquired", "option exercise")
    SELL_TOKENS = ("sell", "sale", "disposed")

    def __init__(self, store: Any,
                 include_insiders: bool = True,
                 insider_lookback_days: int = 90,
                 drop_severity: float = 0.5):
        self.store = store
        self.include_insiders = bool(include_insiders)
        self.insider_lookback_days = int(insider_lookback_days)
        self.drop_severity = float(drop_severity)   # min score decline to flag as a "drop"

    # -- optional-history plumbing --------------------------------------------
    def _table_exists(self, name: str) -> bool:
        conn = getattr(self.store, "conn", None)
        lock = getattr(self.store, "lock", None)
        if conn is None or lock is None:
            return False
        try:
            with lock:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
                return cur.fetchone() is not None
        except Exception:
            return False

    def _history_drops(self, limit: int) -> Optional[List[Dict[str, Any]]]:
        """Real score deltas — honors source quarantine and unscorable rows."""
        if not self._table_exists("sp500_scores_history"):
            return None
        try:
            with self.store.lock:
                cur = self.store.conn.cursor()
                # D7/D8 Reactivation: Filter by scan source and exclude NULL scores
                cur.execute("""
                    SELECT symbol, score, evaluated_at 
                    FROM sp500_scores_history 
                    WHERE source = 'scan' AND score IS NOT NULL
                    ORDER BY symbol ASC, evaluated_at ASC
                """)
                rows = cur.fetchall()
        except Exception as e:
            logger.warning("history query failed: %s", e)
            return None
        by_symbol: Dict[str, List[Tuple[float, str]]] = {}
        for sym, score, ts in rows:
            v = _finite(score)
            if v is None:
                continue
            by_symbol.setdefault(str(sym).upper(), []).append((v, ts or ""))
        drops = []
        for sym, seq in by_symbol.items():
            if len(seq) < 2:
                continue
            prev, latest = seq[-2][0], seq[-1][0]
            delta = latest - prev
            if delta <= -self.drop_severity:
                drops.append({"symbol": sym, "previous": prev, "latest": latest,
                              "delta": delta, "evaluated_at": seq[-1][1]})
        drops.sort(key=lambda d: d["delta"])
        return drops[:limit]

    # -- insider clusters -------------------------------------------------------
    def _insider_cluster(self, symbol: str) -> Optional[str]:
        if not self.include_insiders:
            return None
        try:
            import yfinance as yf  # lazy: keeps the math classes network-free
            ins = yf.Ticker(symbol).insider_transactions
        except Exception as e:
            logger.info("insider fetch failed for %s: %s", symbol, e)
            return None
        if ins is None or getattr(ins, "empty", True):
            return None
        try:
            df = ins.copy().reset_index(drop=False)
            date_col = _find_col(df, ["Start Date", "startDate", "Date", "index"])
            type_col = _find_col(df, ["Transaction", "Transaction Type", "Text"])
            value_col = _find_col(df, ["Value", "Transaction Value"])
            shares_col = _find_col(df, ["Shares", "Shares Traded"])
            price_col = _find_col(df, ["Price", "Transaction Price"])
            cutoff = datetime.now() - timedelta(days=self.insider_lookback_days)
            buys = sells = 0
            net_value = 0.0
            for _, row in df.iterrows():
                d = pd.to_datetime(row.get(date_col), errors="coerce") if date_col else pd.NaT
                if pd.notna(d):
                    d_naive = d.tz_localize(None) if getattr(d, "tzinfo", None) else d
                    if d_naive < cutoff:
                        continue
                text = str(row.get(type_col, "")).lower() if type_col else ""
                if not text:
                    text = " ".join(str(x).lower() for x in row.values)
                is_buy = any(t in text for t in self.BUY_TOKENS)
                is_sell = any(t in text for t in self.SELL_TOKENS)
                if not (is_buy or is_sell):
                    continue
                val = _finite(row.get(value_col)) if value_col else None
                if val is None and shares_col and price_col:
                    sh = _finite(row.get(shares_col))
                    pr = _finite(row.get(price_col))
                    val = abs(sh * pr) if sh is not None and pr is not None else None
                val = val or 0.0
                if is_buy:
                    buys += 1
                    net_value += abs(val)
                else:
                    sells += 1
                    net_value -= abs(val)
            if buys == 0 and sells == 0:
                return f"no insider activity ({self.insider_lookback_days}d)"
            direction = "CLUSTER BUY" if (buys > sells and net_value > 0) else (
                        "CLUSTER SELL" if (sells > buys and net_value < 0) else "MIXED")
            return (f"{direction}: {buys} buy / {sells} sell, "
                    f"net ${net_value / 1e6:+,.1f}M ({self.insider_lookback_days}d)")
        except Exception as e:
            logger.info("insider parse failed for %s: %s", symbol, e)
            return None

    # -- report assembly ---------------------------------------------------------
    def build_report(self, top_n: int = 5, bottom_n: int = 3) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_db": getattr(self.store, "path", "unknown"),
            "evaluated_count": 0,
            "coverage": {"oldest": None, "latest": None},
            "top_compounders": [],
            "severe_quality": [],
            "history_drops": None,
            "notes": [],
        }
        try:
            rows = self.store.list_sp500_scores()  # ordered score DESC
        except Exception as e:
            report["notes"].append(f"score query failed: {e}")
            return report

        valid = []
        for sym, sector, score, model, ts in rows or []:
            v = _finite(score)
            if v is None:
                continue
            valid.append({"symbol": str(sym).upper(), "sector": sector or "Unknown",
                          "score": v, "model": model or "unknown",
                          "evaluated_at": ts or ""})
        report["evaluated_count"] = len(valid)
        if not valid:
            report["notes"].append("sp500_scores is empty — run the scanner first")
            return report

        stamps = sorted(e["evaluated_at"] for e in valid if e["evaluated_at"])
        if stamps:
            report["coverage"] = {"oldest": stamps[0], "latest": stamps[-1]}

        top = valid[:top_n]
        bottom = sorted(valid, key=lambda e: e["score"])[:bottom_n]
        for entry in top + bottom:
            entry["insider"] = self._insider_cluster(entry["symbol"])
        report["top_compounders"] = top
        report["severe_quality"] = bottom

        drops = self._history_drops(bottom_n)
        report["history_drops"] = drops
        if drops is None:
            report["notes"].append(
                "sp500_scores stores latest-only; 'severe_quality' is a snapshot of the "
                "lowest scores, not a decline over time. Create sp500_scores_history"
                "(symbol, score, evaluated_at) to enable true drop detection.")
        if not self.include_insiders:
            report["notes"].append("insider clusters disabled")
        else:
            report["notes"].append(
                "insider clusters fetched live from yfinance (no local insider table yet)")
        return report

    def to_text(self, report: Dict[str, Any]) -> str:
        """Fixed-width text rendering ready for notification/PDF piping."""
        lines: List[str] = []
        lines.append("COMPOUNDEROS — S&P 500 QUALITY REPORT")
        lines.append(f"generated: {report.get('generated_at', '?')}")
        cov = report.get("coverage") or {}
        lines.append(f"evaluated: {report.get('evaluated_count', 0)} symbols "
                     f"(oldest {cov.get('oldest') or '—'} / latest {cov.get('latest') or '—'})")
        lines.append("")

        def block(title: str, entries: List[Dict[str, Any]]):
            lines.append(title)
            lines.append("-" * len(title))
            if not entries:
                lines.append("  (none)")
            for e in entries:
                lines.append(f"  {e['symbol']:<7} {e['score']:>4.1f}/10  {e['sector']:<24} "
                             f"[{e['model']}]")
                if e.get("insider"):
                    lines.append(f"          insiders: {e['insider']}")
            lines.append("")

        block("TOP COMPOUNDERS", report.get("top_compounders") or [])
        block("SEVERE QUALITY (lowest snapshot scores)", report.get("severe_quality") or [])

        drops = report.get("history_drops")
        if drops:
            lines.append("SCORE DROPS (history-based)")
            lines.append("---------------------------")
            for d in drops:
                lines.append(f"  {d['symbol']:<7} {d['previous']:.1f} -> {d['latest']:.1f} "
                             f"({d['delta']:+.1f})  as of {d['evaluated_at']}")
            lines.append("")
        for note in report.get("notes") or []:
            lines.append(f"note: {note}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Synthetic smoke test (no network, no GUI): python -m compounderos_app.analytics_core
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    q_idx = pd.date_range("2023-03-31", periods=12, freq="QE")

    fcf = pd.Series(np.linspace(900e6, 1.3e9, 12), index=q_idx)
    solver = ReverseDCFSolver()
    res = solver.solve(current_price=180.0, shares_outstanding=1.2e9,
                       net_debt=5e9, fcf_series=fcf)
    print(f"[ReverseDCF] implied growth: {res['implied_growth']:.2%} "
          f"(converged={res['converged']}, iters={res['iterations']}, "
          f"baseline={res['baseline_fcf'] / 1e9:.2f}B via {res['baseline_note']})")

    fin = pd.DataFrame({
        "TotalRevenue": np.linspace(10e9, 14e9, 12),
        "GrossProfit": np.linspace(4.5e9, 6.6e9, 12),
        "OperatingIncome": np.linspace(2.0e9, 3.2e9, 12),
        "NetIncome": np.linspace(1.5e9, 2.5e9, 12),
        "EBITDA": np.linspace(2.6e9, 3.9e9, 12),
        "PretaxIncome": np.linspace(1.9e9, 3.1e9, 12),
        "TaxProvision": np.linspace(0.38e9, 0.62e9, 12),
    }, index=q_idx)
    cf = pd.DataFrame({
        "OperatingCashFlow": np.linspace(2.2e9, 3.4e9, 12),
        "CapitalExpenditure": np.linspace(-0.6e9, -0.9e9, 12),
        "FreeCashFlow": np.linspace(1.6e9, 2.5e9, 12),
    }, index=q_idx)
    bal = pd.DataFrame({
        "TotalAssets": np.linspace(30e9, 38e9, 12),
        "CashAndCashEquivalents": np.linspace(4e9, 6e9, 12),
        "CurrentLiabilities": np.linspace(6e9, 7.5e9, 12),
    }, index=q_idx)

    tester = TailRiskScenarioTester()
    shocked_fin, rep1 = tester.apply_obsolescence_shock(fin)
    print(f"[Obsolescence] applied={len(rep1['applied'])} skipped={rep1['skipped']} "
          f"| latest GM {shocked_fin['GrossProfit'].iloc[-1] / shocked_fin['TotalRevenue'].iloc[-1]:.1%} "
          f"(was {fin['GrossProfit'].iloc[-1] / fin['TotalRevenue'].iloc[-1]:.1%})")
    shocked_cf, rep2 = tester.apply_capex_intensification_shock(cf)
    print(f"[CapEx shock]  applied={rep2['applied']} "
          f"| latest FCF {shocked_cf['FreeCashFlow'].iloc[-1] / 1e9:.2f}B "
          f"(was {cf['FreeCashFlow'].iloc[-1] / 1e9:.2f}B)")

    roic = RoicMomentumAnalyzer().analyze(fin, bal)
    slope_val = roic.get('latest_slope')
    slope_str = f"{slope_val:+.4f}/yr" if slope_val is not None else "None"
    
    print(f"[ROIC] latest={roic.get('latest_roic', 0):.1%} slope={slope_str} "
          f"verdict={roic.get('verdict')} ic_method='{roic.get('invested_capital_method')}' "
          f"tax={roic.get('tax_rate', 0):.1%} ({roic.get('tax_rate_source')})")

    class _StubStore:
        path = ":memory-stub:"
        def list_sp500_scores(self):
            return [("MSFT", "Information Technology", 9.1, "gemini-x", "2026-07-01T00:00:00+00:00"),
                    ("AAPL", "Information Technology", 8.7, "gemini-x", "2026-07-02T00:00:00+00:00"),
                    ("KO", "Consumer Staples", 7.9, "gemini-x", "2026-07-03T00:00:00+00:00"),
                    ("F", "Consumer Discretionary", 3.1, "gemini-x", "2026-07-01T12:00:00+00:00"),
                    ("PARA", "Communication Services", 2.4, "gemini-x", "2026-07-02T12:00:00+00:00")]

    reporter = HeadlessDataReporter(_StubStore(), include_insiders=False)
    print()
    print(reporter.to_text(reporter.build_report(top_n=3, bottom_n=2)))
