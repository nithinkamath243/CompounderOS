import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple
from .analytics_core import _find_col, _series, _finite, _is_quarterly
from .scoring_engine import ND_EBITDA_UNSERVICEABLE


@dataclass(frozen=True)
class FinancialBundle:
    info: Mapping[str, Any]
    income_a: Optional[pd.DataFrame]
    income_q: Optional[pd.DataFrame]
    balance_a: Optional[pd.DataFrame]
    balance_q: Optional[pd.DataFrame]
    cashflow_a: Optional[pd.DataFrame]
    cashflow_q: Optional[pd.DataFrame]
    roic_result: Optional[Mapping[str, Any]] = None


def _clean_series(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None or series.empty:
        return None

    s = series.dropna().groupby(level=0).last().sort_index()
    if len(s) < 2:
        return s

    keep = [True] * len(s)
    idx = s.index
    for i in range(len(s) - 1):
        if (idx[i + 1] - idx[i]).days < 45:
            keep[i] = False
    return s[keep]


def _ttm(series: Optional[pd.Series], strict: bool = False) -> Optional[float]:
    s = _clean_series(series)
    if s is None or len(s) < 2:
        return None
    if _is_quarterly(s.index):
        if len(s) >= 4:
            last_4_idx = pd.DatetimeIndex(s.index[-4:])
            span_days = (last_4_idx[-1] - last_4_idx[0]).days
            max_span = 300 if strict else 400
            if span_days <= max_span:
                return float(s.iloc[-4:].sum())
        return None
    return float(s.iloc[-1])


def _aligned_ttm(
    s1: Optional[pd.Series], s2: Optional[pd.Series], strict: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    s1_clean = _clean_series(s1)
    s2_clean = _clean_series(s2)
    if s1_clean is None or s2_clean is None:
        return None, None

    df1 = s1_clean.to_frame(name="v1")
    df2 = s2_clean.to_frame(name="v2")

    try:
        merged = (
            pd.merge_asof(
                df1, df2, left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta(days=45)
            ).dropna()
        )
    except Exception:
        return None, None

    if len(merged) < 4:
        return None, None

    last_4_idx = merged.index[-4:]

    span_days = (last_4_idx[-1] - last_4_idx[0]).days
    max_span = 300 if strict else 400
    if span_days > max_span:
        return None, None

    latest_common = pd.Timestamp(merged.index[-1])
    if (pd.Timestamp(s1_clean.index[-1]) - latest_common).days > 365 or (
        pd.Timestamp(s2_clean.index[-1]) - latest_common
    ).days > 365:
        return None, None

    return float(merged["v1"].loc[last_4_idx].sum()), float(merged["v2"].loc[last_4_idx].sum())


def _ttm_and_prior(series: Optional[pd.Series]) -> Tuple[Optional[float], Optional[float]]:
    s = _clean_series(series)
    if s is None or len(s) < 8:
        return None, None

    last_8_idx = pd.DatetimeIndex(s.index[-8:])

    if (last_8_idx[3] - last_8_idx[0]).days > 300:
        return None, None
    if (last_8_idx[7] - last_8_idx[4]).days > 300:
        return None, None
    if not (330 <= (last_8_idx[4] - last_8_idx[0]).days <= 400):
        return None, None

    return float(s.iloc[-4:].sum()), float(s.iloc[-8:-4].sum())


def _aligned_ttm_and_prior(
    s1: Optional[pd.Series], s2: Optional[pd.Series]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    s1_clean = _clean_series(s1)
    s2_clean = _clean_series(s2)
    if s1_clean is None or s2_clean is None:
        return None, None, None, None

    df1 = s1_clean.to_frame(name="v1")
    df2 = s2_clean.to_frame(name="v2")

    try:
        merged = (
            pd.merge_asof(
                df1, df2, left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta(days=45)
            ).dropna()
        )
    except Exception:
        return None, None, None, None

    if len(merged) < 8:
        return None, None, None, None

    last_8_idx = merged.index[-8:]

    if (last_8_idx[3] - last_8_idx[0]).days > 300:
        return None, None, None, None
    if (last_8_idx[7] - last_8_idx[4]).days > 300:
        return None, None, None, None
    if not (330 <= (last_8_idx[4] - last_8_idx[0]).days <= 400):
        return None, None, None, None

    latest_common = pd.Timestamp(merged.index[-1])
    if (pd.Timestamp(s1_clean.index[-1]) - latest_common).days > 365 or (
        pd.Timestamp(s2_clean.index[-1]) - latest_common
    ).days > 365:
        return None, None, None, None

    return (
        float(merged["v1"].loc[last_8_idx[-4:]].sum()),
        float(merged["v1"].loc[last_8_idx[:4]].sum()),
        float(merged["v2"].loc[last_8_idx[-4:]].sum()),
        float(merged["v2"].loc[last_8_idx[:4]].sum()),
    )


def extract_raw_metrics(bundle: FinancialBundle) -> Dict[str, Any]:
    raw: Dict[str, Any] = {}

    REV_KEYS = ["TotalRevenue", "Total Revenue", "OperatingRevenue", "Revenue"]
    GP_KEYS = ["GrossProfit", "Gross Profit"]
    OPINC_KEYS = ["OperatingIncome", "Operating Income", "OperatingIncomeLoss", "EBIT"]
    CASH_KEYS = [
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash Cash Equivalents And Short Term Investments",
        "CashAndCashEquivalents",
        "Cash",
    ]
    CL_KEYS = ["CurrentLiabilities", "Current Liabilities", "TotalCurrentLiabilities", "Total Current Liabilities"]
    CAPEX_KEYS = ["CapitalExpenditure", "Capital Expenditure", "CapEx", "PaymentsToAcquirePropertyPlantAndEquipment"]

    rev_q = _series(bundle.income_q, REV_KEYS)
    op_q = _series(bundle.income_q, OPINC_KEYS)

    # 1. gross_margin
    gp_q = _series(bundle.income_q, GP_KEYS)
    gp_ttm, gm_rev_ttm = _aligned_ttm(gp_q, rev_q, strict=False)
    if gp_ttm is not None and gm_rev_ttm is not None and gm_rev_ttm > 0:
        raw["gross_margin"] = gp_ttm / gm_rev_ttm
    else:
        raw["gross_margin"] = _finite(bundle.info.get("grossMargins"))

    # 2. revenue_growth
    rev_now, rev_prior = _ttm_and_prior(rev_q)
    if rev_now is not None and rev_prior is not None:
        if rev_prior > 0:
            raw["revenue_growth"] = (rev_now / rev_prior) - 1.0
        else:
            raw["revenue_growth"] = None
            raw["revenue_growth_reason"] = "undefined"
    else:
        rev_a = _series(bundle.income_a, REV_KEYS)
        if rev_a is not None and len(rev_a.dropna()) >= 2:
            s = rev_a.dropna().sort_index()
            prior_rev = float(s.iloc[-2])
            if prior_rev > 0:
                raw["revenue_growth"] = (float(s.iloc[-1]) / prior_rev) - 1.0
            else:
                raw["revenue_growth"] = None
                raw["revenue_growth_reason"] = "undefined"
        else:
            raw["revenue_growth"] = None

    # 3. operating_leverage
    op_now, op_prev, rev_now, rev_prev = _aligned_ttm_and_prior(op_q, rev_q)
    if op_now is not None and op_prev is not None and rev_now is not None and rev_prev is not None:
        if op_prev > 0 and rev_prev > 0:
            raw["operating_leverage"] = (op_now / op_prev - 1.0) - (rev_now / rev_prev - 1.0)
        elif rev_now > 0 and rev_prev > 0:
            margin_delta = (op_now / rev_now) - (op_prev / rev_prev)
            raw["operating_leverage"] = margin_delta
            raw["operating_leverage_fallback"] = True
        else:
            raw["operating_leverage"] = None
            raw["operating_leverage_reason"] = "undefined"
    else:
        raw["operating_leverage"] = None

    # 4. roic & 5. roic_momentum
    if bundle.roic_result:
        raw["roic"] = bundle.roic_result.get("latest_roic")
        raw["roic_momentum"] = bundle.roic_result.get("latest_slope")
    else:
        raw["roic"] = raw["roic_momentum"] = None

    # 6. net_debt_ebitda
    nd_series = _series(bundle.balance_q, ["NetDebt", "Net Debt"])
    if nd_series is not None and not nd_series.empty:
        nd = float(nd_series.dropna().sort_index().iloc[-1])
    else:
        tot_debt = _series(bundle.balance_q, ["TotalDebt", "Total Debt"])
        tot_cash = _series(bundle.balance_q, CASH_KEYS)
        if tot_debt is not None and tot_cash is not None and not tot_debt.empty and not tot_cash.empty:
            nd = float(tot_debt.dropna().sort_index().iloc[-1]) - float(tot_cash.dropna().sort_index().iloc[-1])
        else:
            idb = _finite(bundle.info.get("totalDebt"))
            ich = _finite(bundle.info.get("totalCash"))
            nd = (idb - ich) if (idb is not None and ich is not None) else None

    ebitda = _finite(bundle.info.get("ebitda"))
    if ebitda is None:
        ebitda = _ttm(_series(bundle.income_q, ["EBITDA"]), strict=True)
        if ebitda is None:
            da_q = _series(bundle.cashflow_q, ["ReconciledDepreciation", "Depreciation And Amortization"])
            op_ebitda_ttm, da_ebitda_ttm = _aligned_ttm(op_q, da_q, strict=True)
            ebitda = (
                (op_ebitda_ttm + da_ebitda_ttm)
                if (op_ebitda_ttm is not None and da_ebitda_ttm is not None)
                else None
            )

    if nd is not None and nd <= 0:
        raw["net_debt_ebitda"] = 0.0
    elif nd is not None and nd > 0 and ebitda is not None:
        raw["net_debt_ebitda"] = ND_EBITDA_UNSERVICEABLE if ebitda <= 0 else (nd / ebitda)
    else:
        raw["net_debt_ebitda"] = None

    # 7. current_ratio
    ca = _series(bundle.balance_q, ["CurrentAssets", "Current Assets", "TotalCurrentAssets"])
    cl = _series(bundle.balance_q, CL_KEYS)
    if ca is not None and cl is not None and not ca.empty and not cl.empty:
        curr_cl = float(cl.dropna().sort_index().iloc[-1])
        raw["current_ratio"] = (float(ca.dropna().sort_index().iloc[-1]) / curr_cl) if curr_cl > 0 else None
    else:
        raw["current_ratio"] = _finite(bundle.info.get("currentRatio"))

    # 8. gaap_gap (S3 Denominator Guard)
    ni_a = _series(bundle.income_a, ["NetIncome", "Net Income"])
    norm_a = _series(bundle.income_a, ["NormalizedIncome", "Normalized Income"])
    rev_a = _series(bundle.income_a, REV_KEYS)

    if ni_a is not None and norm_a is not None and not ni_a.empty and not norm_a.empty:
        ni = float(ni_a.dropna().sort_index().iloc[-1])
        norm = float(norm_a.dropna().sort_index().iloc[-1])
        rev_val = float(rev_a.dropna().sort_index().iloc[-1]) if (rev_a is not None and not rev_a.empty) else None

        # S3 Denominator Guard
        if rev_val is not None and rev_val > 0 and abs(ni) < 0.02 * rev_val:
            raw["gaap_gap"] = None
            raw["gaap_gap_reason"] = "denominator_unstable"
        elif abs(ni) >= 1e-6 * max(abs(norm), 1.0):
            raw["gaap_gap"] = abs(norm - ni) / abs(ni)
        else:
            raw["gaap_gap"] = None
            raw["gaap_gap_reason"] = "undefined"
    else:
        raw["gaap_gap"] = None

    # 9. capex_revenue
    capex_q = _series(bundle.cashflow_q, CAPEX_KEYS)
    capex_ttm, cap_rev_ttm = _aligned_ttm(capex_q, rev_q, strict=False)
    if capex_ttm is not None and cap_rev_ttm is not None and cap_rev_ttm > 0:
        raw["capex_revenue"] = abs(capex_ttm) / cap_rev_ttm
    else:
        raw["capex_revenue"] = None

    # 10. earnings_quality (S3 Denominator Guard)
    ni_q = _series(bundle.income_q, ["NetIncome", "Net Income", "NetIncomeLoss"])
    ocf_q = _series(
        bundle.cashflow_q, ["OperatingCashFlow", "Operating Cash Flow", "Total Cash From Operating Activities"]
    )
    ocf_ttm, ni_ttm = _aligned_ttm(ocf_q, ni_q, strict=True)
    rev_ttm = _ttm(rev_q, strict=True)

    if ni_ttm is not None and ocf_ttm is not None:
        # S3 Denominator Guard
        if rev_ttm is not None and rev_ttm > 0 and abs(ni_ttm) < 0.02 * rev_ttm:
            raw["earnings_quality"] = None
            raw["earnings_quality_reason"] = "denominator_unstable"
        elif ni_ttm <= 0:
            if ocf_ttm > 0:
                raw["earnings_quality"] = None
                raw["earnings_quality_reason"] = "undefined"
            else:
                raw["earnings_quality"] = 0.0
        else:
            raw["earnings_quality"] = ocf_ttm / ni_ttm
    else:
        # Annual Fallback
        ni_a_eq = _series(bundle.income_a, ["NetIncome", "Net Income", "NetIncomeLoss"])
        ocf_a_eq = _series(
            bundle.cashflow_a, ["OperatingCashFlow", "Operating Cash Flow", "Total Cash From Operating Activities"]
        )
        if ni_a_eq is not None and ocf_a_eq is not None and not ni_a_eq.empty and not ocf_a_eq.empty:
            ni_latest = float(ni_a_eq.dropna().sort_index().iloc[-1])
            ocf_latest = float(ocf_a_eq.dropna().sort_index().iloc[-1])
            rev_latest = (
                float(rev_a.dropna().sort_index().iloc[-1]) if (rev_a is not None and not rev_a.empty) else None
            )

            if rev_latest is not None and rev_latest > 0 and abs(ni_latest) < 0.02 * rev_latest:
                raw["earnings_quality"] = None
                raw["earnings_quality_reason"] = "denominator_unstable"
            elif ni_latest > 0:
                raw["earnings_quality"] = ocf_latest / ni_latest
                raw["earnings_quality_reason"] = "annual_basis"
                raw["earnings_quality_fallback"] = True
            else:
                if ocf_latest > 0:
                    raw["earnings_quality"] = None
                    raw["earnings_quality_reason"] = "undefined"
                else:
                    raw["earnings_quality"] = 0.0
        else:
            raw["earnings_quality"] = None

    # 11. share_dilution
    shares = _series(bundle.income_a, ["DilutedAverageShares", "Diluted Average Shares"])
    if shares is None or len(shares.dropna()) < 2:
        shares = _series(
            bundle.balance_a, ["OrdinarySharesNumber", "Ordinary Shares Number", "ShareIssued", "Share Issued"]
        )

    if shares is not None and len(shares.dropna()) >= 2:
        s = shares.dropna().sort_index()
        sh_t, sh_t1 = float(s.iloc[-1]), float(s.iloc[-2])
        if sh_t1 > 0:
            delta = (sh_t / sh_t1) - 1.0
            if abs(delta) > 0.25:
                raw["share_dilution"] = None
                raw["share_dilution_reason"] = "unstable"
            else:
                raw["share_dilution"] = delta
        else:
            raw["share_dilution"] = None
            raw["share_dilution_reason"] = "undefined"
    else:
        raw["share_dilution"] = None

    # 12. sbc_revenue
    sbc_q = _series(bundle.cashflow_q, ["StockBasedCompensation", "Stock Based Compensation"])
    sbc_ttm, sbc_rev_ttm = _aligned_ttm(sbc_q, rev_q, strict=False)
    if sbc_ttm is not None and sbc_rev_ttm is not None and sbc_rev_ttm > 0:
        raw["sbc_revenue"] = sbc_ttm / sbc_rev_ttm
    else:
        raw["sbc_revenue"] = None

    return raw


def extract_valuation_multiples(bundle: FinancialBundle) -> Dict[str, Optional[float]]:
    mcap = _finite(bundle.info.get("marketCap"))
    fcf = _finite(bundle.info.get("freeCashflow"))
    if fcf is None:
        fcf = _ttm(_series(bundle.cashflow_q, ["FreeCashFlow", "Free Cash Flow"]), strict=False)

    return {
        "trailing_pe": _finite(bundle.info.get("trailingPE")),
        "forward_pe": _finite(bundle.info.get("forwardPE")),
        "market_cap": mcap,
        "fcf_yield": (fcf / mcap) if (fcf is not None and mcap is not None and mcap > 0) else None,
    }


def compute_owner_earnings(cashflow_q: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    ocf_q = _series(cashflow_q, ["OperatingCashFlow", "Operating Cash Flow"])
    sbc_q = _series(cashflow_q, ["StockBasedCompensation", "Stock Based Compensation"])
    da_q = _series(cashflow_q, ["ReconciledDepreciation", "Depreciation And Amortization"])

    ocf_ttm = _aligned_ttm(ocf_q, ocf_q, strict=False)[0]
    sbc_ttm = _aligned_ttm(sbc_q, sbc_q, strict=False)[0] or 0.0
    da_ttm = _aligned_ttm(da_q, da_q, strict=False)[0]

    if ocf_ttm is None or da_ttm is None:
        return None, None

    oe = ocf_ttm - sbc_ttm - da_ttm
    return oe, da_ttm


def compute_normalized_fcf(cashflow_a: pd.DataFrame) -> Optional[float]:
    ocf_a = _series(
        cashflow_a, ["OperatingCashFlow", "Operating Cash Flow", "NetCashProvidedByUsedInOperatingActivities"]
    )
    capex_a = _series(
        cashflow_a,
        ["CapitalExpenditure", "Capital Expenditure", "CapEx", "PaymentsToAcquirePropertyPlantAndEquipment"],
    )

    if ocf_a is None or capex_a is None:
        return None

    df = pd.DataFrame({"ocf": ocf_a, "capex": capex_a}).dropna()
    df = df.sort_index()

    if len(df) < 3:
        return None

    last_3 = df.iloc[-3:]
    idx = pd.DatetimeIndex(last_3.index)

    if (idx[1] - idx[0]).days > 400 or (idx[2] - idx[1]).days > 400:
        return None

    fcf = last_3["ocf"] - last_3["capex"].abs()
    return float(fcf.mean())
