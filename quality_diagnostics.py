from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


def _pick_series(df: Optional[pd.DataFrame], candidates: Sequence[str]) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    col_map = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        if key in col_map:
            s = pd.to_numeric(df[col_map[key]], errors="coerce").dropna().sort_index()
            if not s.empty:
                return s
    return None


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _ttm_flow(q_series: Optional[pd.Series], a_series: Optional[pd.Series] = None) -> Optional[float]:
    if q_series is not None and len(q_series) >= 4:
        return _safe_float(q_series.iloc[-4:].sum())
    if a_series is not None and len(a_series) >= 1:
        return _safe_float(a_series.iloc[-1])
    if q_series is not None and len(q_series) >= 1:
        return _safe_float(q_series.iloc[-1] * 4.0)
    return None


def _yoy_growth_from_q_or_a(
    q_series: Optional[pd.Series],
    a_series: Optional[pd.Series] = None,
    is_flow: bool = True,
) -> Optional[float]:
    if q_series is not None:
        if is_flow and len(q_series) >= 8:
            curr = float(q_series.iloc[-4:].sum())
            prev = float(q_series.iloc[-8:-4].sum())
            if prev > 0:
                return (curr - prev) / prev
        elif not is_flow and len(q_series) >= 5:
            curr = float(q_series.iloc[-1])
            prev = float(q_series.iloc[-5])
            if prev > 0:
                return (curr - prev) / prev
    if a_series is not None and len(a_series) >= 2:
        curr = float(a_series.iloc[-1])
        prev = float(a_series.iloc[-2])
        if prev > 0:
            return (curr - prev) / prev
    return None


def compute_quality_diagnostics(
    raw_metrics: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None,
    fin_q: Optional[pd.DataFrame] = None,
    fin_a: Optional[pd.DataFrame] = None,
    bal_q: Optional[pd.DataFrame] = None,
    bal_a: Optional[pd.DataFrame] = None,
    cf_q: Optional[pd.DataFrame] = None,
    cf_a: Optional[pd.DataFrame] = None,
    incremental_roic: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Deterministic forensic accounting, working-capital context, and capital-allocation
    diagnostics layered cleanly atop canonical det-1.1.0 without altering engine math.
    """
    raw = dict(raw_metrics or {})
    inf = dict(info or {})

    rev_q = _pick_series(fin_q, ("TotalRevenue", "Total Revenue", "OperatingRevenue"))
    rev_a = _pick_series(fin_a, ("TotalRevenue", "Total Revenue", "OperatingRevenue"))
    gp_q = _pick_series(fin_q, ("GrossProfit", "Gross Profit"))
    gp_a = _pick_series(fin_a, ("GrossProfit", "Gross Profit"))
    cogs_q = _pick_series(fin_q, ("CostOfRevenue", "Cost Of Revenue", "ReconciledCostOfRevenue"))
    cogs_a = _pick_series(fin_a, ("CostOfRevenue", "Cost Of Revenue", "ReconciledCostOfRevenue"))
    ebit_q = _pick_series(fin_q, ("OperatingIncome", "Operating Income", "EBIT"))
    ebit_a = _pick_series(fin_a, ("OperatingIncome", "Operating Income", "EBIT"))
    ni_q = _pick_series(fin_q, ("NetIncome", "Net Income", "NormalizedIncome"))
    ni_a = _pick_series(fin_a, ("NetIncome", "Net Income", "NormalizedIncome"))
    int_q = _pick_series(fin_q, ("InterestExpense", "Interest Expense", "InterestExpenseNonOperating"))
    int_a = _pick_series(fin_a, ("InterestExpense", "Interest Expense", "InterestExpenseNonOperating"))

    ar_q = _pick_series(bal_q, ("AccountsReceivable", "Accounts Receivable", "Receivables"))
    ar_a = _pick_series(bal_a, ("AccountsReceivable", "Accounts Receivable", "Receivables"))
    inv_q = _pick_series(bal_q, ("Inventory", "Inventories"))
    inv_a = _pick_series(bal_a, ("Inventory", "Inventories"))

    ocf_q = _pick_series(cf_q, ("OperatingCashFlow", "Operating Cash Flow"))
    ocf_a = _pick_series(cf_a, ("OperatingCashFlow", "Operating Cash Flow"))
    capex_q = _pick_series(cf_q, ("CapitalExpenditure", "Capital Expenditure"))
    capex_a = _pick_series(cf_a, ("CapitalExpenditure", "Capital Expenditure"))
    sbc_q = _pick_series(cf_q, ("StockBasedCompensation", "Stock Based Compensation"))
    sbc_a = _pick_series(cf_a, ("StockBasedCompensation", "Stock Based Compensation"))
    div_q = _pick_series(cf_q, ("DividendsPaid", "CashDividendsPaid", "CommonStockDividendPaid"))
    div_a = _pick_series(cf_a, ("DividendsPaid", "CashDividendsPaid", "CommonStockDividendPaid"))

    ttm_rev = _ttm_flow(rev_q, rev_a) or _safe_float(inf.get("totalRevenue"))
    ttm_gp = _ttm_flow(gp_q, gp_a)
    ttm_cogs = _ttm_flow(cogs_q, cogs_a)
    if ttm_cogs is None and ttm_rev is not None and ttm_gp is not None and ttm_rev >= ttm_gp:
        ttm_cogs = ttm_rev - ttm_gp
    if ttm_gp is None and ttm_rev is not None:
        gm_raw = _safe_float(raw.get("gross_margin")) or _safe_float(inf.get("grossMargins"))
        if gm_raw is not None:
            ttm_gp = ttm_rev * gm_raw
            if ttm_cogs is None:
                ttm_cogs = max(0.0, ttm_rev - ttm_gp)

    ttm_ebit = _ttm_flow(ebit_q, ebit_a)
    ttm_ni = _ttm_flow(ni_q, ni_a) or _safe_float(inf.get("netIncomeToCommon"))
    ttm_int = _ttm_flow(int_q, int_a)
    ttm_ocf = _ttm_flow(ocf_q, ocf_a) or _safe_float(inf.get("operatingCashflow"))
    ttm_capex = _ttm_flow(capex_q, capex_a)
    ttm_sbc = _ttm_flow(sbc_q, sbc_a)
    ttm_div = _ttm_flow(div_q, div_a)

    ttm_fcf: Optional[float] = None
    if ttm_ocf is not None and ttm_capex is not None:
        ttm_fcf = ttm_ocf - abs(ttm_capex)
    elif inf.get("freeCashflow") is not None:
        ttm_fcf = _safe_float(inf.get("freeCashflow"))

    # 1. Interest Coverage (EBIT / |InterestExpense|)
    interest_coverage: Optional[float] = None
    interest_state = "UNVERIFIED"
    if ttm_ebit is not None and ttm_int is not None:
        abs_int = abs(ttm_int)
        if abs_int <= 1e-6:
            interest_coverage = 100.0 if ttm_ebit > 0 else 0.0
        else:
            interest_coverage = ttm_ebit / abs_int
        if interest_coverage >= 10.0:
            interest_state = "FORTRESS"
        elif interest_coverage >= 4.0:
            interest_state = "ADEQUATE"
        else:
            interest_state = "SOLVENCY_WARNING"

    # 2. Negative Working Capital Moat vs. Liquidity Distress Detector
    curr_ratio = _safe_float(raw.get("current_ratio")) or _safe_float(inf.get("currentRatio"))
    eq_qual = _safe_float(raw.get("earnings_quality"))
    nd_ebitda = _safe_float(raw.get("net_debt_ebitda"))
    wc_regime = "STANDARD"
    if curr_ratio is not None:
        if curr_ratio < 1.2:
            strong_ocf = (eq_qual is not None and eq_qual >= 1.0) or (ttm_ocf is not None and ttm_ocf > 0 and (ttm_fcf or 0.0) > 0)
            clean_debt = nd_ebitda is None or nd_ebitda <= 1.75
            if strong_ocf and clean_debt:
                wc_regime = "NEGATIVE_WC_MOAT"
            elif curr_ratio < 1.0 and ((nd_ebitda is not None and nd_ebitda > 3.0) or not strong_ocf):
                wc_regime = "LIQUIDITY_STRESS"
            else:
                wc_regime = "TIGHT_LIQUIDITY"
        elif curr_ratio >= 1.5:
            wc_regime = "AMPLE_LIQUIDITY"

    # 3. SBC-Adjusted Cash Conversion & True Owner FCF Margin
    sbc_adj_fcf_conversion: Optional[float] = None
    true_fcf_margin: Optional[float] = None
    sbc_cash_state = "UNVERIFIED"
    if ttm_fcf is not None and ttm_sbc is not None:
        true_fcf = ttm_fcf - max(0.0, ttm_sbc)
        if ttm_rev and ttm_rev > 0:
            true_fcf_margin = true_fcf / ttm_rev
        if ttm_ni is not None and ttm_ni > 0:
            sbc_adj_fcf_conversion = true_fcf / ttm_ni
    elif eq_qual is not None and raw.get("sbc_revenue") is not None and raw.get("capex_revenue") is not None:
        # Fallback estimate from canonical raw metrics when statement totals aren't passed
        gm = _safe_float(raw.get("gross_margin"))
        sbc_r = _safe_float(raw.get("sbc_revenue")) or 0.0
        capex_r = _safe_float(raw.get("capex_revenue")) or 0.0
        # Approximate net margin if available from info, else conservative 15% when profitable
        nm = _safe_float(inf.get("profitMargins"))
        if nm and nm > 0.01:
            approx_ocf_r = eq_qual * nm
            true_fcf_margin = approx_ocf_r - capex_r - sbc_r
            sbc_adj_fcf_conversion = true_fcf_margin / nm

    if sbc_adj_fcf_conversion is not None:
        if sbc_adj_fcf_conversion >= 0.85:
            sbc_cash_state = "ELITE_CASH_CONVERSION"
        elif sbc_adj_fcf_conversion >= 0.55:
            sbc_cash_state = "HEALTHY"
        else:
            sbc_cash_state = "SBC_DILUTED_WARNING"

    # 4. Channel-Stuffing Red Flag: Accounts Receivable YoY vs. Revenue YoY
    rev_yoy = _yoy_growth_from_q_or_a(rev_q, rev_a, is_flow=True)
    if rev_yoy is None:
        rev_yoy = _safe_float(raw.get("revenue_growth")) or _safe_float(inf.get("revenueGrowth"))
    ar_yoy = _yoy_growth_from_q_or_a(ar_q, ar_a, is_flow=False)
    ar_rev_divergence: Optional[float] = None
    ar_state = "UNVERIFIED"
    if ar_yoy is not None and rev_yoy is not None:
        ar_rev_divergence = ar_yoy - rev_yoy
        if ar_rev_divergence > 0.15 and ar_yoy > 0.10:
            ar_state = "RED_FLAG_CHANNEL_STUFFING"
        elif ar_rev_divergence > 0.08:
            ar_state = "WATCH"
        else:
            ar_state = "CLEAN"

    # 5. Inventory Bloat, Inventory Turnover (COGS / Avg Inv) & GMROI (Gross Profit / Avg Inv)
    inv_yoy = _yoy_growth_from_q_or_a(inv_q, inv_a, is_flow=False)
    inv_rev_divergence: Optional[float] = None
    inventory_turnover: Optional[float] = None
    gmroi: Optional[float] = None
    inventory_state = "NOT_APPLICABLE"

    avg_inv: Optional[float] = None
    if inv_q is not None and len(inv_q) >= 1:
        avg_inv = float(inv_q.iloc[-4:].mean()) if len(inv_q) >= 4 else float(inv_q.iloc[-1])
    elif inv_a is not None and len(inv_a) >= 1:
        avg_inv = float(inv_a.iloc[-2:].mean()) if len(inv_a) >= 2 else float(inv_a.iloc[-1])

    if avg_inv is not None and avg_inv > 0:
        if ttm_cogs is not None and ttm_cogs >= 0:
            inventory_turnover = ttm_cogs / avg_inv
        if ttm_gp is not None:
            gmroi = ttm_gp / avg_inv
        if inv_yoy is not None and rev_yoy is not None:
            inv_rev_divergence = inv_yoy - rev_yoy
            if inv_rev_divergence > 0.15 and inv_yoy > 0.10:
                inventory_state = "RED_FLAG_INVENTORY_BLOAT"
            elif inv_rev_divergence > 0.08:
                inventory_state = "WATCH"
            else:
                inventory_state = "CLEAN"
        else:
            inventory_state = "TRACKED"

    # 6. Dividend Payout Sustainability (|Dividends| / FCF or Payout Ratio)
    fcf_payout_ratio: Optional[float] = None
    eps_payout_ratio: Optional[float] = _safe_float(inf.get("payoutRatio"))
    dividend_state = "NON_PAYER"
    if ttm_div is not None and abs(ttm_div) > 0:
        abs_div = abs(ttm_div)
        if ttm_fcf is not None:
            if ttm_fcf <= 0:
                fcf_payout_ratio = 9.99
                dividend_state = "YIELD_TRAP_RISK"
            else:
                fcf_payout_ratio = abs_div / ttm_fcf
                if fcf_payout_ratio > 0.80:
                    dividend_state = "YIELD_TRAP_RISK"
                elif 0.15 <= fcf_payout_ratio <= 0.65:
                    dividend_state = "SUSTAINABLE_COMPOUNDER"
                else:
                    dividend_state = "CONSERVATIVE_PAYER"
    elif eps_payout_ratio is not None and eps_payout_ratio > 0:
        if eps_payout_ratio > 0.85:
            dividend_state = "YIELD_TRAP_RISK"
        elif 0.15 <= eps_payout_ratio <= 0.65:
            dividend_state = "SUSTAINABLE_COMPOUNDER"
        else:
            dividend_state = "CONSERVATIVE_PAYER"

    # 7. Intrinsic Compounding Capacity (ROIC_or_IncROIC * Reinvestment Rate)
    roic_val = _safe_float(incremental_roic) if incremental_roic is not None else _safe_float(raw.get("roic"))
    reinvestment_rate: Optional[float] = None
    intrinsic_compounding_rate: Optional[float] = None
    if roic_val is not None and roic_val > 0:
        if fcf_payout_ratio is not None and 0.0 <= fcf_payout_ratio <= 1.5:
            # Retain conservative cap on organic reinvestment rate when capital-light buybacks dominate
            capex_r = _safe_float(raw.get("capex_revenue")) or 0.05
            reinvestment_rate = float(np.clip(max(capex_r * 3.0, 1.0 - min(1.0, fcf_payout_ratio)) * 0.6, 0.10, 0.85))
        else:
            rev_g = _safe_float(raw.get("revenue_growth"))
            if rev_g is not None and rev_g > 0 and roic_val > 0.05:
                reinvestment_rate = float(np.clip(rev_g / max(roic_val, 0.08), 0.10, 0.85))
            else:
                reinvestment_rate = 0.35
        capped_roic = min(roic_val, 0.45)
        intrinsic_compounding_rate = capped_roic * reinvestment_rate

    red_flags: List[str] = []
    strengths: List[str] = []

    if interest_state == "FORTRESS" and interest_coverage is not None:
        strengths.append(f"Fortress interest coverage ({interest_coverage:.1f}x EBIT/Interest).")
    elif interest_state == "SOLVENCY_WARNING" and interest_coverage is not None:
        red_flags.append(f"Thin interest coverage ({interest_coverage:.1f}x EBIT/Interest < 4.0x).")

    if wc_regime == "NEGATIVE_WC_MOAT" and curr_ratio is not None:
        strengths.append(
            f"Negative working-capital moat (current_ratio={curr_ratio:.2f}x backed by strong cash conversion & low net debt — supplier/deferred-revenue float)."
        )
    elif wc_regime == "LIQUIDITY_STRESS" and curr_ratio is not None:
        red_flags.append(
            f"Genuine liquidity tightness (current_ratio={curr_ratio:.2f}x paired with elevated leverage or weak OCF)."
        )

    if sbc_cash_state == "ELITE_CASH_CONVERSION" and sbc_adj_fcf_conversion is not None:
        strengths.append(
            f"High SBC-adjusted owner cash conversion ({sbc_adj_fcf_conversion:.2f}x of Net Income)."
        )
    elif sbc_cash_state == "SBC_DILUTED_WARNING" and sbc_adj_fcf_conversion is not None:
        red_flags.append(
            f"Weak SBC-adjusted FCF conversion ({sbc_adj_fcf_conversion:.2f}x of Net Income after deducting SBC & CapEx)."
        )

    if ar_state == "RED_FLAG_CHANNEL_STUFFING" and ar_rev_divergence is not None:
        red_flags.append(
            f"Receivables red flag: AR YoY outpaced Revenue YoY by {ar_rev_divergence * 100:+.1f}pp (channel-stuffing / credit risk)."
        )

    if inventory_state == "RED_FLAG_INVENTORY_BLOAT" and inv_rev_divergence is not None:
        red_flags.append(
            f"Inventory bloat warning: Inventory YoY outpaced Revenue YoY by {inv_rev_divergence * 100:+.1f}pp."
        )

    if dividend_state == "YIELD_TRAP_RISK":
        payout_disp = f"{fcf_payout_ratio * 100:.0f}% of FCF" if fcf_payout_ratio is not None and fcf_payout_ratio < 9.0 else "exceeds FCF / 85% EPS"
        red_flags.append(f"Dividend sustainability warning (payout {payout_disp} — potential yield-trap risk).")
    elif dividend_state == "SUSTAINABLE_COMPOUNDER":
        payout_disp = f"{fcf_payout_ratio * 100:.0f}% of FCF" if fcf_payout_ratio is not None else f"{(eps_payout_ratio or 0.0) * 100:.0f}% of EPS"
        strengths.append(f"Sustainable dividend compounder profile (healthy payout {payout_disp}).")

    if intrinsic_compounding_rate is not None and intrinsic_compounding_rate >= 0.10:
        strengths.append(
            f"Strong intrinsic compounding capacity (~{intrinsic_compounding_rate * 100:.1f}%/yr from ROIC × Reinvestment)."
        )

    rows = [
        {
            "key": "interest_coverage",
            "label": "Interest Coverage (EBIT/Int)",
            "value_str": f"{interest_coverage:.1f}x" if interest_coverage is not None else "N/A",
            "state": interest_state,
            "note": "Measures operating profit buffer over debt service (>=10x fortress, <4x warning).",
        },
        {
            "key": "working_capital_regime",
            "label": "Working Capital Regime",
            "value_str": f"{curr_ratio:.2f}x" if curr_ratio is not None else "N/A",
            "state": wc_regime,
            "note": "Distinguishes float-funded negative working capital compounders from genuine liquidity distress.",
        },
        {
            "key": "sbc_adj_fcf_conversion",
            "label": "SBC-Adj FCF / Net Income",
            "value_str": f"{sbc_adj_fcf_conversion:.2f}x" if sbc_adj_fcf_conversion is not None else "N/A",
            "state": sbc_cash_state,
            "note": "True owner cash flow conversion after deducting both CapEx and Stock-Based Compensation.",
        },
        {
            "key": "ar_rev_divergence",
            "label": "AR vs. Revenue YoY Spread",
            "value_str": f"{ar_rev_divergence * 100:+.1f}pp" if ar_rev_divergence is not None else "N/A",
            "state": ar_state,
            "note": "Flags channel-stuffing or loosening customer credit when receivables grow >15pp faster than sales.",
        },
        {
            "key": "inventory_efficiency",
            "label": "Inventory Turnover / GMROI",
            "value_str": (
                f"{inventory_turnover:.1f}x / {gmroi:.2f}x"
                if inventory_turnover is not None and gmroi is not None
                else (f"{inventory_turnover:.1f}x" if inventory_turnover is not None else "N/A")
            ),
            "state": inventory_state,
            "note": "COGS / Avg Inventory and Gross Profit / Avg Inventory; flags bloat when inventory outgrows sales by >15pp.",
        },
        {
            "key": "dividend_sustainability",
            "label": "Dividend Payout (FCF / EPS)",
            "value_str": (
                f"{fcf_payout_ratio * 100:.0f}% FCF"
                if fcf_payout_ratio is not None and fcf_payout_ratio < 9.0
                else (f"{eps_payout_ratio * 100:.0f}% EPS" if eps_payout_ratio is not None else "0%")
            ),
            "state": dividend_state,
            "note": "Healthy dividend compounders pay 15%-65% of FCF; >80% FCF payout signals yield-trap fragility.",
        },
        {
            "key": "intrinsic_compounding",
            "label": "Intrinsic Compounding (ROIC×Reinv)",
            "value_str": f"{intrinsic_compounding_rate * 100:.1f}%/yr" if intrinsic_compounding_rate is not None else "N/A",
            "state": "HIGH_COMPOUNDER" if (intrinsic_compounding_rate or 0.0) >= 0.12 else ("MODERATE" if intrinsic_compounding_rate is not None else "UNVERIFIED"),
            "note": "Estimated sustainable organic earnings growth ceiling = min(ROIC, 45%) × Reinvestment Rate.",
        },
    ]

    return {
        "interest_coverage": interest_coverage,
        "interest_state": interest_state,
        "working_capital_regime": wc_regime,
        "sbc_adj_fcf_conversion": sbc_adj_fcf_conversion,
        "true_fcf_margin": true_fcf_margin,
        "sbc_cash_state": sbc_cash_state,
        "ar_rev_divergence": ar_rev_divergence,
        "ar_state": ar_state,
        "inv_rev_divergence": inv_rev_divergence,
        "inventory_turnover": inventory_turnover,
        "gmroi": gmroi,
        "inventory_state": inventory_state,
        "fcf_payout_ratio": fcf_payout_ratio,
        "eps_payout_ratio": eps_payout_ratio,
        "dividend_state": dividend_state,
        "reinvestment_rate": reinvestment_rate,
        "intrinsic_compounding_rate": intrinsic_compounding_rate,
        "strengths": strengths,
        "red_flags": red_flags,
        "rows": rows,
    }


def assess_moat_and_five_forces(
    raw_metrics: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quantitative Economic Moat & Porter's Five Forces corroboration.
    Never awards a moat label without quantitative ROIC and margin evidence.
    """
    raw = dict(raw_metrics or {})
    diag = diagnostics or compute_quality_diagnostics(raw_metrics=raw)

    gm = _safe_float(raw.get("gross_margin"))
    roic = _safe_float(raw.get("roic"))
    roic_mom = _safe_float(raw.get("roic_momentum"))
    op_lev = _safe_float(raw.get("operating_leverage"))
    capex_r = _safe_float(raw.get("capex_revenue"))
    rev_g = _safe_float(raw.get("revenue_growth"))
    wc_regime = diag.get("working_capital_regime")

    pillars: List[Dict[str, str]] = []
    moat_points = 0

    # Pillar 1: Pricing Power / Switching Costs & Intangible Assets
    if gm is not None and gm >= 0.50:
        pillars.append({
            "pillar": "Pricing Power & Switching Costs",
            "status": "STRONG",
            "evidence": f"Gross margin of {gm * 100:.1f}% (>=50%) signals differentiated IP, brand, or customer lock-in.",
        })
        moat_points += 2
    elif gm is not None and gm >= 0.32:
        pillars.append({
            "pillar": "Pricing Power & Switching Costs",
            "status": "MODERATE",
            "evidence": f"Gross margin of {gm * 100:.1f}% indicates defendable value-add above commodity levels.",
        })
        moat_points += 1
    else:
        gm_s = f"{gm * 100:.1f}%" if gm is not None else "N/A"
        pillars.append({
            "pillar": "Pricing Power & Switching Costs",
            "status": "LIMITED",
            "evidence": f"Gross margin ({gm_s}) relies on volume/cost efficiency rather than standalone pricing power.",
        })

    # Pillar 2: Economic Excess Returns (ROIC Durability)
    if roic is not None and roic >= 0.18 and (roic_mom is None or roic_mom >= -0.02):
        pillars.append({
            "pillar": "Excess Return on Capital (Moat Proof)",
            "status": "STRONG",
            "evidence": f"ROIC of {roic * 100:.1f}% comfortably clears cost of capital with stable/expanding trajectory.",
        })
        moat_points += 2
    elif roic is not None and roic >= 0.11:
        pillars.append({
            "pillar": "Excess Return on Capital (Moat Proof)",
            "status": "MODERATE",
            "evidence": f"ROIC of {roic * 100:.1f}% exceeds typical WACC (~9%), supporting a narrow economic moat.",
        })
        moat_points += 1
    else:
        roic_s = f"{roic * 100:.1f}%" if roic is not None else "N/A"
        pillars.append({
            "pillar": "Excess Return on Capital (Moat Proof)",
            "status": "UNPROVEN",
            "evidence": f"ROIC ({roic_s}) does not yet demonstrate sustained excess economic returns above WACC.",
        })

    # Pillar 3: Cost Scale & Bargaining Float
    if (op_lev is not None and op_lev >= 1.2) or wc_regime == "NEGATIVE_WC_MOAT":
        detail = "Negative working-capital float" if wc_regime == "NEGATIVE_WC_MOAT" else f"Operating leverage {op_lev:.2f}x"
        pillars.append({
            "pillar": "Scale & Ecosystem Bargaining Power",
            "status": "STRONG",
            "evidence": f"{detail} confirms fixed-cost scalability and supplier/customer bargaining advantage.",
        })
        moat_points += 1
    else:
        pillars.append({
            "pillar": "Scale & Ecosystem Bargaining Power",
            "status": "NEUTRAL",
            "evidence": "Standard working-capital and operating-cost scaling.",
        })

    # Enforce strict rule: no quantitative ROIC >= 10% => cannot claim WIDE or NARROW moat
    if roic is None or roic < 0.10:
        verdict = "NO_QUANT_MOAT"
        verdict_label = "No Quantitative Moat Confirmed"
    elif moat_points >= 4 and roic >= 0.18:
        verdict = "WIDE_MOAT_CANDIDATE"
        verdict_label = "Wide Moat Candidate (Quantitatively Corroborated)"
    elif moat_points >= 2:
        verdict = "NARROW_MOAT"
        verdict_label = "Narrow Moat (Quantitatively Corroborated)"
    else:
        verdict = "NO_QUANT_MOAT"
        verdict_label = "No Quantitative Moat Confirmed"

    five_forces = {
        "Buyer Power": "LOW (favorable)" if (gm is not None and gm >= 0.45) else ("HIGH (price-sensitive buyers)" if (gm is not None and gm < 0.25) else "MODERATE"),
        "Supplier Power": "LOW (favorable float)" if wc_regime == "NEGATIVE_WC_MOAT" else ("ELEVATED" if (capex_r is not None and capex_r > 0.12) else "MODERATE"),
        "Barriers to Entry": "HIGH (favorable)" if (roic is not None and roic >= 0.16) else "MODERATE / CONTESTABLE",
        "Substitution Risk": "LOW" if (rev_g is not None and rev_g >= 0.08 and (roic_mom or 0.0) >= -0.01) else "MONITOR",
        "Competitive Rivalry": f"DISCIPLINED ({sector or 'Sector'})" if (roic is not None and roic >= 0.15 and (gm or 0.0) >= 0.35) else f"INTENSE ({sector or 'Sector'})",
    }

    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "pillars": pillars,
        "five_forces": five_forces,
    }


def compute_garp_conviction_score(row_dict: Dict[str, Any], us10y: float = 0.0425) -> float:
    """
    Unified 0-100 conviction score blending Shrunk Quality (55%), Trajectory Momentum (25%),
    and Dynamic Equity Risk Premium / Valuation (20%).
    """
    comp = _safe_float(row_dict.get("composite_shrunk"))
    if comp is None:
        comp = _safe_float(row_dict.get("score")) or 0.0
    qual_pct = float(np.clip(comp * 10.0, 0.0, 100.0))

    traj = _safe_float(row_dict.get("trajectory"))
    traj_pct = float(np.clip(traj if traj is not None else 50.0, 0.0, 100.0))

    fy = _safe_float(row_dict.get("fcf_yield"))
    pe_f = _safe_float(row_dict.get("pe_forward"))
    pe_t = _safe_float(row_dict.get("pe_trailing"))
    rf = float(us10y if us10y is not None else 0.0425)

    if fy is not None:
        erp = fy - rf
        # erp = 0 bps -> 55, +200 bps -> 85, -200 bps -> 25
        val_pct = float(np.clip(55.0 + (erp * 1500.0), 0.0, 100.0))
    elif pe_f is not None and pe_f > 0:
        val_pct = float(np.clip(100.0 - (pe_f - 12.0) * 2.5, 0.0, 100.0))
    elif pe_t is not None and pe_t > 0:
        val_pct = float(np.clip(100.0 - (pe_t - 14.0) * 2.2, 0.0, 100.0))
    else:
        val_pct = 40.0

    return round(0.55 * qual_pct + 0.25 * traj_pct + 0.20 * val_pct, 2)


def passes_garp_preset(row_dict: Dict[str, Any], us10y: float = 0.0425) -> bool:
    """
    GARP ('Compounders at Fair Value') filter:
    - Gate state == 'pass'
    - Shrunk quality composite >= 7.0 and confidence >= 0.70
    - Non-deteriorating trajectory (None or >= 48.0)
    - Fair-or-better valuation relative to live 10Y Treasury (ERP >= -50 bps) or Forward P/E <= 26x
    """
    gs = str(row_dict.get("gate_state") or "").strip().lower()
    if gs != "pass":
        return False
    comp = _safe_float(row_dict.get("composite_shrunk"))
    if comp is None:
        comp = _safe_float(row_dict.get("score"))
    if comp is None or comp < 7.0:
        return False
    conf = _safe_float(row_dict.get("confidence"))
    if conf is not None and conf < 0.70:
        return False
    traj = _safe_float(row_dict.get("trajectory"))
    if traj is not None and traj < 48.0:
        return False

    fy = _safe_float(row_dict.get("fcf_yield"))
    pe_f = _safe_float(row_dict.get("pe_forward"))
    pe_t = _safe_float(row_dict.get("pe_trailing"))
    rf = float(us10y if us10y is not None else 0.0425)

    fair_erp = fy is not None and (fy - rf) >= -0.005
    fair_pe_fwd = pe_f is not None and 0.0 < pe_f <= 26.0
    fair_pe_trail = pe_t is not None and 0.0 < pe_t <= 28.0 and (fy is None or fy >= 0.032)
    return bool(fair_erp or fair_pe_fwd or fair_pe_trail)


def _fmt_raw_metric_headless(key: str, val: Any) -> str:
    from .scoring_engine import DEFAULT_REGISTRY

    spec_by_key = {spec.key: spec for spec in DEFAULT_REGISTRY}
    if val is None:
        return "N/A"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    spec = spec_by_key.get(key)
    fmt = (
        spec.fmt
        if spec
        else (
            "pct"
            if abs(f) <= 1.5
            and any(tok in key for tok in ("margin", "roic", "growth", "cagr", "dilution", "yield", "sbc", "capex"))
            else "mult"
        )
    )
    if fmt == "pct":
        return f"{f * 100:.1f}%"
    if fmt == "pp":
        return f"{f * 100:+.1f}pp"
    return f"{f:.2f}x"


def synthesize_bull_bear_views(
    symbol: str,
    score_res: Any,
    raw_metrics: Optional[Dict[str, Any]] = None,
    traj_row: Optional[Dict[str, Any]] = None,
    implied_growth: Optional[float] = None,
    info: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    import json
    from .scoring_engine import DEFAULT_REGISTRY

    spec_by_key = {spec.key: spec for spec in DEFAULT_REGISTRY}
    sym = (symbol or "TICKER").strip().upper()
    raw = dict(raw_metrics or {})
    info_dict = dict(info or {})
    subscores: Dict[str, float] = {}
    buckets: Dict[str, Optional[float]] = {}
    gates_fired: List[str] = []
    gate_state = "pass"
    gate_cap_applied: Optional[float] = None
    composite_val: Optional[float] = None
    fcf_yield_val: Optional[float] = None

    if score_res is not None:
        if isinstance(score_res, dict):
            if not raw and score_res.get("raw_json"):
                try:
                    raw.update(json.loads(score_res["raw_json"]))
                except Exception:
                    pass
            if isinstance(score_res.get("subscores"), dict):
                for k, v in score_res["subscores"].items():
                    if v is not None:
                        subscores[k] = float(v)
            elif score_res.get("subscores_json"):
                try:
                    for k, v in json.loads(score_res["subscores_json"]).items():
                        if v is not None:
                            subscores[k] = float(v)
                except Exception:
                    pass
            buckets["balance_sheet"] = score_res.get("b_balance_sheet")
            buckets["profitability"] = score_res.get("b_profitability")
            buckets["efficiency"] = score_res.get("b_efficiency")
            buckets["capital_alloc"] = score_res.get("b_capital_alloc")
            gf = score_res.get("gates_fired")
            if isinstance(gf, str):
                gates_fired = [g.strip() for g in gf.split(",") if g.strip()]
            elif isinstance(gf, (list, tuple)):
                gates_fired = [str(g).strip() for g in gf if str(g).strip()]
            gate_state = str(score_res.get("gate_state") or ("fired" if gates_fired else "pass"))
            gate_cap_applied = score_res.get("gate_cap_applied")
            composite_val = (
                score_res.get("composite_shrunk")
                if score_res.get("composite_shrunk") is not None
                else score_res.get("score")
            )
            fcf_yield_val = score_res.get("fcf_yield")
        else:
            if hasattr(score_res, "subscores") and isinstance(score_res.subscores, dict):
                for k, v in score_res.subscores.items():
                    if v is not None:
                        subscores[k] = float(v)
            if hasattr(score_res, "metrics") and isinstance(score_res.metrics, dict):
                for k, detail in score_res.metrics.items():
                    sub_v = getattr(detail, "subscore", detail if isinstance(detail, (int, float)) else None)
                    if sub_v is not None and k not in subscores:
                        subscores[k] = float(sub_v)
                    raw_v = getattr(detail, "raw", None)
                    if raw_v is not None and k not in raw:
                        raw[k] = raw_v
            if hasattr(score_res, "buckets") and isinstance(score_res.buckets, dict):
                buckets.update(score_res.buckets)
            if getattr(score_res, "b_balance_sheet", None) is not None:
                buckets["balance_sheet"] = getattr(score_res, "b_balance_sheet")
            gf = getattr(score_res, "gates_fired", ())
            if isinstance(gf, str):
                gates_fired = [g.strip() for g in gf.split(",") if g.strip()]
            elif isinstance(gf, (list, tuple)):
                gates_fired = [str(g).strip() for g in gf if str(g).strip()]
            gs_attr = getattr(score_res, "gate_state", None)
            if gs_attr:
                gate_state = str(gs_attr)
            else:
                gate_state = (
                    "fired"
                    if gates_fired
                    else ("unverified" if getattr(score_res, "gates_unverified", ()) else "pass")
                )
            gate_cap_applied = getattr(score_res, "gate_cap_applied", None)
            composite_val = getattr(score_res, "composite_shrunk", None)
            if composite_val is None:
                composite_val = getattr(score_res, "composite", None)
            fcf_yield_val = getattr(score_res, "fcf_yield", None)

    if fcf_yield_val is None:
        fcf_yield_val = (
            raw.get("fcf_yield") if raw.get("fcf_yield") is not None else info_dict.get("freeCashflowYield")
        )

    hist_growth = (
        raw.get("rev_growth_3y")
        if raw.get("rev_growth_3y") is not None
        else (
            raw.get("rev_cagr_3y")
            if raw.get("rev_cagr_3y") is not None
            else (
                raw.get("revenue_growth")
                if raw.get("revenue_growth") is not None
                else info_dict.get("revenueGrowth")
            )
        )
    )

    bullish: List[str] = []
    bearish: List[str] = []

    for k, sub in subscores.items():
        spec = spec_by_key.get(k)
        label = spec.label if spec else k.replace("_", " ").title()
        raw_str = _fmt_raw_metric_headless(k, raw.get(k))
        if sub >= 7.5:
            bullish.append(f"Elite {label} ({k}: raw {raw_str}, subscore {sub:.1f}/10).")
        elif sub <= 4.0:
            bearish.append(f"Weak {label} ({k}: raw {raw_str}, subscore {sub:.1f}/10).")

    b_bal = buckets.get("balance_sheet")
    if gate_state == "pass" and b_bal is not None and float(b_bal) >= 7.0:
        bullish.append(
            f"Clean balance sheet with gate_state='pass' and balance_sheet bucket {float(b_bal):.1f}/10."
        )
    if gates_fired:
        cap_str = f"{float(gate_cap_applied):.1f}/10" if gate_cap_applied is not None else "applied"
        bearish.append(
            f"Fundamental risk gates fired ({', '.join(gates_fired)}) with score cap {cap_str}."
        )

    if isinstance(traj_row, dict) and traj_row:
        t_val = traj_row.get("trajectory")
        derived: Dict[str, Any] = {}
        if isinstance(traj_row.get("derived_json"), str) and traj_row["derived_json"]:
            try:
                derived = json.loads(traj_row["derived_json"])
            except Exception:
                derived = {}
        elif isinstance(traj_row.get("derived_json"), dict):
            derived = dict(traj_row["derived_json"])

        self_rev = (
            traj_row.get("self_revision")
            if traj_row.get("self_revision") is not None
            else derived.get("self_revision")
        )
        peer_rev = (
            traj_row.get("peer_median_revision")
            if traj_row.get("peer_median_revision") is not None
            else derived.get("peer_median_revision")
        )

        pos_traj = False
        neg_traj = False
        if t_val is not None:
            tf = float(t_val)
            if tf > 10.0:
                pos_traj = tf >= 55.0
                neg_traj = tf < 45.0
            else:
                pos_traj = tf >= 5.5
                neg_traj = tf < 4.5

        pos_rev = self_rev is not None and peer_rev is not None and float(self_rev) > float(peer_rev)
        neg_rev = self_rev is not None and float(self_rev) < 0.0

        if pos_traj or pos_rev:
            parts = []
            if t_val is not None:
                parts.append(f"trajectory={float(t_val):.1f}")
            if self_rev is not None and peer_rev is not None:
                parts.append(
                    f"self_revision={float(self_rev):+.1%} vs peer_median_revision={float(peer_rev):+.1%}"
                )
            elif self_rev is not None:
                parts.append(f"self_revision={float(self_rev):+.1%}")
            bullish.append(f"Positive trajectory momentum ({', '.join(parts)}).")

        if neg_traj or neg_rev:
            parts = []
            if t_val is not None:
                parts.append(f"trajectory={float(t_val):.1f}")
            if self_rev is not None:
                parts.append(f"self_revision={float(self_rev):+.1%}")
            bearish.append(f"Negative trajectory revision momentum ({', '.join(parts)}).")

    if implied_growth is not None and hist_growth is not None:
        ig = float(implied_growth)
        hg = float(hist_growth)
        if ig < hg:
            bullish.append(
                f"Valuation margin of safety: Reverse-DCF implied_growth ({ig:.1%}) is below historical rev_growth_3y ({hg:.1%})."
            )
        elif (ig - hg) > 0.02:
            bearish.append(
                f"Valuation expectation risk: Reverse-DCF implied_growth ({ig:.1%}) exceeds historical rev_growth_3y ({hg:.1%}) by {(ig - hg) * 100:.1f}pp."
            )

    if fcf_yield_val is not None:
        fy = float(fcf_yield_val)
        if fy >= 0.045:
            bullish.append(f"Attractive cash flow valuation with fcf_yield={fy:.2%} (>= 4.50%).")
        elif fy < 0.025:
            bearish.append(f"Rich cash flow multiple with compressed fcf_yield={fy:.2%} (< 2.50%).")

    diag = diagnostics if isinstance(diagnostics, dict) else compute_quality_diagnostics(raw_metrics=raw, info=info_dict)
    for st_item in diag.get("strengths") or []:
        if st_item not in bullish:
            bullish.append(st_item)
    for rf_item in diag.get("red_flags") or []:
        if rf_item not in bearish:
            bearish.append(rf_item)

    if len(bullish) < 2:
        c_str = f"{float(composite_val):.2f}/10" if composite_val is not None else "N/A"
        bullish.append(
            f"{sym} maintains deterministic quality composite score of {c_str} with gate_state='{gate_state}'."
        )
    if len(bullish) < 2:
        sec_str = info_dict.get("sector") or "its sector"
        b_prof = buckets.get("profitability")
        p_str = f"{float(b_prof):.1f}/10" if b_prof is not None else "tracked"
        bullish.append(f"Operating scale in {sec_str} supported by profitability bucket score ({p_str}).")

    if len(bearish) < 2:
        c_str = f"{float(composite_val):.2f}/10" if composite_val is not None else "N/A"
        bearish.append(
            f"Ongoing execution required to defend {sym} quality composite ({c_str}) against margin mean-reversion."
        )
    if len(bearish) < 2:
        fy_fallback = f"{float(fcf_yield_val):.2%}" if fcf_yield_val is not None else "current multiple"
        bearish.append(
            f"Multiple compression sensitivity at fcf_yield={fy_fallback} if forward growth or ROIC decelerates."
        )

    return {"bullish": bullish, "bearish": bearish}
