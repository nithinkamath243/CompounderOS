# filepath: compounderos_app/test_scoring_engine.py
import math
from dataclasses import replace
import pytest
import numpy as np
import pandas as pd

from compounderos_app.scoring_engine import (
    band_score, gaap_gap_subscore, validate_registry, score_metrics,
    registry_fingerprint, DEFAULT_REGISTRY, BUCKET_WEIGHTS, MetricSpec,
    ENGINE_VERSION, CONFIDENCE_WEIGHTS
)
from compounderos_app.gics_map import universe_verdict
from compounderos_app.metric_extraction import extract_raw_metrics, FinancialBundle


def get_t_ref_raw():
    return {
        "gross_margin": 0.55, "revenue_growth": 0.12, "operating_leverage": 0.04,
        "roic": 0.22, "roic_momentum": 0.01, "net_debt_ebitda": 1.0,
        "current_ratio": 1.6, "gaap_gap": 0.05, "capex_revenue": 0.06,
        "earnings_quality": 1.15, "share_dilution": -0.01, "sbc_revenue": 0.03
    }


def test_v1_1_version():
    assert ENGINE_VERSION == "det-1.1.0"


def test_confidence_weights_sum_to_one():
    total_weight = sum(CONFIDENCE_WEIGHTS.values())
    assert math.isclose(total_weight, 1.0, abs_tol=1e-9)


def test_explicit_weights_zero_change():
    raw1 = get_t_ref_raw()
    raw2 = dict(raw1)
    raw2["revenue_growth"] = None
    raw3 = dict(raw1)
    raw3["roic_momentum"] = None

    implicit_registry = tuple(replace(s, weight=None) for s in DEFAULT_REGISTRY)

    for raw in (raw1, raw2, raw3):
        res_default = score_metrics(raw, registry=DEFAULT_REGISTRY)
        res_implicit = score_metrics(raw, registry=implicit_registry)
        assert res_default.composite == pytest.approx(res_implicit.composite, abs=1e-9)
        assert res_default.composite_pre_gate == pytest.approx(res_implicit.composite_pre_gate, abs=1e-9)
        assert res_default.buckets == res_implicit.buckets


def test_s4_graduated_gate_ramp_crossovers():
    raw_cap_nd = get_t_ref_raw()
    raw_cap_nd["net_debt_ebitda"] = 3.6
    res_36 = score_metrics(raw_cap_nd)
    assert res_36.composite == 4.0
    assert res_36.gate_cap_applied == 4.0

    raw_rel_nd = get_t_ref_raw()
    raw_rel_nd["net_debt_ebitda"] = 2.8
    res_28 = score_metrics(raw_rel_nd)
    assert res_28.gate_cap_applied is None

    raw_cap_eq = get_t_ref_raw()
    raw_cap_eq["earnings_quality"] = 0.84
    res_84 = score_metrics(raw_cap_eq)
    assert res_84.composite == 4.0
    assert res_84.gate_cap_applied == 4.0

    raw_rel_eq = get_t_ref_raw()
    raw_rel_eq["earnings_quality"] = 0.92
    res_92 = score_metrics(raw_rel_eq)
    assert res_92.gate_cap_applied is None


def test_s1_unverified_cap():
    raw = get_t_ref_raw()
    raw["net_debt_ebitda"] = None
    res = score_metrics(raw)
    assert res.composite <= 6.0
    assert "net_debt_ebitda" in res.gates_unverified

    raw["earnings_quality"] = 0.5
    res_both = score_metrics(raw)
    assert res_both.composite == 4.0


def test_min_bucket_rule_s8():
    raw = get_t_ref_raw()
    raw["capex_revenue"] = None
    raw["earnings_quality"] = None
    raw["share_dilution"] = None
    res = score_metrics(raw)
    assert res.buckets["capital_alloc"] is None


def test_universe_verdict():
    assert universe_verdict("Transaction & Payment Processing Services", "Financials") == "SCORED"
    assert universe_verdict("Financial Exchanges & Data", "Financials") == "SCORED"
    assert universe_verdict("Regional Banks", "Financials") == "NOT_APPLICABLE"
    assert universe_verdict("Commercial Banks", "Financials") == "NOT_APPLICABLE"
    assert universe_verdict("Industrial REITs", "Real Estate") == "NOT_APPLICABLE"
    assert universe_verdict("Data Center REITs", "Real Estate") == "NOT_APPLICABLE"
    assert universe_verdict("Multi-Family Residential REITs", "Real Estate") == "NOT_APPLICABLE"
    assert universe_verdict("Commercial & Residential Mortgage Finance", "Financials") == "NOT_APPLICABLE"
    assert universe_verdict("Unknown Sub Industry", "Financials") == "NOT_APPLICABLE_FALLBACK"
    assert universe_verdict("Unknown Sub Industry", "Real Estate") == "NOT_APPLICABLE_FALLBACK"
    assert universe_verdict("Systems Software", "Information Technology") == "SCORED"


def test_denominator_guards():
    idx = pd.date_range("2023-01-01", periods=4, freq="QE")
    inc_q = pd.DataFrame({"TotalRevenue": [100e6]*4, "NetIncome": [1e6]*4}, index=idx)
    inc_a = pd.DataFrame({"TotalRevenue": [400e6]*2, "NetIncome": [2e6]*2, "NormalizedIncome": [20e6]*2}, index=idx[:2])
    cf_q = pd.DataFrame({"OperatingCashFlow": [10e6]*4}, index=idx)
    cf_a = pd.DataFrame({"OperatingCashFlow": [40e6]*2}, index=idx[:2])

    bundle = FinancialBundle(info={}, income_a=inc_a, income_q=inc_q, balance_a=None, balance_q=None, cashflow_a=cf_a, cashflow_q=cf_q)
    raw = extract_raw_metrics(bundle)

    assert raw.get("gaap_gap") is None
    assert raw.get("gaap_gap_reason") == "denominator_unstable"
    assert raw.get("earnings_quality") is None
    assert raw.get("earnings_quality_reason") == "denominator_unstable"


def test_fingerprint_provenance_guard():
    fp1 = registry_fingerprint(DEFAULT_REGISTRY, BUCKET_WEIGHTS, gate_ramp_slope=3.0)
    fp2 = registry_fingerprint(DEFAULT_REGISTRY, BUCKET_WEIGHTS, gate_ramp_slope=4.0)
    assert fp1 != fp2


def test_band_score_none_nan_inf():
    assert band_score(None, 0.0, 5.0, 10.0) is None
    assert band_score(np.nan, 0.0, 5.0, 10.0) is None
    assert band_score(np.inf, 0.0, 5.0, 10.0) is None


def test_band_score_increasing_normal():
    assert band_score(0.0, 0.0, 5.0, 10.0) == 0.0
    assert band_score(5.0, 0.0, 5.0, 10.0) == 5.0
    assert band_score(10.0, 0.0, 5.0, 10.0) == 10.0
    assert band_score(2.5, 0.0, 5.0, 10.0) == 2.5


def test_band_score_decreasing_normal():
    assert band_score(10.0, 10.0, 5.0, 0.0) == 0.0
    assert band_score(5.0, 10.0, 5.0, 0.0) == 5.0
    assert band_score(0.0, 10.0, 5.0, 0.0) == 10.0


def test_band_score_clipping():
    assert band_score(-5.0, 0.0, 5.0, 10.0) == 0.0
    assert band_score(20.0, 0.0, 5.0, 10.0) == 10.0


def test_gaap_gap_subscore_normal():
    assert gaap_gap_subscore(100.0, 100.0) == 10.0
    assert gaap_gap_subscore(100.0, 110.0) == 5.0
    assert gaap_gap_subscore(100.0, 130.0) == 0.0


def test_gaap_gap_subscore_zero_or_tiny_ni():
    assert gaap_gap_subscore(0.0, 100.0) is None
    assert gaap_gap_subscore(1e-7, 100.0) is None


def test_gaap_gap_subscore_none():
    assert gaap_gap_subscore(None, 100.0) is None
    assert gaap_gap_subscore(100.0, None) is None


def test_nd_ebitda_unserviceable_score():
    raw = get_t_ref_raw()
    raw["net_debt_ebitda"] = 99.0
    res = score_metrics(raw)
    assert res.metrics["net_debt_ebitda"].subscore == 0.0


def test_nd_ebitda_negative_net_debt():
    raw = get_t_ref_raw()
    raw["net_debt_ebitda"] = 0.0
    res = score_metrics(raw)
    assert res.metrics["net_debt_ebitda"].subscore == 10.0


def test_earnings_quality_positive_ratio():
    raw = get_t_ref_raw()
    raw["earnings_quality"] = 1.0
    res = score_metrics(raw)
    assert res.metrics["earnings_quality"].subscore == 5.0


def test_earnings_quality_none_on_non_cash_losses():
    raw = get_t_ref_raw()
    raw["earnings_quality"] = None
    raw["earnings_quality_reason"] = "undefined"
    res = score_metrics(raw)
    assert res.metrics["earnings_quality"].subscore is None


def test_earnings_quality_zero_when_both_negative():
    raw = get_t_ref_raw()
    raw["earnings_quality"] = 0.0
    res = score_metrics(raw)
    assert res.metrics["earnings_quality"].subscore == 0.0


def test_share_dilution_unstable_delta():
    raw = get_t_ref_raw()
    raw["share_dilution"] = None
    raw["share_dilution_reason"] = "unstable"
    res = score_metrics(raw)
    assert res.metrics["share_dilution"].subscore is None


def test_share_dilution_normal():
    raw = get_t_ref_raw()
    raw["share_dilution"] = 0.0
    res = score_metrics(raw)
    assert res.metrics["share_dilution"].subscore == 5.0


def test_drop_and_reweight_profitability():
    raw = get_t_ref_raw()
    raw["operating_leverage"] = None
    res = score_metrics(raw)
    gm_sub = res.metrics["gross_margin"].subscore
    rg_sub = res.metrics["revenue_growth"].subscore
    assert res.buckets["profitability"] == pytest.approx((gm_sub + rg_sub) / 2.0, abs=1e-9)


def test_drop_and_reweight_efficiency():
    raw = get_t_ref_raw()
    raw["roic_momentum"] = None
    res = score_metrics(raw)
    assert res.buckets["efficiency"] == pytest.approx(res.metrics["roic"].subscore, abs=1e-9)


def test_drop_and_reweight_balance_sheet():
    raw = get_t_ref_raw()
    raw["gaap_gap"] = None
    res = score_metrics(raw)
    nd_sub = res.metrics["net_debt_ebitda"].subscore
    cr_sub = res.metrics["current_ratio"].subscore
    assert res.buckets["balance_sheet"] == pytest.approx((nd_sub + cr_sub) / 2.0, abs=1e-9)


def test_validate_registry_valid():
    validate_registry(DEFAULT_REGISTRY, BUCKET_WEIGHTS)


def test_validate_registry_invalid_bucket_weights():
    bad_weights = {"profitability": 0.50, "efficiency": 0.30, "balance_sheet": 0.20, "capital_alloc": 0.10}
    with pytest.raises(ValueError, match="Bucket weights must sum to 1.0"):
        validate_registry(DEFAULT_REGISTRY, bad_weights)


def test_validate_registry_non_monotonic_anchors():
    bad_spec = MetricSpec("gross_margin", "Gross Margin", "profitability", 0.50, 0.40, 0.60, weight=1.0/3.0, fmt="pct")
    bad_reg = (bad_spec,) + DEFAULT_REGISTRY[1:]
    with pytest.raises(ValueError, match="anchors must be strictly monotonic"):
        validate_registry(bad_reg, BUCKET_WEIGHTS)


def test_score_metrics_all_missing():
    res = score_metrics({})
    for b in res.buckets.values():
        assert b is None
    assert res.composite is None
    assert res.confidence == 0.0


def test_operating_leverage_fallback_anchors():
    raw = get_t_ref_raw()
    raw["operating_leverage"] = 0.005
    raw["operating_leverage_fallback"] = True
    res = score_metrics(raw)
    assert res.metrics["operating_leverage"].subscore == 5.0
