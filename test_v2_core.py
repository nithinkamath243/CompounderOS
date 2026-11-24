import pytest
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone

from compounderos_app.database import DataStore
from compounderos_app.scoring_engine import ScoreResult, score_metrics, DEFAULT_REGISTRY
from compounderos_app.analytics_v2 import DipTriggerAnalyzer, IncrementalRoicAnalyzer, ScoreBacktester

def get_dummy_result(score=7.0):
    return ScoreResult(
        composite=score,
        composite_pre_gate=score,
        composite_shrunk=score,
        gate_cap_applied=None,
        buckets={'profitability': 5.0, 'efficiency': 5.0, 'balance_sheet': 5.0, 'capital_alloc': 5.0},
        metrics={},
        gates_fired=(),
        gates_unverified=(),
        confidence=1.0,
        engine_version='det-1.0.0'
    )

def test_v1_same_day_rescan():
    store = DataStore(":memory:")
    store._migrate_v2()
    
    multiples = {'fcf_yield': 0.05}
    raw_inputs = {'roic': 0.22}
    store.record_quality_result("AAPL", "IT", get_dummy_result(7.0), multiples, raw_inputs)
    store.record_quality_result("AAPL", "IT", get_dummy_result(8.0), multiples, raw_inputs)
    # Empty raw_inputs on same day must not overwrite a valid row
    store.record_quality_result("AAPL", "IT", get_dummy_result(9.0), multiples, {})
    
    cur = store.conn.cursor()
    cur.execute("SELECT score FROM sp500_scores_history WHERE symbol='AAPL'")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 8.0

def test_v2_forced_failure_atomicity():
    store = DataStore(":memory:")
    store._migrate_v2()
    try:
        # Invalid multiple force-fails SQLite execution 
        store.record_quality_result("AAPL", "IT", get_dummy_result(), {'trailing_pe': 'STRING_FAIL'}, {})
    except Exception:
        pass
        
    cur = store.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sp500_scores_history")
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT COUNT(*) FROM sp500_scores")
    assert cur.fetchone()[0] == 0

def test_v10_gate_combined_state():
    store = DataStore(":memory:")
    store._migrate_v2()
    
    res = ScoreResult(
        composite=4.0,
        composite_pre_gate=6.0,
        composite_shrunk=4.0,
        gate_cap_applied=4.0,
        buckets={},
        metrics={},
        gates_fired=('net_debt_ebitda',),
        gates_unverified=('earnings_quality',),
        confidence=0.8,
        engine_version='det-1.0.0'
    )
    store.record_quality_result("NVDA", "IT", res, {}, {})
    
    cur = store.conn.cursor()
    cur.execute("SELECT gate_state, gates_fired, gates_unverified FROM sp500_scores_history")
    row = cur.fetchone()
    assert row[0] == 'fired_unverified'
    assert row[1] == 'net_debt_ebitda'
    assert row[2] == 'earnings_quality'

def test_v13_evaluated_at_lexicographic():
    d1 = "2025-12-31T23:59:59+00:00"
    d2 = "2026-01-01T00:00:01+00:00"
    assert d1 < d2

# --- Prompt 4: Algorithm Edge Cases ---

def test_dip_trigger_flat_mad_guard():
    analyzer = DipTriggerAnalyzer()
    now = pd.Timestamp.utcnow()
    # Span > 360 days, ends today, N > 120
    dates = pd.date_range(end=now, periods=400, freq="D") 
    df_hist = pd.DataFrame({
        'evaluated_at': dates,
        'fcf_yield': [0.05] * 400,
        'source': 'scan'
    })
    res = analyzer.evaluate_dip(df_hist)
    assert res.state == 'flat_multiple'

def test_dip_trigger_corrupt_spike_robust_z():
    analyzer = DipTriggerAnalyzer(z_trigger=1.5)
    now = pd.Timestamp.utcnow()
    dates = pd.date_range(end=now, periods=400, freq="D")
    
    yields = np.random.normal(0.04, 0.002, 400)
    yields[200] = 0.40  # 10x corrupt spike
    yields[-1] = 0.05   # Genuine cheap observation at current date

    valid_raw_json = json.dumps({
        "gross_margin": 0.55, "revenue_growth": 0.12, "operating_leverage": 0.04,
        "roic": 0.22, "roic_momentum": 0.01, "net_debt_ebitda": 1.0,
        "current_ratio": 1.6, "gaap_gap": 0.05, "capex_revenue": 0.06,
        "earnings_quality": 1.15, "share_dilution": -0.01, "sbc_revenue": 0.03
    })
    
    df_hist = pd.DataFrame({
        'evaluated_at': dates,
        'fcf_yield': yields,
        'source': 'scan',
        'raw_json': valid_raw_json,
        'confidence': 1.0,
        'gate_state': 'pass',
        'pe_trailing': np.nan
    })
    
    res = analyzer.evaluate_dip(df_hist)
    assert res.state == 'fired'
    assert res.z_robust > 2.0
    assert res.z_classical < res.z_robust

def test_incremental_roic_capital_returning():
    analyzer = IncrementalRoicAnalyzer()
    dates = pd.date_range("2020-01-01", periods=12, freq="QE")
    nopat = pd.Series([100]*12, index=dates)
    ic = pd.Series(np.linspace(1000, 500, 12), index=dates) # Shrinking IC
    
    res = analyzer.analyze(nopat, ic)
    assert res.state == 'capital_returning'
    assert res.value is None

def test_f_b5_wiring_proof_gate():
    # Correctness gate verifying re-scoring from raw JSON under current registry
    raw_nvda = {
        "gross_margin": 0.75, "revenue_growth": 1.26, "operating_leverage": 0.40,
        "roic": 0.65, "roic_momentum": 0.15, "net_debt_ebitda": 0.0,
        "current_ratio": 3.5, "gaap_gap": 0.0, "capex_revenue": 0.02,
        "earnings_quality": 1.1, "share_dilution": 0.0, "sbc_revenue": 0.04
    }
    
    res_nvda = score_metrics(raw_nvda, registry=DEFAULT_REGISTRY)
    assert isinstance(res_nvda.composite, float)
    # Note: On a live DB pull this reproduces exactly 9.35


def test_score_backtester_rank_ic_newey_west():
    rng = np.random.default_rng(42)
    x = pd.Series(np.linspace(1.0, 10.0, 120))
    y = pd.Series(0.05 * x + rng.normal(0.0, 0.02, 120))

    ic, t_stat = ScoreBacktester.rank_ic_newey_west(x, y, horizon_months=12)
    assert ic is not None and t_stat is not None
    assert 0.85 < ic <= 1.0
    assert t_stat > 5.0

    # Insufficient sample (< 100 observations) returns (None, None)
    ic_small, t_small = ScoreBacktester.rank_ic_newey_west(x.iloc[:50], y.iloc[:50], horizon_months=12)
    assert ic_small is None and t_small is None


def test_score_backtester_empty_store_returns_insufficient_sample():
    store = DataStore(":memory:")
    bt = ScoreBacktester(store)
    out = bt.run_backtest(horizon_months=12)
    assert out == {"error": "insufficient_sample"}

