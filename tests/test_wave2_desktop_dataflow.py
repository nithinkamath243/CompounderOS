import os
import ssl
import tempfile
import threading
import pandas as pd
import numpy as np

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.analytics_core import RoicMomentumAnalyzer
from compounderos_app.charts import ChartPayload
from compounderos_app.database import DataStore
from compounderos_app.gui.services.data_service import DataService
from compounderos_app.gui.tabs.valuation import ValuationTabMixin
from compounderos_app.metric_extraction import (
    FinancialBundle,
    compute_normalized_fcf,
    compute_owner_earnings,
    extract_raw_metrics,
)


def _make_sec_facts():
    q_dates = pd.date_range("2023-03-31", periods=12, freq="QE")
    a_dates = pd.date_range("2023-12-31", periods=3, freq="YE")

    def _q_items(vals, unit_scale=1.0):
        out = []
        for d, v in zip(q_dates, vals):
            start = (d - pd.Timedelta(days=89)).strftime("%Y-%m-%d")
            end = d.strftime("%Y-%m-%d")
            out.append({"start": start, "end": end, "val": float(v * unit_scale), "fp": f"Q{d.quarter}", "form": "10-Q"})
        return out

    def _a_items(vals, unit_scale=1.0):
        out = []
        for d, v in zip(a_dates, vals):
            start = (d - pd.Timedelta(days=364)).strftime("%Y-%m-%d")
            end = d.strftime("%Y-%m-%d")
            out.append({"start": start, "end": end, "val": float(v * unit_scale), "fp": "FY", "form": "10-K"})
        return out

    us_gaap = {
        "Revenues": {"units": {"USD": _q_items(np.linspace(100, 160, 12), 1e6) + _a_items([420, 500, 600], 1e6)}},
        "GrossProfit": {"units": {"USD": _q_items(np.linspace(60, 96, 12), 1e6) + _a_items([250, 300, 360], 1e6)}},
        "OperatingIncomeLoss": {"units": {"USD": _q_items(np.linspace(25, 45, 12), 1e6) + _a_items([105, 130, 165], 1e6)}},
        "NetIncomeLoss": {"units": {"USD": _q_items(np.linspace(20, 36, 12), 1e6) + _a_items([84, 104, 132], 1e6)}},
        "IncomeTaxExpenseBenefit": {"units": {"USD": _q_items(np.linspace(5, 9, 12), 1e6) + _a_items([21, 26, 33], 1e6)}},
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": {
            "units": {"USD": _q_items(np.linspace(25, 45, 12), 1e6) + _a_items([105, 130, 165], 1e6)}
        },
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {"shares": _q_items(np.linspace(105, 100, 12), 1e6) + _a_items([104, 102, 100], 1e6)}
        },
        "DepreciationDepletionAndAmortization": {
            "units": {"USD": _q_items(np.linspace(4, 6, 12), 1e6) + _a_items([16, 20, 22], 1e6)}
        },
        "Assets": {"units": {"USD": _q_items(np.linspace(400, 520, 12), 1e6) + _a_items([430, 480, 520], 1e6)}},
        "AssetsCurrent": {"units": {"USD": _q_items(np.linspace(150, 210, 12), 1e6) + _a_items([165, 190, 210], 1e6)}},
        "LiabilitiesCurrent": {"units": {"USD": _q_items(np.linspace(80, 100, 12), 1e6) + _a_items([85, 92, 100], 1e6)}},
        "CashAndCashEquivalentsAtCarryingValue": {
            "units": {"USD": _q_items(np.linspace(60, 90, 12), 1e6) + _a_items([65, 78, 90], 1e6)}
        },
        "LongTermDebt": {"units": {"USD": _q_items(np.linspace(50, 40, 12), 1e6) + _a_items([48, 44, 40], 1e6)}},
        "StockholdersEquity": {"units": {"USD": _q_items(np.linspace(250, 360, 12), 1e6) + _a_items([270, 315, 360], 1e6)}},
        "CommonStockSharesOutstanding": {
            "units": {"shares": _q_items(np.linspace(105, 100, 12), 1e6) + _a_items([104, 102, 100], 1e6)}
        },
        "NetCashProvidedByUsedInOperatingActivities": {
            "units": {"USD": _q_items(np.linspace(28, 48, 12), 1e6) + _a_items([115, 145, 180], 1e6)}
        },
        "PaymentsToAcquirePropertyPlantAndEquipment": {
            "units": {"USD": _q_items(np.linspace(5, 8, 12), 1e6) + _a_items([20, 25, 30], 1e6)}
        },
        "ShareBasedCompensation": {
            "units": {"USD": _q_items(np.linspace(2, 3, 12), 1e6) + _a_items([8, 10, 12], 1e6)}
        },
    }
    return {"facts": {"us-gaap": us_gaap}}


def test_build_sec_dataframes_extracts_full_columns_for_metrics_owner_earnings_and_roic():
    facts = _make_sec_facts()
    ds = DataService(None, "CompounderOS/1.0 (test@example.com)", lambda *a, **kw: "")
    fin_q, fin_a, bal_q, bal_a, cf_q, cf_a = ds.build_sec_dataframes(facts)

    for col in ("TaxProvision", "PretaxIncome", "DilutedAverageShares", "NormalizedIncome", "EBITDA"):
        assert col in fin_q.columns, f"Missing {col} in fin_q"
        assert col in fin_a.columns, f"Missing {col} in fin_a"
        assert not fin_q[col].dropna().empty
        assert not fin_a[col].dropna().empty

    for col in ("TotalAssets", "CurrentAssets", "CurrentLiabilities"):
        assert col in bal_q.columns, f"Missing {col} in bal_q"
        assert col in bal_a.columns, f"Missing {col} in bal_a"
        assert not bal_q[col].dropna().empty
        assert not bal_a[col].dropna().empty

    assert "ReconciledDepreciation" in cf_q.columns
    assert "ReconciledDepreciation" in cf_a.columns
    assert not cf_q["ReconciledDepreciation"].dropna().empty
    assert not cf_a["ReconciledDepreciation"].dropna().empty

    # Verify EBITDA = OperatingIncome + DA when EBITDA tag is not separately reported
    latest_q = fin_q.index[-1]
    assert np.isclose(
        fin_q.loc[latest_q, "EBITDA"],
        fin_q.loc[latest_q, "OperatingIncome"] + cf_q.loc[latest_q, "ReconciledDepreciation"],
    )

    # Verify RoicMomentumAnalyzer uses primary IC formula and effective tax rate
    roic_res = RoicMomentumAnalyzer().analyze(fin_q, bal_q)
    assert roic_res["invested_capital_method"] == "TotalAssets - Cash - CurrentLiabilities"
    assert roic_res["tax_rate_source"] == "effective (from statements, clipped 0-50%)"
    assert roic_res["latest_roic"] is not None

    # Verify compute_owner_earnings and compute_normalized_fcf succeed
    oe, maint_capex = compute_owner_earnings(cf_q)
    assert oe is not None and maint_capex is not None
    norm_fcf = compute_normalized_fcf(cf_a)
    assert norm_fcf is not None and norm_fcf > 0

    # Verify extract_raw_metrics computes all 12 metrics without None fallbacks
    bundle = FinancialBundle(
        info={},
        income_q=fin_q,
        income_a=fin_a,
        balance_q=bal_q,
        balance_a=bal_a,
        cashflow_q=cf_q,
        cashflow_a=cf_a,
        roic_result=roic_res,
    )
    raw = extract_raw_metrics(bundle)
    for k in (
        "gross_margin",
        "revenue_growth",
        "operating_leverage",
        "roic",
        "roic_momentum",
        "net_debt_ebitda",
        "current_ratio",
        "gaap_gap",
        "capex_revenue",
        "earnings_quality",
        "share_dilution",
        "sbc_revenue",
    ):
        assert raw.get(k) is not None, f"Expected non-None metric for {k}, got {raw}"


def test_apply_search_results_assigns_annual_dataframes():
    from compounderos_app.gui.app import CompounderOSApp

    facts = _make_sec_facts()
    ds = DataService(None, "CompounderOS/1.0 (test@example.com)", lambda *a, **kw: "")
    fin_q, fin_a, bal_q, bal_a, cf_q, cf_a = ds.build_sec_dataframes(facts)

    class _DummyLabel:
        def configure(self, **_kw):
            pass

    class _DummyHarness:
        request_id = 1
        hist_period = "1y"
        df_fin_q = None
        df_bal_q = None
        df_cf_q = None
        df_fin_a = None
        df_bal_a = None
        df_cf_a = None
        df_dividends = None
        lbl_status = _DummyLabel()
        fig = object()
        ax = object()

        def fill_insights(self, *_a):
            pass

        def fill_quality_checklist(self, *_a):
            self.captured_bundle = FinancialBundle(
                info={},
                income_q=self.df_fin_q,
                income_a=self.df_fin_a,
                balance_q=self.df_bal_q,
                balance_a=self.df_bal_a,
                cashflow_q=self.df_cf_q,
                cashflow_a=self.df_cf_a,
            )

        def fill_insider_calendar(self, *_a):
            pass

        def load_kpis(self):
            pass

        def update_kpis_from_sec(self, *_a):
            pass

        def load_transcripts(self):
            pass

        def refresh_historical_valuation(self):
            self.captured_norm_fcf = compute_normalized_fcf(self.df_cf_a)

        def ensure_history_period(self, *_a):
            pass

        def run_ai_report_logic(self, *_a):
            pass

    import compounderos_app.gui.app as app_mod

    class _FakeTicker:
        def __init__(self, sym):
            self.symbol = sym
            self.info = {}
            self.insider_transactions = None
            self.calendar = None
            self.dividends = pd.Series(dtype=float)

    orig_ticker = app_mod.yf.Ticker
    app_mod.yf.Ticker = _FakeTicker
    try:
        harness = _DummyHarness()
        CompounderOSApp._apply_search_results(
            harness, "AAPL", {}, fin_q, bal_q, cf_q, pd.Series(dtype=float), 1, fin_a, bal_a, cf_a
        )
    finally:
        app_mod.yf.Ticker = orig_ticker

    assert harness.df_fin_a is fin_a
    assert harness.df_bal_a is bal_a
    assert harness.df_cf_a is cf_a
    assert harness.captured_norm_fcf is not None

    raw = extract_raw_metrics(harness.captured_bundle)
    assert raw["gaap_gap"] is not None
    assert raw["share_dilution"] is not None

    payload = ChartPayload.from_app(harness)
    assert payload.df_fin_a is fin_a
    assert payload.df_bal_a is bal_a
    assert payload.df_cf_a is cf_a


def test_proc_historical_valuation_reads_from_scan_store_under_lock():
    store = DataStore(":memory:", role="sor")
    scan_store = DataStore(":memory:", role="replica")

    dates = pd.date_range("2025-01-01", periods=70, freq="3D")
    with scan_store.lock:
        for i, d in enumerate(dates):
            iso = d.strftime("%Y-%m-%dT12:00:00+00:00")
            scan_store.conn.execute(
                """INSERT INTO sp500_scores_history
                   (symbol, scan_date, evaluated_at, sector, score, confidence, gate_state, fcf_yield, engine_version, source, raw_json, subscores_json)
                   VALUES (?, ?, ?, 'IT', 8.0, 0.9, 'pass', ?, 'det-1.1.0', 'scan', '{}', '{}')""",
                ("AAPL", d.strftime("%Y-%m-%d"), iso, 0.03 + 0.0002 * i),
            )
        scan_store.conn.commit()

    lock_acquired = {"count": 0}
    orig_lock = scan_store.lock

    class _TrackingLock:
        def __enter__(self):
            lock_acquired["count"] += 1
            return orig_lock.__enter__()

        def __exit__(self, *args):
            return orig_lock.__exit__(*args)

    scan_store.lock = _TrackingLock()

    facts = _make_sec_facts()
    ds = DataService(None, "CompounderOS/1.0 (test@example.com)", lambda *a, **kw: "")
    _, _, _, _, cf_q, cf_a = ds.build_sec_dataframes(facts)

    captured_text = {}

    class _ValHarness(ValuationTabMixin):
        def __init__(self):
            self.store = store
            self.scan_store = scan_store
            self.df_cf_q = cf_q
            self.df_cf_a = cf_a

        def ui(self, fn, text):
            captured_text["text"] = text

    vh = _ValHarness()
    vh._proc_historical_valuation("AAPL")

    assert lock_acquired["count"] >= 1
    assert "Current FCF Yield:" in captured_text["text"]
    assert "Historical Percentile:" in captured_text["text"]


def test_scoped_ssl_context_defaults_to_verified(monkeypatch):
    import compounderos_app.gui.app as app_mod

    monkeypatch.delenv("COMPOUNDEROS_ALLOW_UNVERIFIED_SSL", raising=False)
    assert hasattr(app_mod, "_build_https_context"), "Expected _build_https_context helper in compounderos_app.gui.app"
    ctx = app_mod._build_https_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED

    monkeypatch.setenv("COMPOUNDEROS_ALLOW_UNVERIFIED_SSL", "1")
    ctx_unv = app_mod._build_https_context()
    assert ctx_unv.verify_mode == ssl.CERT_NONE
