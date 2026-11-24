import json
import os
import sqlite3
import tempfile
import threading
import numpy as np
import pandas as pd

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.gui.app import CompounderOSApp
from compounderos_app.gui.services.data_service import DataService
from compounderos_app.gui.tabs.transcripts import TranscriptsTabMixin
from compounderos_app.gui.tabs.watchlist import WatchlistTabMixin


def _build_minimal_sec_facts() -> dict:
    q_dates = pd.date_range("2023-03-31", periods=12, freq="QE")
    a_dates = pd.date_range("2023-12-31", periods=3, freq="YE")

    def _q_items(vals):
        return [
            {
                "start": (d - pd.Timedelta(days=89)).strftime("%Y-%m-%d"),
                "end": d.strftime("%Y-%m-%d"),
                "val": float(v * 1e6),
                "fp": f"Q{d.quarter}",
                "form": "10-Q",
            }
            for d, v in zip(q_dates, vals)
        ]

    def _a_items(vals):
        return [
            {
                "start": (d - pd.Timedelta(days=364)).strftime("%Y-%m-%d"),
                "end": d.strftime("%Y-%m-%d"),
                "val": float(v * 1e6),
                "fp": "FY",
                "form": "10-K",
            }
            for d, v in zip(a_dates, vals)
        ]

    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": _q_items(np.linspace(100, 160, 12)) + _a_items([420, 500, 600])}},
                "GrossProfit": {"units": {"USD": _q_items(np.linspace(60, 96, 12)) + _a_items([250, 300, 360])}},
                "OperatingIncomeLoss": {"units": {"USD": _q_items(np.linspace(25, 45, 12)) + _a_items([105, 130, 165])}},
                "NetIncomeLoss": {"units": {"USD": _q_items(np.linspace(20, 36, 12)) + _a_items([84, 104, 132])}},
                "IncomeTaxExpenseBenefit": {"units": {"USD": _q_items(np.linspace(5, 9, 12)) + _a_items([21, 26, 33])}},
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": {
                    "units": {"USD": _q_items(np.linspace(25, 45, 12)) + _a_items([105, 130, 165])}
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": _q_items(np.linspace(105, 100, 12)) + _a_items([104, 102, 100])}
                },
                "DepreciationDepletionAndAmortization": {
                    "units": {"USD": _q_items(np.linspace(4, 6, 12)) + _a_items([16, 20, 22])}
                },
                "Assets": {"units": {"USD": _q_items(np.linspace(400, 520, 12)) + _a_items([430, 480, 520])}},
                "AssetsCurrent": {"units": {"USD": _q_items(np.linspace(150, 210, 12)) + _a_items([165, 190, 210])}},
                "LiabilitiesCurrent": {"units": {"USD": _q_items(np.linspace(80, 100, 12)) + _a_items([85, 92, 100])}},
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {"USD": _q_items(np.linspace(60, 90, 12)) + _a_items([65, 78, 90])}
                },
                "LongTermDebt": {"units": {"USD": _q_items(np.linspace(50, 40, 12)) + _a_items([48, 44, 40])}},
                "StockholdersEquity": {"units": {"USD": _q_items(np.linspace(250, 360, 12)) + _a_items([270, 315, 360])}},
                "CommonStockSharesOutstanding": {
                    "units": {"shares": _q_items(np.linspace(105, 100, 12)) + _a_items([104, 102, 100])}
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": _q_items(np.linspace(28, 48, 12)) + _a_items([115, 145, 180])}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": _q_items(np.linspace(5, 8, 12)) + _a_items([20, 25, 30])}
                },
                "ShareBasedCompensation": {
                    "units": {"USD": _q_items(np.linspace(2, 3, 12)) + _a_items([8, 10, 12])}
                },
            }
        }
    }


def test_legacy_database_schema_migrates_ticker_column_and_truncated_window(tmp_path):
    db_file = tmp_path / "legacy_compounder.db"
    raw_conn = sqlite3.connect(str(db_file))
    raw_conn.execute(
        """
        CREATE TABLE ticker_map (
            ticker TEXT PRIMARY KEY,
            cik TEXT NOT NULL,
            name TEXT,
            updated_at TEXT
        )
        """
    )
    raw_conn.execute(
        "INSERT INTO ticker_map (ticker, cik, name, updated_at) VALUES ('AAPL', '0000320193', 'Apple Inc.', '2026-01-01T00:00:00Z')"
    )
    raw_conn.execute(
        """
        CREATE TABLE price_history (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT NOT NULL,
            PRIMARY KEY (symbol, date, source)
        )
        """
    )
    raw_conn.commit()
    raw_conn.close()

    store = DataStore(str(db_file))
    try:
        row = store.get_ticker_map("AAPL")
        assert row is not None
        assert row[0] == "0000320193"
        assert row[1] == "Apple Inc."

        assert store.resolve_symbol_from_db("apple") == "AAPL"
        assert store.resolve_symbol_from_db("Apple Inc.") == "AAPL"

        df_prices = pd.DataFrame(
            {"Open": [210.0], "High": [215.0], "Low": [209.0], "Close": [214.0], "Volume": [1000000.0]},
            index=pd.to_datetime(["2026-03-01"]),
        )
        store.upsert_price_history("AAPL", df_prices, source="test")
        store.flag_price_history_truncated("AAPL")
        saved = store.conn.execute(
            "SELECT close, truncated_window FROM price_history WHERE symbol = 'AAPL'"
        ).fetchone()
        assert saved is not None
        assert float(saved[0]) == 214.0
        assert int(saved[1]) == 1
    finally:
        store.close()


def test_resolve_symbol_company_names_aliases_and_sec_json(tmp_path):
    store = DataStore(str(tmp_path / "resolve.db"))
    sec_payload = json.dumps(
        {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
            "2": {"cik_str": 1800, "ticker": "ABT", "title": "ABBOTT LABORATORIES"},
        }
    )

    def fake_http(url, **_kw):
        if "company_tickers.json" in url:
            return sec_payload
        return ""

    try:
        ds = DataService(store, "CompounderOS/1.0 (test@example.com)", fake_http)
        assert ds.resolve_symbol("apple") == "AAPL"
        assert ds.resolve_symbol("Apple Inc.") == "AAPL"
        assert ds.resolve_symbol("brk.b") == "BRK-B"
        assert ds.resolve_symbol("abbott laboratories") == "ABT"
        assert ds.get_cik("AAPL") == "0000320193"
    finally:
        store.close()


def test_build_sec_dataframes_handles_empty_or_non_us_gaap_facts():
    ds = DataService(None, "CompounderOS/1.0 (test@example.com)", lambda *a, **kw: "")
    fin_q, fin_a, bal_q, bal_a, cf_q, cf_a = ds.build_sec_dataframes({"facts": {"ifrs-full": {}}})
    for df in (fin_q, fin_a, bal_q, bal_a, cf_q, cf_a):
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert isinstance(df.index, pd.DatetimeIndex)


def test_search_worker_resolves_apple_and_survives_yfinance_failure(tmp_path, monkeypatch):
    import compounderos_app.gui.app as app_mod

    store = DataStore(str(tmp_path / "worker.db"))
    store.upsert_ticker_map("AAPL", "0000320193", "Apple Inc.", "Information Technology", "Consumer Electronics")
    df_prices = pd.DataFrame(
        {"Open": [220.0, 224.0], "High": [222.0, 226.0], "Low": [219.0, 223.0], "Close": [221.0, 225.5], "Volume": [1e6, 1.2e6]},
        index=pd.to_datetime(["2026-03-01", "2026-03-02"]),
    )
    store.upsert_price_history("AAPL", df_prices, source="yfinance")
    facts = _build_minimal_sec_facts()

    def fake_http(url, **_kw):
        if "companyfacts/CIK0000320193.json" in url:
            return json.dumps(facts)
        if "submissions/CIK0000320193.json" in url:
            return json.dumps({"cik": "320193", "filings": {"recent": {}}})
        return ""

    class _BrokenYFTicker:
        def __init__(self, sym):
            self.symbol = sym

        @property
        def info(self):
            raise RuntimeError("429 Too Many Requests")

        @property
        def dividends(self):
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(app_mod.yf, "Ticker", _BrokenYFTicker)

    class _StatusStub:
        def __init__(self):
            self.text = ""

        def configure(self, **kw):
            if "text" in kw:
                self.text = kw["text"]

    class _VarStub:
        def __init__(self, val=""):
            self.val = val

        def get(self):
            return self.val

        def set(self, val):
            self.val = val

    class _WorkerHarness:
        def __init__(self):
            self.store = store
            self.scan_store = None
            self.data_service = DataService(store, "CompounderOS/1.0 (test@example.com)", fake_http)
            self.cache_lock = threading.Lock()
            self.cache = {}
            self.request_id = 1
            self.current_symbol = "APPLE"
            self.search_var = _VarStub("apple")
            self.lbl_status = _StatusStub()
            self.applied = None

        _enrich_info_from_local_db = CompounderOSApp._enrich_info_from_local_db
        _safe_yf_df = staticmethod(CompounderOSApp._safe_yf_df)

        def ui(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

        def _apply_search_results(self, symbol, info, fin_q, bal_q, cf_q, divs, req_id, *extra):
            self.applied = (symbol, info, fin_q, bal_q, cf_q, divs, req_id, extra)
            self.lbl_status.configure(text=f"Loaded {symbol}")

    try:
        harness = _WorkerHarness()
        CompounderOSApp._search_worker(harness, "apple", 1)

        assert harness.current_symbol == "AAPL"
        assert harness.search_var.get() == "AAPL"
        assert harness.applied is not None
        resolved_sym, info, fin_q, _bal_q, _cf_q, _divs, _req_id, _extra = harness.applied
        assert resolved_sym == "AAPL"
        assert info.get("longName") == "Apple Inc."
        assert info.get("currentPrice") == 225.5
        assert not fin_q.empty
        assert harness.lbl_status.text == "Loaded AAPL"
    finally:
        store.close()


def test_watchlist_and_transcripts_delegate_to_data_service(tmp_path):
    store = DataStore(str(tmp_path / "wl_tr.db"))
    store.upsert_ticker_map("AAPL", "0000320193", "Apple Inc.")
    facts = _build_minimal_sec_facts()

    def fake_http(url, **_kw):
        if "companyfacts/CIK0000320193.json" in url:
            return json.dumps(facts)
        if "submissions/CIK0000320193.json" in url:
            return json.dumps({"cik": "320193", "filings": {"recent": {"form": [], "accessionNumber": [], "filingDate": []}}})
        return ""

    class _RootStub:
        def after(self, _ms, fn):
            fn()

    class _TabHarness(WatchlistTabMixin, TranscriptsTabMixin):
        def __init__(self):
            self.store = store
            self.scan_store = None
            self.data_service = DataService(store, "CompounderOS/1.0 (test@example.com)", fake_http)
            self.sec_user_agent = "CompounderOS/1.0 (test@example.com)"
            self.http_get_text = fake_http
            self.transcript_fetch_cache = {}
            self.transcript_fetch_status = ""
            self.request_id = 1
            self.root = _RootStub()
            self.done_sources = []

        def _mark_transcript_fetch_done(self, src):
            self.done_sources.append(src)

    try:
        h = _TabHarness()
        score = h._compute_symbol_quality("AAPL")
        assert score is not None
        assert 0.0 <= score <= 10.0

        h.fetch_sec_earnings_release("AAPL", 1, force=True)
        assert "SEC" in h.done_sources
        assert "No recent 8-K" in h.transcript_fetch_status
    finally:
        store.close()
