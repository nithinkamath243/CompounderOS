import os
import tempfile
from unittest.mock import patch
import pandas as pd

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

import sp500_scanner
from compounderos_app.database import DataStore
from scanner_service.yf_gateway import TokenBucket, YFGateway


def test_token_bucket_never_deadlocks_when_cost_exceeds_burst():
    # Reproduces the exact Home Assistant config.yaml bug: burst=5 vs fetch_fundamentals cost=8
    bucket = TokenBucket(rate=40.0, burst=5)
    ok = bucket.acquire(cost=8, timeout=0.5)
    assert ok is True
    assert bucket.requests_last_hour == 8


def test_evaluate_symbol_falls_back_to_history_close_when_info_omits_price(tmp_path):
    store = DataStore(str(tmp_path / "scanner_price_fb.db"), role="sor", check_same_thread=False)
    try:
        dates = pd.date_range("2025-03-31", periods=4, freq="QE")
        inc = pd.DataFrame(
            {d: [120.0, 72.0, 34.0, 26.0] for d in dates},
            index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
        )
        bal = pd.DataFrame(
            {d: [250.0, 100.0, 50.0, 60.0, 20.0, 130.0, 10.0] for d in dates},
            index=[
                "Total Assets",
                "Current Assets",
                "Current Liabilities",
                "Cash And Cash Equivalents",
                "Total Debt",
                "Stockholders Equity",
                "Ordinary Shares Number",
            ],
        )
        cf = pd.DataFrame(
            {d: [36.0, -6.0, 2.0] for d in dates},
            index=["Operating Cash Flow", "Capital Expenditure", "Stock Based Compensation"],
        )
        hist = pd.DataFrame(
            {"Close": [210.0, 215.5], "Volume": [1000, 1200]},
            index=pd.date_range("2026-03-01", periods=2, freq="D"),
        )

        def fake_fetch(_self, _sym):
            return {
                "info": {
                    # Deliberately omit currentPrice and regularMarketPrice
                    "marketCap": 3_200_000_000_000,
                    "trailingPE": 30.0,
                    "sector": "Information Technology",
                    "subIndustry": "Technology Hardware, Storage & Peripherals",
                },
                "history": hist,
                "income_q": inc,
                "income_a": inc,
                "balance_q": bal,
                "balance_a": bal,
                "cashflow_q": cf,
                "cashflow_a": cf,
            }

        bucket = TokenBucket(rate=50.0, burst=5)
        gw = YFGateway(bucket)
        with patch.object(YFGateway, "fetch_fundamentals", fake_fetch):
            ok = sp500_scanner.evaluate_symbol(
                store,
                gw,
                "AAPL",
                "Information Technology",
                "Technology Hardware, Storage & Peripherals",
                True,
            )
        assert ok is True
        cnt = store.conn.execute("SELECT COUNT(*) FROM sp500_scores_history WHERE symbol='AAPL'").fetchone()[0]
        assert cnt == 1
    finally:
        store.close()


def test_headless_scanner_imports_without_tkinter():
    """Verify scanner_service.__main__ and web views import cleanly in slim containers lacking libtk8.6.so."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "sys.modules['tkinter'] = None; "
        "sys.modules['_tkinter'] = None; "
        "import scanner_service.api; "
        "import scanner_service.web.views; "
        "import compounderos_app.research_copilot; "
        "print('HEADLESS_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "HEADLESS_OK" in proc.stdout
