import asyncio
import os
import tempfile
import threading
import pandas as pd

import sp500_scanner
from compounderos_app.database import DataStore
from compounderos_app.scoring_engine import ENGINE_VERSION
from scanner_service.api import create_app
from scanner_service.pipeline import ScannerPipeline
from scanner_service.yf_gateway import TokenBucket, YFGateway


def _asgi_request(app, method: str, path: str, headers=None, client=("172.30.32.2", 50000)):
    headers_list = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": headers_list,
        "client": client,
        "server": ("127.0.0.1", 8321),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        messages.append(msg)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    resp_headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in start.get("headers", [])}
    return start["status"], resp_headers, body


def test_ingress_static_and_ui_routes_under_starlette():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = DataStore(db_path, role="sor", check_same_thread=False)
        app = create_app(store)

        # 1. Direct request without X-Ingress-Path
        status, _, body = _asgi_request(app, "GET", "/ui/static/app.css")
        assert status == 200
        assert b"background" in body

        # 2. Request under Home Assistant Ingress X-Ingress-Path (Starlette >= 0.33 Mount regression)
        ingress_headers = {"X-Ingress-Path": "/api/hassio_ingress/tok_abc123"}
        status_ing, _, body_ing = _asgi_request(
            app, "GET", "/ui/static/app.css", headers=ingress_headers
        )
        assert status_ing == 200
        assert b"background" in body_ing

        # 3. UI Ranking and System pages under Ingress from Supervisor IP
        status_rank, _, body_rank = _asgi_request(
            app, "GET", "/ui/ranking", headers=ingress_headers, client=("172.30.32.2", 50000)
        )
        assert status_rank == 200
        assert b"S&P 500 Quality Ranking" in body_rank

        status_sys, _, body_sys = _asgi_request(
            app, "GET", "/ui/system", headers=ingress_headers, client=("172.30.32.2", 50000)
        )
        assert status_sys == 200
        assert b"System Status" in body_sys
        assert ENGINE_VERSION.encode("utf-8") in body_sys

        # 4. Non-Supervisor IP rejected with 403 on UI routes
        status_forbid, _, _ = _asgi_request(
            app, "GET", "/ui/ranking", headers=ingress_headers, client=("10.0.0.99", 50000)
        )
        assert status_forbid == 403

        # 5. Root relative redirect and health endpoint
        status_root, hdrs_root, _ = _asgi_request(app, "GET", "/", headers=ingress_headers)
        assert status_root == 307
        assert hdrs_root.get("location") == "ui/ranking"

        status_health, _, body_health = _asgi_request(app, "GET", "/v1/health")
        assert status_health == 200
        assert b'"ok"' in body_health

        store.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_scanner_pipeline_pick_next_symbol_no_historical_starvation_and_watchlist():
    store = DataStore(":memory:", role="sor", check_same_thread=False)
    stop_event = threading.Event()
    pipeline = ScannerPipeline(store, {}, stop_event)
    pipeline.universe = [("AAPL", "Information Technology", "Technology Hardware, Storage & Peripherals")]

    # Insert an already-evaluated snapshot for AAPL
    with store.lock:
        store.conn.execute(
            "INSERT INTO sp500_scores (symbol, sector, score, composite_shrunk, verdict, model, evaluated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", "Information Technology", 8.0, 8.0, "SCORED", ENGINE_VERSION, "2026-01-01T00:00:00+00:00"),
        )
        # Insert a delisted historical symbol LEHM in sp500_scores_history only (not in active universe)
        store.conn.execute(
            """INSERT INTO sp500_scores_history
               (symbol, scan_date, evaluated_at, sector, score, confidence, gate_state, engine_version, source, raw_json, subscores_json, registry_hash)
               VALUES ('LEHM', '2020-01-01', '2020-01-01T00:00:00+00:00', 'Financials', 3.0, 0.8, 'pass', 'det-1.1.0', 'scan', '{}', '{}', '')"""
        )
        # Insert a non-S&P-500 watchlist symbol SHOP in __mac_push__
        store.conn.execute(
            "INSERT INTO watchlists (name, symbol) VALUES ('__mac_push__', 'SHOP')"
        )
        store.conn.commit()

    # 1. Watchlist symbol SHOP should be prioritized first and marked active
    picked1 = pipeline._pick_next_symbol()
    assert picked1 is not None
    assert picked1[0] == "SHOP"
    assert picked1[3] is True

    # 2. If SHOP is in-flight, AAPL should be picked next — delisted LEHM must never be picked for fundamental scoring
    pipeline.in_flight.add("SHOP")
    picked2 = pipeline._pick_next_symbol()
    assert picked2 is not None
    assert picked2[0] == "AAPL"
    assert picked2[3] is True

    pipeline.in_flight.add("AAPL")
    picked3 = pipeline._pick_next_symbol()
    assert picked3 is None


from unittest.mock import patch


def test_scanner_pipeline_worker_records_failure_when_evaluate_returns_false():
    store = DataStore(":memory:", role="sor", check_same_thread=False)
    stop_event = threading.Event()
    pipeline = ScannerPipeline(store, {}, stop_event)

    def fake_evaluate(store_arg, gateway_arg, symbol, sector, sub_ind, is_active):
        stop_event.set()
        return False

    with patch.object(sp500_scanner, "evaluate_symbol", fake_evaluate):
        pipeline.in_flight.add("BADTICK")
        pipeline.task_queue.put(("BADTICK", "Information Technology", "Systems Software", True))
        pipeline._worker_loop()

    assert "BADTICK" in pipeline.failures
    assert "BADTICK" not in pipeline.in_flight


def test_fetch_sp500_universe_uses_token_bucket_and_db_fallback():
    store = DataStore(":memory:", role="sor", check_same_thread=False)
    store.upsert_ticker_classification(
        "TSLA", "Tesla Inc.", "Consumer Discretionary", "Automobile Manufacturers", "wikipedia_gics"
    )

    acquired = {"count": 0}
    bucket = TokenBucket(rate=100.0, burst=5)
    orig_acquire = bucket.acquire

    def spy_acquire(tokens=1.0):
        acquired["count"] += 1
        return orig_acquire(tokens)

    with patch.object(bucket, "acquire", spy_acquire), patch.object(
        sp500_scanner, "http_get_text", lambda *a, **kw: ""
    ):
        universe = sp500_scanner.fetch_sp500_universe(store, bucket=bucket)

    assert acquired["count"] == 1
    assert ("TSLA", "Consumer Discretionary", "Automobile Manufacturers") in universe


def test_evaluate_symbol_standalone_routes_through_yf_gateway():
    store = DataStore(":memory:", role="sor", check_same_thread=False)
    gateway_called = {"count": 0}

    def fake_fetch_fundamentals(self, symbol):
        gateway_called["count"] += 1
        dates = pd.date_range("2024-01-01", periods=4, freq="QE")
        inc = pd.DataFrame(
            {
                dates[0]: [100.0, 60.0, 25.0, 20.0],
                dates[1]: [105.0, 63.0, 27.0, 21.0],
                dates[2]: [110.0, 66.0, 29.0, 22.0],
                dates[3]: [120.0, 72.0, 32.0, 25.0],
            },
            index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
        )
        bal = pd.DataFrame(
            {
                dates[0]: [200.0, 80.0, 40.0, 50.0, 20.0, 100.0, 10.0],
                dates[1]: [210.0, 82.0, 40.0, 52.0, 20.0, 105.0, 10.0],
                dates[2]: [220.0, 85.0, 40.0, 55.0, 20.0, 110.0, 10.0],
                dates[3]: [230.0, 90.0, 40.0, 60.0, 20.0, 115.0, 10.0],
            },
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
            {
                dates[0]: [30.0, -5.0, 2.0],
                dates[1]: [32.0, -5.0, 2.0],
                dates[2]: [34.0, -6.0, 2.0],
                dates[3]: [36.0, -6.0, 2.0],
            },
            index=["Operating Cash Flow", "Capital Expenditure", "Stock Based Compensation"],
        )
        hist = pd.DataFrame(
            {"Close": [180.0, 185.0], "Volume": [1000, 1200]},
            index=pd.date_range("2026-01-01", periods=2, freq="D"),
        )
        return {
            "info": {
                "currentPrice": 185.0,
                "marketCap": 2_800_000_000_000,
                "trailingPE": 28.0,
                "forwardPE": 24.0,
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

    with patch.object(YFGateway, "fetch_fundamentals", fake_fetch_fundamentals):
        ok = sp500_scanner.evaluate_symbol(
            store,
            None,
            "AAPL",
            "Information Technology",
            "Technology Hardware, Storage & Peripherals",
            True,
        )

    assert ok is True
    assert gateway_called["count"] == 1

