import json
import os
import tempfile
import threading

from fastapi.testclient import TestClient

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.gui.services.data_service import DataService
from compounderos_app.gui.tabs.due_diligence import synthesize_bull_bear_views
from compounderos_app.scoring_engine import CANONICAL_REGISTRY_ID, score_metrics
from scanner_service.api import create_app
from scanner_service.pipeline import ScannerPipeline


def _sample_sec_submissions() -> dict:
    return {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-25-000106",
                    "0000320193-25-000079",
                    "0000320193-25-000075",
                    "0000320193-25-000050",
                    "0000320193-25-000012",
                ],
                "form": [
                    "10-K",
                    "10-Q",
                    "8-K",
                    "4",
                    "S-8",
                ],
                "filingDate": [
                    "2025-10-31",
                    "2025-08-01",
                    "2025-07-31",
                    "2025-05-15",
                    "2025-02-10",
                ],
                "reportDate": [
                    "2025-09-27",
                    "2025-06-28",
                    "2025-07-31",
                    "2025-05-13",
                    "",
                ],
                "primaryDocument": [
                    "aapl-20250927.htm",
                    "aapl-20250628.htm",
                    "aapl-20250731.htm",
                    "xslF345X05/wk-form4_174734.xml",
                    "s8.htm",
                ],
                "items": [
                    [],
                    [],
                    ["2.02", "9.01"],
                    [],
                    [],
                ],
            }
        },
    }


def test_extract_recent_sec_filings_and_sqlite_roundtrip(tmp_path):
    sub = _sample_sec_submissions()
    filings = DataService.extract_recent_sec_filings(sub, "0000320193", limit=25)

    # S-8 must be filtered out; 10-K, 10-Q, 8-K, 4 must be retained in descending filing_date order
    assert [f["form"] for f in filings] == ["10-K", "10-Q", "8-K", "4"]
    assert filings[0]["accession_number"] == "0000320193-25-000106"
    assert filings[0]["edgar_url"] == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000106/aapl-20250927.htm"
    assert filings[2]["items"] == "2.02, 9.01"
    assert filings[3]["edgar_url"] == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000050/xslF345X05/wk-form4_174734.xml"

    store = DataStore(str(tmp_path / "sec_filings.db"))
    try:
        inserted = store.upsert_sec_filings("aapl", filings)
        assert inserted == 4

        stored_all = store.get_sec_filings("AAPL", limit=10)
        assert len(stored_all) == 4
        assert stored_all[0]["form"] == "10-K"
        assert stored_all[0]["edgar_url"] == filings[0]["edgar_url"]

        only_10k_8k = store.get_sec_filings("AAPL", forms=["10-K", "8-K"], limit=10)
        assert [r["form"] for r in only_10k_8k] == ["10-K", "8-K"]
    finally:
        store.close()


def test_synthesize_bull_bear_views_high_quality_vs_gated():
    elite_raw = {
        "gross_margin": 0.72,
        "revenue_growth": 0.18,
        "rev_growth_3y": 0.18,
        "operating_leverage": 0.12,
        "roic": 0.34,
        "roic_momentum": 0.04,
        "net_debt_ebitda": 0.3,
        "current_ratio": 2.4,
        "gaap_gap": 0.04,
        "capex_revenue": 0.04,
        "earnings_quality": 1.15,
        "share_dilution": -0.02,
        "sbc_revenue": 0.02,
        "fcf_yield": 0.052,
    }
    elite_score = score_metrics(elite_raw)
    elite_traj = {
        "trajectory": 62.0,
        "self_revision": 0.085,
        "peer_median_revision": 0.025,
    }
    elite_views = synthesize_bull_bear_views(
        symbol="NVDA",
        score_res=elite_score,
        raw_metrics=elite_raw,
        traj_row=elite_traj,
        implied_growth=0.11,
        info={"sector": "Information Technology"},
    )
    assert len(elite_views["bullish"]) >= 4
    assert len(elite_views["bearish"]) >= 2
    bull_joined = " ".join(elite_views["bullish"])
    assert "roic" in bull_joined
    assert "/10" in bull_joined
    assert "trajectory=62.0" in bull_joined
    assert "implied_growth (11.0%)" in bull_joined
    assert "fcf_yield=5.20%" in bull_joined
    assert "gate_state='pass'" in bull_joined

    gated_raw = {
        "gross_margin": 0.18,
        "revenue_growth": 0.02,
        "rev_growth_3y": 0.03,
        "operating_leverage": -0.05,
        "roic": 0.04,
        "roic_momentum": -0.03,
        "net_debt_ebitda": 5.5,
        "current_ratio": 0.85,
        "gaap_gap": 0.45,
        "capex_revenue": 0.18,
        "earnings_quality": 0.55,
        "share_dilution": 0.05,
        "sbc_revenue": 0.12,
        "fcf_yield": 0.012,
    }
    gated_score = score_metrics(gated_raw)
    gated_traj = {
        "trajectory": 35.0,
        "self_revision": -0.06,
        "peer_median_revision": 0.01,
    }
    gated_views = synthesize_bull_bear_views(
        symbol="BA",
        score_res=gated_score,
        raw_metrics=gated_raw,
        traj_row=gated_traj,
        implied_growth=0.09,
        info={"sector": "Industrials"},
    )
    assert len(gated_views["bullish"]) >= 2
    assert len(gated_views["bearish"]) >= 4
    bear_joined = " ".join(gated_views["bearish"])
    assert "net_debt_ebitda" in bear_joined
    assert "trajectory=35.0" in bear_joined
    assert "self_revision=-6.0%" in bear_joined
    assert "implied_growth (9.0%) exceeds historical rev_growth_3y (3.0%)" in bear_joined
    assert "fcf_yield=1.20%" in bear_joined


def test_web_symbol_detail_renders_bull_bear_and_sec_filings(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    store = DataStore(str(tmp_path / "web_sec.db"), role="sor")
    pipeline = ScannerPipeline(store, {"api_token": "tok", "scan_workers": 1}, threading.Event())

    raw = {
        "gross_margin": 0.62,
        "revenue_growth": 0.14,
        "operating_leverage": 0.06,
        "roic": 0.28,
        "roic_momentum": 0.02,
        "net_debt_ebitda": 0.8,
        "current_ratio": 1.9,
        "gaap_gap": 0.05,
        "capex_revenue": 0.05,
        "earnings_quality": 1.10,
        "share_dilution": -0.01,
        "sbc_revenue": 0.03,
    }
    res = score_metrics(raw)
    subscores = {k: v.subscore for k, v in res.metrics.items()}

    with store.lock:
        store.conn.execute(
            """
            INSERT INTO sp500_scores_history (
                symbol, scan_date, evaluated_at, sector, score, score_pre_gate, composite_shrunk,
                b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                confidence, gate_state, gates_fired, gates_unverified, gate_cap_applied,
                verdict, pe_trailing, pe_forward, fcf_yield, market_cap,
                engine_version, source, raw_json, subscores_json, registry_id, registry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAPL",
                "2026-01-20",
                "2026-01-20T12:00:00+00:00",
                "Information Technology",
                res.composite,
                res.composite_pre_gate,
                res.composite_shrunk,
                res.buckets.get("profitability"),
                res.buckets.get("efficiency"),
                res.buckets.get("balance_sheet"),
                res.buckets.get("capital_alloc"),
                res.confidence,
                "pass",
                "",
                "",
                None,
                "SCORED",
                28.0,
                24.5,
                0.046,
                3.3e12,
                "det-1.1.0",
                "scan",
                json.dumps(raw),
                json.dumps(subscores),
                CANONICAL_REGISTRY_ID,
                res.registry_hash,
            ),
        )
        store.conn.commit()

    filings = DataService.extract_recent_sec_filings(_sample_sec_submissions(), "0000320193")
    store.upsert_sec_filings("AAPL", filings)

    app = create_app(store, pipeline)
    client = TestClient(app)
    try:
        r = client.get("/ui/symbol/AAPL")
        assert r.status_code == 200
        assert "Bullish Catalysts" in r.text
        assert "Bearish Headwinds" in r.text
        assert "Official SEC Filings" in r.text
        assert "https://www.sec.gov/Archives/edgar/data/320193/000032019325000106/aapl-20250927.htm" in r.text
        assert "0000320193-25-000106" in r.text
    finally:
        store.close()
