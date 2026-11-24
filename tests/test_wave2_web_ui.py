import json
import os
import tempfile
import threading

from fastapi.testclient import TestClient

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.scoring_engine import CANONICAL_METRIC_SPECS, CANONICAL_REGISTRY_ID, score_metrics
from scanner_service.api import create_app
from scanner_service.pipeline import ScannerPipeline


def _build_seeded_web_client(tmp_path):
    db_path = str(tmp_path / "web_ui.db")
    store = DataStore(db_path, role="sor")
    cfg = {"api_token": "secret-token", "scan_workers": 1}
    pipeline = ScannerPipeline(store, cfg, threading.Event())

    aapl_raw = {
        "roic": 0.28,
        "operating_margin": 0.30,
        "fcf_margin": 0.25,
        "rev_cagr_3y": 0.12,
        "eps_cagr_3y": 0.15,
        "fcf_conversion": 0.96,
        "net_debt_to_fcf": 0.6,
        "interest_coverage": 28.0,
        "share_dilution": -0.02,
        "gaap_gap": 0.92,
        "reinvestment_rate": 0.25,
        "incremental_roic": 0.26,
    }
    aapl_res = score_metrics(aapl_raw)
    aapl_subscores = {k: v.subscore for k, v in aapl_res.metrics.items()}

    ba_raw = {
        "roic": 0.04,
        "operating_margin": 0.05,
        "fcf_margin": 0.02,
        "rev_cagr_3y": 0.02,
        "eps_cagr_3y": 0.01,
        "fcf_conversion": 0.50,
        "net_debt_to_fcf": 6.5,
        "interest_coverage": 1.2,
        "share_dilution": 0.04,
        "gaap_gap": 0.55,
        "reinvestment_rate": 0.10,
        "incremental_roic": 0.04,
    }
    ba_res = score_metrics(ba_raw)
    ba_subscores = {k: v.subscore for k, v in ba_res.metrics.items()}

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
                "2026-01-15",
                "2026-01-15T12:00:00+00:00",
                "Information Technology",
                aapl_res.composite,
                aapl_res.composite_pre_gate,
                aapl_res.composite_shrunk,
                aapl_res.buckets.get("profitability"),
                aapl_res.buckets.get("efficiency"),
                aapl_res.buckets.get("balance_sheet"),
                aapl_res.buckets.get("capital_alloc"),
                aapl_res.confidence,
                "pass",
                "",
                "",
                None,
                "SCORED",
                28.5,
                25.0,
                0.038,
                3.2e12,
                "det-1.1.0",
                "scan",
                json.dumps(aapl_raw),
                json.dumps(aapl_subscores),
                CANONICAL_REGISTRY_ID,
                aapl_res.registry_hash,
            ),
        )
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
                "BA",
                "2026-01-15",
                "2026-01-15T12:00:00+00:00",
                "Industrials",
                ba_res.composite,
                ba_res.composite_pre_gate,
                ba_res.composite_shrunk,
                ba_res.buckets.get("profitability"),
                ba_res.buckets.get("efficiency"),
                ba_res.buckets.get("balance_sheet"),
                ba_res.buckets.get("capital_alloc"),
                ba_res.confidence,
                "fired",
                "leverage_gate,coverage_gate",
                "",
                5.0,
                "SCORED",
                45.0,
                32.0,
                0.015,
                1.1e11,
                "det-1.1.0",
                "scan",
                json.dumps(ba_raw),
                json.dumps(ba_subscores),
                CANONICAL_REGISTRY_ID,
                ba_res.registry_hash,
            ),
        )
        store.conn.execute(
            """
            INSERT INTO sp500_scores (
                symbol, sector, score, composite_shrunk, verdict, model,
                evaluated_at, registry_id, registry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAPL",
                "Information Technology",
                aapl_res.composite,
                aapl_res.composite_shrunk,
                "SCORED",
                "det-1.1.0",
                "2026-01-15T12:00:00+00:00",
                CANONICAL_REGISTRY_ID,
                aapl_res.registry_hash,
            ),
        )
        store.conn.execute(
            """
            INSERT INTO sp500_scores (
                symbol, sector, score, composite_shrunk, verdict, model,
                evaluated_at, registry_id, registry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BA",
                "Industrials",
                ba_res.composite,
                ba_res.composite_shrunk,
                "SCORED",
                "det-1.1.0",
                "2026-01-15T12:00:00+00:00",
                CANONICAL_REGISTRY_ID,
                ba_res.registry_hash,
            ),
        )
        store.conn.commit()

    store.record_trajectory_result(
        symbol="AAPL",
        trajectory=6.8,
        confidence=0.92,
        composition="AGB",
        granularity="sub_industry",
        peer_set=["DELL", "HPQ", "IBM", "NTAP", "STX"],
        raw_inputs={"eps_cur_0y": 6.2, "eps_90d_0y": 5.8, "rev_growth_1y": 0.12},
        engine_version="traj-1.1.0",
        peer_state="OK",
        derived_json={
            "a_score": 7.1,
            "g_score": 6.5,
            "b_score": 6.6,
            "self_revision": 0.068,
            "peer_median_revision": 0.032,
            "peer_breadth": 0.80,
            "median_peer_growth": 0.10,
            "peer_n": 6,
            "peer_pos_n": 5,
        },
        extraction_version="trajx-1.0.0",
    )

    store.record_or_extend_alert(
        symbol="AAPL",
        kind="dip_trigger",
        instance="2026-01-15",
        registry_tag="canonical",
        payload={"fcf_yield": 0.038, "percentile": 85.0},
    )

    app = create_app(store, pipeline)
    return TestClient(app), store


def test_ui_ranking_filters_and_enriched_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    client, store = _build_seeded_web_client(tmp_path)
    try:
        r_all = client.get("/ui/ranking")
        assert r_all.status_code == 200
        assert "AAPL" in r_all.text
        assert "BA" in r_all.text
        assert "./symbol/AAPL" in r_all.text or "/ui/symbol/AAPL" in r_all.text
        assert "6.8" in r_all.text

        r_q = client.get("/ui/ranking?q=AAP")
        assert r_q.status_code == 200
        assert "./symbol/AAPL" in r_q.text or "/ui/symbol/AAPL" in r_q.text
        assert "./symbol/BA" not in r_q.text and "/ui/symbol/BA" not in r_q.text

        r_sec = client.get("/ui/ranking?sector=Industrials")
        assert r_sec.status_code == 200
        assert "./symbol/BA" in r_sec.text or "/ui/symbol/BA" in r_sec.text
        assert "./symbol/AAPL" not in r_sec.text and "/ui/symbol/AAPL" not in r_sec.text

        r_ungated = client.get("/ui/ranking?hide_gated=true")
        assert r_ungated.status_code == 200
        assert "./symbol/AAPL" in r_ungated.text or "/ui/symbol/AAPL" in r_ungated.text
        assert "./symbol/BA" not in r_ungated.text and "/ui/symbol/BA" not in r_ungated.text
    finally:
        store.close()


def test_ui_symbol_drilldown_and_404(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    client, store = _build_seeded_web_client(tmp_path)
    try:
        r = client.get("/ui/symbol/AAPL")
        assert r.status_code == 200
        assert "AAPL" in r.text
        assert "Information Technology" in r.text
        for metric_key in CANONICAL_METRIC_SPECS.keys():
            assert metric_key in r.text, f"Expected canonical metric {metric_key} in symbol drilldown"
        assert "7.1" in r.text
        assert "6.5" in r.text
        assert "6.6" in r.text

        r_404 = client.get("/ui/symbol/DOESNOTEXIST")
        assert r_404.status_code == 404
    finally:
        store.close()


def test_ui_alerts_center_and_acknowledge(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    client, store = _build_seeded_web_client(tmp_path)
    try:
        r = client.get("/ui/alerts")
        assert r.status_code == 200
        assert "AAPL" in r.text
        assert "dip_trigger" in r.text

        open_alerts = store.list_alerts(status="open")
        assert len(open_alerts) == 1
        aid = open_alerts[0]["alert_id"]

        r_ack = client.post(f"/ui/alerts/{aid}/ack", follow_redirects=False)
        assert r_ack.status_code == 303
        assert "alerts" in r_ack.headers.get("location", "")

        open_after = store.list_alerts(status="open")
        assert len(open_after) == 0
        acked_after = store.list_alerts(status="acked")
        assert len(acked_after) == 1
    finally:
        store.close()
