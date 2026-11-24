import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from compounderos_app.database import DataStore
from compounderos_app.research_copilot import ResearchCopilot
from scanner_service.alerts import AlertEngine
from scanner_service.api import create_app


def _seed_store(store: DataStore):
    with store.lock:
        # MSFT: high compounder + high trajectory + dip candidate (fcf_yield = 0.050 >= 0.045, pass)
        store.conn.execute(
            """
            INSERT INTO sp500_scores_history (
                scan_date, symbol, evaluated_at, engine_version, registry_hash,
                sector, score, composite_shrunk,
                b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                gate_state, gates_fired, confidence,
                fcf_yield, raw_json, subscores_json
            ) VALUES (
                '2026-03-01', 'MSFT', '2026-03-01T12:00:00Z', 'det-1.1.0', 'hash',
                'Information Technology', 8.8, 8.8,
                9.1, 8.9, 8.9, 8.2,
                'pass', '', 1.0,
                0.050, ?, ?
            )
            """,
            (
                json.dumps({"roic": 0.31, "fcf_yield": 0.050, "rev_growth_3y": 0.14}),
                json.dumps({"roic": 9.2, "fcf_margin": 8.9}),
            ),
        )
        store.conn.execute(
            """
            INSERT INTO trajectory_history (
                symbol, scan_date, evaluated_at, engine_version, trajectory, composition, peer_state,
                granularity_used, confidence, derived_json
            ) VALUES (
                'MSFT', '2026-03-01', '2026-03-01T12:00:00Z', 'traj-1.1.0', 68.5, 'AGB', 'OK',
                'GICS_SUB_INDUSTRY', 0.90, ?
            )
            """,
            (json.dumps({"self_revision": 0.035, "peer_median_revision": 0.01}),),
        )

        # INTC: lower score, fired gate_state, low trajectory
        store.conn.execute(
            """
            INSERT INTO sp500_scores_history (
                scan_date, symbol, evaluated_at, engine_version, registry_hash,
                sector, score, composite_shrunk,
                b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                gate_state, gates_fired, confidence,
                fcf_yield, raw_json, subscores_json
            ) VALUES (
                '2026-03-01', 'INTC', '2026-03-01T12:00:00Z', 'det-1.1.0', 'hash',
                'Information Technology', 3.8, 3.8,
                3.5, 4.0, 3.9, 3.8,
                'fired', 'negative_fcf', 1.0,
                -0.01, ?, ?
            )
            """,
            (
                json.dumps({"roic": 0.02, "fcf_yield": -0.01}),
                json.dumps({"roic": 2.0}),
            ),
        )
        store.conn.execute(
            """
            INSERT INTO trajectory_history (
                symbol, scan_date, evaluated_at, engine_version, trajectory, composition, peer_state,
                granularity_used, confidence, derived_json
            ) VALUES (
                'INTC', '2026-03-01', '2026-03-01T12:00:00Z', 'traj-1.1.0', 32.0, 'AGB', 'OK',
                'GICS_SUB_INDUSTRY', 0.85, ?
            )
            """,
            (json.dumps({"self_revision": -0.04, "peer_median_revision": 0.01}),),
        )
        store.conn.commit()

    store.upsert_sec_filings(
        "MSFT",
        [
            {
                "form": "10-K",
                "filing_date": "2025-07-30",
                "report_date": "2025-06-30",
                "accession_number": "0000950170-25-100001",
                "primary_document": "msft-20250630.htm",
                "items": "",
                "edgar_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017025100001/msft-20250630.htm",
            }
        ],
    )
    store.upsert_macro_quotes(
        [
            {
                "symbol": "^TNX",
                "label": "US 10Y Treasury",
                "category": "yields",
                "price": 4.20,
                "change_pct": 0.0,
                "unit": "pct",
            }
        ]
    )


def test_research_copilot_chat_and_deep_search_multi_turn(tmp_path):
    db_path = tmp_path / "copilot.db"
    store = DataStore(str(db_path))
    _seed_store(store)

    # Turn 1: Offline deterministic chat mode
    res1 = ResearchCopilot.ask(
        store,
        symbol="MSFT",
        question="How is Microsoft's balance sheet and ROIC?",
        thread_id="test_thread_msft",
        mode="chat",
    )
    assert res1["thread_id"] == "test_thread_msft"
    assert res1["symbol"] == "MSFT"
    assert res1["mode"] == "chat"
    assert "8.80/10" in res1["answer"]
    assert any(c["source"] == "det-1.1.0" for c in res1["citations"])
    assert any(c["source"] == "traj-1.1.0" for c in res1["citations"])
    assert any(c["source"] == "SEC EDGAR" for c in res1["citations"])

    # Turn 2: Deep search mode with mocked gemini_fn
    res2 = ResearchCopilot.ask(
        store,
        symbol="MSFT",
        question="Analyze trajectory revisions and valuation margin of safety.",
        thread_id="test_thread_msft",
        mode="deep_search",
        gemini_fn=lambda prompt: "Cloud commercial bookings accelerated while FCF yield offers +80 bps ERP over 10Y.",
    )
    assert res2["mode"] == "deep_search"
    assert len(res2["plan"]) == 4
    assert "RESEARCH PLAN" in res2["answer"]
    assert "QUANTITATIVE EVIDENCE" in res2["answer"]
    assert "SYNTHESIS & ANSWER" in res2["answer"]
    assert "CITATIONS" in res2["answer"]
    assert "Cloud commercial bookings accelerated" in res2["answer"]

    # Verify multi-turn persistence in research_threads (2 user + 2 assistant = 4 messages)
    thread = store.get_research_thread("test_thread_msft")
    assert len(thread) == 4
    assert thread[0]["role"] == "user"
    assert thread[0]["mode"] == "chat"
    assert thread[1]["role"] == "assistant"
    assert thread[2]["role"] == "user"
    assert thread[2]["mode"] == "deep_search"
    assert thread[3]["role"] == "assistant"
    assert len(thread[3]["citations"]) >= 3


def test_alert_engine_daily_digest_and_ha_push(tmp_path):
    db_path = tmp_path / "alerts.db"
    store = DataStore(str(db_path))
    _seed_store(store)

    # Seed dip trigger and gate alert
    store.record_or_extend_alert("MSFT", "dip_trigger", "2026-03-01", "canonical", {"magnitude": -1.8})
    store.record_or_extend_alert("INTC", "gate_fired", "negative_fcf", "canonical", {"gate": "negative_fcf"})

    cfg = {
        "alert_notify_service": "mobile_app_ipad",
        "SUPERVISOR_TOKEN": "test_ha_supervisor_token",
    }
    engine = AlertEngine(store, cfg)

    digest = engine.build_daily_digest(top_n=5)
    assert any(r["symbol"] == "MSFT" for r in digest["top_compounders"])
    assert all(r["symbol"] != "INTC" for r in digest["top_compounders"])
    assert digest["top_trajectory_improvers"][0]["symbol"] == "MSFT"
    assert len(digest["active_dip_triggers"]) == 1
    assert len(digest["open_gate_alerts"]) == 1
    assert "CompounderOS Daily Digest:" in digest["summary_text"]
    assert "MSFT" in digest["summary_text"]

    with patch("scanner_service.alerts.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        sent = engine.send_daily_digest()
        assert sent["summary_text"] == digest["summary_text"]
        assert mock_post.called
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://supervisor/core/api/services/notify/mobile_app_ipad"
        assert "CompounderOS Daily Digest:" in call_args[1]["json"]["message"]

    with store.lock:
        row = store.conn.execute(
            "SELECT value FROM scanner_state WHERE key = 'last_daily_digest_at'"
        ).fetchone()
    assert row is not None and len(row[0]) > 10


def test_web_ranking_presets_and_ask_copilot_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    db_path = tmp_path / "web_w3_5.db"
    store = DataStore(str(db_path))
    _seed_store(store)

    app = create_app(store)
    client = TestClient(app)

    # 1. Preset: top_compounders (only MSFT passes gate_state == 'pass' & composite >= 7.0)
    r_top = client.get("/ui/ranking?preset=top_compounders")
    assert r_top.status_code == 200
    assert "MSFT" in r_top.text
    assert "INTC" not in r_top.text

    # 2. Preset: trajectory_improvers (only MSFT has trajectory >= 55.0)
    r_traj = client.get("/ui/ranking?preset=trajectory_improvers")
    assert r_traj.status_code == 200
    assert "MSFT" in r_traj.text
    assert "INTC" not in r_traj.text

    # 3. Preset: dip_candidates (MSFT has fcf_yield 0.050 >= 0.045 and gate_state == 'pass')
    r_dip = client.get("/ui/ranking?preset=dip_candidates")
    assert r_dip.status_code == 200
    assert "MSFT" in r_dip.text
    assert "INTC" not in r_dip.text

    # 4. POST /ui/symbol/MSFT/ask in deep_search mode
    r_ask = client.post(
        "/ui/symbol/MSFT/ask",
        data={
            "question": "What are the primary drivers of MSFT quality and ERP?",
            "mode": "deep_search",
        },
        follow_redirects=True,
    )
    assert r_ask.status_code == 200
    assert "Ask CompounderOS" in r_ask.text
    assert "What are the primary drivers of MSFT quality and ERP?" in r_ask.text
    assert "RESEARCH PLAN" in r_ask.text
    assert "CITATIONS" in r_ask.text
