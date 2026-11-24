import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from compounderos_app.database import DataStore
from compounderos_app.portfolio_analytics import (
    PortfolioAnalyticsEngine,
    import_portfolio_dataframe,
    parse_natural_language_holdings,
)
from scanner_service.api import create_app


def test_parse_natural_language_holdings_variants():
    # 1. Conversational sentence with 'shares of ... at $...'
    h1 = parse_natural_language_holdings(
        "Create a portfolio with 25 shares of NVDA at $118.50 and 100 shares of SPY at $510"
    )
    assert len(h1) == 2
    assert h1[0]["symbol"] == "NVDA"
    assert h1[0]["shares"] == pytest.approx(25.0)
    assert h1[0]["cost_basis"] == pytest.approx(118.50)
    assert h1[1]["symbol"] == "SPY"
    assert h1[1]["shares"] == pytest.approx(100.0)
    assert h1[1]["cost_basis"] == pytest.approx(510.0)

    # 2. Shorthand '@' syntax
    h2 = parse_natural_language_holdings("Buy 50 AAPL @ 185.25, 10 MSFT @ $420")
    assert len(h2) == 2
    assert h2[0] == {"symbol": "AAPL", "shares": 50.0, "cost_basis": 185.25, "trade_date": None}
    assert h2[1] == {"symbol": "MSFT", "shares": 10.0, "cost_basis": 420.0, "trade_date": None}

    # 3. Line / CSV-style syntax
    h3 = parse_natural_language_holdings("NVDA 25 118.50\nMSFT, 15, 410.00")
    assert len(h3) == 2
    assert h3[0]["symbol"] == "NVDA"
    assert h3[0]["shares"] == pytest.approx(25.0)
    assert h3[0]["cost_basis"] == pytest.approx(118.50)
    assert h3[1]["symbol"] == "MSFT"
    assert h3[1]["shares"] == pytest.approx(15.0)
    assert h3[1]["cost_basis"] == pytest.approx(410.0)

    # 4. Missing cost syntax sets cost_basis = 0.0
    h4 = parse_natural_language_holdings("25 shares of NVDA and 100 shares of SPY")
    assert len(h4) == 2
    assert h4[0] == {"symbol": "NVDA", "shares": 25.0, "cost_basis": 0.0, "trade_date": None}
    assert h4[1] == {"symbol": "SPY", "shares": 100.0, "cost_basis": 0.0, "trade_date": None}


def test_import_portfolio_dataframe_brokerage_aliases():
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "GOOGL"],
            "Qty": ["50", "20", "15"],
            "Avg_Cost": ["$180.00", "400.50", "150.25"],
            "Trade_Date": ["2025-01-15", "2025-02-01", "2025-02-10"],
            "Fee": ["1.50", "0.00", "2.00"],
        }
    )
    rows = import_portfolio_dataframe(df)
    assert len(rows) == 3
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["shares"] == pytest.approx(50.0)
    assert rows[0]["cost_basis"] == pytest.approx(180.0)
    assert rows[0]["trade_date"] == "2025-01-15"
    assert rows[0]["commission"] == pytest.approx(1.50)


def test_portfolio_analytics_engine_math_and_concentration_alerts(tmp_path):
    store = DataStore(str(tmp_path / "test_port.db"))

    # Add two lots for MSFT (verify weighted-average cost synchronization) and one for AAPL and JNJ
    store.add_portfolio_lot("Main", "MSFT", 10.0, 300.0, trade_date="2025-01-01", realized_pnl=150.0)
    store.add_portfolio_lot("Main", "MSFT", 10.0, 400.0, trade_date="2025-02-01", realized_pnl=50.0)
    store.add_portfolio_lot("Main", "AAPL", 20.0, 150.0, trade_date="2025-01-10")
    store.add_portfolio_lot("Main", "JNJ", 10.0, 150.0, trade_date="2025-01-15")

    lots = store.get_portfolio_lots("Main")
    assert len(lots) == 4
    agg = dict((sym, (sh, cb)) for sym, sh, cb in store.get_portfolio("Main"))
    assert agg["MSFT"][0] == pytest.approx(20.0)
    assert agg["MSFT"][1] == pytest.approx(350.0)

    # Seed latest price_history for MSFT ($450 -> MV $9,000), AAPL ($200 -> MV $4,000), JNJ ($160 -> MV $1,600)
    # Total MV = $14,600; Total Cost = $7,000 + $3,000 + $1,500 = $11,500; Unrealized PnL = +$3,100
    for sym, close_px in [("MSFT", 450.0), ("AAPL", 200.0), ("JNJ", 160.0)]:
        p_df = pd.DataFrame({"Close": [close_px]}, index=pd.to_datetime(["2026-03-01"]))
        store.upsert_price_history(sym, p_df)

    # Seed sp500_scores_history and trajectory_history
    with store.lock:
        for sym, sector, shrunk, roic, fy, traj in [
            ("MSFT", "Information Technology", 8.8, 0.32, 0.035, 72.0),
            ("AAPL", "Information Technology", 8.2, 0.45, 0.040, 64.0),
            ("JNJ", "Health Care", 7.4, 0.18, 0.048, 52.0),
        ]:
            store.conn.execute(
                """
                INSERT INTO sp500_scores_history (
                    scan_date, symbol, evaluated_at, engine_version, registry_hash,
                    sector, score, composite_shrunk, gate_state, confidence,
                    fcf_yield, raw_json, subscores_json
                ) VALUES ('2026-03-01', ?, ?, 'det-1.1.0', 'hash', ?, ?, ?, 'pass', 1.0, ?, ?, '{}')
                """,
                (sym, "2026-03-01T00:00:00Z", sector, shrunk, shrunk, fy, json.dumps({"roic": roic})),
            )
            store.conn.execute(
                """
                INSERT INTO trajectory_history (
                    symbol, evaluated_at, engine_version, trajectory, composition,
                    peer_state, granularity_used, confidence, derived_json
                ) VALUES (?, ?, 'traj-1.1.0', ?, 'AGB', 'OK', 'GICS_SUB_INDUSTRY', 1.0, '{}')
                """,
                (sym, "2026-03-01T00:00:00Z", traj),
            )
        store.conn.commit()

    res = PortfolioAnalyticsEngine.analyze_portfolio(store, "Main")
    assert res["total_market_value"] == pytest.approx(14600.0)
    assert res["total_cost_value"] == pytest.approx(11500.0)
    assert res["total_unrealized_pnl"] == pytest.approx(3100.0)
    assert res["total_unrealized_pnl_pct"] == pytest.approx(3100.0 / 11500.0)
    assert res["total_realized_pnl"] == pytest.approx(200.0)

    # Sector weights sum to 1.0; Information Technology = (9000 + 4000) / 14600 = 13000 / 14600 (~89.04%)
    assert sum(res["sector_weights"].values()) == pytest.approx(1.0)
    assert res["sector_weights"]["Information Technology"] == pytest.approx(13000.0 / 14600.0)

    # Value-weighted look-through composite_shrunk
    expected_wq = (9000.0 * 8.8 + 4000.0 * 8.2 + 1600.0 * 7.4) / 14600.0
    expected_roic = (9000.0 * 0.32 + 4000.0 * 0.45 + 1600.0 * 0.18) / 14600.0
    expected_fy = (9000.0 * 0.035 + 4000.0 * 0.040 + 1600.0 * 0.048) / 14600.0
    assert res["look_through"]["composite_shrunk"] == pytest.approx(expected_wq)
    assert res["look_through"]["roic"] == pytest.approx(expected_roic)
    assert res["look_through"]["fcf_yield"] == pytest.approx(expected_fy)

    # Concentration alerts: MSFT (>15%), AAPL (>15%), and Information Technology (>35%)
    kinds = [(a["kind"], a["target"]) for a in res["concentration_alerts"]]
    assert ("single_name", "MSFT") in kinds
    assert ("single_name", "AAPL") in kinds
    assert ("sector", "Information Technology") in kinds


def test_web_portfolio_get_and_post_import(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    store = DataStore(str(tmp_path / "web_port.db"))

    with store.lock:
        store.conn.execute(
            """
            INSERT INTO sp500_scores_history (
                scan_date, symbol, evaluated_at, engine_version, registry_hash,
                sector, score, composite_shrunk, gate_state, confidence,
                fcf_yield, raw_json, subscores_json
            ) VALUES ('2026-03-01', 'NVDA', '2026-03-01T00:00:00Z', 'det-1.1.0', 'hash',
                      'Information Technology', 9.0, 8.9, 'pass', 1.0, 0.025, '{"roic": 0.48}', '{}')
            """
        )
        store.conn.commit()

    app = create_app(store)
    client = TestClient(app)

    # POST natural language import
    resp_post = client.post(
        "/ui/portfolio/import",
        data={
            "portfolio_name": "Growth",
            "nl_prompt": "Create a portfolio with 25 shares of NVDA at $118.50 and 10 MSFT @ $420",
        },
        follow_redirects=False,
    )
    assert resp_post.status_code == 303
    assert resp_post.headers["location"] == "/ui/portfolio?name=Growth"

    # Follow to GET /ui/portfolio?name=Growth
    resp_get = client.get("/ui/portfolio?name=Growth")
    assert resp_get.status_code == 200
    assert "Portfolio Intelligence — Growth" in resp_get.text
    assert "NVDA" in resp_get.text
    assert "MSFT" in resp_get.text
    assert "$7162.50" in resp_get.text
