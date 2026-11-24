import pytest
from fastapi.testclient import TestClient

from compounderos_app.database import DataStore
from compounderos_app.macro_tracker import (
    CANONICAL_MACRO_BENCHMARKS,
    compute_equity_risk_premium,
    compute_yield_curve_spread,
)
from compounderos_app.prediction_markets import PredictionMarketClient
from scanner_service.api import create_app


def test_compute_equity_risk_premium_and_us10y_normalization(tmp_path):
    assert "yields" in CANONICAL_MACRO_BENCHMARKS
    assert "indices" in CANONICAL_MACRO_BENCHMARKS
    assert "commodities" in CANONICAL_MACRO_BENCHMARKS
    assert "fx" in CANONICAL_MACRO_BENCHMARKS
    assert "crypto" in CANONICAL_MACRO_BENCHMARKS

    # Both decimal (0.055) and percentage (5.5) normalize to 0.055
    erp1 = compute_equity_risk_premium(0.055, 0.0425)
    erp2 = compute_equity_risk_premium(5.5, 4.25)
    erp3 = compute_equity_risk_premium(0.055, 4.25)
    assert erp1 == pytest.approx(0.0125)
    assert erp2 == pytest.approx(0.0125)
    assert erp3 == pytest.approx(0.0125)
    assert compute_equity_risk_premium(None, 0.0425) is None

    store = DataStore(str(tmp_path / "macro.db"))
    # Default fallback when macro_quotes is empty
    assert store.get_us10y_yield(default=0.0425) == pytest.approx(0.0425)

    # Upsert ^TNX as 4.35 (percentage points) and ^IRX as 4.60
    store.upsert_macro_quotes(
        [
            {"symbol": "^TNX", "label": "US 10Y Treasury", "category": "yields", "price": 4.35, "change_pct": 0.02, "unit": "pct"},
            {"symbol": "^IRX", "label": "US 13W T-Bill", "category": "yields", "price": 4.60, "change_pct": -0.01, "unit": "pct"},
        ]
    )
    assert store.get_us10y_yield() == pytest.approx(0.0435)
    quotes = store.get_macro_quotes(category="yields")
    assert len(quotes) == 2
    spread = compute_yield_curve_spread(quotes)
    assert spread == pytest.approx(-0.25)


def test_prediction_market_normalization_and_sqlite_roundtrip(tmp_path):
    store = DataStore(str(tmp_path / "odds.db"))

    kalshi_payload = {
        "markets": [
            {
                "ticker": "KXFED-26MAY-T425",
                "title": "Fed cuts rates by 25 bps in May",
                "yes_bid": 62,
                "yes_ask": 66,
                "last_price": 64,
                "volume": 250000,
            }
        ]
    }
    poly_payload = [
        {
            "id": "99123",
            "question": "US recession in 2026?",
            "outcomePrices": '["0.28", "0.72"]',
            "volume": "1850000",
        }
    ]

    k_rows = PredictionMarketClient.normalize_kalshi_markets(kalshi_payload)
    p_rows = PredictionMarketClient.normalize_polymarket_markets(poly_payload)
    assert len(k_rows) == 1
    assert k_rows[0]["source"] == "Kalshi"
    assert k_rows[0]["probability_pct"] == pytest.approx(64.0)
    assert k_rows[0]["category"] == "rates"

    assert len(p_rows) == 1
    assert p_rows[0]["source"] == "Polymarket"
    assert p_rows[0]["probability_pct"] == pytest.approx(28.0)
    assert p_rows[0]["category"] == "growth"

    inserted = store.upsert_prediction_odds(k_rows + p_rows)
    assert inserted == 2

    cached = store.get_prediction_odds()
    assert len(cached) == 2
    # Ordered by volume DESC -> Polymarket (1.85M) first, Kalshi (250k) second
    assert cached[0]["source"] == "Polymarket"
    assert cached[1]["source"] == "Kalshi"


def test_web_macro_view_and_symbol_detail_erp(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    store = DataStore(str(tmp_path / "web_macro.db"))

    store.upsert_macro_quotes(
        [
            {"symbol": "^TNX", "label": "US 10Y Treasury", "category": "yields", "price": 4.20, "change_pct": 0.01, "unit": "pct"},
            {"symbol": "^IRX", "label": "US 13W T-Bill", "category": "yields", "price": 4.40, "change_pct": -0.02, "unit": "pct"},
            {"symbol": "^GSPC", "label": "S&P 500", "category": "indices", "price": 5880.0, "change_pct": 0.45, "unit": "usd"},
        ]
    )
    store.upsert_prediction_odds(
        [
            {
                "contract_id": "KALSHI-FOMC-CUT",
                "source": "Kalshi",
                "category": "rates",
                "title": "FOMC cuts 25 bps at next meeting",
                "probability_pct": 71.5,
                "volume": 950000.0,
                "url": "https://kalshi.com/markets/fomc",
            }
        ]
    )

    with store.lock:
        store.conn.execute(
            """
            INSERT INTO sp500_scores_history (
                scan_date, symbol, evaluated_at, engine_version, registry_hash,
                sector, score, composite_shrunk, gate_state, confidence,
                fcf_yield, raw_json, subscores_json
            ) VALUES ('2026-03-01', 'AAPL', '2026-03-01T00:00:00Z', 'det-1.1.0', 'hash',
                      'Information Technology', 8.5, 8.3, 'pass', 1.0, 0.0545, '{"roic": 0.42}', '{}')
            """
        )
        store.conn.commit()

    app = create_app(store)
    client = TestClient(app)

    # 1. GET /ui/macro
    resp_macro = client.get("/ui/macro")
    assert resp_macro.status_code == 200
    assert "Macro Ribbon" in resp_macro.text
    assert "US 10Y Treasury" in resp_macro.text
    assert "4.20%" in resp_macro.text
    assert "FOMC cuts 25 bps at next meeting" in resp_macro.text
    assert "71.5%" in resp_macro.text

    # 2. GET /ui/symbol/AAPL -> FCF yield 5.45% vs 10Y 4.20% => ERP = +1.25% (+125 bps)
    resp_sym = client.get("/ui/symbol/AAPL")
    assert resp_sym.status_code == 200
    assert "10Y Treasury: 4.20%" in resp_sym.text
    assert "Equity Risk Premium (ERP):" in resp_sym.text
    assert "+1.25%" in resp_sym.text
    assert "+125 bps" in resp_sym.text
