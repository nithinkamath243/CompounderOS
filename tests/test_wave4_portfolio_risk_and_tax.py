import os
import tempfile
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.portfolio_analytics import (
    PortfolioAnalyticsEngine,
    analyze_tax_lots,
    simulate_lot_sale,
)
from scanner_service.api import create_app


def _make_ohlcv(prices: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.full(len(prices), 1_000_000.0),
        },
        index=dates,
    )


def test_portfolio_volatility_sharpe_beta_r2_and_pairwise_correlation_alert(tmp_path):
    store = DataStore(str(tmp_path / "risk_port.db"))
    try:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2026-01-02", periods=60)
        spy_rets = rng.normal(0.0008, 0.01, size=len(dates))
        nvda_rets = 1.50 * spy_rets + rng.normal(0.0002, 0.001, size=len(dates))
        avgo_rets = 1.45 * spy_rets + rng.normal(0.0002, 0.001, size=len(dates))

        spy_px = 500.0 * np.cumprod(1.0 + spy_rets)
        nvda_px = 120.0 * np.cumprod(1.0 + nvda_rets)
        avgo_px = 160.0 * np.cumprod(1.0 + avgo_rets)

        store.upsert_price_history("SPY", _make_ohlcv(spy_px, dates), source="test")
        store.upsert_price_history("NVDA", _make_ohlcv(nvda_px, dates), source="test")
        store.upsert_price_history("AVGO", _make_ohlcv(avgo_px, dates), source="test")

        store.add_portfolio_lot("Semi", "NVDA", 50.0, 110.0, trade_date="2024-05-01")
        store.add_portfolio_lot("Semi", "AVGO", 40.0, 150.0, trade_date="2024-06-01")

        res = PortfolioAnalyticsEngine.analyze_portfolio(store, "Semi", as_of_date="2026-04-01")
        rm = res["risk_metrics"]
        assert rm["benchmark_symbol"] == "SPY"
        assert rm["annualized_volatility"] is not None and rm["annualized_volatility"] > 0.05
        assert rm["sharpe_ratio"] is not None
        assert rm["beta"] is not None and 1.35 <= rm["beta"] <= 1.60
        assert rm["r_squared"] is not None and rm["r_squared"] > 0.90
        assert rm["max_pair_correlation"] is not None and rm["max_pair_correlation"] > 0.95
        assert any(a["kind"] == "correlation" and "AVGO/NVDA" in a["message"] for a in res["concentration_alerts"])
    finally:
        store.close()


def test_tax_lot_ltcg_holding_days_harvesting_and_specific_id_vs_fifo_tax_savings():
    lots = [
        {"lot_id": 1, "symbol": "AAPL", "shares": 10.0, "cost_basis": 100.0, "trade_date": "2023-01-10"},
        {"lot_id": 2, "symbol": "AAPL", "shares": 10.0, "cost_basis": 195.0, "trade_date": "2024-06-15"},
        {"lot_id": 3, "symbol": "AAPL", "shares": 10.0, "cost_basis": 250.0, "trade_date": "2026-01-15"},
    ]
    tax_summary = analyze_tax_lots(lots, {"AAPL": 220.0}, as_of_date="2026-04-01")
    by_id = {l["lot_id"]: l for l in tax_summary["lots"]}
    assert by_id[1]["tax_status"] == "LTCG"
    assert by_id[2]["tax_status"] == "LTCG"
    assert by_id[3]["tax_status"] == "STCG"
    assert by_id[3]["days_to_ltcg"] > 200
    assert np.isclose(tax_summary["harvestable_loss_total"], 300.0)
    assert np.isclose(tax_summary["ordinary_income_offset_cap"], 300.0)

    fifo_sale = simulate_lot_sale(
        lots, "AAPL", shares_to_sell=10.0, sale_price=220.0, method="fifo", as_of_date="2026-04-01"
    )
    spec_sale = simulate_lot_sale(
        lots, "AAPL", shares_to_sell=10.0, sale_price=220.0, method="specific_id", as_of_date="2026-04-01"
    )
    assert fifo_sale["selected_lots"][0]["lot_id"] == 1
    assert fifo_sale["total_realized_gain"] == 1200.0
    assert fifo_sale["estimated_tax"] > 0

    assert spec_sale["selected_lots"][0]["lot_id"] == 3
    assert spec_sale["total_realized_gain"] == -300.0
    assert spec_sale["estimated_tax"] < 0
    assert spec_sale["estimated_tax"] < fifo_sale["estimated_tax"]


def test_web_portfolio_renders_risk_beta_and_tax_lot_intelligence(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    store = DataStore(str(tmp_path / "web_tax.db"))
    try:
        store.add_portfolio_lot("TaxAware", "MSFT", 15.0, 310.0, trade_date="2024-01-15")
        store.add_portfolio_lot("TaxAware", "MSFT", 5.0, 460.0, trade_date="2026-02-10")
        dates = pd.bdate_range("2026-01-02", periods=25)
        msft_px = np.linspace(410.0, 425.0, len(dates))
        spy_px = np.linspace(510.0, 525.0, len(dates))
        store.upsert_price_history("MSFT", _make_ohlcv(msft_px, dates), source="test")
        store.upsert_price_history("SPY", _make_ohlcv(spy_px, dates), source="test")

        client = TestClient(create_app(store))
        resp = client.get("/ui/portfolio?name=TaxAware")
        assert resp.status_code == 200
        assert "Risk, Beta &amp; Tax-Lot Intelligence" in resp.text
        assert "Tax-Loss Harvesting Pool" in resp.text
        assert "LTCG" in resp.text
        assert "STCG" in resp.text
    finally:
        store.close()
