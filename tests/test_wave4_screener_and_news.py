import os
import tempfile
import time
from fastapi.testclient import TestClient

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.gui.tabs.insights import InsightsTabMixin
from compounderos_app.news_classifier import classify_news_headline, rank_and_tag_news
from compounderos_app.quality_diagnostics import compute_garp_conviction_score, passes_garp_preset
from scanner_service.api import create_app


def test_news_classifier_separates_primary_catalysts_from_clickbait_noise():
    sig = classify_news_headline(
        "Apple Reports Q1 Quarterly Results and Raises Quarterly Dividend by 5%",
        provider="Business Wire",
    )
    assert sig["signal_tier"] == "PRIMARY_SIGNAL"
    assert sig["signal_badge"] == "[SIGNAL]"

    noise = classify_news_headline(
        "3 No-Brainer Millionaire-Maker Stocks to Buy Right Now",
        provider="The Motley Fool",
    )
    assert noise["signal_tier"] == "SPECULATIVE_NOISE"
    assert noise["signal_badge"] == "[NOISE]"

    now_ts = int(time.time())
    ranked = rank_and_tag_news(
        [
            {"ts": now_ts, "title": "Forget Apple, Buy These 2 Stocks Instead", "provider": "Motley Fool"},
            {"ts": now_ts - 3600, "title": "Apple Files Annual Report on Form 10-K with SEC", "provider": "Reuters"},
        ]
    )
    assert ranked[0]["signal_tier"] == "PRIMARY_SIGNAL"
    assert "10-K" in ranked[0]["title"]
    assert ranked[1]["signal_tier"] == "SPECULATIVE_NOISE"


def test_insights_collect_week_news_survives_broken_ticker_news(monkeypatch):
    import compounderos_app.gui.tabs.insights as insights_mod

    now_ts = int(time.time())

    class _BrokenNewsTicker:
        @property
        def news(self):
            raise RuntimeError("Yahoo news endpoint unavailable")

    class _FakeSearch:
        def __init__(self, _q):
            self.news = [
                {
                    "title": "AAPL Announces $110B Share Repurchase and Quarterly Results",
                    "publisher": "Business Wire",
                    "providerPublishTime": now_ts - 600,
                },
                {
                    "title": "Should You Buy AAPL Stock Right Now?",
                    "publisher": "InvestorPlace",
                    "providerPublishTime": now_ts - 60,
                },
            ]

    monkeypatch.setattr(insights_mod.yf, "Search", _FakeSearch)

    h = InsightsTabMixin()
    items = h._collect_week_news(_BrokenNewsTicker(), "AAPL", "APPLE")
    assert len(items) == 2
    assert items[0]["signal_badge"] == "[SIGNAL]"
    assert items[1]["signal_badge"] == "[NOISE]"


def test_garp_compounder_fair_value_screener_preset_and_conviction_sort(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    store = DataStore(str(tmp_path / "garp_rank.db"))
    try:
        store.upsert_macro_quotes(
            [{"symbol": "^TNX", "label": "US 10Y Treasury", "category": "rates", "price": 4.20, "change_pct": 0.0, "unit": "%"}]
        )
        rows = [
            # 1. GOOGL: high quality, strong trajectory, positive ERP (4.6% vs 4.2% US10Y) => PASS (#1)
            ("GOOGL", "Communication Services", 8.5, 8.4, 0.95, "pass", 0.046, 22.0, 19.5, 68.0),
            # 2. V: high quality, positive trajectory, reasonable forward P/E 23.5x => PASS (#2)
            ("V", "Financials", 8.2, 8.1, 0.90, "pass", 0.038, 26.5, 23.5, 58.0),
            # 3. BUBBL: high quality, but extreme valuation (1.1% FCF yield, 68x fwd P/E) => EXCLUDED
            ("BUBBL", "Information Technology", 8.6, 8.5, 0.95, "pass", 0.011, 75.0, 68.0, 64.0),
            # 4. DETER: cheap FCF yield, but deteriorating trajectory (34.0 < 48.0) => EXCLUDED
            ("DETER", "Industrials", 7.6, 7.5, 0.90, "pass", 0.058, 15.0, 14.0, 34.0),
        ]
        with store.lock:
            for sym, sec, sc, shrunk, conf, gs, fy, pet, pef, traj in rows:
                store.conn.execute(
                    """
                    INSERT INTO sp500_scores_history (
                        symbol, scan_date, evaluated_at, sector, score, composite_shrunk,
                        b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                        confidence, gate_state, fcf_yield, pe_trailing, pe_forward,
                        engine_version, raw_json, subscores_json, registry_hash
                    ) VALUES (?, '2026-03-01', '2026-03-01T00:00:00Z', ?, ?, ?, 8.2, 8.4, 8.1, 8.0,
                              ?, ?, ?, ?, ?, 'det-1.1.0', '{}', '{}', 'hash')
                    """,
                    (sym, sec, sc, shrunk, conf, gs, fy, pet, pef),
                )
                store.conn.execute(
                    """
                    INSERT INTO trajectory_history (
                        symbol, scan_date, trajectory, confidence, composition, peer_state,
                        granularity_used, engine_version, evaluated_at
                    ) VALUES (?, '2026-03-01', ?, 0.9, 'AGB', 'OK', 'sub_industry', 'traj-1.1.0', '2026-03-01T00:00:00Z')
                    """,
                    (sym, traj),
                )
            store.conn.commit()

        client = TestClient(create_app(store))
        resp = client.get("/ui/ranking?preset=compounder_fair_value")
        assert resp.status_code == 200
        assert "<h1>S&P 500 Quality Ranking</h1>" in resp.text
        assert "GARP: Compounders @ Fair Value" in resp.text
        assert "GOOGL" in resp.text
        assert "V" in resp.text
        assert "BUBBL" not in resp.text
        assert "DETER" not in resp.text
        # Verify GOOGL ranks ahead of V by unified conviction score
        assert resp.text.index("./symbol/GOOGL") < resp.text.index("./symbol/V")
    finally:
        store.close()
