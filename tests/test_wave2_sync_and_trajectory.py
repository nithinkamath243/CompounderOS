import json
import os
import sqlite3
import tempfile
import threading

import pandas as pd

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.gui.services.scanner_gateway import ScannerSyncClient
from compounderos_app.gui.tabs.quality import QualityTabMixin
from compounderos_app.scoring_engine import CANONICAL_REGISTRY_ID
from rescore_history import rescore_database
from scanner_service.pipeline import ScannerPipeline


def test_trajectory_pass_persists_peer_prospects_in_derived_json_only(tmp_path):
    db_path = str(tmp_path / "traj_test.db")
    store = DataStore(db_path, role="sor")
    pipeline = ScannerPipeline(store, {"scan_workers": 1}, threading.Event())

    pipeline.universe = [
        ("AAPL", "Information Technology", "Technology Hardware, Storage & Peripherals"),
        ("HPQ", "Information Technology", "Technology Hardware, Storage & Peripherals"),
        ("DELL", "Information Technology", "Technology Hardware, Storage & Peripherals"),
    ]

    def _fake_estimates(sym):
        offset = {"AAPL": 0.0, "HPQ": 0.02, "DELL": 0.04}[sym]
        eps_trend = pd.DataFrame(
            {
                "current": [6.20 + offset, 7.10 + offset],
                "7daysAgo": [6.10, 6.95],
                "30daysAgo": [6.00, 6.80],
                "60daysAgo": [5.90, 6.70],
                "90daysAgo": [5.80, 6.60],
            },
            index=["0y", "+1y"],
        )
        eps_revisions = pd.DataFrame(
            {
                "upLast7days": [12, 11],
                "upLast30days": [15, 14],
                "downLast7days": [1, 2],
                "downLast30days": [2, 2],
            },
            index=["0y", "+1y"],
        )
        earnings_estimate = pd.DataFrame(
            {
                "numberOfAnalysts": [30, 28],
                "growth": [0.12 + offset, 0.14 + offset],
                "yearAgoEps": [5.50, 6.20],
            },
            index=["0y", "+1y"],
        )
        revenue_estimate = pd.DataFrame(
            {
                "growth": [0.10 + offset, 0.12 + offset],
                "avg": [4.0e11, 4.48e11],
                "yearAgoRevenue": [3.6e11, 4.0e11],
            },
            index=["0y", "+1y"],
        )
        return {
            "eps_trend": eps_trend,
            "eps_revisions": eps_revisions,
            "earnings_estimate": earnings_estimate,
            "revenue_estimate": revenue_estimate,
        }

    pipeline.gateway.fetch_estimates = _fake_estimates
    ok = pipeline._run_trajectory_stage("2026-01-15")
    assert ok is True

    traj_row = store.get_latest_trajectory("AAPL")
    assert traj_row is not None

    raw_inputs = json.loads(traj_row["raw_inputs_json"])
    derived = json.loads(traj_row["derived_json"])

    forbidden_peer_keys = {
        "median_revision",
        "breadth",
        "median_peer_growth",
        "peer_median_revision",
        "peer_breadth",
        "peer_n",
        "peer_pos_n",
    }
    assert forbidden_peer_keys.isdisjoint(set(raw_inputs.keys())), (
        f"raw_inputs_json must stay pure self_raw only; found {forbidden_peer_keys & set(raw_inputs.keys())}"
    )

    assert derived.get("peer_median_revision") is not None
    assert derived.get("peer_breadth") is not None
    assert derived.get("median_peer_growth") is not None
    assert derived.get("peer_n") == 2
    assert derived.get("peer_pos_n") == 2

    class _FakeLabel:
        def __init__(self):
            self.text = ""
            self.text_color = None

        def configure(self, **kwargs):
            if "text" in kwargs:
                self.text = kwargs["text"]
            if "text_color" in kwargs:
                self.text_color = kwargs["text_color"]

    class _DummyQualHost(QualityTabMixin):
        def __init__(self, scan_store):
            self.scan_store = scan_store
            self.lbl_traj_value = _FakeLabel()
            self.lbl_traj_mode = _FakeLabel()
            self.lbl_traj_prospects = _FakeLabel()
            self.lbl_traj_asof = _FakeLabel()

    host = _DummyQualHost(store)
    host._fill_trajectory_panel("AAPL")

    assert "Future Prospects (peer set):" in host.lbl_traj_prospects.text
    assert "median est. revision" in host.lbl_traj_prospects.text
    assert "breadth" in host.lbl_traj_prospects.text
    assert "peer growth" in host.lbl_traj_prospects.text
    store.close()


def test_price_history_composite_cursor_and_multipage_sync(tmp_path):
    server_store = DataStore(str(tmp_path / "server.db"), role="sor")
    symbols = [f"SYM{i:02d}" for i in range(15)]

    with server_store.lock:
        for d in ["2026-01-02", "2026-01-03"]:
            for idx, sym in enumerate(symbols):
                server_store.conn.execute(
                    """
                    INSERT INTO price_history (symbol, date, open, high, low, close, volume, source, truncated_window)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (sym, d, 100.0 + idx, 101.0 + idx, 99.0 + idx, 100.0 + idx, 1000000 + idx, "yfinance"),
                )
        server_store.conn.commit()

    collected = []
    cursor = "2026-01-01"
    pages = 0
    while True:
        batch = server_store.get_prices_since(cursor, limit=4, source="yfinance")
        if not batch:
            break
        collected.extend(batch)
        cursor = f"{batch[-1]['date']}:{batch[-1]['symbol']}"
        pages += 1
        assert pages <= 20

    assert len(collected) == 30, f"Expected all 30 rows across same-date pages, got {len(collected)}"
    keys = [(r["date"], r["symbol"]) for r in collected]
    assert len(set(keys)) == 30

    replica_store = DataStore(str(tmp_path / "replica.db"), role="replica")
    client = ScannerSyncClient("http://scanner.local", "secret-token", replica_store)

    def _mock_get_json(path):
        if path.startswith("/v1/sync/prices"):
            since_val = path.split("since=")[1].split("&")[0]
            rows = server_store.get_prices_since(since_val, limit=7, source="yfinance")
            next_since = f"{rows[-1]['date']}:{rows[-1]['symbol']}" if rows else None
            return {"rows": rows, "next_since": next_since}
        return {"rows": [], "next_since": None}

    client._get_json = _mock_get_json
    report = client.pull_once()
    assert report.tables.get("prices") == 30

    synced_rows = replica_store.get_prices_since("1970-01-01", limit=100, source="yfinance")
    assert len(synced_rows) == 30
    assert client.cursors["last_sync_prices"] == "2026-01-03:SYM14"

    server_store.close()
    replica_store.close()


def test_rescore_history_updates_registry_id_and_sp500_scores_snapshot(tmp_path):
    db_path = str(tmp_path / "rescore.db")
    store = DataStore(db_path, role="sor")

    raw_metrics = {
        "roic": 0.25,
        "operating_margin": 0.28,
        "fcf_margin": 0.22,
        "rev_cagr_3y": 0.14,
        "eps_cagr_3y": 0.16,
        "fcf_conversion": 0.95,
        "net_debt_to_fcf": 0.8,
        "interest_coverage": 20.0,
        "share_dilution": -0.01,
        "gaap_gap": 0.90,
        "reinvestment_rate": 0.30,
        "incremental_roic": 0.24,
    }

    with store.lock:
        store.conn.execute(
            """
            INSERT INTO ticker_map (symbol, cik, name, sector, industry, classification_as_of)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("MSFT", "0000789019", "Microsoft Corp", "Information Technology", "Systems Software", "2026-01-01"),
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
                "MSFT",
                "2026-01-10",
                "2026-01-10T00:00:00+00:00",
                "Information Technology",
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                "unverified",
                "",
                "",
                None,
                "SCORED",
                30.0,
                26.0,
                0.035,
                3e12,
                "det-1.0.0",
                "scan",
                json.dumps(raw_metrics),
                "{}",
                "legacy-0.9",
                "oldhash",
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
                "MSFT",
                "Information Technology",
                1.0,
                1.0,
                "SCORED",
                "det-1.0.0",
                "2026-01-10T00:00:00+00:00",
                "legacy-0.9",
                "oldhash",
            ),
        )
        store.conn.commit()
    store.close()

    rescore_database(db_path, dry_run=False)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    hist_row = conn.execute("SELECT * FROM sp500_scores_history WHERE symbol = 'MSFT'").fetchone()
    snap_row = conn.execute("SELECT * FROM sp500_scores WHERE symbol = 'MSFT'").fetchone()
    conn.close()

    assert hist_row["registry_id"] == CANONICAL_REGISTRY_ID
    assert hist_row["score"] > 5.0
    assert snap_row is not None
    assert abs(snap_row["score"] - hist_row["score"]) < 1e-6
    assert abs(snap_row["composite_shrunk"] - hist_row["composite_shrunk"]) < 1e-6
    assert snap_row["registry_id"] == CANONICAL_REGISTRY_ID
    assert snap_row["registry_hash"] == hist_row["registry_hash"]
