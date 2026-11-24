import json
import os
import tempfile

import pandas as pd

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.gui.tabs.due_diligence import DueDiligenceTabMixin
from compounderos_app.gui.tabs.quality import QualityTabMixin
from compounderos_app.gui.tabs.sp500_ranking import Sp500RankingTabMixin
from compounderos_app.scoring_engine import CANONICAL_REGISTRY_ID


class _FakeTextbox:
    def __init__(self):
        self.content = ""

    def configure(self, **kwargs):
        pass

    def delete(self, *_args):
        self.content = ""

    def insert(self, _pos, text):
        self.content += str(text)


class _FakeLabel:
    def __init__(self):
        self.text = ""
        self.text_color = None

    def configure(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "text_color" in kwargs:
            self.text_color = kwargs["text_color"]


class _FakeTreeview:
    def __init__(self):
        self.items = {}
        self.headings = {}
        self.columns = {}
        self._counter = 0

    def delete(self, *iids):
        for iid in iids:
            self.items.pop(iid, None)

    def get_children(self):
        return list(self.items.keys())

    def heading(self, col, **kwargs):
        self.headings[col] = kwargs

    def column(self, col, **kwargs):
        self.columns[col] = kwargs

    def insert(self, _parent, _index, values=(), tags=()):
        self._counter += 1
        iid = f"I{self._counter:03d}"
        self.items[iid] = {"values": values, "tags": tags}
        return iid

    def item(self, iid, **kwargs):
        if kwargs:
            self.items[iid].update(kwargs)
        return self.items[iid]

    def tag_configure(self, *_a, **_kw):
        pass

    def xview_moveto(self, *_a):
        pass


def test_quality_tab_renders_deterministic_swot_and_bucket_breakdown(tmp_path):
    store = DataStore(str(tmp_path / "qual.db"), role="replica")

    class _QualHarness(QualityTabMixin, DueDiligenceTabMixin):
        def __init__(self):
            self.scan_store = store
            self.current_symbol = "AAPL"
            self.gemini_api_key = ""
            self.macro_swot_cache = {}
            self.macro_swot_inflight = set()
            self.tr_check = _FakeTreeview()
            self.lbl_score = _FakeLabel()
            self.txt_verdict = _FakeTextbox()
            self.qual_Grouping_box = self.txt_verdict
            self.lbl_traj_value = _FakeLabel()
            self.lbl_traj_mode = _FakeLabel()
            self.lbl_traj_prospects = _FakeLabel()
            self.lbl_traj_asof = _FakeLabel()
            self.df_fin_a = None
            self.df_bal_a = None
            self.df_cf_a = None

        def _sync_watchlist_score(self, *_a):
            pass

        def _render_quality_radar(self, *_a):
            pass

    q_dates = pd.date_range("2024-03-31", periods=8, freq="QE")
    fin_q = pd.DataFrame(
        {
            "TotalRevenue": [100e9 + i * 3e9 for i in range(8)],
            "GrossProfit": [45e9 + i * 1.5e9 for i in range(8)],
            "OperatingIncome": [30e9 + i * 1.1e9 for i in range(8)],
            "NetIncome": [24e9 + i * 0.9e9 for i in range(8)],
            "NormalizedIncome": [24e9 + i * 0.9e9 for i in range(8)],
            "EBITDA": [33e9 + i * 1.2e9 for i in range(8)],
            "TaxProvision": [5e9 for _ in range(8)],
            "PretaxIncome": [29e9 + i * 0.9e9 for i in range(8)],
        },
        index=q_dates,
    )
    bal_q = pd.DataFrame(
        {
            "CashAndCashEquivalents": [50e9 for _ in range(8)],
            "TotalDebt": [80e9 for _ in range(8)],
            "TotalStockholderEquity": [70e9 for _ in range(8)],
            "TotalAssets": [300e9 for _ in range(8)],
            "CurrentAssets": [120e9 for _ in range(8)],
            "CurrentLiabilities": [100e9 for _ in range(8)],
        },
        index=q_dates,
    )
    cf_q = pd.DataFrame(
        {
            "OperatingCashFlow": [28e9 for _ in range(8)],
            "CapitalExpenditure": [-3e9 for _ in range(8)],
            "FreeCashFlow": [25e9 for _ in range(8)],
            "StockBasedCompensation": [2.5e9 for _ in range(8)],
        },
        index=q_dates,
    )

    harness = _QualHarness()
    harness.fill_quality_checklist(
        {"sector": "Technology", "industry": "Consumer Electronics", "longName": "Apple Inc."},
        fin_q,
        bal_q,
        cf_q,
    )

    verdict_text = harness.txt_verdict.content
    assert "Deterministic Bucket Breakdown:" in verdict_text
    assert "Macro & Sector SWOT (Deterministic):" in verdict_text
    assert "Strengths:" in verdict_text
    assert "Threats:" in verdict_text
    store.close()


def test_sp500_ranking_tab_displays_v1_1_shrunk_buckets_gates_and_trajectory(tmp_path):
    scan_store = DataStore(str(tmp_path / "rank_replica.db"), role="replica")

    with scan_store.lock:
        scan_store.conn.execute(
            """
            INSERT INTO sp500_scores (
                symbol, sector, score, composite_shrunk, verdict, model,
                evaluated_at, registry_id, registry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "Information Technology", 8.4, 8.1, "SCORED", "det-1.1.0", "2026-01-15T12:00:00+00:00", CANONICAL_REGISTRY_ID, "h1"),
        )
        scan_store.conn.execute(
            """
            INSERT INTO sp500_scores (
                symbol, sector, score, composite_shrunk, verdict, model,
                evaluated_at, registry_id, registry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("BA", "Industrials", 3.8, 3.6, "SCORED", "det-1.1.0", "2026-01-15T12:00:00+00:00", CANONICAL_REGISTRY_ID, "h1"),
        )
        scan_store.conn.execute(
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
                "AAPL", "2026-01-15", "2026-01-15T12:00:00+00:00", "Information Technology",
                8.4, 8.4, 8.1, 8.5, 8.8, 7.6, 8.2, 1.0, "pass", "", "", None,
                "SCORED", 28.0, 25.0, 0.038, 3.2e12, "det-1.1.0", "scan", "{}", "{}", CANONICAL_REGISTRY_ID, "h1",
            ),
        )
        scan_store.conn.execute(
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
                "BA", "2026-01-15", "2026-01-15T12:00:00+00:00", "Industrials",
                3.8, 5.2, 3.6, 4.0, 3.5, 2.0, 4.5, 0.85, "fired", "net_debt_ebitda", "", 4.0,
                "SCORED", 40.0, 30.0, 0.015, 1.1e11, "det-1.1.0", "scan", "{}", "{}", CANONICAL_REGISTRY_ID, "h1",
            ),
        )
        scan_store.conn.execute(
            """
            INSERT INTO trajectory_history (
                symbol, scan_date, trajectory, confidence, composition, peer_state,
                granularity_used, peer_set_json, raw_inputs_json, derived_json,
                extraction_version, engine_version, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAPL", "2026-01-15", 6.8, 0.92, "AGB", "OK",
                "sub_industry", "[]", "{}", "{}", "trajx-1.0.0", "traj-1.1.0", "2026-01-15T12:00:00+00:00",
            ),
        )
        scan_store.conn.commit()

    class _FakeRoot:
        def winfo_exists(self):
            return True

    class _RankHarness(Sp500RankingTabMixin):
        def __init__(self):
            self.scan_store = scan_store
            self.root = _FakeRoot()
            self.tr_sp500 = _FakeTreeview()
            self.lbl_sp500_status = _FakeLabel()
            self.gateway = None

        def ui(self, fn, *args):
            fn(*args)

    harness = _RankHarness()
    harness._proc_sp500_ranking()

    children = harness.tr_sp500.get_children()
    assert len(children) == 2

    row_aapl = harness.tr_sp500.item(children[0])["values"]
    row_ba = harness.tr_sp500.item(children[1])["values"]

    row_aapl_str = " | ".join(str(x) for x in row_aapl)
    row_ba_str = " | ".join(str(x) for x in row_ba)

    assert "AAPL" in row_aapl_str
    assert "8.1" in row_aapl_str
    assert "P:8.5" in row_aapl_str
    assert "PASS" in row_aapl_str
    assert "6.8" in row_aapl_str and "AGB" in row_aapl_str

    assert "BA" in row_ba_str
    assert "CAPPED" in row_ba_str
    scan_store.close()
