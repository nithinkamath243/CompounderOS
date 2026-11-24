import json
import os
import tempfile
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

os.environ.setdefault("COMPOUNDEROS_BASE_DIR", os.path.join(tempfile.gettempdir(), "compounderos_test_base"))

from compounderos_app.database import DataStore
from compounderos_app.gui.services.data_service import DataService
from compounderos_app.gui.tabs.due_diligence import synthesize_bull_bear_views
from compounderos_app.quality_diagnostics import (
    assess_moat_and_five_forces,
    compute_quality_diagnostics,
)
from scanner_service.api import create_app


def test_sec_extraction_of_forensic_tags():
    q_dates = pd.date_range("2024-03-31", periods=8, freq="QE")

    def _q_items(vals):
        return [
            {
                "start": (d - pd.Timedelta(days=89)).strftime("%Y-%m-%d"),
                "end": d.strftime("%Y-%m-%d"),
                "val": float(v * 1e6),
                "fp": f"Q{d.quarter}",
                "form": "10-Q",
            }
            for d, v in zip(q_dates, vals)
        ]

    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": _q_items(np.linspace(100, 120, 8))}},
                "GrossProfit": {"units": {"USD": _q_items(np.linspace(60, 72, 8))}},
                "CostOfGoodsAndServicesSold": {"units": {"USD": _q_items(np.linspace(40, 48, 8))}},
                "OperatingIncomeLoss": {"units": {"USD": _q_items(np.linspace(30, 38, 8))}},
                "NetIncomeLoss": {"units": {"USD": _q_items(np.linspace(24, 30, 8))}},
                "InterestExpense": {"units": {"USD": _q_items(np.linspace(2, 2, 8))}},
                "AccountsReceivableNetCurrent": {"units": {"USD": _q_items(np.linspace(20, 22, 8))}},
                "InventoryNet": {"units": {"USD": _q_items(np.linspace(10, 11, 8))}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": _q_items(np.linspace(28, 35, 8))}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": _q_items(np.linspace(4, 5, 8))}},
                "ShareBasedCompensation": {"units": {"USD": _q_items(np.linspace(2, 2.5, 8))}},
                "PaymentsOfDividendsCommonStock": {"units": {"USD": _q_items(np.linspace(8, 9, 8))}},
            }
        }
    }

    fin_q, _, bal_q, _, cf_q, _ = DataService.build_sec_dataframes(facts)
    assert "CostOfRevenue" in fin_q.columns and not fin_q["CostOfRevenue"].dropna().empty
    assert "InterestExpense" in fin_q.columns and not fin_q["InterestExpense"].dropna().empty
    assert "AccountsReceivable" in bal_q.columns and not bal_q["AccountsReceivable"].dropna().empty
    assert "Inventory" in bal_q.columns and not bal_q["Inventory"].dropna().empty
    assert "DividendsPaid" in cf_q.columns and not cf_q["DividendsPaid"].dropna().empty


def test_negative_wc_moat_vs_liquidity_stress_and_sbc_cash_conversion():
    # Apple/Costco style float-funded compounder
    diag_moat = compute_quality_diagnostics(
        raw_metrics={
            "current_ratio": 0.92,
            "earnings_quality": 1.22,
            "net_debt_ebitda": 0.45,
            "roic": 0.34,
            "gross_margin": 0.46,
            "revenue_growth": 0.09,
        }
    )
    assert diag_moat["working_capital_regime"] == "NEGATIVE_WC_MOAT"
    assert any("Negative working-capital moat" in s for s in diag_moat["strengths"])

    # Levered liquidity distress
    diag_stress = compute_quality_diagnostics(
        raw_metrics={
            "current_ratio": 0.78,
            "earnings_quality": 0.55,
            "net_debt_ebitda": 4.20,
            "roic": 0.05,
        }
    )
    assert diag_stress["working_capital_regime"] == "LIQUIDITY_STRESS"
    assert any("liquidity tightness" in r for r in diag_stress["red_flags"])


def test_channel_stuffing_inventory_bloat_and_yield_trap_red_flags():
    q_dates = pd.date_range("2024-03-31", periods=8, freq="QE")
    # Revenue grows +5% YoY (400 -> 420), while AR balloons +45% (80 -> 116) and Inventory balloons +40% (50 -> 70)
    fin_q = pd.DataFrame(
        {
            "TotalRevenue": [100, 100, 100, 100, 105, 105, 105, 105],
            "GrossProfit": [40, 40, 40, 40, 42, 42, 42, 42],
            "CostOfRevenue": [60, 60, 60, 60, 63, 63, 63, 63],
            "OperatingIncome": [12, 12, 12, 12, 10, 10, 10, 10],
            "NetIncome": [8, 8, 8, 8, 6, 6, 6, 6],
            "InterestExpense": [4, 4, 4, 4, 4, 4, 4, 4],
        },
        index=q_dates,
    )
    bal_q = pd.DataFrame(
        {
            "AccountsReceivable": [75, 78, 80, 80, 95, 102, 110, 116],
            "Inventory": [48, 49, 50, 50, 58, 62, 66, 70],
        },
        index=q_dates,
    )
    # TTM OCF = 32, CapEx = 12 -> TTM FCF = 20; Dividends = 18 (90% FCF payout => yield trap) and SBC = 12 (True FCF = 8 / NI 24 = 0.33x)
    cf_q = pd.DataFrame(
        {
            "OperatingCashFlow": [8, 8, 8, 8, 8, 8, 8, 8],
            "CapitalExpenditure": [-3, -3, -3, -3, -3, -3, -3, -3],
            "StockBasedCompensation": [2, 2, 2, 2, 3, 3, 3, 3],
            "DividendsPaid": [-4, -4, -4, -4, -4.5, -4.5, -4.5, -4.5],
        },
        index=q_dates,
    )

    diag = compute_quality_diagnostics(
        raw_metrics={"current_ratio": 1.4, "roic": 0.08},
        fin_q=fin_q,
        bal_q=bal_q,
        cf_q=cf_q,
    )
    assert diag["interest_state"] == "SOLVENCY_WARNING"
    assert np.isclose(diag["interest_coverage"], 40.0 / 16.0)
    assert diag["ar_state"] == "RED_FLAG_CHANNEL_STUFFING"
    assert diag["inventory_state"] == "RED_FLAG_INVENTORY_BLOAT"
    # Verify sound Inventory Turnover = TTM COGS (252) / Avg 4Q Inv (64) and GMROI = TTM GP (168) / Avg 4Q Inv (64)
    assert np.isclose(diag["inventory_turnover"], 252.0 / 64.0)
    assert np.isclose(diag["gmroi"], 168.0 / 64.0)
    assert diag["dividend_state"] == "YIELD_TRAP_RISK"
    assert diag["sbc_cash_state"] == "SBC_DILUTED_WARNING"

    bb = synthesize_bull_bear_views(
        symbol="STUFF",
        score_res={"score": 4.5, "gate_state": "pass"},
        raw_metrics={"roic": 0.08},
        diagnostics=diag,
    )
    assert any("Receivables red flag" in b for b in bb["bearish"])
    assert any("Dividend sustainability warning" in b for b in bb["bearish"])


def test_moat_and_five_forces_requires_quantitative_roic_and_renders_on_web(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOUNDEROS_LOCAL_DEV", "1")
    low_moat = assess_moat_and_five_forces(
        raw_metrics={"roic": 0.06, "gross_margin": 0.65, "operating_leverage": 1.8},
        sector="Information Technology",
    )
    assert low_moat["verdict"] == "NO_QUANT_MOAT"

    wide_moat = assess_moat_and_five_forces(
        raw_metrics={"roic": 0.29, "roic_momentum": 0.02, "gross_margin": 0.68, "operating_leverage": 1.4, "revenue_growth": 0.14},
        sector="Information Technology",
    )
    assert wide_moat["verdict"] == "WIDE_MOAT_CANDIDATE"

    store = DataStore(str(tmp_path / "web_moat.db"))
    try:
        raw_json = json.dumps(
            {
                "roic": 0.31,
                "roic_momentum": 0.02,
                "gross_margin": 0.69,
                "operating_leverage": 1.35,
                "current_ratio": 0.95,
                "earnings_quality": 1.25,
                "net_debt_ebitda": 0.20,
                "revenue_growth": 0.15,
            }
        )
        with store.lock:
            store.conn.execute(
                """
                INSERT INTO sp500_scores_history (
                    symbol, scan_date, evaluated_at, sector, score, composite_shrunk,
                    b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                    confidence, gate_state, fcf_yield, engine_version, raw_json, subscores_json, registry_hash
                ) VALUES (
                    'AAPL', '2026-03-01', '2026-03-01T00:00:00Z', 'Information Technology',
                    8.7, 8.6, 8.8, 9.1, 8.2, 8.4, 1.0, 'pass', 0.038, 'det-1.1.0', ?, '{}', 'hash'
                )
                """,
                (raw_json,),
            )
            store.conn.commit()

        client = TestClient(create_app(store))
        resp = client.get("/ui/symbol/AAPL")
        assert resp.status_code == 200
        assert "Economic Moat" in resp.text
        assert "Wide Moat Candidate" in resp.text
        assert "NEGATIVE_WC_MOAT" in resp.text
    finally:
        store.close()
