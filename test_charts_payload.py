import ast
import os
from dataclasses import is_dataclass
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import compounderos_app.charts as charts_mod
from compounderos_app.config import CONFIG_FILE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PALETTE = {
    "C_ACCENT": "#38bdf8",
    "C_BORDER": "#334155",
    "C_GREEN": "#22c55e",
    "C_RED": "#ef4444",
    "C_YELLOW": "#eab308",
    "C_PINK": "#ec4899",
    "C_BG_CARD": "#0f172a",
    "C_TEXT_MAIN": "#f8fafc",
    "C_TEXT_SUB": "#94a3b8",
    "C_PLOT_FRAME": "#1e293b",
    "C_SERIES_NEUTRAL": "#64748b",
}


def test_chart_payload_dataclass_and_no_self_app_in_renderer():
    assert hasattr(charts_mod, "ChartPayload"), "ChartPayload dataclass must exist in compounderos_app.charts"
    ChartPayload = charts_mod.ChartPayload
    assert is_dataclass(ChartPayload)

    # AST proof: ChartRenderer must have zero self.app attribute accesses (Invariant 8)
    charts_path = os.path.join(REPO_ROOT, "compounderos_app", "charts.py")
    with open(charts_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=charts_path)

    renderer_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChartRenderer"
    )
    self_app_refs = [
        node.lineno
        for node in ast.walk(renderer_cls)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "app"
    ]
    assert self_app_refs == [], f"ChartRenderer still accesses self.app at lines: {self_app_refs}"

    # Render directly from ChartPayload without any GUI App instance
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    dates_d = pd.date_range("2025-01-01", periods=30, freq="D")
    hist_df = pd.DataFrame(
        {"Close": np.linspace(150.0, 180.0, 30), "SMA_50": np.linspace(148.0, 175.0, 30)},
        index=dates_d,
    )
    dates_q = pd.date_range("2024-03-31", periods=4, freq="QE")
    fin_q = pd.DataFrame(
        {
            "TotalRevenue": [10e9, 11e9, 12e9, 13e9],
            "EBITDA": [3e9, 3.2e9, 3.5e9, 4e9],
            "GrossProfit": [6e9, 6.6e9, 7.2e9, 7.8e9],
            "OperatingIncome": [2.5e9, 2.7e9, 3.0e9, 3.4e9],
            "NetIncome": [2.0e9, 2.2e9, 2.4e9, 2.8e9],
            "BasicEPS": [1.5, 1.6, 1.7, 1.9],
        },
        index=dates_q,
    )
    bal_q = pd.DataFrame(
        {
            "CashAndCashEquivalents": [20e9, 21e9, 22e9, 24e9],
            "TotalDebt": [10e9, 10e9, 9.5e9, 9.0e9],
            "CommonStockSharesOutstanding": [1e9, 1e9, 1e9, 1e9],
        },
        index=dates_q,
    )
    cf_q = pd.DataFrame(
        {
            "FreeCashFlow": [2.2e9, 2.4e9, 2.6e9, 3.0e9],
            "StockBasedCompensation": [0.3e9, 0.3e9, 0.35e9, 0.4e9],
        },
        index=dates_q,
    )

    recorded_summaries = []
    for mode in ("Price", "Revenue & EBITDA", "Margins (Gross/Op/Net)", "FCF Per Share"):
        payload = ChartPayload(
            fig=fig,
            ax=ax,
            mode=mode,
            freq="Quarterly",
            price_range="1Y",
            hist_period="1y",
            hist_data=hist_df,
            df_fin_q=fin_q,
            df_bal_q=bal_q,
            df_cf_q=cf_q,
            anim_enabled=False,
            on_analysis=lambda key, summary: recorded_summaries.append((key, summary)),
        )
        renderer = charts_mod.ChartRenderer(payload, palette=PALETTE, anim_frames=10)
        renderer.render()

    assert len(recorded_summaries) == 4


def test_gitignore_covers_build_artifacts():
    gitignore_path = os.path.join(REPO_ROOT, ".gitignore")
    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()
    for required in ("venv/", "build/", "dist/", ".pytest_cache/", ".hypothesis/", "repomix-output.xml"):
        assert required in content, f"Missing {required} in .gitignore"


def test_diag_gemini_uses_canonical_config_path():
    diag_path = os.path.join(REPO_ROOT, "diag_gemini.py")
    with open(diag_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "from compounderos_app.config import CONFIG_FILE" in src, "diag_gemini.py must resolve config via compounderos_app.config.CONFIG_FILE"
    assert os.path.normpath(str(CONFIG_FILE)).endswith(".compounderos_config.json")

