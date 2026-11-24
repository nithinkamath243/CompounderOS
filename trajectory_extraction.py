# filepath: compounderos_app/trajectory_extraction.py
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import pandas as pd
from compounderos_app.gics_map import GICS_MAP_VERSION

EXTRACTION_VERSION = "trajx-1.0.0"


def _cell(df: Optional[pd.DataFrame], row_label: str, col_label: str) -> Optional[float]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if row_label not in df.index or col_label not in df.columns:
        return None
    try:
        val = df.loc[row_label, col_label]
        if val is None:
            return None
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    except (TypeError, ValueError, KeyError):
        return None


def _cell_case_insensitive(
    df: Optional[pd.DataFrame],
    row_label: str,
    dir_token: str,
    days_token: str,
) -> Optional[float]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if row_label not in df.index:
        return None
    for col in df.columns:
        cl = str(col).lower()
        if dir_token.lower() in cl and days_token.lower() in cl:
            try:
                val = df.loc[row_label, col]
                if val is None:
                    return None
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    return None
                return fval
            except (TypeError, ValueError, KeyError):
                return None
    return None


def extract_trajectory_inputs(estimates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract flat trajectory raw input metrics from yfinance estimate DataFrames.
    Every expected field maps to None if missing/NaN/unparseable, never 0.0 and never omitted.
    """
    eps_trend = estimates.get("eps_trend")
    eps_revisions = estimates.get("eps_revisions")
    earnings_estimate = estimates.get("earnings_estimate")
    revenue_estimate = estimates.get("revenue_estimate")

    out: Dict[str, Any] = {
        "extraction_version": EXTRACTION_VERSION,
        "gics_map_version": GICS_MAP_VERSION,
        "rev_window_used": "90d",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    horizon_map = {
        "0y": "0y",
        "1y": "+1y",
    }

    for h_key, row_label in horizon_map.items():
        # 1. eps_trend
        out[f"eps_cur_{h_key}"] = _cell(eps_trend, row_label, "current")
        out[f"eps_7d_{h_key}"] = _cell(eps_trend, row_label, "7daysAgo")
        out[f"eps_30d_{h_key}"] = _cell(eps_trend, row_label, "30daysAgo")
        out[f"eps_60d_{h_key}"] = _cell(eps_trend, row_label, "60daysAgo")
        out[f"eps_90d_{h_key}"] = _cell(eps_trend, row_label, "90daysAgo")

        # 2. eps_revisions (case-insensitive matching on direction & days)
        out[f"up30_{h_key}"] = _cell_case_insensitive(eps_revisions, row_label, "up", "30")
        out[f"down30_{h_key}"] = _cell_case_insensitive(eps_revisions, row_label, "down", "30")
        out[f"up7_{h_key}"] = _cell_case_insensitive(eps_revisions, row_label, "up", "7")
        out[f"down7_{h_key}"] = _cell_case_insensitive(eps_revisions, row_label, "down", "7")

        # 3. earnings_estimate
        out[f"n_analysts_{h_key}"] = _cell(earnings_estimate, row_label, "numberOfAnalysts")
        out[f"eps_growth_{h_key}"] = _cell(earnings_estimate, row_label, "growth")
        out[f"year_ago_eps_{h_key}"] = _cell(earnings_estimate, row_label, "yearAgoEps")

        # 4. revenue_estimate
        out[f"rev_growth_{h_key}"] = _cell(revenue_estimate, row_label, "growth")
        out[f"rev_avg_{h_key}"] = _cell(revenue_estimate, row_label, "avg")
        out[f"year_ago_rev_{h_key}"] = _cell(revenue_estimate, row_label, "yearAgoRevenue")

    return out
