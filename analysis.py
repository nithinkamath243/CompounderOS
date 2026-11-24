from typing import Optional

import numpy as np
import pandas as pd


def clean_numeric_series(series: Optional[pd.Series]):
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        pass
    return s.sort_index()


def clean_datetime_index_df(df: Optional[pd.DataFrame]):
    if df is None or df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def safe_div(n, d, default=0.0):
    """Safely divide values while filtering invalid inputs."""
    try:
        if d == 0 or pd.isna(d) or pd.isna(n) or np.isinf(d) or np.isinf(n):
            return default
        return n / d
    except Exception:
        return default


def calculate_technical_indicators(df: Optional[pd.DataFrame]):
    if df is None or df.empty or "Close" not in df.columns:
        return df
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    out["SMA_200"] = close.rolling(window=200, min_periods=1).mean()
    return out


def align_to_quarter_end(series: Optional[pd.Series]):
    """Snap SEC fact dates to quarter end to avoid near-date misalignment."""
    if series is None or series.empty:
        return series
    s = pd.to_numeric(series, errors="coerce").dropna().copy()
    idx = pd.to_datetime(s.index).tz_localize(None)
    s.index = idx + pd.tseries.offsets.QuarterEnd(0)
    return s.groupby(level=0).last().sort_index()


def align_to_year_end(series: Optional[pd.Series]):
    if series is None or series.empty:
        return series
    s = pd.to_numeric(series, errors="coerce").dropna().copy()
    idx = pd.to_datetime(s.index).tz_localize(None)
    s.index = idx + pd.tseries.offsets.YearEnd(0)
    return s.groupby(level=0).last().sort_index()


def _pick_series(df: Optional[pd.DataFrame], keys):
    if df is None or df.empty:
        return None
    for col in df.columns:
        norm_col = "".join(str(col).split()).lower()
        for key in keys:
            if "".join(str(key).split()).lower() in norm_col:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    return s.sort_index()
    return None


def calculate_piotroski_f_score(
    fin_a: Optional[pd.DataFrame],
    bal_a: Optional[pd.DataFrame],
    cf_a: Optional[pd.DataFrame],
):
    """Return Piotroski F-score details using annual data.

    Output: {"score": int, "max_score": int, "checks": {name: bool|None}}
    """
    if fin_a is None or fin_a.empty or bal_a is None or bal_a.empty or cf_a is None or cf_a.empty:
        return None

    ni = _pick_series(fin_a, ["NetIncome", "NetIncomeLoss"])
    gp = _pick_series(fin_a, ["GrossProfit"])
    rev = _pick_series(fin_a, ["TotalRevenue", "Revenue"])
    ocf = _pick_series(cf_a, ["OperatingCashFlow", "NetCashProvidedByUsedInOperatingActivities"])
    assets = _pick_series(bal_a, ["TotalAssets", "Assets"])
    debt = _pick_series(bal_a, ["LongTermDebt", "TotalDebt", "Debt"])
    ca = _pick_series(bal_a, ["CurrentAssets"])
    cl = _pick_series(bal_a, ["CurrentLiabilities"])
    shares = _pick_series(bal_a, ["ShareIssued", "CommonStockSharesOutstanding", "OrdinarySharesNumber"])

    def latest_two(series: Optional[pd.Series]):
        if series is None or len(series.dropna()) < 2:
            return None, None
        s = series.dropna().sort_index()
        return float(s.iloc[-1]), float(s.iloc[-2])

    score = 0
    max_score = 0
    checks = {}

    def add_check(name: str, value):
        nonlocal score, max_score
        checks[name] = value
        if value is None:
            return
        max_score += 1
        if bool(value):
            score += 1

    # Profitability
    ni_c, ni_p = latest_two(ni)
    ocf_c, _ = latest_two(ocf)
    assets_c, assets_p = latest_two(assets)
    add_check("positive_net_income", ni_c > 0 if ni_c is not None else None)
    add_check("positive_operating_cash_flow", ocf_c > 0 if ocf_c is not None else None)
    if None not in (ni_c, ni_p, assets_c, assets_p) and assets_c != 0 and assets_p != 0:
        roa_c = ni_c / assets_c
        roa_p = ni_p / assets_p
        add_check("roa_improving", roa_c > roa_p)
    else:
        add_check("roa_improving", None)
    add_check("accrual_quality", (ocf_c > ni_c) if (ocf_c is not None and ni_c is not None) else None)

    # Leverage / liquidity / dilution
    debt_c, debt_p = latest_two(debt)
    add_check("leverage_down", debt_c < debt_p if (debt_c is not None and debt_p is not None) else None)
    ca_c, ca_p = latest_two(ca)
    cl_c, cl_p = latest_two(cl)
    if None not in (ca_c, ca_p, cl_c, cl_p) and cl_c != 0 and cl_p != 0:
        add_check("current_ratio_up", (ca_c / cl_c) > (ca_p / cl_p))
    else:
        add_check("current_ratio_up", None)
    sh_c, sh_p = latest_two(shares)
    add_check("no_dilution", sh_c <= sh_p if (sh_c is not None and sh_p is not None) else None)

    # Operating efficiency
    gp_c, gp_p = latest_two(gp)
    rev_c, rev_p = latest_two(rev)
    if None not in (gp_c, gp_p, rev_c, rev_p) and rev_c != 0 and rev_p != 0:
        add_check("gross_margin_up", (gp_c / rev_c) > (gp_p / rev_p))
    else:
        add_check("gross_margin_up", None)
    if None not in (rev_c, rev_p, assets_c, assets_p) and assets_c != 0 and assets_p != 0:
        add_check("asset_turnover_up", (rev_c / assets_c) > (rev_p / assets_p))
    else:
        add_check("asset_turnover_up", None)

    if max_score == 0:
        return None
    return {"score": score, "max_score": max_score, "checks": checks}
