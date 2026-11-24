import json
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .database import DataStore

_NL_STOPWORDS = {
    "CREATE",
    "A",
    "AN",
    "THE",
    "PORTFOLIO",
    "WITH",
    "SHARES",
    "SHARE",
    "OF",
    "AT",
    "AND",
    "BUY",
    "SELL",
    "HOLD",
    "ADD",
    "FOR",
    "IN",
    "TO",
    "ON",
    "PER",
    "USD",
    "SH",
    "SHS",
    "QTY",
    "QUANTITY",
    "COST",
    "PRICE",
    "BASIS",
    "SYMBOL",
    "TICKER",
    "STOCK",
    "UNITS",
    "UNIT",
}

_ROW_SYM_FIRST_RE = re.compile(
    r"^\s*(?P<symbol>[A-Za-z]{1,6}(?:[\.-][A-Za-z]{1,2})?)\s*[,:\s]+\s*(?P<shares>\d+(?:\.\d+)?)"
    r"(?:\s*(?:shares?|shs?|sh|units?)?)?"
    r"(?:\s*(?:[,:\s]+|@|at\s+)\s*\$?(?P<cost>\d+(?:\.\d+)?))?"
    r"(?:\s*[,\s]+\s*(?P<date>\d{4}-\d{2}-\d{2}))?\s*$",
    re.IGNORECASE,
)

_SHARES_FIRST_RE = re.compile(
    r"(?<!\$)\b(?P<shares>\d+(?:\.\d+)?)\s*(?:shares?|shs?|sh|units?)?\s+(?:of\s+)?(?P<symbol>[A-Za-z]{1,6}(?:[\.-][A-Za-z]{1,2})?)\b"
    r"(?:\s*(?:@|at|for)\s*\$?(?P<cost>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)


def parse_natural_language_holdings(text: str) -> List[Dict[str, Any]]:
    if not text or not str(text).strip():
        return []

    holdings: List[Dict[str, Any]] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 1. Check if line is a single symbol-first row (e.g., "NVDA 25 118.50" or "MSFT, 15, 410.00")
        m_row = _ROW_SYM_FIRST_RE.match(line)
        if m_row:
            sym_cand = m_row.group("symbol").upper()
            if sym_cand not in _NL_STOPWORDS:
                sh_val = float(m_row.group("shares"))
                cost_str = m_row.group("cost")
                cost_val = float(cost_str) if cost_str is not None else 0.0
                t_date = m_row.group("date")
                if sh_val > 0:
                    holdings.append(
                        {
                            "symbol": sym_cand,
                            "shares": sh_val,
                            "cost_basis": cost_val,
                            "trade_date": t_date,
                        }
                    )
                continue

        # 2. Conversational / multi-clause shares-first patterns
        for m in _SHARES_FIRST_RE.finditer(line):
            sym_cand = m.group("symbol").upper()
            if sym_cand in _NL_STOPWORDS:
                continue
            sh_val = float(m.group("shares"))
            if sh_val <= 0:
                continue
            cost_str = m.group("cost")
            cost_val = float(cost_str) if cost_str is not None else 0.0
            holdings.append(
                {
                    "symbol": sym_cand,
                    "shares": sh_val,
                    "cost_basis": cost_val,
                    "trade_date": None,
                }
            )

    return holdings


def _clean_float(val: Any, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("$", "").replace(",", "")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def import_portfolio_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    col_map: Dict[str, str] = {}
    for col in df.columns:
        norm = re.sub(r"[^a-z0-9]", "_", str(col).strip().lower()).strip("_")
        col_map[norm] = col

    def _find_col(aliases: Tuple[str, ...]) -> Optional[str]:
        for a in aliases:
            if a in col_map:
                return col_map[a]
        return None

    sym_col = _find_col(("symbol", "ticker", "instrument", "stock"))
    sh_col = _find_col(("shares", "qty", "quantity", "units", "position"))
    cost_col = _find_col(("cost_basis", "costbasis", "cost", "avg_cost", "avgcost", "average_cost", "price", "purchase_price"))
    date_col = _find_col(("trade_date", "tradedate", "date"))
    comm_col = _find_col(("commission", "fee", "fees"))
    rpnl_col = _find_col(("realized_pnl", "realizedpnl", "realized"))

    if not sym_col or not sh_col:
        return []

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_sym = row.get(sym_col)
        if raw_sym is None or (isinstance(raw_sym, float) and pd.isna(raw_sym)):
            continue
        sym = str(raw_sym).strip().upper()
        if not sym or sym in _NL_STOPWORDS:
            continue
        sh = _clean_float(row.get(sh_col), 0.0)
        if sh <= 0:
            continue
        cb = _clean_float(row.get(cost_col), 0.0) if cost_col else 0.0
        t_date = None
        if date_col and row.get(date_col) is not None and not pd.isna(row.get(date_col)):
            t_date = str(row.get(date_col)).strip() or None
        comm = _clean_float(row.get(comm_col), 0.0) if comm_col else 0.0
        rpnl = _clean_float(row.get(rpnl_col), 0.0) if rpnl_col else 0.0
        out.append(
            {
                "symbol": sym,
                "shares": sh,
                "cost_basis": cb,
                "trade_date": t_date,
                "commission": comm,
                "realized_pnl": rpnl,
            }
        )
    return out


def _parse_date_safe(date_str: Any) -> Optional[pd.Timestamp]:
    if not date_str:
        return None
    try:
        ts = pd.to_datetime(str(date_str).strip(), utc=False)
        if pd.isna(ts):
            return None
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    except Exception:
        return None


def analyze_tax_lots(
    lots: List[Dict[str, Any]],
    price_by_symbol: Dict[str, float],
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    ref_ts = _parse_date_safe(as_of_date) or pd.Timestamp.now("UTC").tz_localize(None).normalize()
    enriched_lots: List[Dict[str, Any]] = []
    ltcg_unrealized = 0.0
    stcg_unrealized = 0.0
    harvest_candidates: List[Dict[str, Any]] = []
    near_ltcg_lots: List[Dict[str, Any]] = []

    for lot in lots or []:
        sym = str(lot.get("symbol") or "").strip().upper()
        shares = float(lot.get("shares") or 0.0)
        if not sym or shares <= 0:
            continue
        cost_basis = float(lot.get("cost_basis") or 0.0)
        curr_price = float(price_by_symbol.get(sym, cost_basis))
        unrealized_gain = (curr_price - cost_basis) * shares
        t_date_raw = lot.get("trade_date")
        t_ts = _parse_date_safe(t_date_raw)

        holding_days: Optional[int] = None
        tax_status = "UNSPECIFIED"
        days_to_ltcg: Optional[int] = None
        if t_ts is not None:
            holding_days = max(0, int((ref_ts - t_ts.normalize()).days))
            if holding_days > 365:
                tax_status = "LTCG"
                days_to_ltcg = 0
                ltcg_unrealized += unrealized_gain
            else:
                tax_status = "STCG"
                days_to_ltcg = max(0, 366 - holding_days)
                stcg_unrealized += unrealized_gain
                if 0 < days_to_ltcg <= 45 and unrealized_gain > 0:
                    near_ltcg_lots.append(
                        {
                            "lot_id": lot.get("lot_id"),
                            "symbol": sym,
                            "shares": shares,
                            "days_to_ltcg": days_to_ltcg,
                            "unrealized_gain": unrealized_gain,
                        }
                    )
        else:
            stcg_unrealized += unrealized_gain

        item = {
            "lot_id": lot.get("lot_id"),
            "symbol": sym,
            "shares": shares,
            "cost_basis": cost_basis,
            "price": curr_price,
            "trade_date": str(t_date_raw) if t_date_raw else None,
            "holding_days": holding_days,
            "tax_status": tax_status,
            "days_to_ltcg": days_to_ltcg,
            "unrealized_gain": unrealized_gain,
        }
        enriched_lots.append(item)
        if unrealized_gain < -1e-6:
            harvest_candidates.append(item)

    harvestable_loss_total = abs(sum(c["unrealized_gain"] for c in harvest_candidates))
    return {
        "lots": enriched_lots,
        "ltcg_unrealized": ltcg_unrealized,
        "stcg_unrealized": stcg_unrealized,
        "harvest_candidates": harvest_candidates,
        "harvestable_loss_total": harvestable_loss_total,
        "ordinary_income_offset_cap": min(harvestable_loss_total, 3000.0),
        "near_ltcg_lots": near_ltcg_lots,
    }


def simulate_lot_sale(
    lots: List[Dict[str, Any]],
    symbol: str,
    shares_to_sell: float,
    sale_price: float,
    method: str = "specific_id",
    as_of_date: Optional[str] = None,
    stcg_rate: float = 0.35,
    ltcg_rate: float = 0.15,
) -> Dict[str, Any]:
    """
    Simulate selling `shares_to_sell` of `symbol` at `sale_price` under either:
      - 'fifo': oldest trade_date first
      - 'specific_id': tax-minimizing Specific Identification (ST losses -> LT losses -> LT lowest gains -> ST lowest gains)
    """
    sym = (symbol or "").strip().upper()
    target_sh = float(shares_to_sell or 0.0)
    px = float(sale_price or 0.0)
    m_norm = (method or "specific_id").strip().lower()
    tax_info = analyze_tax_lots(lots, {sym: px}, as_of_date=as_of_date)
    sym_lots = [dict(l) for l in tax_info["lots"] if l["symbol"] == sym and l["shares"] > 0]

    def _fifo_key(l: Dict[str, Any]):
        return (l.get("trade_date") or "9999-12-31", int(l.get("lot_id") or 0))

    def _spec_id_key(l: Dict[str, Any]):
        gain_ps = px - float(l["cost_basis"])
        is_lt = l["tax_status"] == "LTCG"
        if gain_ps < 0 and not is_lt:
            bucket = 0  # STCG loss first (highest tax shield)
        elif gain_ps < 0 and is_lt:
            bucket = 1  # LTCG loss second
        elif is_lt:
            bucket = 2  # LTCG gain third (lower tax rate)
        else:
            bucket = 3  # STCG gain last (highest tax rate)
        # Within losses, most negative gain_ps first; within gains, smallest gain_ps first => ascending gain_ps!
        return (bucket, gain_ps, int(l.get("lot_id") or 0))

    ordered = sorted(sym_lots, key=_fifo_key if m_norm == "fifo" else _spec_id_key)

    remaining = target_sh
    selected: List[Dict[str, Any]] = []
    realized_stcg = 0.0
    realized_ltcg = 0.0

    for lot in ordered:
        if remaining <= 1e-9:
            break
        take = min(remaining, float(lot["shares"]))
        gain = (px - float(lot["cost_basis"])) * take
        if lot["tax_status"] == "LTCG":
            realized_ltcg += gain
        else:
            realized_stcg += gain
        selected.append(
            {
                "lot_id": lot.get("lot_id"),
                "trade_date": lot.get("trade_date"),
                "tax_status": lot["tax_status"],
                "shares_sold": take,
                "cost_basis": float(lot["cost_basis"]),
                "realized_gain": gain,
            }
        )
        remaining -= take

    # IRS netting across ST and LT buckets
    net_gain = realized_stcg + realized_ltcg
    if realized_stcg >= 0 and realized_ltcg >= 0:
        est_tax = realized_stcg * stcg_rate + realized_ltcg * ltcg_rate
    elif net_gain <= 0:
        # Net capital loss offsets up to $3,000 of ordinary income in current year
        est_tax = -min(abs(net_gain), 3000.0) * stcg_rate
    elif realized_stcg > 0 and realized_ltcg < 0:
        est_tax = net_gain * stcg_rate
    else:
        est_tax = net_gain * ltcg_rate

    return {
        "symbol": sym,
        "method": "fifo" if m_norm == "fifo" else "specific_id",
        "shares_requested": target_sh,
        "shares_sold": target_sh - max(0.0, remaining),
        "sale_price": px,
        "realized_stcg": round(realized_stcg, 2),
        "realized_ltcg": round(realized_ltcg, 2),
        "total_realized_gain": round(net_gain, 2),
        "estimated_tax": round(est_tax, 2),
        "selected_lots": selected,
    }


class PortfolioAnalyticsEngine:
    @staticmethod
    def _compute_risk_and_correlations(
        store: DataStore,
        weights_by_sym: Dict[str, float],
        correlation_limit: float = 0.80,
    ) -> Dict[str, Any]:
        import numpy as np

        empty_risk = {
            "annualized_volatility": None,
            "annualized_return": None,
            "sharpe_ratio": None,
            "beta": None,
            "r_squared": None,
            "benchmark_symbol": None,
            "max_pair_correlation": None,
            "correlated_pairs": [],
        }
        if not weights_by_sym:
            return empty_risk

        syms = sorted(weights_by_sym.keys())
        bench_candidates = ("SPY", "^GSPC", "VOO")
        query_syms = list(dict.fromkeys(syms + list(bench_candidates)))
        placeholders = ",".join("?" for _ in query_syms)

        with store.lock:
            rows = store.conn.execute(
                f"""
                SELECT symbol, date, close
                FROM price_history
                WHERE symbol IN ({placeholders}) AND close > 0
                ORDER BY date ASC
                """,
                tuple(query_syms),
            ).fetchall()

        if not rows:
            return empty_risk

        df_raw = pd.DataFrame([dict(r) for r in rows])
        # Deduplicate multiple sources per (symbol, date) keeping last
        df_piv = (
            df_raw.drop_duplicates(subset=["date", "symbol"], keep="last")
            .pivot(index="date", columns="symbol", values="close")
            .sort_index()
            .tail(260)
            .ffill(limit=2)
        )
        rets = df_piv.pct_change().dropna(how="all")
        held_cols = [s for s in syms if s in rets.columns and rets[s].dropna().shape[0] >= 10]
        if not held_cols:
            return empty_risk

        held_rets = rets[held_cols]
        w_series = pd.Series({s: float(weights_by_sym[s]) for s in held_cols})
        # Weighted daily portfolio return, renormalizing across non-NaN symbols each day
        valid_mask = held_rets.notna()
        day_w = valid_mask.mul(w_series, axis=1)
        day_w_sum = day_w.sum(axis=1)
        valid_days = day_w_sum > 0
        if int(valid_days.sum()) < 10:
            return empty_risk

        norm_w = day_w.loc[valid_days].div(day_w_sum.loc[valid_days], axis=0)
        port_ret = (held_rets.loc[valid_days].fillna(0.0) * norm_w).sum(axis=1)

        ann_vol = float(port_ret.std(ddof=1) * np.sqrt(252.0))
        ann_ret = float(port_ret.mean() * 252.0)
        rf = store.get_us10y_yield() if hasattr(store, "get_us10y_yield") else 0.0425
        sharpe = ((ann_ret - rf) / ann_vol) if ann_vol > 1e-8 else None

        beta_val: Optional[float] = None
        r2_val: Optional[float] = None
        bench_used: Optional[str] = None
        for b_sym in bench_candidates:
            if b_sym in rets.columns:
                aligned = pd.concat([port_ret.rename("p"), rets[b_sym].rename("b")], axis=1).dropna()
                if len(aligned) >= 10:
                    b_var = float(aligned["b"].var(ddof=1))
                    if b_var > 1e-12:
                        cov_pb = float(aligned["p"].cov(aligned["b"]))
                        beta_val = cov_pb / b_var
                        corr_pb = float(aligned["p"].corr(aligned["b"]))
                        if np.isfinite(corr_pb):
                            r2_val = corr_pb * corr_pb
                        bench_used = b_sym
                        break

        max_corr: Optional[float] = None
        corr_pairs: List[Dict[str, Any]] = []
        if len(held_cols) >= 2:
            corr_mat = held_rets.corr(min_periods=10)
            for idx_i in range(len(held_cols)):
                for idx_j in range(idx_i + 1, len(held_cols)):
                    s1, s2 = held_cols[idx_i], held_cols[idx_j]
                    c_val = corr_mat.loc[s1, s2]
                    if pd.isna(c_val) or not np.isfinite(float(c_val)):
                        continue
                    cf = float(c_val)
                    if max_corr is None or cf > max_corr:
                        max_corr = cf
                    comb_w = float(weights_by_sym.get(s1, 0.0) + weights_by_sym.get(s2, 0.0))
                    if cf > correlation_limit and comb_w >= 0.10:
                        corr_pairs.append(
                            {
                                "pair": f"{s1}/{s2}",
                                "symbol_a": s1,
                                "symbol_b": s2,
                                "correlation": cf,
                                "combined_weight": comb_w,
                            }
                        )
            corr_pairs.sort(key=lambda x: -x["correlation"])

        return {
            "annualized_volatility": ann_vol,
            "annualized_return": ann_ret,
            "sharpe_ratio": sharpe,
            "beta": beta_val,
            "r_squared": r2_val,
            "benchmark_symbol": bench_used,
            "max_pair_correlation": max_corr,
            "correlated_pairs": corr_pairs,
        }

    @staticmethod
    def analyze_portfolio(
        store: DataStore,
        portfolio_name: str = "Main",
        price_overrides: Optional[Dict[str, float]] = None,
        single_name_limit: float = 0.15,
        sector_limit: float = 0.35,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        p_name = (portfolio_name or "Main").strip() or "Main"
        overrides = {str(k).strip().upper(): float(v) for k, v in (price_overrides or {}).items() if v is not None}

        base_holdings = store.get_portfolio(p_name)
        lots = store.get_portfolio_lots(p_name) if hasattr(store, "get_portfolio_lots") else []

        lot_comm_by_sym: Dict[str, float] = {}
        lot_rpnl_by_sym: Dict[str, float] = {}
        for lot in lots:
            lsym = str(lot.get("symbol") or "").strip().upper()
            if not lsym:
                continue
            lot_comm_by_sym[lsym] = lot_comm_by_sym.get(lsym, 0.0) + float(lot.get("commission") or 0.0)
            lot_rpnl_by_sym[lsym] = lot_rpnl_by_sym.get(lsym, 0.0) + float(lot.get("realized_pnl") or 0.0)

        if not base_holdings and lots:
            agg_sh: Dict[str, float] = {}
            agg_cost_val: Dict[str, float] = {}
            for lot in lots:
                lsym = str(lot.get("symbol") or "").strip().upper()
                lsh = float(lot.get("shares") or 0.0)
                lcb = float(lot.get("cost_basis") or 0.0)
                if lsym and lsh > 0:
                    agg_sh[lsym] = agg_sh.get(lsym, 0.0) + lsh
                    agg_cost_val[lsym] = agg_cost_val.get(lsym, 0.0) + lsh * lcb
            base_holdings = [
                (sym, sh, agg_cost_val[sym] / sh if sh > 0 else 0.0)
                for sym, sh in sorted(agg_sh.items())
            ]

        positions: List[Dict[str, Any]] = []
        total_market_value = 0.0
        total_cost_value = 0.0
        total_realized_pnl = sum(lot_rpnl_by_sym.values())
        total_commission = sum(lot_comm_by_sym.values())

        with store.lock:
            cur = store.conn.cursor()
            for sym_raw, shares_raw, cost_raw in base_holdings:
                sym = str(sym_raw).strip().upper()
                shares = float(shares_raw or 0.0)
                cost_basis = float(cost_raw or 0.0)
                if not sym or shares <= 0:
                    continue

                price = overrides.get(sym)
                if price is None:
                    prow = cur.execute(
                        "SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                        (sym,),
                    ).fetchone()
                    if prow and prow["close"] is not None:
                        price = float(prow["close"])
                if price is None:
                    price = cost_basis

                srow = cur.execute(
                    """
                    SELECT sector, score, composite_shrunk, fcf_yield, raw_json
                    FROM sp500_scores_history
                    WHERE symbol = ?
                    ORDER BY evaluated_at DESC
                    LIMIT 1
                    """,
                    (sym,),
                ).fetchone()

                sector = None
                comp_shrunk: Optional[float] = None
                roic_val: Optional[float] = None
                fcf_yield_val: Optional[float] = None

                if srow:
                    sector = srow["sector"]
                    comp_shrunk = (
                        float(srow["composite_shrunk"])
                        if srow["composite_shrunk"] is not None
                        else (float(srow["score"]) if srow["score"] is not None else None)
                    )
                    if srow["fcf_yield"] is not None:
                        fcf_yield_val = float(srow["fcf_yield"])
                    if srow["raw_json"]:
                        try:
                            rdict = json.loads(srow["raw_json"])
                            if rdict.get("roic") is not None:
                                roic_val = float(rdict["roic"])
                            if fcf_yield_val is None and rdict.get("fcf_yield") is not None:
                                fcf_yield_val = float(rdict["fcf_yield"])
                        except Exception:
                            pass
                else:
                    snap = cur.execute(
                        "SELECT sector, score, composite_shrunk FROM sp500_scores WHERE symbol = ?",
                        (sym,),
                    ).fetchone()
                    if snap:
                        sector = snap["sector"]
                        comp_shrunk = (
                            float(snap["composite_shrunk"])
                            if snap["composite_shrunk"] is not None
                            else (float(snap["score"]) if snap["score"] is not None else None)
                        )

                if not sector:
                    tmap = cur.execute(
                        "SELECT sector FROM ticker_map WHERE symbol = ?",
                        (sym,),
                    ).fetchone()
                    if tmap and tmap["sector"]:
                        sector = tmap["sector"]
                sector = sector or "Unknown"

                trow = cur.execute(
                    """
                    SELECT trajectory FROM trajectory_history
                    WHERE symbol = ?
                    ORDER BY evaluated_at DESC
                    LIMIT 1
                    """,
                    (sym,),
                ).fetchone()
                traj_val = float(trow["trajectory"]) if (trow and trow["trajectory"] is not None) else None

                m_val = shares * price
                c_val = shares * cost_basis
                u_pnl = m_val - c_val
                u_pnl_pct = (u_pnl / c_val) if c_val > 0 else 0.0

                total_market_value += m_val
                total_cost_value += c_val

                positions.append(
                    {
                        "symbol": sym,
                        "shares": shares,
                        "cost_basis": cost_basis,
                        "price": price,
                        "market_value": m_val,
                        "cost_value": c_val,
                        "unrealized_pnl": u_pnl,
                        "unrealized_pnl_pct": u_pnl_pct,
                        "realized_pnl": lot_rpnl_by_sym.get(sym, 0.0),
                        "commission": lot_comm_by_sym.get(sym, 0.0),
                        "weight": 0.0,
                        "sector": sector,
                        "composite_shrunk": comp_shrunk,
                        "roic": roic_val,
                        "fcf_yield": fcf_yield_val,
                        "trajectory": traj_val,
                    }
                )

        sector_totals: Dict[str, float] = {}
        weights_by_sym: Dict[str, float] = {}
        price_by_sym: Dict[str, float] = {}
        for pos in positions:
            w = (pos["market_value"] / total_market_value) if total_market_value > 0 else 0.0
            pos["weight"] = w
            weights_by_sym[pos["symbol"]] = w
            price_by_sym[pos["symbol"]] = float(pos["price"])
            sec = pos["sector"] or "Unknown"
            sector_totals[sec] = sector_totals.get(sec, 0.0) + pos["market_value"]

        sector_weights: Dict[str, float] = {}
        if total_market_value > 0:
            for sec, val in sorted(sector_totals.items(), key=lambda kv: -kv[1]):
                sector_weights[sec] = val / total_market_value

        def _weighted_metric(key: str) -> Optional[float]:
            num = 0.0
            den = 0.0
            for pos in positions:
                mv = pos["market_value"]
                val = pos.get(key)
                if mv > 0 and val is not None:
                    num += mv * float(val)
                    den += mv
            return (num / den) if den > 0 else None

        look_through = {
            "composite_shrunk": _weighted_metric("composite_shrunk"),
            "roic": _weighted_metric("roic"),
            "fcf_yield": _weighted_metric("fcf_yield"),
            "trajectory": _weighted_metric("trajectory"),
        }

        concentration_alerts: List[Dict[str, Any]] = []
        for pos in sorted(positions, key=lambda p: -p["weight"]):
            if pos["weight"] > single_name_limit:
                concentration_alerts.append(
                    {
                        "kind": "single_name",
                        "target": pos["symbol"],
                        "symbol": pos["symbol"],
                        "weight": pos["weight"],
                        "limit": single_name_limit,
                        "message": f"Single-name concentration: {pos['symbol']} is {pos['weight']:.1%} of portfolio (limit {single_name_limit:.0%}).",
                    }
                )

        for sec, sec_w in sorted(sector_weights.items(), key=lambda kv: -kv[1]):
            if sec_w > sector_limit:
                concentration_alerts.append(
                    {
                        "kind": "sector",
                        "target": sec,
                        "sector": sec,
                        "weight": sec_w,
                        "limit": sector_limit,
                        "message": f"Sector concentration: {sec} is {sec_w:.1%} of portfolio (limit {sector_limit:.0%}).",
                    }
                )

        risk_metrics = PortfolioAnalyticsEngine._compute_risk_and_correlations(
            store, weights_by_sym
        )
        for cp in risk_metrics.get("correlated_pairs") or []:
            concentration_alerts.append(
                {
                    "kind": "correlation",
                    "target": cp["pair"],
                    "weight": cp["combined_weight"],
                    "limit": 0.80,
                    "message": (
                        f"Factor correlation cluster: {cp['pair']} daily return correlation is "
                        f"{cp['correlation']:.2f} (>0.80) across {cp['combined_weight']:.1%} combined weight."
                    ),
                }
            )

        tax_lots_summary = analyze_tax_lots(lots, price_by_sym, as_of_date=as_of_date)

        total_unrealized_pnl = total_market_value - total_cost_value
        total_unrealized_pnl_pct = (
            (total_unrealized_pnl / total_cost_value) if total_cost_value > 0 else 0.0
        )

        return {
            "portfolio_name": p_name,
            "positions": positions,
            "total_market_value": total_market_value,
            "total_cost_value": total_cost_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
            "total_realized_pnl": total_realized_pnl,
            "total_commission": total_commission,
            "sector_weights": sector_weights,
            "look_through": look_through,
            "risk_metrics": risk_metrics,
            "tax_lots": tax_lots_summary,
            "concentration_alerts": concentration_alerts,
        }
