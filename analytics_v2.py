import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .scoring_engine import score_metrics, DEFAULT_REGISTRY


@dataclass(frozen=True)
class DipTriggerResult:
    state: str
    magnitude: Optional[float]
    z_robust: Optional[float]
    z_classical: Optional[float]
    multiple_used: str
    corroboration: Optional[str]
    window: Dict[str, float]
    quality: Dict[str, float]
    signal_class: str = "research_screen"


class DipTriggerAnalyzer:
    """§B Algorithm 1: Dip-Trigger."""

    def __init__(self, z_trigger: float = 1.5, epsilon: float = 0.5, gate_policy: str = "strict"):
        self.z_trigger = z_trigger
        self.epsilon = epsilon
        self.gate_policy = gate_policy

    def evaluate_dip(self, df_history: pd.DataFrame) -> DipTriggerResult:
        """
        df_history must contain: evaluated_at, fcf_yield, pe_trailing, raw_json,
        gate_state, confidence, engine_version, source, gate_cap_applied. Filtered to W=2 yr by caller.
        """
        if df_history.empty:
            return self._fail("insufficient_history")

        # 1. Validity
        df_scan_seeded = df_history[df_history["source"].isin(["scan", "seeded"])].dropna(
            subset=["fcf_yield"]
        )
        if len(df_scan_seeded) < 120:
            return self._fail("insufficient_history")

        first_obs = pd.to_datetime(df_scan_seeded["evaluated_at"].iloc[0])
        last_obs = pd.to_datetime(df_scan_seeded["evaluated_at"].iloc[-1])
        now = pd.Timestamp.utcnow()

        if (last_obs - first_obs).days < 360:
            return self._fail("insufficient_history")
        if (now - last_obs).days > 7:
            return self._fail("stale_data")

        # 2. Dispersion (robust z)
        fcf_series = df_scan_seeded["fcf_yield"].astype(float)
        median_w = fcf_series.median()
        mad_w = (fcf_series - median_w).abs().median()

        if 1.4826 * mad_w < max(1e-4, 0.02 * abs(median_w)):
            return self._fail("flat_multiple")

        x_now = fcf_series.iloc[-1]
        z_robust = (x_now - median_w) / (1.4826 * mad_w)
        z_classical = (x_now - fcf_series.mean()) / fcf_series.std()

        c = z_robust
        if c < self.z_trigger:
            return self._fail("no_trigger")

        # 3. Quality Hold Check (Rescore from raw_json under current registry)
        df_scan = df_history[df_history["source"] == "scan"]
        if df_scan.empty:
            return self._fail("quality_unverifiable")

        q_series = []
        for raw_str in df_scan["raw_json"]:
            raw_dict = json.loads(raw_str) if pd.notna(raw_str) else {}
            res = score_metrics(raw_dict, registry=DEFAULT_REGISTRY)
            # S7 Fix: SKIP THE ROW if un-scorable instead of coercing to 0.0
            if res.composite is not None:
                q_series.append(res.composite)

        if not q_series:
            return self._fail("quality_unverifiable")

        q_series_pd = pd.Series(q_series)
        older_half = q_series_pd.iloc[: len(q_series_pd) // 2]
        recent_5 = q_series_pd.iloc[-5:]

        q_base = older_half.median()
        q_now = recent_5.median()
        delta_q = q_now - q_base

        if delta_q < -self.epsilon:
            return self._fail("quality_falling")

        last_scan = df_scan.iloc[-1]
        if last_scan["confidence"] < 0.75:
            return self._fail("quality_unverifiable")

        gs = last_scan["gate_state"]
        if gs in ["fired", "fired_unverified"]:
            return self._fail("quality_falling")
        if self.gate_policy == "strict" and gs == "unverified":
            return self._fail("quality_unverifiable")

        # S4 Fix: Gate Cap Applied treating bound composite as quality-falling
        if "gate_cap_applied" in last_scan and pd.notna(last_scan["gate_cap_applied"]):
            return self._fail("quality_falling")

        # 4. Corroboration (PE trailing)
        corroboration = "unavailable"
        pe_scan = df_scan["pe_trailing"].dropna()
        pe_c = None
        if not pe_scan.empty and len(pe_scan) > 60:
            pe_median = pe_scan.median()
            pe_mad = (pe_scan - pe_median).abs().median()
            if 1.4826 * pe_mad >= max(1e-4, 0.02 * abs(pe_median)):
                pe_now = pe_scan.iloc[-1]
                pe_c = -(pe_now - pe_median) / (1.4826 * pe_mad)
                corroboration = (
                    "agrees" if pe_c >= 1.0 else ("diverges" if pe_c < 0 else "unavailable")
                )

        return DipTriggerResult(
            state="fired",
            magnitude=c,
            z_robust=z_robust,
            z_classical=z_classical,
            multiple_used="fcf_yield",
            corroboration=corroboration,
            window={
                "n_valid": len(fcf_series),
                "span_days": (last_obs - first_obs).days,
                "median": median_w,
                "mad": mad_w,
            },
            quality={
                "q_now": q_now,
                "q_base": q_base,
                "delta": delta_q,
                "confidence": last_scan["confidence"],
                "gate_state": gs,
            },
        )

    def _fail(self, state: str) -> DipTriggerResult:
        return DipTriggerResult(
            state=state,
            magnitude=None,
            z_robust=None,
            z_classical=None,
            multiple_used="fcf_yield",
            corroboration=None,
            window={},
            quality={},
        )


@dataclass(frozen=True)
class IncrementalRoicResult:
    value: Optional[float]
    state: str
    n: int
    ic_range_rel: float
    nopat_trend: float
    window: str = "12q"
    flags: List[str] = None


class IncrementalRoicAnalyzer:
    """§D Algorithm 3: Incremental ROIC (Diagnostic Only)."""

    def analyze(self, nopat: pd.Series, ic: pd.Series) -> IncrementalRoicResult:
        flags = []
        union = nopat.index.union(ic.index).sort_values()
        ic_ff = ic.reindex(union).ffill()
        avg_ic = ic_ff.rolling(2, min_periods=1).mean()

        aligned = pd.concat({"nopat": nopat.reindex(union), "ic": avg_ic}, axis=1).dropna()
        aligned = aligned[aligned["ic"] > 0]
        aligned = aligned.tail(12)
        n = len(aligned)

        if n < 8:
            return IncrementalRoicResult(None, "insufficient_points", n, 0.0, 0.0, flags=flags)

        ic_vals = aligned["ic"].values
        ic_range_rel = (ic_vals.max() - ic_vals.min()) / ic_vals.mean()

        if ic_range_rel < 0.05:
            return IncrementalRoicResult(None, "capital_base_static", n, ic_range_rel, 0.0, flags=flags)

        x_days = (pd.DatetimeIndex(aligned.index) - pd.DatetimeIndex(aligned.index)[0]).days.values
        ic_slope = np.polyfit(x_days, ic_vals, 1)[0]
        nopat_slope = np.polyfit(x_days, aligned["nopat"].values, 1)[0]
        nopat_trend_yr = nopat_slope * 365.25

        if ic_slope < 0:
            return IncrementalRoicResult(
                None, "capital_returning", n, ic_range_rel, nopat_trend_yr, flags=flags
            )

        marginal_roic = np.polyfit(ic_vals, aligned["nopat"].values, 1)[0]

        if marginal_roic < -1.0 or marginal_roic > 1.0:
            flags.append("clamped")
            marginal_roic = max(-1.0, min(1.0, marginal_roic))

        return IncrementalRoicResult(
            marginal_roic, "valid", n, ic_range_rel, nopat_trend_yr, flags=flags
        )


class ScoreBacktester:
    """§C Algorithm 2: Score Backtest with Newey-West standard errors."""

    def __init__(self, store):
        self.store = store

    def run_backtest(self, horizon_months: int = 12) -> Dict[str, Any]:
        df_history = self._load_scan_history()
        if df_history.empty:
            return {"error": "insufficient_sample"}

        df_history["month_idx"] = df_history["evaluated_at"].dt.to_period("M")
        grid_dates = df_history.groupby("month_idx")["evaluated_at"].min()

        results_full = []
        results_filtered = []

        for month, t_start in grid_dates.items():
            mask = df_history["evaluated_at"] <= t_start
            period_data = (
                df_history[mask].sort_values("evaluated_at").groupby("symbol").last().reset_index()
            )

            if len(period_data) < 100:
                continue

            scores = []
            for raw_str in period_data["raw_json"]:
                raw_dict = json.loads(raw_str) if pd.notna(raw_str) else {}
                res = score_metrics(raw_dict, registry=DEFAULT_REGISTRY)
                scores.append(res.composite)
            period_data["pit_score"] = scores

            outcomes = self._get_forward_returns(period_data["symbol"], t_start, horizon_months)
            if outcomes.empty:
                continue

            merged = period_data.merge(outcomes, on="symbol").dropna(
                subset=["pit_score", "forward_return"]
            )

            if not merged.empty:
                results_full.append(self._calc_grid_metrics(merged, horizon_months))

            subset = merged[(merged["confidence"] >= 0.75) & (merged["gate_state"] == "pass")]
            if not subset.empty:
                results_filtered.append(self._calc_grid_metrics(subset, horizon_months))

        if len(results_full) < 6:
            return {"error": "insufficient_sample"}

        return {
            "horizon_months": horizon_months,
            "full_universe": self._aggregate_grid_results(results_full),
            "high_quality_subset": self._aggregate_grid_results(results_filtered),
        }

    def _calc_grid_metrics(self, df: pd.DataFrame, horizon_months: int) -> Dict[str, Any]:
        ic, t_stat = self.rank_ic_newey_west(df["pit_score"], df["forward_return"], horizon_months)

        df["quintile"] = pd.qcut(df["pit_score"], 5, labels=False, duplicates="drop")
        q_means = df.groupby("quintile")["forward_return"].mean().to_dict()
        top_minus_bottom = q_means.get(4, 0) - q_means.get(0, 0)

        return {
            "ic": ic,
            "t_stat": t_stat,
            "quintile_means": q_means,
            "hit": top_minus_bottom > 0,
        }

    def _aggregate_grid_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_ic = [r["ic"] for r in results if r["ic"] is not None]
        valid_t = [r["t_stat"] for r in results if r["t_stat"] is not None]
        hits = sum(1 for r in results if r["hit"])

        return {
            "mean_ic": np.mean(valid_ic) if valid_ic else None,
            "mean_t_stat": np.mean(valid_t) if valid_t else None,
            "hit_rate": hits / len(results) if results else 0.0,
            "grid_dates_evaluated": len(results),
        }

    def _load_scan_history(self) -> pd.DataFrame:
        query = "SELECT * FROM sp500_scores_history WHERE source = 'scan'"
        try:
            with self.store.lock:
                df = pd.read_sql_query(query, self.store.conn)
            if not df.empty:
                df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)
            return df
        except Exception:
            return pd.DataFrame()

    def _get_forward_returns(
        self, symbols: pd.Series, t_start: pd.Timestamp, horizon_months: int
    ) -> pd.DataFrame:
        t_end = t_start + pd.DateOffset(months=horizon_months)
        sym_list = tuple(symbols.tolist())

        query = f"""
            SELECT symbol, date, close, truncated_window 
            FROM price_history 
            WHERE symbol IN ({','.join(['?']*len(sym_list))})
            AND date >= ? AND date <= ?
        """
        try:
            with self.store.lock:
                prices = pd.read_sql_query(
                    query,
                    self.store.conn,
                    params=(*sym_list, t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d")),
                )
        except Exception:
            return pd.DataFrame()

        if prices.empty:
            return pd.DataFrame()

        prices["date"] = pd.to_datetime(prices["date"], utc=True)
        returns = []

        for sym, group in prices.groupby("symbol"):
            group = group.sort_values("date")
            if len(group) < 2:
                continue

            entry_price = group.iloc[0]["close"]
            exit_row = group.iloc[-1]
            exit_price = exit_row["close"]

            if exit_row["date"] < t_end and not exit_row.get("truncated_window", 0):
                continue

            returns.append({
                "symbol": sym,
                "forward_return": (exit_price - entry_price) / entry_price,
            })

        return pd.DataFrame(returns)

    @staticmethod
    def rank_ic_newey_west(predictor: pd.Series, outcome: pd.Series, horizon_months: int) -> Tuple[Optional[float], Optional[float]]:
        df = pd.concat({"x": predictor, "y": outcome}, axis=1).dropna()
        if len(df) < 100:
            return None, None

        rx = df["x"].rank(method="average")
        ry = df["y"].rank(method="average")
        ic_val = rx.corr(ry, method="pearson")
        ic = float(ic_val) if pd.notna(ic_val) else None
        if ic is None:
            return None, None

        lag = max(0, int(horizon_months) - 1)
        try:
            import statsmodels.api as sm  # type: ignore[import-untyped]

            X_sm = sm.add_constant(rx)
            model = sm.OLS(ry, X_sm)
            results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
            return ic, float(results.tvalues.iloc[1])
        except ImportError:
            n = len(df)
            x_arr = rx.to_numpy(dtype=float)
            y_arr = ry.to_numpy(dtype=float)
            X = np.column_stack([np.ones(n, dtype=float), x_arr])
            beta, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
            u = y_arr - X @ beta
            scores = X * u[:, None]
            omega = scores.T @ scores
            max_lag = min(lag, n - 1)
            for ell in range(1, max_lag + 1):
                weight = 1.0 - (ell / (lag + 1.0))
                gamma = scores[ell:].T @ scores[:-ell]
                omega = omega + weight * (gamma + gamma.T)
            bread = np.linalg.pinv(X.T @ X)
            dof_scale = n / max(1, n - X.shape[1])
            cov_hac = dof_scale * (bread @ omega @ bread)
            var_slope = float(cov_hac[1, 1])
            if var_slope <= 0.0:
                return ic, 0.0
            t_stat = float(beta[1] / math.sqrt(var_slope))
            return ic, t_stat

