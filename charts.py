from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .analysis import _pick_series, safe_div

CHART_OPTIONS = [
    "Price",
    "Revenue & EBITDA",
    "EPS & Dividends",
    "Cash vs Debt",
    "Margins (Gross/Op/Net)",
    "Valuation (P/E)",
    "Valuation (FCF Yield)",
    "Valuation (P/S)",
    "Cash Flow Detail (FCF vs SBC)",
    "FCF Per Share",
]
PRICE_PERIOD_TO_YF = {
    "1D": "1d",
    "5D": "5d",
    "1W": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
    "10Y": "10y",
    "YTD": "ytd",
    "ALL": "max",
    "MAX": "max",
}

def price_period_to_yf(label: str) -> str:
    return PRICE_PERIOD_TO_YF.get(label, "1y")


def detect_key_moments(
    price_df: Optional[pd.DataFrame],
    z_threshold: float = 3.0,
    max_moments: int = 5,
    events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Detect statistically significant daily return anomalies using robust MAD z-scores
    and attach nearby catalyst labels within +/- 2 calendar days.
    """
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return []
    close = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    close = close[close > 0]
    if len(close) < 3:
        return []

    returns = close.pct_change().dropna()
    if returns.empty:
        return []

    r_vals = returns.values.astype(float)
    med = float(np.median(r_vals))
    mad = float(np.median(np.abs(r_vals - med)))
    denom = 1.4826 * mad
    if denom <= 1e-12:
        std = float(np.std(r_vals))
        denom = std if std > 1e-12 else 1.0

    z_scores = (r_vals - med) / denom

    parsed_events: List[Dict[str, Any]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        raw_dt = ev.get("date") or ev.get("filing_date")
        if not raw_dt:
            continue
        try:
            dt_val = pd.Timestamp(raw_dt).tz_localize(None)
        except Exception:
            continue
        lbl = ev.get("label") or ev.get("catalyst") or ev.get("form") or "Event"
        parsed_events.append({"dt": dt_val, "label": str(lbl)})

    moments: List[Dict[str, Any]] = []
    for idx_ts, r_t, z_t in zip(returns.index, r_vals, z_scores):
        if not np.isfinite(z_t) or abs(z_t) < z_threshold:
            continue
        try:
            ts_naive = pd.Timestamp(idx_ts).tz_localize(None)
        except Exception:
            ts_naive = pd.Timestamp(idx_ts)

        matched_label: Optional[str] = None
        best_delta = pd.Timedelta(days=999)
        for pev in parsed_events:
            delta = abs(ts_naive - pev["dt"])
            if delta <= pd.Timedelta(days=2) and delta < best_delta:
                best_delta = delta
                matched_label = f"{pev['label']} ({r_t:+.1%}, z={z_t:+.1f}σ)"

        if not matched_label:
            matched_label = f"Return spike ({r_t:+.1%}, z={z_t:+.1f}σ)"

        moments.append(
            {
                "date": idx_ts,
                "close": float(close.loc[idx_ts]),
                "return_pct": float(r_t),
                "z_score": float(z_t),
                "direction": "up" if r_t >= 0 else "down",
                "label": matched_label,
            }
        )

    moments.sort(key=lambda m: abs(m["z_score"]), reverse=True)
    return moments[: max(0, int(max_moments))]


def normalize_peer_overlays(
    primary_close: pd.Series,
    peer_Map: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None,
    **kwargs: Any,
) -> Dict[str, pd.Series]:
    """
    Align primary_close and each peer series to their overlapping date window
    and normalize each series to percentage change from its initial value:
    pct_t = ((P_t / P_0) - 1.0) * 100.0
    """
    peers = peer_Map if peer_Map is not None else kwargs.get("peer_map") or {}
    if primary_close is None:
        return {}
    if isinstance(primary_close, pd.DataFrame):
        if "Close" not in primary_close.columns:
            return {}
        p_s = pd.to_numeric(primary_close["Close"], errors="coerce").dropna()
    else:
        p_s = pd.to_numeric(primary_close, errors="coerce").dropna()

    p_s = p_s[p_s > 0].copy()
    if p_s.empty:
        return {}
    p_s.index = pd.to_datetime(p_s.index).tz_localize(None)
    p_s = p_s[~p_s.index.duplicated(keep="last")].sort_index()

    cleaned_peers: Dict[str, pd.Series] = {}
    for sym, raw_s in (peers or {}).items():
        if raw_s is None:
            continue
        if isinstance(raw_s, pd.DataFrame):
            if "Close" not in raw_s.columns:
                continue
            s = pd.to_numeric(raw_s["Close"], errors="coerce").dropna()
        else:
            s = pd.to_numeric(raw_s, errors="coerce").dropna()
        s = s[s > 0].copy()
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        cleaned_peers[str(sym)] = s

    # Determine common start date across primary and peers with overlap
    common_start = p_s.index.min()
    common_end = p_s.index.max()
    for s in cleaned_peers.values():
        if s.index.max() >= common_start and s.index.min() <= common_end:
            common_start = max(common_start, s.index.min())
            common_end = min(common_end, s.index.max())

    p_win = p_s[(p_s.index >= common_start) & (p_s.index <= common_end)]
    if p_win.empty:
        p_win = p_s
    p0 = float(p_win.iloc[0])
    out: Dict[str, pd.Series] = {
        "__primary__": ((p_win / p0) - 1.0) * 100.0 if p0 != 0 else p_win * 0.0
    }

    for sym, s in cleaned_peers.items():
        s_win = s[(s.index >= p_win.index.min()) & (s.index <= p_win.index.max())]
        if s_win.empty:
            continue
        s0 = float(s_win.iloc[0])
        if s0 == 0:
            continue
        out[sym] = ((s_win / s0) - 1.0) * 100.0

    return out


def apply_chart_layout(fig):
    try:
        fig.subplots_adjust(bottom=0.20, left=0.11, right=0.97, top=0.88)
    except Exception:
        pass

def _freq_code(freq: str):
    return "Y" if freq == "Annual" else "Q"

def group_series(s: Optional[pd.Series], freq: str, agg: str = "last"):
    if s is None or s.empty:
        return s
    idx = pd.to_datetime(s.index).tz_localize(None)
    pidx = pd.PeriodIndex(idx, freq=_freq_code(freq))
    df = pd.DataFrame({"v": s.values}, index=pidx)
    if agg == "sum":
        out = df.groupby(level=0).sum()["v"]
    else:
        out = df.groupby(level=0).last()["v"]
    out.index = out.index.to_timestamp()
    return out

def last_n_periods(s: Optional[pd.Series], n: int = 4, freq: str = "Quarterly"):
    if s is None or s.empty:
        return s
    freq_code = _freq_code(freq)
    pidx = pd.PeriodIndex(pd.to_datetime(s.index).tz_localize(None), freq=freq_code)
    last_p = pidx.max()
    if last_p is pd.NaT:
        return s
    periods = pd.period_range(end=last_p, periods=n, freq=freq_code)
    s2 = pd.Series(s.values, index=pidx).groupby(level=0).last()
    s2 = s2.reindex(periods)
    s2.index = s2.index.to_timestamp()
    return s2

def ttm_series(s_q: Optional[pd.Series]):
    s_q = group_series(s_q, "Quarterly", agg="last")
    if s_q is None or s_q.empty:
        return s_q
    pidx = pd.PeriodIndex(pd.to_datetime(s_q.index).tz_localize(None), freq="Q")
    full = pd.period_range(start=pidx.min(), end=pidx.max(), freq="Q")
    s_full = pd.Series(s_q.values, index=pidx).reindex(full)
    ttm = s_full.rolling(4, min_periods=4).sum().dropna()
    if ttm.empty:
        return ttm
    ttm = ttm.tail(4)
    ttm.index = ttm.index.to_timestamp()
    return ttm

def series_for_flow(s_q: Optional[pd.Series], s_a: Optional[pd.Series], freq: str):
    if freq == "Annual":
        if s_a is not None and not s_a.empty:
            return group_series(s_a, "Annual", agg="sum")
        return group_series(s_q, "Annual", agg="sum")
    if freq == "TTM":
        return ttm_series(s_q)
    return group_series(s_q, "Quarterly", agg="last")

def series_for_balance(s_q: Optional[pd.Series], s_a: Optional[pd.Series], freq: str):
    if freq == "Annual":
        if s_a is not None and not s_a.empty:
            return group_series(s_a, "Annual", agg="last")
        return group_series(s_q, "Annual", agg="last")
    s = group_series(s_q, "Quarterly", agg="last")
    if freq == "TTM":
        return last_n_periods(s, n=4, freq="Quarterly")
    return s

def dividend_series(divs: Optional[pd.Series], freq: str):
    if divs is None or divs.empty:
        return None
    base_freq = "Quarterly" if freq == "TTM" else freq
    idx = pd.to_datetime(divs.index).tz_localize(None)
    pidx = pd.PeriodIndex(idx, freq=_freq_code(base_freq))
    s = pd.Series(divs.values, index=pidx).groupby(level=0).sum()
    if freq == "TTM":
        s = s.rolling(4, min_periods=4).sum().dropna()
        if s.empty:
            return s
        s = s.tail(4)
    s.index = s.index.to_timestamp()
    return s


@dataclass(frozen=True)
class ChartPayload:
    fig: Any
    ax: Any
    mode: str = "Price"
    freq: str = "Quarterly"
    price_range: str = "1Y"
    hist_period: Optional[str] = "1y"
    hist_data: Optional[pd.DataFrame] = None
    hist_data_long: Optional[pd.DataFrame] = None
    df_fin_q: Optional[pd.DataFrame] = None
    df_fin_a: Optional[pd.DataFrame] = None
    df_bal_q: Optional[pd.DataFrame] = None
    df_bal_a: Optional[pd.DataFrame] = None
    df_cf_q: Optional[pd.DataFrame] = None
    df_cf_a: Optional[pd.DataFrame] = None
    df_dividends: Optional[pd.Series] = None
    anim_enabled: bool = False
    price_chart_style: str = "line"
    show_ema_20: bool = False
    show_volume: bool = False
    show_key_moments: bool = False
    show_events: bool = False
    peer_overlays: Optional[Dict[str, pd.DataFrame]] = None
    event_markers: Optional[List[Dict[str, Any]]] = None
    on_reset: Optional[Callable[[], None]] = None
    on_price_change: Optional[Callable[[str, Optional[str]], None]] = None
    on_ensure_history_period: Optional[Callable[[str], None]] = None
    on_analysis: Optional[Callable[[str, str], None]] = None
    on_draw_idle: Optional[Callable[[], None]] = None
    on_anim_created: Optional[Callable[[Any], None]] = None
    on_ax2_created: Optional[Callable[[Any], None]] = None

    @classmethod
    def from_app(cls, app: Any) -> "ChartPayload":
        def _reset_app():
            app._redraw_job = None
            if hasattr(app, "txt_chart_ai"):
                app._update_chart_ai("Refreshing chart insights...")
            if getattr(app, "anim", None) and app.anim.event_source:
                try:
                    app.anim.event_source.stop()
                except Exception:
                    pass
            app.anim = None
            if getattr(app, "_ax2", None) is not None:
                try:
                    app._ax2.remove()
                except Exception:
                    pass
                app._ax2 = None
            if hasattr(app, "lbl_price_change"):
                app.lbl_price_change.configure(text="")

        def _price_change(text: str, text_color: Optional[str] = None):
            if hasattr(app, "lbl_price_change"):
                if text_color is not None:
                    app.lbl_price_change.configure(text=text, text_color=text_color)
                else:
                    app.lbl_price_change.configure(text=text)

        return cls(
            fig=app.fig,
            ax=app.ax,
            mode=app.chart_var.get() if hasattr(app, "chart_var") else "Price",
            freq=app.freq_var.get() if hasattr(app, "freq_var") else "Quarterly",
            price_range=app.price_range_var.get() if hasattr(app, "price_range_var") else "1Y",
            hist_period=getattr(app, "hist_period", "1y"),
            hist_data=getattr(app, "hist_data", None),
            hist_data_long=getattr(app, "hist_data_long", None),
            df_fin_q=getattr(app, "df_fin_q", None),
            df_fin_a=getattr(app, "df_fin_a", None),
            df_bal_q=getattr(app, "df_bal_q", None),
            df_bal_a=getattr(app, "df_bal_a", None),
            df_cf_q=getattr(app, "df_cf_q", None),
            df_cf_a=getattr(app, "df_cf_a", None),
            df_dividends=getattr(app, "df_dividends", None),
            anim_enabled=bool(app.anim_enabled.get()) if hasattr(app, "anim_enabled") else False,
            price_chart_style=getattr(app, "price_chart_style", "line"),
            show_ema_20=bool(getattr(app, "show_ema_20", False)),
            show_volume=bool(getattr(app, "show_volume", False)),
            show_key_moments=bool(getattr(app, "show_key_moments", False)),
            show_events=bool(getattr(app, "show_events", False)),
            peer_overlays=getattr(app, "peer_overlays", None),
            event_markers=getattr(app, "event_markers", None),
            on_reset=_reset_app,
            on_price_change=_price_change,
            on_ensure_history_period=getattr(app, "ensure_history_period", None),
            on_analysis=getattr(app, "_set_chart_analysis", None),
            on_draw_idle=app.canvas.draw_idle if hasattr(app, "canvas") else None,
            on_anim_created=lambda anim_obj: setattr(app, "anim", anim_obj),
            on_ax2_created=lambda ax2_obj: setattr(app, "_ax2", ax2_obj),
        )


class ChartRenderer:
    def __init__(self, payload: Union[ChartPayload, Any], *, palette, anim_frames: int):
        self.payload: ChartPayload = (
            payload if isinstance(payload, ChartPayload) else ChartPayload.from_app(payload)
        )
        self._ax2 = None
        self.anim = None
        self.C_ACCENT = palette["C_ACCENT"]
        self.C_BORDER = palette["C_BORDER"]
        self.C_GREEN = palette["C_GREEN"]
        self.C_RED = palette["C_RED"]
        self.C_YELLOW = palette["C_YELLOW"]
        self.C_PINK = palette["C_PINK"]
        self.C_BG_CARD = palette["C_BG_CARD"]
        self.C_TEXT_MAIN = palette["C_TEXT_MAIN"]
        self.C_TEXT_SUB = palette["C_TEXT_SUB"]
        self.C_PLOT_FRAME = palette["C_PLOT_FRAME"]
        self.C_SERIES_NEUTRAL = palette["C_SERIES_NEUTRAL"]
        self.anim_frames = anim_frames

    def _get_series(self, df: Optional[pd.DataFrame], keys) -> Optional[pd.Series]:
        return _pick_series(df, keys)

    def _get_freq_df(self, kind: str, freq: str) -> Optional[pd.DataFrame]:
        is_annual = freq == "Annual"
        if kind == "fin":
            return self.payload.df_fin_a if is_annual else self.payload.df_fin_q
        if kind == "bal":
            return self.payload.df_bal_a if is_annual else self.payload.df_bal_q
        if kind == "cf":
            return self.payload.df_cf_a if is_annual else self.payload.df_cf_q
        return None

    def _create_ax2(self):
        ax2 = self.payload.ax.twinx()
        self._ax2 = ax2
        if self.payload.on_ax2_created:
            self.payload.on_ax2_created(ax2)
        return ax2

    def _set_anim(self, anim_obj):
        self.anim = anim_obj
        if self.payload.on_anim_created:
            self.payload.on_anim_created(anim_obj)

    def _freq_code(self, freq: str):
        return "Y" if freq == "Annual" else "Q"

    def _format_period_labels(self, periods: pd.PeriodIndex, freq: str):
        if freq == "Annual":
            return [str(p.year) for p in periods]
        return [f"{p.year}-Q{p.quarter}" for p in periods]

    def apply_axes_frame(self, axis):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(self.C_PLOT_FRAME)
            spine.set_linewidth(1.0)

    def style_axes(self, title=None, y_label=None):
        ax = self.payload.ax
        ax.set_facecolor(self.C_BG_CARD)
        ax.grid(True, linestyle="--", alpha=0.18)
        self.apply_axes_frame(ax)
        ax.tick_params(axis="both", colors=self.C_TEXT_SUB, labelsize=9)
        if title:
            ax.set_title(title, fontweight="bold", color=self.C_TEXT_MAIN, pad=8)
        if y_label:
            ax.set_ylabel(y_label, color=self.C_TEXT_SUB)

    def set_period_ticks(self, x_idx, labels, max_ticks=8):
        ax = self.payload.ax
        if len(labels) == 0:
            return
        step = max(1, int(len(labels) / max_ticks))
        ticks = x_idx[::step]
        tick_labels = [labels[i] for i in range(0, len(labels), step)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    def set_date_ticks(self, x_index):
        ax = self.payload.ax
        if x_index is None or len(x_index) == 0:
            return
        try:
            x = pd.DatetimeIndex(pd.to_datetime(x_index)).tz_localize(None)
        except Exception:
            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            return
        x = x.sort_values().unique()
        if len(x) == 0:
            return
        if len(x) == 1:
            locator = mdates.MinuteLocator(interval=30)
            formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            return
        span = x[-1] - x[0]
        if span <= pd.Timedelta(days=2):
            locator = mdates.HourLocator(interval=2)
        elif span <= pd.Timedelta(days=31):
            locator = mdates.DayLocator(interval=max(1, int(np.ceil(span.days / 8))))
        elif span <= pd.Timedelta(days=365 * 3):
            locator = mdates.MonthLocator(interval=max(1, int(np.ceil(span.days / 240))))
        else:
            years = max(1, int(np.ceil(span.days / 365)))
            locator = mdates.YearLocator(base=max(1, int(np.ceil(years / 10))))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def set_date_xlim(self, x_index, pad_hours: int = 6):
        if x_index is None or len(x_index) == 0:
            return
        try:
            x = pd.DatetimeIndex(pd.to_datetime(x_index)).tz_localize(None).sort_values()
        except Exception:
            return
        if len(x) == 0:
            return
        left = x[0]
        right = x[-1]
        if left == right:
            pad = pd.Timedelta(hours=pad_hours)
            left = left - pad
            right = right + pad
        self.payload.ax.set_xlim(left, right)

    def add_vertical_guides(self, x_positions):
        if len(x_positions) > 40:
            return
        for x in x_positions:
            self.payload.ax.axvline(x, color=self.C_BORDER, lw=0.5, alpha=0.12, zorder=0, linestyle="--")

    def to_period_index(self, idx, freq: str):
        freq_code = self._freq_code(freq)
        return pd.PeriodIndex(pd.to_datetime(idx).tz_localize(None), freq=freq_code)

    def safe_ylim(self, y_min, y_max, pad=0.08, zero_floor=False):
        ax = self.payload.ax
        if not np.isfinite(y_min):
            y_min = 0.0
        if not np.isfinite(y_max):
            y_max = 1.0
        if zero_floor:
            y_min = min(0.0, y_min)
        if y_max == y_min:
            y_max = y_min + 1.0
        span = y_max - y_min
        ax.set_ylim(y_min - span * pad, y_max + span * pad)

    def align_data(self, primary_series, secondary_series=None, fill_value=0.0, freq="Quarterly"):
        if primary_series is None or primary_series.empty:
            return [], np.array([]), np.array([])
        freq_code = self._freq_code(freq)
        def clean_series(s, col_name):
            s = s.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            df = s.to_frame(name=col_name)
            df["period"] = df.index.to_period(freq_code)
            return df.groupby("period").last()
        df_p = clean_series(primary_series, "p")
        if secondary_series is not None and not secondary_series.empty:
            df_s = clean_series(secondary_series, "s")
            df_merged = pd.merge(df_p, df_s, on="period", how="left")
        else:
            df_merged = df_p.copy()
            df_merged["s"] = np.nan
        df_merged["s"] = df_merged["s"].fillna(fill_value)
        df_merged = df_merged.sort_index()
        periods = df_merged.index
        labels = self._format_period_labels(periods, freq)
        return labels, df_merged["p"].values, df_merged["s"].values

    def series_stats(self, values, label: str, unit: str = ""):
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return f"{label}: no data."
        start = arr[0]
        end = arr[-1]
        minv = arr.min()
        maxv = arr.max()
        pct = (end - start) / abs(start) * 100 if start not in (0, 0.0) else 0.0
        return f"{label}: start {start:.2f}{unit}, end {end:.2f}{unit}, change {pct:+.1f}%, min {minv:.2f}{unit}, max {maxv:.2f}{unit}."

    def draw_no_data(self, msg="No data available"):
        self.style_axes()
        self.payload.ax.text(0.5, 0.5, msg, color=self.C_TEXT_SUB, ha="center", va="center", transform=self.payload.ax.transAxes)

    def _group_series(self, s, freq: str, agg: str = "last"):
        return group_series(s, freq, agg)

    def _ttm_series(self, s_q):
        return ttm_series(s_q)

    def _last_n_periods(self, s, n: int = 4, freq: str = "Quarterly"):
        return last_n_periods(s, n=n, freq=freq)

    def series_for_flow(self, s_q, s_a, freq: str):
        return series_for_flow(s_q, s_a, freq)

    def series_for_balance(self, s_q, s_a, freq: str):
        return series_for_balance(s_q, s_a, freq)

    def dividend_series(self, freq: str):
        return dividend_series(self.payload.df_dividends, freq)

    def to_ttm(self, s):
        if s is None or s.empty:
            return s
        s = s.sort_index()
        return s.rolling(4).sum()

    def _reset(self):
        p = self.payload
        if p.on_reset:
            p.on_reset()
        self.anim = None
        if self._ax2 is not None:
            try:
                self._ax2.remove()
            except Exception:
                pass
            self._ax2 = None
        try:
            for extra_ax in list(p.fig.axes):
                if extra_ax is not p.ax:
                    extra_ax.remove()
        except Exception:
            pass
        p.ax.clear()

    def _legend(self, handles=None, labels=None):
        if handles is not None and labels is not None:
            self.payload.ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
        else:
            self.payload.ax.legend(loc="upper left", frameon=False, fontsize=8)

    def render(self):
        self._reset()
        p = self.payload
        mode = p.mode
        freq = p.freq
        dispatch = {
            "Price": lambda: self._render_price(),
            "Revenue & EBITDA": lambda: self._render_revenue_ebitda(freq),
            "EPS & Dividends": lambda: self._render_eps_dividends(freq),
            "Cash vs Debt": lambda: self._render_cash_vs_debt(freq),
            "Margins (Gross/Op/Net)": lambda: self._render_margins(freq),
            "Valuation (P/E)": lambda: self._render_valuation(mode, freq),
            "Valuation (FCF Yield)": lambda: self._render_valuation(mode, freq),
            "Valuation (P/S)": lambda: self._render_valuation(mode, freq),
            "Cash Flow Detail (FCF vs SBC)": lambda: self._render_cash_flow_detail(freq),
            "FCF Per Share": lambda: self._render_fcf_per_share(freq),
        }
        handler = dispatch.get(mode)
        if handler:
            handler()
        else:
            self.draw_no_data("Unsupported chart mode")
            apply_chart_layout(p.fig)
        if p.on_draw_idle:
            p.on_draw_idle()

    def _render_price(self):
        p = self.payload
        if p.on_price_change:
            p.on_price_change("", getattr(self, "C_TEXT_SEC", self.C_TEXT_SUB))
        desired = price_period_to_yf(p.price_range)
        if p.hist_period != desired:
            if p.on_ensure_history_period:
                p.on_ensure_history_period(desired)
            self.draw_no_data(f"Loading {p.price_range} data...")
            return
        base = p.hist_data
        if (base is None or base.empty) and desired == "max":
            hist = p.hist_data_long
            if hist is not None and not hist.empty:
                base = hist
        if base is None or base.empty:
            self.draw_no_data("No Price Data")
            return
        valid_data = base[base["Close"] > 0].copy()
        if valid_data.empty:
            self.draw_no_data("No Price Data")
            apply_chart_layout(p.fig)
            return
        x = valid_data.index
        y = valid_data["Close"].values.astype(float)
        y = np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = np.isfinite(y)
        if mask.sum() == 0:
            self.draw_no_data("No Price Data")
            return
        x = x[mask]
        y = y[mask]
        if y.max() <= 0:
            self.draw_no_data("No Positive Price Data")
            return

        # 1. Peer percentage overlays mode
        if p.peer_overlays:
            norm_map = normalize_peer_overlays(valid_data["Close"], p.peer_overlays)
            if norm_map:
                self.style_axes("RELATIVE PERFORMANCE (% CHANGE)", "% Change")
                primary_s = norm_map.get("__primary__")
                if primary_s is not None and not primary_s.empty:
                    self.set_date_ticks(primary_s.index)
                    p.ax.plot(
                        primary_s.index,
                        primary_s.values,
                        color=self.C_ACCENT,
                        linewidth=2.0,
                        label="Primary",
                    )
                peer_colors = [self.C_GREEN, self.C_YELLOW, self.C_PINK, self.C_RED, self.C_SERIES_NEUTRAL]
                all_vals: List[float] = list(primary_s.values) if primary_s is not None else []
                c_idx = 0
                for peer_sym, peer_s in norm_map.items():
                    if peer_sym == "__primary__" or peer_s.empty:
                        continue
                    col = peer_colors[c_idx % len(peer_colors)]
                    c_idx += 1
                    p.ax.plot(peer_s.index, peer_s.values, color=col, linewidth=1.5, alpha=0.9, label=str(peer_sym))
                    all_vals.extend(list(peer_s.values))
                p.ax.axhline(0.0, color=self.C_BORDER, linestyle="--", linewidth=0.9, alpha=0.7)
                if all_vals:
                    arr_all = np.array(all_vals, dtype=float)
                    self.safe_ylim(float(np.nanmin(arr_all)), float(np.nanmax(arr_all)), pad=0.08, zero_floor=False)
                if primary_s is not None and not primary_s.empty:
                    self.set_date_xlim(primary_s.index, pad_hours=6)
                handles, labels = p.ax.get_legend_handles_labels()
                if handles:
                    self._legend(handles, labels)
                apply_chart_layout(p.fig)
                return

        self.style_axes("PRICE HISTORY", "Price ($)")
        self.set_date_ticks(x)

        is_candlestick = (
            str(p.price_chart_style or "line").lower() == "candlestick"
            and all(col in valid_data.columns for col in ("Open", "High", "Low", "Close"))
        )
        line = None
        if is_candlestick:
            cdf = valid_data.loc[x]
            opens = pd.to_numeric(cdf["Open"], errors="coerce").values.astype(float)
            highs = pd.to_numeric(cdf["High"], errors="coerce").values.astype(float)
            lows = pd.to_numeric(cdf["Low"], errors="coerce").values.astype(float)
            closes = pd.to_numeric(cdf["Close"], errors="coerce").values.astype(float)
            up_mask = closes >= opens
            down_mask = ~up_mask
            if up_mask.any():
                p.ax.vlines(x[up_mask], lows[up_mask], highs[up_mask], color=self.C_GREEN, linewidth=1.0, alpha=0.85)
                p.ax.vlines(
                    x[up_mask],
                    np.minimum(opens[up_mask], closes[up_mask]),
                    np.maximum(opens[up_mask], closes[up_mask]),
                    color=self.C_GREEN,
                    linewidth=3.5,
                    label="OHLC (Up)",
                )
            if down_mask.any():
                p.ax.vlines(x[down_mask], lows[down_mask], highs[down_mask], color=self.C_RED, linewidth=1.0, alpha=0.85)
                p.ax.vlines(
                    x[down_mask],
                    np.minimum(opens[down_mask], closes[down_mask]),
                    np.maximum(opens[down_mask], closes[down_mask]),
                    color=self.C_RED,
                    linewidth=3.5,
                    label="OHLC (Down)",
                )
        else:
            line_kwargs = {"color": self.C_ACCENT, "linewidth": 1.8}
            if len(x) == 1:
                line_kwargs.update({"marker": "o", "markersize": 5})
            line, = p.ax.plot([], [], **line_kwargs)
            line.set_data(x[:1], y[:1])

        if "SMA_50" in valid_data.columns:
            s50 = pd.to_numeric(valid_data["SMA_50"], errors="coerce").reindex(x)
            if np.isfinite(s50.values).any():
                p.ax.plot(x, s50.values, color=self.C_GREEN, linewidth=1.0, alpha=0.8, label="SMA 50")
        if "SMA_200" in valid_data.columns:
            s200 = pd.to_numeric(valid_data["SMA_200"], errors="coerce").reindex(x)
            if np.isfinite(s200.values).any():
                p.ax.plot(x, s200.values, color=self.C_YELLOW, linewidth=1.0, alpha=0.8, label="SMA 200")
        if p.show_ema_20 or "EMA_20" in valid_data.columns:
            if "EMA_20" in valid_data.columns:
                ema20 = pd.to_numeric(valid_data["EMA_20"], errors="coerce").reindex(x)
            else:
                ema20 = pd.Series(y, index=x).ewm(span=20, adjust=False).mean()
            if np.isfinite(ema20.values).any():
                p.ax.plot(x, ema20.values, color=self.C_PINK, linewidth=1.1, linestyle="--", alpha=0.9, label="EMA 20")

        if p.show_volume and "Volume" in valid_data.columns:
            vol_s = pd.to_numeric(valid_data["Volume"], errors="coerce").reindex(x).fillna(0.0)
            max_vol = float(vol_s.max()) if not vol_s.empty else 0.0
            if max_vol > 0:
                ax2 = self._create_ax2()
                ax2.set_facecolor("none")
                ax2.bar(x, vol_s.values, color=self.C_SERIES_NEUTRAL, alpha=0.22, label="Volume")
                ax2.set_ylim(0.0, max_vol * 4.0)
                ax2.set_yticks([])

        if p.show_key_moments:
            moments = detect_key_moments(valid_data.loc[x], events=p.event_markers)
            if moments:
                m_dates = [m["date"] for m in moments]
                m_closes = [m["close"] for m in moments]
                p.ax.scatter(
                    m_dates,
                    m_closes,
                    color=self.C_YELLOW,
                    edgecolors=self.C_TEXT_MAIN,
                    s=42,
                    zorder=5,
                    label="Key Moments",
                )
                for m in moments[:3]:
                    p.ax.annotate(
                        m["label"],
                        xy=(m["date"], m["close"]),
                        xytext=(0, 8 if m["direction"] == "up" else -12),
                        textcoords="offset points",
                        fontsize=7,
                        color=self.C_TEXT_MAIN,
                        ha="center",
                    )

        if p.show_events:
            ev_dates = []
            if p.df_dividends is not None and not p.df_dividends.empty:
                try:
                    div_idx = pd.to_datetime(p.df_dividends.index).tz_localize(None)
                    x_naive = pd.to_datetime(x).tz_localize(None)
                    for d in div_idx:
                        if x_naive.min() <= d <= x_naive.max():
                            ev_dates.append(d)
                except Exception:
                    pass
            for ev in p.event_markers or []:
                raw_d = ev.get("date") or ev.get("filing_date") if isinstance(ev, dict) else None
                if raw_d:
                    try:
                        ev_dates.append(pd.Timestamp(raw_d).tz_localize(None))
                    except Exception:
                        pass
            for d in ev_dates[:15]:
                p.ax.axvline(d, color=self.C_PINK, linestyle=":", linewidth=0.9, alpha=0.6)

        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > 0 and (y_max / y_min) > 50:
            q1, q99 = np.nanpercentile(y, [1, 99])
            if np.isfinite(q1) and np.isfinite(q99) and q99 > q1:
                y_min, y_max = q1, q99
        self.safe_ylim(y_min, y_max, pad=0.06, zero_floor=False)
        self.set_date_xlim(x, pad_hours=6)
        if len(y) >= 2 and y[0] != 0:
            pct = (y[-1] - y[0]) / abs(y[0]) * 100
            if p.on_price_change:
                p.on_price_change(
                    f"{p.price_range} Change: {pct:+.2f}%",
                    self.C_GREEN if pct >= 0 else self.C_RED,
                )
        if not is_candlestick and line is not None:
            if p.anim_enabled:
                def update(frame):
                    if len(x) == 0:
                        return line,
                    prog = min(1.0, (frame + 1) / self.anim_frames)
                    limit = int(prog * len(x))
                    limit = max(1, min(len(x), limit))
                    line.set_data(x[:limit], y[:limit])
                    return line,
                self._set_anim(animation.FuncAnimation(
                    p.fig, update, frames=self.anim_frames + 5, interval=20, blit=False, repeat=False
                ))
            else:
                line.set_data(x, y)
                p.ax.fill_between(x, y, color=self.C_ACCENT, alpha=0.08)
        handles, labels = p.ax.get_legend_handles_labels()
        if self._ax2 is not None:
            h2, l2 = self._ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            self._legend(handles, labels)
        summary = self.series_stats(y, "Price", "$")
        if p.on_analysis:
            p.on_analysis(f"Price|{p.price_range}", summary)
        apply_chart_layout(p.fig)

    def _render_revenue_ebitda(self, freq: str):
        p = self.payload
        rev_q = self._get_series(p.df_fin_q, ["TotalRevenue", "OperatingRevenue", "Revenue"])
        rev_a = self._get_series(p.df_fin_a, ["TotalRevenue", "OperatingRevenue", "Revenue"])
        ebitda_q = self._get_series(p.df_fin_q, ["EBITDA", "NormalizedEBITDA"])
        ebitda_a = self._get_series(p.df_fin_a, ["EBITDA", "NormalizedEBITDA"])
        ebitda_label = "EBITDA"
        if ebitda_q is None or ebitda_q.empty:
            ebitda_q = self._get_series(p.df_fin_q, ["OperatingIncome", "OperatingIncomeLoss", "EBIT"])
        if ebitda_a is None or ebitda_a.empty:
            ebitda_a = self._get_series(p.df_fin_a, ["OperatingIncome", "OperatingIncomeLoss", "EBIT"])
        if (ebitda_q is not None and not ebitda_q.empty) or (ebitda_a is not None and not ebitda_a.empty):
            if ebitda_a is None and ebitda_q is not None:
                ebitda_label = "Operating Income"
        rev = self.series_for_flow(rev_q, rev_a, freq)
        ebitda = self.series_for_flow(ebitda_q, ebitda_a, freq)
        if rev is None:
            self.draw_no_data("Revenue data unavailable")
            apply_chart_layout(p.fig)
            return
        dates, vals_rev, vals_ebitda = self.align_data(
            rev / 1e9,
            ebitda / 1e9 if ebitda is not None else None,
            fill_value=np.nan,
            freq=freq,
        )
        has_ebitda = ebitda is not None and np.isfinite(vals_ebitda).any()
        vals_rev = np.nan_to_num(vals_rev, nan=0.0, posinf=0.0, neginf=0.0)
        vals_ebitda = np.nan_to_num(vals_ebitda, nan=0.0, posinf=0.0, neginf=0.0)
        if len(dates) == 0:
            self.draw_no_data("Revenue data unavailable")
            apply_chart_layout(p.fig)
            return
        x_idx = np.arange(len(dates))
        self.style_axes("REVENUE & EBITDA", "USD ($B)")
        self.set_period_ticks(x_idx, dates)
        p.ax.set_xlim(-0.5, len(dates) - 0.5)
        self.add_vertical_guides(x_idx)
        show_ebitda = has_ebitda and np.nanmax(np.abs(vals_ebitda)) > 0
        bars = p.ax.bar(x_idx, [0] * len(dates), color=self.C_SERIES_NEUTRAL, label="Revenue")
        ebitda_line = None
        use_twin = False
        if show_ebitda:
            rev_max = np.nanmax(np.abs(vals_rev)) if len(vals_rev) else 0
            ebitda_max = np.nanmax(np.abs(vals_ebitda)) if len(vals_ebitda) else 0
            if rev_max > 0 and ebitda_max > 0:
                ratio = max(rev_max, ebitda_max) / min(rev_max, ebitda_max)
                use_twin = ratio >= 6
            if use_twin:
                ax2 = self._create_ax2()
                ax2.set_facecolor("none")
                self.apply_axes_frame(ax2)
                ax2.tick_params(axis="y", colors=self.C_GREEN, labelsize=9)
                ax2.set_ylabel(f"{ebitda_label} ($B)", color=self.C_GREEN)
                ebitda_line, = ax2.plot([], [], color=self.C_GREEN, marker="o", label=ebitda_label, linewidth=2)
            else:
                ebitda_line, = p.ax.plot([], [], color=self.C_GREEN, marker="o", label=ebitda_label, linewidth=2)
        y_max = max(vals_rev.max() if len(vals_rev) else 0, vals_ebitda.max() if show_ebitda else 0)
        y_min = min(0, vals_rev.min() if len(vals_rev) else 0, vals_ebitda.min() if show_ebitda else 0)
        self.safe_ylim(y_min, y_max, pad=0.15, zero_floor=True)
        p.ax.axhline(0, color=self.C_BORDER, lw=1, alpha=0.6)
        if self._ax2 is not None:
            emax = np.nanmax(np.abs(vals_ebitda)) if len(vals_ebitda) else 0
            if not np.isfinite(emax) or emax <= 0:
                emax = 1.0
            self._ax2.set_ylim(-emax * 1.2 if np.nanmin(vals_ebitda) < 0 else 0, emax * 1.2)
        if p.anim_enabled:
            def update(frame):
                prog = min(1.0, (frame + 1) / self.anim_frames)
                for rect, h in zip(bars, vals_rev):
                    rect.set_height(h * prog)
                if show_ebitda and ebitda_line is not None:
                    limit = int(prog * len(dates))
                    limit = max(1, min(len(dates), limit))
                    ebitda_line.set_data(x_idx[:limit], vals_ebitda[:limit])
                return bars
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            for rect, h in zip(bars, vals_rev):
                rect.set_height(h)
            if show_ebitda and ebitda_line is not None:
                ebitda_line.set_data(x_idx, vals_ebitda)
        handles, labels = p.ax.get_legend_handles_labels()
        if self._ax2 is not None:
            h2, l2 = self._ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        self._legend(handles, labels)
        summary = "\n".join([
            self.series_stats(vals_rev, "Revenue", "B"),
            self.series_stats(vals_ebitda, ebitda_label, "B") if show_ebitda else f"{ebitda_label}: no data.",
        ])
        if p.on_analysis:
            p.on_analysis(f"Revenue & EBITDA|{freq}", summary)
        apply_chart_layout(p.fig)

    def _render_eps_dividends(self, freq: str):
        p = self.payload
        eps_q = self._get_series(p.df_fin_q, ["BasicEPS", "DilutedEPS"])
        eps_a = self._get_series(p.df_fin_a, ["BasicEPS", "DilutedEPS"])
        eps = self.series_for_flow(eps_q, eps_a, freq)
        if eps is None:
            self.draw_no_data("EPS data unavailable")
            apply_chart_layout(p.fig)
            return
        df = pd.DataFrame({"eps": eps}).sort_index()
        if df["eps"].dropna().empty:
            self.draw_no_data("EPS data unavailable")
            apply_chart_layout(p.fig)
            return
        div_s = self.dividend_series(freq)
        if freq == "Annual":
            pidx = pd.PeriodIndex(pd.to_datetime(df.index), freq="Y")
            eps_series = pd.Series(df["eps"].values, index=pidx).groupby(level=0).sum()
            df = pd.DataFrame({"eps": eps_series.values}, index=eps_series.index.to_timestamp())
            if div_s is not None and not div_s.empty:
                d_pidx = pd.PeriodIndex(pd.to_datetime(div_s.index), freq="Y")
                d_series = pd.Series(div_s.values, index=d_pidx).groupby(level=0).sum()
                df["div"] = d_series.reindex(eps_series.index).fillna(0).values
        elif freq == "TTM":
            pidx = pd.PeriodIndex(pd.to_datetime(df.index), freq="Q")
            if not pidx.empty:
                periods = pd.period_range(end=pidx.max(), periods=4, freq="Q")
                eps_series = pd.Series(df["eps"].values, index=pidx).reindex(periods)
                df = pd.DataFrame({"eps": eps_series.values}, index=periods.to_timestamp())
                if div_s is not None and not div_s.empty:
                    d_pidx = pd.PeriodIndex(pd.to_datetime(div_s.index), freq="Q")
                    d_series = pd.Series(div_s.values, index=d_pidx).reindex(periods).fillna(0)
                    df["div"] = d_series.values
        else:
            if div_s is not None and not div_s.empty:
                df["div"] = div_s.reindex(df.index).fillna(0).values
        periods = self.to_period_index(df.index, freq)
        dates = self._format_period_labels(periods, freq)
        x_idx = np.arange(len(dates))
        y_eps = df["eps"].values.astype(float)
        y_div = np.nan_to_num(df["div"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0) if "div" in df.columns else None
        self.style_axes("EPS & DIVIDENDS", "Per Share ($)")
        self.set_period_ticks(x_idx, dates)
        p.ax.set_xlim(-0.5, len(dates) - 0.5)
        p.ax.axhline(0, color=self.C_BORDER, lw=1, alpha=0.6)
        self.add_vertical_guides(x_idx)
        eps_line, = p.ax.plot([], [], color=self.C_ACCENT, marker="o", label="EPS", linewidth=2)
        div_bars = None
        if y_div is not None and np.nanmax(y_div) > 0:
            ax2 = self._create_ax2()
            ax2.set_facecolor("none")
            self.apply_axes_frame(ax2)
            ax2.tick_params(axis="y", colors=self.C_YELLOW, labelsize=9)
            ax2.set_ylabel("Dividends ($)", color=self.C_YELLOW)
            div_bars = ax2.bar(x_idx, [0] * len(dates), color=self.C_YELLOW, alpha=0.35, label="Dividends")
        if np.isfinite(y_eps).any():
            y_max = np.nanmax(y_eps)
            y_min = np.nanmin(y_eps)
        else:
            y_max = 1.0
            y_min = 0.0
        self.safe_ylim(min(0, y_min), y_max, pad=0.2, zero_floor=False)
        if self._ax2 is not None:
            div_max = y_div.max() if len(y_div) else 0
            if not np.isfinite(div_max) or div_max <= 0:
                div_max = 1.0
            self._ax2.set_ylim(0, div_max * 1.25)
        if p.anim_enabled:
            def update(frame):
                prog = min(1.0, (frame + 1) / self.anim_frames)
                limit = int(prog * len(dates))
                limit = max(1, min(len(dates), limit))
                eps_line.set_data(x_idx[:limit], y_eps[:limit])
                if div_bars is not None:
                    for rect, h in zip(div_bars, y_div):
                        rect.set_height(h * prog)
                return eps_line,
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            eps_line.set_data(x_idx, y_eps)
            if div_bars is not None:
                for rect, h in zip(div_bars, y_div):
                    rect.set_height(h)
        handles, labels = p.ax.get_legend_handles_labels()
        if self._ax2 is not None:
            h2, l2 = self._ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        self._legend(handles, labels)
        summary = self.series_stats(y_eps, "EPS", "")
        if y_div is not None:
            summary += "\n" + self.series_stats(y_div, "Dividends", "")
        if p.on_analysis:
            p.on_analysis(f"EPS & Dividends|{freq}", summary)
        apply_chart_layout(p.fig)

    def _render_cash_vs_debt(self, freq: str):
        p = self.payload
        cash_q = self._get_series(p.df_bal_q, ["CashAndCashEquivalents", "Cash"])
        cash_a = self._get_series(p.df_bal_a, ["CashAndCashEquivalents", "Cash"])
        debt_q = self._get_series(p.df_bal_q, ["TotalDebt", "LongTermDebt"])
        debt_a = self._get_series(p.df_bal_a, ["TotalDebt", "LongTermDebt"])
        cash = self.series_for_balance(cash_q, cash_a, freq)
        debt = self.series_for_balance(debt_q, debt_a, freq)
        if cash is None:
            self.draw_no_data("Cash/Debt data unavailable")
            apply_chart_layout(p.fig)
            return
        dates, v_cash, v_debt = self.align_data(cash / 1e9, debt / 1e9 if debt is not None else None, freq=freq)
        if len(dates) == 0:
            self.draw_no_data("Cash/Debt data unavailable")
            apply_chart_layout(p.fig)
            return
        x_idx = np.arange(len(dates))
        v_cash = np.nan_to_num(v_cash.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        v_debt = np.nan_to_num(v_debt.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        self.style_axes("CASH vs DEBT", "USD ($B)")
        self.set_period_ticks(x_idx, dates)
        p.ax.set_xlim(-0.5, len(dates) - 0.5)
        self.add_vertical_guides(x_idx)
        l1, = p.ax.plot([], [], color=self.C_GREEN, marker="o", label="Cash")
        l2 = None
        use_twin = False
        if debt is not None:
            cash_max = np.nanmax(np.abs(v_cash)) if len(v_cash) else 0
            debt_max = np.nanmax(np.abs(v_debt)) if len(v_debt) else 0
            if cash_max > 0 and debt_max > 0:
                ratio = max(cash_max, debt_max) / min(cash_max, debt_max)
                use_twin = ratio >= 6
            if use_twin:
                ax2 = self._create_ax2()
                ax2.set_facecolor("none")
                self.apply_axes_frame(ax2)
                ax2.tick_params(axis="y", colors=self.C_RED, labelsize=9)
                ax2.set_ylabel("Debt ($B)", color=self.C_RED)
                l2, = ax2.plot([], [], color=self.C_RED, marker="x", label="Debt")
            else:
                l2, = p.ax.plot([], [], color=self.C_RED, marker="x", label="Debt")
        max_val = max(v_cash.max() if len(v_cash) > 0 else 0, v_debt.max() if len(v_debt) > 0 else 0)
        self.safe_ylim(0, max_val, pad=0.2, zero_floor=True)
        if self._ax2 is not None:
            dmax = np.nanmax(np.abs(v_debt)) if len(v_debt) else 0
            if not np.isfinite(dmax) or dmax <= 0:
                dmax = 1.0
            self._ax2.set_ylim(0, dmax * 1.2)
        if p.anim_enabled:
            def update(frame):
                prog = min(1.0, (frame + 1) / self.anim_frames)
                limit = int(prog * len(dates))
                limit = max(1, min(len(dates), limit))
                l1.set_data(x_idx[:limit], v_cash[:limit])
                if l2 is not None:
                    l2.set_data(x_idx[:limit], v_debt[:limit])
                return l1,
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            l1.set_data(x_idx, v_cash)
            if l2 is not None:
                l2.set_data(x_idx, v_debt)
        handles, labels = p.ax.get_legend_handles_labels()
        if self._ax2 is not None:
            h2, l2 = self._ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        self._legend(handles, labels)
        summary = "\n".join([
            self.series_stats(v_cash, "Cash", "B"),
            self.series_stats(v_debt, "Debt", "B") if debt is not None else "Debt: no data.",
        ])
        if p.on_analysis:
            p.on_analysis(f"Cash vs Debt|{freq}", summary)
        apply_chart_layout(p.fig)

    def _render_margins(self, freq: str):
        p = self.payload
        rev_q = self._get_series(p.df_fin_q, ["TotalRevenue"])
        rev_a = self._get_series(p.df_fin_a, ["TotalRevenue"])
        rev = self.series_for_flow(rev_q, rev_a, freq)
        if rev is None:
            self.draw_no_data("Margins data unavailable")
            apply_chart_layout(p.fig)
            return
        gp_q = self._get_series(p.df_fin_q, ["GrossProfit"])
        gp_a = self._get_series(p.df_fin_a, ["GrossProfit"])
        op_q = self._get_series(p.df_fin_q, ["OperatingIncome", "OperatingIncomeLoss"])
        op_a = self._get_series(p.df_fin_a, ["OperatingIncome", "OperatingIncomeLoss"])
        ni_q = self._get_series(p.df_fin_q, ["NetIncome"])
        ni_a = self._get_series(p.df_fin_a, ["NetIncome"])
        gp = self.series_for_flow(gp_q, gp_a, freq)
        op = self.series_for_flow(op_q, op_a, freq)
        ni = self.series_for_flow(ni_q, ni_a, freq)
        df = pd.DataFrame({"r": rev, "g": gp, "o": op, "n": ni}).dropna(subset=["r"])
        df["g"] = df["g"].fillna(0)
        df["o"] = df["o"].fillna(0)
        df["n"] = df["n"].fillna(0)
        df = df.sort_index()
        if df.empty:
            self.draw_no_data("Margins data unavailable")
            apply_chart_layout(p.fig)
            return
        periods = self.to_period_index(df.index, freq)
        dates = self._format_period_labels(periods, freq)
        v_gross = np.nan_to_num(((df["g"] / df["r"]) * 100).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        v_op = np.nan_to_num(((df["o"] / df["r"]) * 100).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        v_net = np.nan_to_num(((df["n"] / df["r"]) * 100).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        x_idx = np.arange(len(dates))
        self.style_axes("MARGINS", "Percent (%)")
        self.set_period_ticks(x_idx, dates)
        p.ax.set_xlim(-0.5, len(dates) - 0.5)
        p.ax.axhline(0, color=self.C_BORDER, lw=1, alpha=0.6)
        self.add_vertical_guides(x_idx)
        l1, = p.ax.plot([], [], color=self.C_YELLOW, label="Gross %")
        l2, = p.ax.plot([], [], color=self.C_ACCENT, label="Op %")
        l3, = p.ax.plot([], [], color=self.C_GREEN, label="Net %")
        y_min = min(0, v_net.min(), v_op.min(), v_gross.min())
        y_max = max(5, v_gross.max(), v_op.max(), v_net.max())
        self.safe_ylim(y_min, y_max, pad=0.2, zero_floor=False)
        if p.anim_enabled:
            def update(frame):
                prog = min(1.0, (frame + 1) / self.anim_frames)
                limit = int(prog * len(dates))
                limit = max(1, min(len(dates), limit))
                l1.set_data(x_idx[:limit], v_gross[:limit])
                l2.set_data(x_idx[:limit], v_op[:limit])
                l3.set_data(x_idx[:limit], v_net[:limit])
                return l1, l2, l3
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            l1.set_data(x_idx, v_gross)
            l2.set_data(x_idx, v_op)
            l3.set_data(x_idx, v_net)
        self._legend()
        summary = "\n".join([
            self.series_stats(v_gross, "Gross Margin", "%"),
            self.series_stats(v_op, "Operating Margin", "%"),
            self.series_stats(v_net, "Net Margin", "%"),
        ])
        if p.on_analysis:
            p.on_analysis(f"Margins|{freq}", summary)
        apply_chart_layout(p.fig)

    def _render_cash_flow_detail(self, freq: str):
        p = self.payload
        fcf_q = self._get_series(p.df_cf_q, ["FreeCashFlow", "Free Cash Flow"])
        fcf_a = self._get_series(p.df_cf_a, ["FreeCashFlow", "Free Cash Flow"])
        fcf = self.series_for_flow(fcf_q, fcf_a, freq)
        if fcf is None:
            ocf_q = self._get_series(p.df_cf_q, ["OperatingCashFlow"])
            ocf_a = self._get_series(p.df_cf_a, ["OperatingCashFlow"])
            cap_q = self._get_series(p.df_cf_q, ["CapitalExpenditure"])
            cap_a = self._get_series(p.df_cf_a, ["CapitalExpenditure"])
            ocf = self.series_for_flow(ocf_q, ocf_a, freq)
            cap = self.series_for_flow(cap_q, cap_a, freq)
            if ocf is not None and cap is not None:
                fcf = ocf - cap.abs()
        sbc_q = self._get_series(p.df_cf_q, ["StockBasedCompensation", "ShareBasedCompensation"])
        sbc_a = self._get_series(p.df_cf_a, ["StockBasedCompensation", "ShareBasedCompensation"])
        sbc = self.series_for_flow(sbc_q, sbc_a, freq)
        if fcf is None:
            self.draw_no_data("Cash flow data unavailable")
            apply_chart_layout(p.fig)
            return
        dates, v_fcf, v_sbc = self.align_data(fcf / 1e9, sbc / 1e9 if sbc is not None else None, fill_value=0.0, freq=freq)
        if len(dates) == 0:
            self.draw_no_data("Cash flow data unavailable")
            apply_chart_layout(p.fig)
            return
        v_fcf = np.nan_to_num(v_fcf.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        v_sbc = np.nan_to_num(v_sbc.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        x_idx = np.arange(len(dates))
        self.style_axes("CASH FLOW DETAIL", "USD ($B)")
        self.set_period_ticks(x_idx, dates)
        p.ax.set_xlim(-0.5, len(dates) - 0.5)
        self.add_vertical_guides(x_idx)
        p.ax.axhline(0, color=self.C_BORDER, lw=1, alpha=0.6)
        bars = p.ax.bar(x_idx, [0] * len(dates), color=self.C_GREEN, label="FCF", alpha=0.6)
        sbc_line = None
        use_twin = False
        if sbc is not None:
            fcf_max = np.nanmax(np.abs(v_fcf)) if len(v_fcf) else 0
            sbc_max = np.nanmax(np.abs(v_sbc)) if len(v_sbc) else 0
            if fcf_max > 0 and sbc_max > 0:
                ratio = max(fcf_max, sbc_max) / min(fcf_max, sbc_max)
                use_twin = ratio >= 6
            if use_twin:
                ax2 = self._create_ax2()
                ax2.set_facecolor("none")
                self.apply_axes_frame(ax2)
                ax2.tick_params(axis="y", colors=self.C_PINK, labelsize=9)
                ax2.set_ylabel("SBC ($B)", color=self.C_PINK)
                sbc_line, = ax2.plot([], [], color=self.C_PINK, marker="o", label="SBC", linewidth=2)
            else:
                sbc_line, = p.ax.plot([], [], color=self.C_PINK, marker="o", label="SBC", linewidth=2)
        max_val = max(v_fcf.max() if len(v_fcf) > 0 else 0, v_sbc.max() if len(v_sbc) > 0 else 0)
        min_val = min(0, v_fcf.min() if len(v_fcf) > 0 else 0, v_sbc.min() if len(v_sbc) > 0 else 0)
        self.safe_ylim(min_val, max_val, pad=0.2, zero_floor=False)
        if self._ax2 is not None:
            smax = np.nanmax(np.abs(v_sbc)) if len(v_sbc) else 0
            if not np.isfinite(smax) or smax <= 0:
                smax = 1.0
            self._ax2.set_ylim(-smax * 1.2 if np.nanmin(v_sbc) < 0 else 0, smax * 1.2)
        if p.anim_enabled:
            def update(frame):
                if len(dates) == 0:
                    return bars
                prog = min(1.0, (frame + 1) / self.anim_frames)
                for rect, h in zip(bars, v_fcf):
                    rect.set_height(h * prog)
                limit = int(prog * len(dates))
                limit = max(1, min(len(dates), limit))
                if sbc_line is not None:
                    sbc_line.set_data(x_idx[:limit], v_sbc[:limit])
                return bars
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            for rect, h in zip(bars, v_fcf):
                rect.set_height(h)
            if sbc_line is not None:
                sbc_line.set_data(x_idx, v_sbc)
        handles, labels = p.ax.get_legend_handles_labels()
        if self._ax2 is not None:
            h2, l2 = self._ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        self._legend(handles, labels)
        summary = "\n".join([
            self.series_stats(v_fcf, "FCF", "B"),
            self.series_stats(v_sbc, "SBC", "B") if sbc is not None else "SBC: no data.",
        ])
        if p.on_analysis:
            p.on_analysis(f"Cash Flow Detail|{freq}", summary)
        apply_chart_layout(p.fig)

    def _render_valuation(self, mode: str, freq: str):
        p = self.payload
        if p.hist_data is None or p.df_fin_q is None:
            self.draw_no_data("Insufficient data for valuation")
            apply_chart_layout(p.fig)
            return
        fin_df = self._get_freq_df("fin", freq)
        eps = self._get_series(fin_df, ["BasicEPS"])
        rev = self._get_series(fin_df, ["TotalRevenue"])
        fcf = self._get_series(self._get_freq_df("cf", freq), ["FreeCashFlow", "Free Cash Flow"])
        shares = self._get_series(self._get_freq_df("bal", freq), ["ShareIssued", "CommonStockSharesOutstanding"])
        price = pd.to_numeric(p.hist_data.get("Close"), errors="coerce")
        if freq == "TTM":
            eps = self.to_ttm(eps)
            rev = self.to_ttm(rev)
            fcf = self.to_ttm(fcf)
        if mode == "Valuation (P/E)" and eps is not None:
            aligned_eps = pd.to_numeric(eps.reindex(p.hist_data.index, method="ffill"), errors="coerce")
            pe = price / (aligned_eps * 4)
            pe = pe.replace([np.inf, -np.inf], np.nan)
            pe = pe[(pe > 0) & (pe < 250)].dropna()
            series = pe
            label = "P/E Ratio"
            color = self.C_ACCENT
        elif mode == "Valuation (P/S)" and rev is not None and shares is not None:
            aligned_rev = pd.to_numeric(rev.reindex(p.hist_data.index, method="ffill"), errors="coerce")
            aligned_shares = pd.to_numeric(shares.reindex(p.hist_data.index, method="ffill"), errors="coerce")
            ps = (price * aligned_shares) / (aligned_rev * 4)
            ps = ps.replace([np.inf, -np.inf], np.nan)
            ps = ps[(ps > 0) & (ps < 50)].dropna()
            series = ps
            label = "P/S Ratio"
            color = self.C_YELLOW
        elif mode == "Valuation (FCF Yield)" and fcf is not None and shares is not None:
            aligned_fcf = pd.to_numeric(fcf.reindex(p.hist_data.index, method="ffill"), errors="coerce")
            aligned_shares = pd.to_numeric(shares.reindex(p.hist_data.index, method="ffill"), errors="coerce")
            mcap = price * aligned_shares
            fy = ((aligned_fcf * 4) / mcap.replace(0, np.nan)) * 100
            fy = fy.replace([np.inf, -np.inf], np.nan).dropna()
            series = fy
            label = "FCF Yield %"
            color = self.C_GREEN
        else:
            series = None
        if series is None or series.empty:
            self.draw_no_data("Insufficient data for valuation")
            apply_chart_layout(p.fig)
            return
        x = series.index
        y = np.nan_to_num(series.values.astype(float), nan=np.nan, posinf=np.nan, neginf=np.nan)
        self.style_axes(label.upper(), label)
        self.set_date_ticks(x)
        line, = p.ax.plot([], [], color=color, label=label, linewidth=1.8)
        self.set_date_xlim(x, pad_hours=24)
        self.safe_ylim(np.nanmin(y), np.nanmax(y), pad=0.15, zero_floor=False)
        if p.anim_enabled:
            def update(frame):
                prog = min(1.0, (frame + 1) / self.anim_frames)
                limit = int(prog * len(x))
                limit = max(1, min(len(x), limit))
                line.set_data(x[:limit], y[:limit])
                return line,
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            line.set_data(x, y)
        self._legend()
        summary = self.series_stats(y, label, "")
        if p.on_analysis:
            p.on_analysis(f"{label}|{freq}", summary)
        apply_chart_layout(p.fig)

    def _render_fcf_per_share(self, freq: str):
        p = self.payload
        fcf_q = self._get_series(p.df_cf_q, ["FreeCashFlow", "Free Cash Flow"])
        fcf_a = self._get_series(p.df_cf_a, ["FreeCashFlow", "Free Cash Flow"])
        shares_q = self._get_series(p.df_bal_q, ["ShareIssued", "CommonStockSharesOutstanding"])
        shares_a = self._get_series(p.df_bal_a, ["ShareIssued", "CommonStockSharesOutstanding"])
        fcf = self.series_for_flow(fcf_q, fcf_a, freq)
        shares = self.series_for_balance(shares_q, shares_a, freq)
        if fcf is None or shares is None:
            self.draw_no_data("FCF/share unavailable")
            apply_chart_layout(p.fig)
            return
        try:
            fps = (fcf / shares).replace([np.inf, -np.inf], np.nan)
        except Exception:
            fps = self._safe_div(fcf, shares)
        if not isinstance(fps, pd.Series):
            try:
                fps = pd.Series(fps)
            except Exception:
                fps = None
        df = pd.DataFrame({"fps": fps}).dropna().sort_index() if fps is not None else pd.DataFrame()
        if df.empty:
            self.draw_no_data("FCF/share unavailable")
            apply_chart_layout(p.fig)
            return
        periods = self.to_period_index(df.index, freq)
        dates = self._format_period_labels(periods, freq)
        x_idx = np.arange(len(dates))
        y = np.nan_to_num(df["fps"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        self.style_axes("FCF PER SHARE", "Per Share ($)")
        self.set_period_ticks(x_idx, dates)
        p.ax.set_xlim(-0.5, len(dates) - 0.5)
        self.add_vertical_guides(x_idx)
        line, = p.ax.plot([], [], color=self.C_GREEN, marker="o", label="FCF/Share", linewidth=2)
        self.safe_ylim(min(0, y.min()), max(0.1, y.max()), pad=0.2, zero_floor=False)
        if p.anim_enabled:
            def update(frame):
                prog = min(1.0, (frame + 1) / self.anim_frames)
                limit = int(prog * len(x_idx))
                limit = max(1, min(len(x_idx), limit))
                line.set_data(x_idx[:limit], y[:limit])
                return line,
            self._set_anim(animation.FuncAnimation(p.fig, update, frames=self.anim_frames, interval=20, blit=False, repeat=False))
        else:
            line.set_data(x_idx, y)
        self._legend()
        summary = self.series_stats(y, "FCF/Share", "")
        if p.on_analysis:
            p.on_analysis(f"FCF/Share|{freq}", summary)
        apply_chart_layout(p.fig)


def redraw_chart_now(payload_or_app, *, palette, anim_frames):
    payload = payload_or_app if isinstance(payload_or_app, ChartPayload) else ChartPayload.from_app(payload_or_app)
    renderer = ChartRenderer(payload, palette=palette, anim_frames=anim_frames)
    renderer.render()

