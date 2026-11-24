import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from compounderos_app.charts import (
    PRICE_PERIOD_TO_YF,
    ChartPayload,
    ChartRenderer,
    detect_key_moments,
    normalize_peer_overlays,
    price_period_to_yf,
)


PALETTE = {
    "C_ACCENT": "#58a6ff",
    "C_BORDER": "#30363d",
    "C_GREEN": "#3fb950",
    "C_RED": "#f85149",
    "C_YELLOW": "#d29922",
    "C_PINK": "#db61a2",
    "C_BG_CARD": "#161b22",
    "C_TEXT_MAIN": "#e6edf3",
    "C_TEXT_SUB": "#8b949e",
    "C_PLOT_FRAME": "#30363d",
    "C_SERIES_NEUTRAL": "#6e7681",
}


def test_price_period_mapping_5d_and_max():
    assert PRICE_PERIOD_TO_YF["5D"] == "5d"
    assert PRICE_PERIOD_TO_YF["1W"] == "5d"
    assert PRICE_PERIOD_TO_YF["MAX"] == "max"
    assert PRICE_PERIOD_TO_YF["ALL"] == "max"
    assert price_period_to_yf("5D") == "5d"
    assert price_period_to_yf("MAX") == "max"


def test_detect_key_moments_mad_zscore_and_catalyst_matching():
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    # Baseline small alternating +0.2% / -0.2% returns
    returns = np.array([0.002 if i % 2 == 0 else -0.002 for i in range(40)], dtype=float)
    # Inject +12% spike at index 15 and -10% drop at index 28
    returns[15] = 0.12
    returns[28] = -0.10

    closes = [100.0]
    for r in returns[1:]:
        closes.append(closes[-1] * (1.0 + r))

    df = pd.DataFrame({"Close": closes, "Volume": [1_000_000] * 40}, index=dates)

    # Provide an 8-K catalyst within 1 day of dates[15]
    catalyst_date = (dates[15] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    events = [{"date": catalyst_date, "label": "8-K Earnings Release"}]

    moments = detect_key_moments(df, z_threshold=3.0, max_moments=5, events=events)
    assert len(moments) >= 2

    by_dir = {m["direction"]: m for m in moments}
    assert "up" in by_dir
    assert "down" in by_dir

    up_m = by_dir["up"]
    assert up_m["date"] == dates[15]
    assert up_m["return_pct"] == pytest.approx(0.12, rel=1e-3)
    assert up_m["z_score"] > 3.0
    assert "8-K Earnings Release" in up_m["label"]

    down_m = by_dir["down"]
    assert down_m["date"] == dates[28]
    assert down_m["return_pct"] == pytest.approx(-0.10, rel=1e-3)
    assert down_m["z_score"] < -3.0
    assert "Return spike" in down_m["label"]


def test_normalize_peer_overlays_starts_at_zero_and_tracks_pct():
    dates = pd.date_range("2026-02-01", periods=5, freq="D")
    primary = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    peer_spy = pd.Series([500.0, 505.0, 510.0, 500.0, 525.0], index=dates)
    peer_qqq = pd.DataFrame({"Close": [400.0, 408.0, 420.0, 412.0, 440.0]}, index=dates)

    out = normalize_peer_overlays(primary, {"SPY": peer_spy, "QQQ": peer_qqq})
    assert set(out.keys()) == {"__primary__", "SPY", "QQQ"}
    assert out["__primary__"].iloc[0] == pytest.approx(0.0)
    assert out["SPY"].iloc[0] == pytest.approx(0.0)
    assert out["QQQ"].iloc[0] == pytest.approx(0.0)

    assert out["__primary__"].iloc[-1] == pytest.approx(20.0)
    assert out["SPY"].iloc[-1] == pytest.approx(5.0)
    assert out["QQQ"].iloc[-1] == pytest.approx(10.0)


def test_chart_renderer_candlestick_overlays_ema_volume_and_key_moments():
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    closes = np.linspace(100.0, 115.0, 30)
    closes[12] = closes[11] * 1.14  # +14% anomaly
    opens = closes * 0.99
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.985
    volumes = np.linspace(1e6, 2.5e6, 30)

    hist_df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )

    # 1. Candlestick + EMA20 + Volume + Key Moments + Events
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    payload1 = ChartPayload(
        fig=fig1,
        ax=ax1,
        mode="Price",
        price_range="1Y",
        hist_period="1y",
        hist_data=hist_df,
        price_chart_style="candlestick",
        show_ema_20=True,
        show_volume=True,
        show_key_moments=True,
        show_events=True,
        event_markers=[{"date": dates[12].strftime("%Y-%m-%d"), "label": "10-K Filing"}],
    )
    renderer1 = ChartRenderer(payload1, palette=PALETTE, anim_frames=10)
    renderer1.render()

    _, labels1 = ax1.get_legend_handles_labels()
    assert "EMA 20" in labels1
    assert "Key Moments" in labels1
    assert renderer1._ax2 is not None
    plt.close(fig1)

    # 2. Peer Overlays mode
    peer_df = pd.DataFrame({"Close": np.linspace(200.0, 210.0, 30)}, index=dates)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    payload2 = ChartPayload(
        fig=fig2,
        ax=ax2,
        mode="Price",
        price_range="1Y",
        hist_period="1y",
        hist_data=hist_df,
        peer_overlays={"SPY": peer_df},
    )
    renderer2 = ChartRenderer(payload2, palette=PALETTE, anim_frames=10)
    renderer2.render()

    _, labels2 = ax2.get_legend_handles_labels()
    assert "Primary" in labels2
    assert "SPY" in labels2
    assert ax2.get_ylabel() == "% Change"
    plt.close(fig2)
