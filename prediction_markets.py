import json
from datetime import datetime
from typing import Any, Dict, List
import numpy as np

from .database import DataStore


DEFAULT_MACRO_ODDS: List[Dict[str, Any]] = [
    {
        "contract_id": "KALSHI-FED-CUT-NEXT",
        "source": "Kalshi",
        "category": "rates",
        "title": "Fed cuts target rate by >=25 bps at next FOMC meeting",
        "probability_pct": 68.0,
        "volume": 1450000.0,
        "url": "https://kalshi.com/markets/fed",
    },
    {
        "contract_id": "POLY-US-RECESSION-2026",
        "source": "Polymarket",
        "category": "growth",
        "title": "US NBER Recession declared before year-end",
        "probability_pct": 22.0,
        "volume": 3200000.0,
        "url": "https://polymarket.com",
    },
    {
        "contract_id": "KALSHI-CPI-CORE-BELOW-3",
        "source": "Kalshi",
        "category": "inflation",
        "title": "US Core CPI YoY prints below 3.0% next release",
        "probability_pct": 74.0,
        "volume": 890000.0,
        "url": "https://kalshi.com/markets/cpi",
    },
]


def _infer_macro_category(title: str, default: str = "macro") -> str:
    t = (title or "").lower()
    if any(k in t for k in ("fed", "fomc", "rate", "bps", "powell", "yield")):
        return "rates"
    if any(k in t for k in ("cpi", "inflation", "pce", "prices")):
        return "inflation"
    if any(k in t for k in ("gdp", "recession", "nber", "payroll", "unemployment", "growth")):
        return "growth"
    return default


class PredictionMarketClient:
    """Normalizes public Kalshi and Polymarket prediction market payloads into canonical SQLite rows."""

    @staticmethod
    def normalize_kalshi_markets(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        markets = payload.get("markets")
        if not isinstance(markets, list):
            return []
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out: List[Dict[str, Any]] = []
        for m in markets:
            if not isinstance(m, dict):
                continue
            ticker = str(m.get("ticker") or m.get("id") or "").strip()
            title = str(m.get("title") or m.get("subtitle") or "").strip()
            if not ticker or not title:
                continue

            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            last_px = m.get("last_price")
            prob_raw = None
            if yes_bid is not None and yes_ask is not None:
                try:
                    prob_raw = 0.5 * (float(yes_bid) + float(yes_ask))
                except (TypeError, ValueError):
                    prob_raw = None
            if prob_raw is None and last_px is not None:
                try:
                    prob_raw = float(last_px)
                except (TypeError, ValueError):
                    prob_raw = None
            if prob_raw is None:
                prob_raw = 0.0

            # If given as decimal in (0, 1], scale to percentage; clamp to [0, 100]
            if 0.0 < prob_raw <= 1.0:
                prob_pct = prob_raw * 100.0
            else:
                prob_pct = prob_raw
            prob_pct = float(np.clip(prob_pct, 0.0, 100.0))

            vol = m.get("volume")
            try:
                vol_f = float(vol) if vol is not None else 0.0
            except (TypeError, ValueError):
                vol_f = 0.0

            cat = str(m.get("category") or _infer_macro_category(title)).strip().lower()
            url = str(m.get("url") or f"https://kalshi.com/markets/{ticker.lower()}").strip()
            out.append(
                {
                    "contract_id": ticker,
                    "source": "Kalshi",
                    "category": cat,
                    "title": title,
                    "probability_pct": round(prob_pct, 2),
                    "volume": vol_f,
                    "url": url,
                    "updated_at": now_iso,
                }
            )
        return out

    @staticmethod
    def normalize_polymarket_markets(payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out: List[Dict[str, Any]] = []
        for m in payload:
            if not isinstance(m, dict):
                continue
            cid = str(m.get("id") or m.get("conditionId") or m.get("slug") or "").strip()
            question = str(m.get("question") or m.get("title") or "").strip()
            if not cid or not question:
                continue

            outcome_prices = m.get("outcomePrices")
            prices_list: List[Any] = []
            if isinstance(outcome_prices, str):
                try:
                    parsed = json.loads(outcome_prices)
                    if isinstance(parsed, list):
                        prices_list = parsed
                except Exception:
                    prices_list = []
            elif isinstance(outcome_prices, list):
                prices_list = outcome_prices

            prob_pct = 0.0
            if prices_list:
                try:
                    first_px = float(prices_list[0])
                    prob_pct = first_px * 100.0 if first_px <= 1.0 else first_px
                except (TypeError, ValueError):
                    prob_pct = 0.0
            prob_pct = float(np.clip(prob_pct, 0.0, 100.0))

            vol = m.get("volume") or m.get("volumeNum")
            try:
                vol_f = float(vol) if vol is not None else 0.0
            except (TypeError, ValueError):
                vol_f = 0.0

            cat = str(m.get("category") or _infer_macro_category(question)).strip().lower()
            slug = str(m.get("slug") or cid).strip()
            url = str(m.get("url") or f"https://polymarket.com/event/{slug}").strip()
            out.append(
                {
                    "contract_id": f"POLY-{cid}" if not cid.upper().startswith("POLY") else cid,
                    "source": "Polymarket",
                    "category": cat,
                    "title": question,
                    "probability_pct": round(prob_pct, 2),
                    "volume": vol_f,
                    "url": url,
                    "updated_at": now_iso,
                }
            )
        return out

    @staticmethod
    def get_cached_or_default_odds(store: DataStore) -> List[Dict[str, Any]]:
        rows = store.get_prediction_odds(limit=25)
        if rows:
            return rows
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seeded = [{**row, "updated_at": now_iso} for row in DEFAULT_MACRO_ODDS]
        store.upsert_prediction_odds(seeded)
        return store.get_prediction_odds(limit=25)
