import re
from typing import Any, Dict, List


_PRIMARY_PATTERNS = [
    r"\b10-k\b",
    r"\b10-q\b",
    r"\b8-k\b",
    r"\bform\s*4\b",
    r"\bsec\s+filing\b",
    r"\bearnings\b",
    r"\bquarterly\s+results\b",
    r"\bfull[- ]year\s+results\b",
    r"\brevenue\s+(beat|miss|rose|grew|guidance)\b",
    r"\bguidance\b",
    r"\boutlook\b",
    r"\bdividend\b",
    r"\bbuyback\b",
    r"\brepurchase\b",
    r"\bacquires?\b",
    r"\bacquisition\b",
    r"\bmerger\b",
    r"\bfda\b",
    r"\bdoj\b",
    r"\bftc\b",
    r"\bantitrust\b",
    r"\bsubpoena\b",
    r"\bsettlement\b",
    r"\bceo\b",
    r"\bcfo\b",
    r"\bfederal\s+reserve\b",
    r"\bfomc\b",
    r"\bcpi\b",
    r"\btreasury\b",
]

_NOISE_PATTERNS = [
    r"\b\d+\s+(stocks?|reasons?|things?)\s+to\s+(buy|sell|watch|avoid)\b",
    r"\bforget\s+[a-z0-9\s,&]+,\s*buy\b",
    r"\bturn\s+\$[\d,]+\s+into\s+\$[\d,]+\b",
    r"\bmillionaire[- ]maker\b",
    r"\bno[- ]brainer\b",
    r"\bwhere\s+will\s+.+\s+stock\s+be\s+in\s+\d+\s+years?\b",
    r"\bif\s+you\s+invested\s+\$[\d,]+\b",
    r"\bis\s+.+\s+stock\s+a\s+buy\b",
    r"\bshould\s+you\s+buy\s+.+\s+stock\b",
    r"\bhere'?s\s+why\s+.+\s+(jumped|plunged|soared|tanked|moved)\b",
    r"\bcathie\s+wood\b",
    r"\bjim\s+cramer\b",
    r"\bprediction:\b",
]

_PRIMARY_PROVIDERS = {
    "pr newswire",
    "business wire",
    "globenewswire",
    "sec edgar",
    "reuters",
    "bloomberg",
    "wall street journal",
    "wsj",
    "financial times",
    "associated press",
    "dow jones",
}

_NOISE_PROVIDERS = {
    "motley fool",
    "the motley fool",
    "insider monkey",
    "24/7 wall st.",
    "zacks",
    "investorplace",
    "benzinga",
}


def classify_news_headline(title: str, provider: str = "", summary: str = "") -> Dict[str, Any]:
    """
    Classify financial news into verifiable PRIMARY_SIGNAL, FUNDAMENTAL_CONTEXT, or SPECULATIVE_NOISE.
    """
    t_clean = (title or "").strip()
    s_clean = (summary or "").strip()
    p_low = (provider or "").strip().lower()
    blob = f"{t_clean} {s_clean}".lower()

    for pat in _NOISE_PATTERNS:
        if re.search(pat, blob):
            return {
                "signal_tier": "SPECULATIVE_NOISE",
                "signal_score": 0,
                "signal_badge": "[NOISE]",
            }

    for pat in _PRIMARY_PATTERNS:
        if re.search(pat, blob):
            return {
                "signal_tier": "PRIMARY_SIGNAL",
                "signal_score": 3,
                "signal_badge": "[SIGNAL]",
            }

    if any(np_name in p_low for np_name in _NOISE_PROVIDERS):
        return {
            "signal_tier": "SPECULATIVE_NOISE",
            "signal_score": 0,
            "signal_badge": "[NOISE]",
        }

    if any(pp_name in p_low for pp_name in _PRIMARY_PROVIDERS):
        return {
            "signal_tier": "PRIMARY_SIGNAL",
            "signal_score": 3,
            "signal_badge": "[SIGNAL]",
        }

    return {
        "signal_tier": "FUNDAMENTAL_CONTEXT",
        "signal_score": 2,
        "signal_badge": "[CONTEXT]",
    }


def rank_and_tag_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Annotate news dicts with signal_tier / signal_badge and rank verifiable signals ahead of noise,
    preserving timestamp recency within each signal tier.
    """
    enriched: List[Dict[str, Any]] = []
    for it in items or []:
        row = dict(it)
        cls_info = classify_news_headline(
            title=str(row.get("title") or ""),
            provider=str(row.get("provider") or ""),
            summary=str(row.get("summary") or ""),
        )
        row.update(cls_info)
        enriched.append(row)

    enriched.sort(
        key=lambda x: (int(x.get("signal_score", 1)), int(x.get("ts") or 0)),
        reverse=True,
    )
    return enriched
