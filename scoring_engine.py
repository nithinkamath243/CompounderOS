import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

ENGINE_VERSION = "det-1.1.0"
ND_EBITDA_UNSERVICEABLE = 99.0
OPLEV_FALLBACK_ANCHORS = (-0.02, 0.005, 0.03)
GAAP_GAP_ANCHORS = (0.30, 0.10, 0.0)

CANONICAL_REGISTRY_ID = "canonical"
LENS_REGISTRY_ID = "custom"

DEFAULT_GATE_TRIGGER = 1.0
DEFAULT_GATE_CAP = 4.0
DEFAULT_GATE_RAMP_SLOPE = 3.0
DEFAULT_GATE_RAMP_CEIL = 3.0
DEFAULT_UNVERIFIED_CAP = 6.0
DEFAULT_CONFIDENCE_PRIOR = 5.0


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    bucket: str
    floor: float
    target: float
    elite: float
    is_gate: bool = False
    weight: Optional[float] = None
    fmt: str = "ratio"


@dataclass(frozen=True)
class MetricDetail:
    key: str
    raw: Optional[float]
    subscore: Optional[float]
    reason: Optional[str]


@dataclass(frozen=True)
class ScoreResult:
    composite: Optional[float]
    composite_pre_gate: Optional[float]
    composite_shrunk: Optional[float]
    gate_cap_applied: Optional[float]
    buckets: Dict[str, Optional[float]]
    metrics: Dict[str, MetricDetail]
    gates_fired: Tuple[str, ...]
    gates_unverified: Tuple[str, ...]
    confidence: float
    engine_version: str
    registry_id: str = CANONICAL_REGISTRY_ID
    registry_hash: str = ""


# S9: Explicit equal weights within each bucket summing to 1.0 per bucket
DEFAULT_REGISTRY = (
    MetricSpec("gross_margin", "Gross Margin", "profitability", 0.20, 0.40, 0.60, weight=1.0 / 3.0, fmt="pct"),
    MetricSpec("revenue_growth", "Revenue Growth (YoY)", "profitability", 0.00, 0.08, 0.18, weight=1.0 / 3.0, fmt="pct"),
    MetricSpec("operating_leverage", "Operating Leverage", "profitability", -0.05, 0.02, 0.10, weight=1.0 / 3.0, fmt="pp"),
    MetricSpec("roic", "Absolute ROIC", "efficiency", 0.08, 0.15, 0.30, weight=0.5, fmt="pct"),
    MetricSpec("roic_momentum", "ROIC Momentum", "efficiency", -0.03, 0.00, 0.03, weight=0.5, fmt="pp"),
    MetricSpec("net_debt_ebitda", "Net Debt / EBITDA", "balance_sheet", 4.0, 2.0, 0.0, is_gate=True, weight=1.0 / 3.0, fmt="ratio"),
    MetricSpec("current_ratio", "Current Ratio", "balance_sheet", 0.8, 1.2, 2.0, weight=1.0 / 3.0, fmt="ratio"),
    MetricSpec("gaap_gap", "GAAP Gap", "balance_sheet", 0.30, 0.10, 0.0, weight=1.0 / 3.0, fmt="pct"),
    MetricSpec("capex_revenue", "CapEx / Revenue", "capital_alloc", 0.20, 0.10, 0.03, weight=0.25, fmt="pct"),
    MetricSpec("earnings_quality", "Earnings Quality", "capital_alloc", 0.80, 1.00, 1.30, is_gate=True, weight=0.25, fmt="ratio"),
    MetricSpec("share_dilution", "Share Dilution", "capital_alloc", 0.03, 0.00, -0.03, weight=0.25, fmt="pct"),
    MetricSpec("sbc_revenue", "SBC / Revenue", "capital_alloc", 0.15, 0.05, 0.0, weight=0.25, fmt="pct"),
)

BUCKET_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "profitability": 0.35,
    "efficiency": 0.30,
    "balance_sheet": 0.20,
    "capital_alloc": 0.15,
})

# S7: Weighted confidence scheme parameters derived from bucket weights and canonical cardinalities
BUCKET_CARDINALITY: Mapping[str, int] = MappingProxyType({
    "profitability": 3,
    "efficiency": 2,
    "balance_sheet": 3,
    "capital_alloc": 4,
})

CONFIDENCE_WEIGHTS: Mapping[str, float] = MappingProxyType({
    s.key: BUCKET_WEIGHTS[s.bucket] / float(BUCKET_CARDINALITY[s.bucket])
    for s in DEFAULT_REGISTRY
})

_CANONICAL_BY_KEY = {s.key: s for s in DEFAULT_REGISTRY}
CANONICAL_METRIC_SPECS = _CANONICAL_BY_KEY


def registry_fingerprint(
    registry: Sequence[MetricSpec],
    bucket_weights: Mapping[str, float],
    gate_trigger: float = DEFAULT_GATE_TRIGGER,
    gate_cap: float = DEFAULT_GATE_CAP,
    gate_ramp_slope: float = DEFAULT_GATE_RAMP_SLOPE,
    gate_ramp_ceil: float = DEFAULT_GATE_RAMP_CEIL,
    unverified_cap: float = DEFAULT_UNVERIFIED_CAP,
    confidence_prior: float = DEFAULT_CONFIDENCE_PRIOR,
) -> str:
    """L8 canonical serialization including v1.1 ramp parameters and confidence parameters."""
    payload = {
        "specs": sorted(
            (
                {
                    "key": s.key,
                    "bucket": s.bucket,
                    "floor": float(s.floor),
                    "target": float(s.target),
                    "elite": float(s.elite),
                    "is_gate": bool(s.is_gate),
                    "weight": None if s.weight is None else float(s.weight),
                }
                for s in registry
            ),
            key=lambda d: d["key"],
        ),
        "bucket_weights": {str(k): float(v) for k, v in bucket_weights.items()},
        "gate_trigger": float(gate_trigger),
        "gate_cap": float(gate_cap),
        "gate_ramp_slope": float(gate_ramp_slope),
        "gate_ramp_ceil": float(gate_ramp_ceil),
        "unverified_cap": float(unverified_cap),
        "confidence_prior": float(confidence_prior),
        "confidence_weights": {str(k): float(v) for k, v in CONFIDENCE_WEIGHTS.items()},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def registry_hash(
    registry: Sequence[MetricSpec],
    bucket_weights: Mapping[str, float],
    gate_trigger: float = DEFAULT_GATE_TRIGGER,
    gate_cap: float = DEFAULT_GATE_CAP,
    gate_ramp_slope: float = DEFAULT_GATE_RAMP_SLOPE,
    gate_ramp_ceil: float = DEFAULT_GATE_RAMP_CEIL,
    unverified_cap: float = DEFAULT_UNVERIFIED_CAP,
    confidence_prior: float = DEFAULT_CONFIDENCE_PRIOR,
) -> str:
    return hashlib.sha256(
        registry_fingerprint(
            registry,
            bucket_weights,
            gate_trigger,
            gate_cap,
            gate_ramp_slope,
            gate_ramp_ceil,
            unverified_cap,
            confidence_prior,
        ).encode("utf-8")
    ).hexdigest()


CANONICAL_REGISTRY_HASH = registry_hash(DEFAULT_REGISTRY, BUCKET_WEIGHTS)


def band_score(x: Optional[float], floor: float, target: float, elite: float) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    pairs = sorted(zip((floor, target, elite), (0.0, 5.0, 10.0)))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return float(np.interp(x, xs, ys))


def gaap_gap_subscore(net_income: Optional[float], normalized_income: Optional[float]) -> Optional[float]:
    if net_income is None or normalized_income is None:
        return None
    ni, norm = float(net_income), float(normalized_income)
    if not (math.isfinite(ni) and math.isfinite(norm)):
        return None
    if abs(ni) < 1e-6 * max(abs(norm), 1.0):
        return None
    gap = abs(norm - ni) / abs(ni)
    return band_score(gap, *GAAP_GAP_ANCHORS)


def validate_registry(registry: Sequence[MetricSpec], bucket_weights: Mapping[str, float]) -> None:
    w_sum = sum(bucket_weights.values())
    if not math.isclose(w_sum, 1.0, abs_tol=1e-9):
        raise ValueError(f"Bucket weights must sum to 1.0, got {w_sum}")

    bucket_metric_weights: Dict[str, float] = {}
    for spec in registry:
        if spec.weight is not None:
            bucket_metric_weights[spec.bucket] = bucket_metric_weights.get(spec.bucket, 0.0) + spec.weight

        anchors = [spec.floor, spec.target, spec.elite]
        is_increasing = all(x < y for x, y in zip(anchors, anchors[1:]))
        is_decreasing = all(x > y for x, y in zip(anchors, anchors[1:]))
        if not (is_increasing or is_decreasing):
            raise ValueError(f"Metric {spec.key} anchors must be strictly monotonic: {anchors}")

    for b_key, w_sum in bucket_metric_weights.items():
        if not math.isclose(w_sum, 1.0, abs_tol=1e-9):
            raise ValueError(f"Intra-bucket weights for '{b_key}' must sum to 1.0, got {w_sum}")


def validate_lens(
    registry: Sequence[MetricSpec],
    bucket_weights: Mapping[str, float],
    gate_trigger: float = DEFAULT_GATE_TRIGGER,
    gate_cap: float = DEFAULT_GATE_CAP,
    gate_ramp_slope: float = DEFAULT_GATE_RAMP_SLOPE,
    gate_ramp_ceil: float = DEFAULT_GATE_RAMP_CEIL,
    unverified_cap: float = DEFAULT_UNVERIFIED_CAP,
    confidence_prior: float = DEFAULT_CONFIDENCE_PRIOR,
) -> None:
    keys = [s.key for s in registry]
    if len(keys) != len(set(keys)):
        raise ValueError("L1: duplicate metric keys in lens registry")
    canonical_keys = set(_CANONICAL_BY_KEY)
    if set(keys) != canonical_keys:
        extra = sorted(set(keys) - canonical_keys)
        missing = sorted(canonical_keys - set(keys))
        raise ValueError(
            f"L1: lens metric set must equal the canonical {len(canonical_keys)} keys "
            f"(extra={extra}, missing={missing}); soft-drop is weight 0, not removal"
        )

    if set(bucket_weights) != set(BUCKET_WEIGHTS):
        raise ValueError(
            f"L2: lens bucket_weights keys must be exactly {sorted(BUCKET_WEIGHTS)}, "
            f"got {sorted(bucket_weights)}"
        )
    if any(float(v) < 0 for v in bucket_weights.values()):
        raise ValueError("L6: bucket weights must be >= 0")
    total = sum(float(v) for v in bucket_weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"L6: bucket weights must sum to 1.0, got {total}")

    per_bucket: Dict[str, List[Optional[float]]] = {}
    for spec in registry:
        canon = _CANONICAL_BY_KEY[spec.key]

        if spec.bucket != canon.bucket:
            raise ValueError(f"L2: metric '{spec.key}' must stay in bucket '{canon.bucket}', got '{spec.bucket}'")

        canon_dir = 1.0 if canon.elite > canon.floor else -1.0
        lens_span = float(spec.elite) - float(spec.floor)
        if lens_span == 0.0 or (lens_span > 0) != (canon_dir > 0):
            raise ValueError(
                f"L3: metric '{spec.key}' anchor direction must match canonical "
                f"({'increasing' if canon_dir > 0 else 'decreasing'})"
            )

        if bool(spec.is_gate) != bool(canon.is_gate):
            raise ValueError(f"L4: is_gate flag for '{spec.key}' is canonical-fixed in v1")
        if canon.is_gate and (float(spec.floor), float(spec.target), float(spec.elite)) != (
            float(canon.floor),
            float(canon.target),
            float(canon.elite),
        ):
            raise ValueError(
                f"L4: gate metric '{spec.key}' anchors are canonical-fixed in v1 "
                "(a lens softening them converts a risk floor into an opinion)"
            )

        if spec.weight is not None and float(spec.weight) < 0:
            raise ValueError(f"L5: metric weight for '{spec.key}' must be >= 0")
        per_bucket.setdefault(spec.bucket, []).append(spec.weight)

    if not (
        math.isclose(float(gate_trigger), DEFAULT_GATE_TRIGGER, abs_tol=1e-12)
        and math.isclose(float(gate_cap), DEFAULT_GATE_CAP, abs_tol=1e-12)
        and math.isclose(float(gate_ramp_slope), DEFAULT_GATE_RAMP_SLOPE, abs_tol=1e-12)
        and math.isclose(float(gate_ramp_ceil), DEFAULT_GATE_RAMP_CEIL, abs_tol=1e-12)
        and math.isclose(float(unverified_cap), DEFAULT_UNVERIFIED_CAP, abs_tol=1e-12)
        and math.isclose(float(confidence_prior), DEFAULT_CONFIDENCE_PRIOR, abs_tol=1e-12)
    ):
        raise ValueError("L4: gate_trigger/gate_cap/ramp/unverified parameters are canonical-fixed in v1")

    for b_key, ws in per_bucket.items():
        explicit = [w for w in ws if w is not None]
        if explicit and len(explicit) != len(ws):
            raise ValueError(
                f"L5: bucket '{b_key}' mixes explicit and None metric weights; "
                "weights are all-None (equal) or all-explicit summing to 1.0"
            )

    validate_registry(registry, bucket_weights)


def lens_to_definition(
    registry: Sequence[MetricSpec],
    bucket_weights: Mapping[str, float],
    gate_trigger: float = DEFAULT_GATE_TRIGGER,
    gate_cap: float = DEFAULT_GATE_CAP,
) -> Dict[str, Any]:
    return {
        "specs": [
            {
                "key": s.key,
                "label": s.label,
                "bucket": s.bucket,
                "floor": float(s.floor),
                "target": float(s.target),
                "elite": float(s.elite),
                "is_gate": bool(s.is_gate),
                "weight": None if s.weight is None else float(s.weight),
                "fmt": s.fmt,
            }
            for s in registry
        ],
        "bucket_weights": {str(k): float(v) for k, v in bucket_weights.items()},
        "gate_trigger": float(gate_trigger),
        "gate_cap": float(gate_cap),
    }


def lens_from_definition(
    definition: Mapping[str, Any],
) -> Tuple[Tuple[MetricSpec, ...], Dict[str, float], float, float]:
    specs = tuple(
        MetricSpec(
            key=d["key"],
            label=d.get("label") or d["key"],
            bucket=d["bucket"],
            floor=float(d["floor"]),
            target=float(d["target"]),
            elite=float(d["elite"]),
            is_gate=bool(d.get("is_gate", False)),
            weight=None if d.get("weight") is None else float(d["weight"]),
            fmt=d.get("fmt", "ratio"),
        )
        for d in definition["specs"]
    )
    bw = {str(k): float(v) for k, v in definition["bucket_weights"].items()}
    gt = float(definition.get("gate_trigger", DEFAULT_GATE_TRIGGER))
    gc = float(definition.get("gate_cap", DEFAULT_GATE_CAP))
    return specs, bw, gt, gc


def score_metrics(
    raw: Mapping[str, Any],
    registry: Sequence[MetricSpec] = DEFAULT_REGISTRY,
    bucket_weights: Mapping[str, float] = BUCKET_WEIGHTS,
    gate_trigger: float = DEFAULT_GATE_TRIGGER,
    gate_cap: float = DEFAULT_GATE_CAP,
    gate_ramp_slope: float = DEFAULT_GATE_RAMP_SLOPE,
    gate_ramp_ceil: float = DEFAULT_GATE_RAMP_CEIL,
    unverified_cap: float = DEFAULT_UNVERIFIED_CAP,
    confidence_prior: float = DEFAULT_CONFIDENCE_PRIOR,
) -> ScoreResult:
    validate_registry(registry, bucket_weights)

    _rhash = registry_hash(
        registry,
        bucket_weights,
        gate_trigger,
        gate_cap,
        gate_ramp_slope,
        gate_ramp_ceil,
        unverified_cap,
        confidence_prior,
    )
    _rid = CANONICAL_REGISTRY_ID if _rhash == CANONICAL_REGISTRY_HASH else LENS_REGISTRY_ID

    metrics: Dict[str, MetricDetail] = {}
    bucket_sums: Dict[str, float] = {}
    bucket_weights_used: Dict[str, float] = {}
    bucket_counts: Dict[str, int] = {}

    weighted_confidence = 0.0

    for spec in registry:
        val = raw.get(spec.key)

        if spec.key == "operating_leverage" and raw.get(f"{spec.key}_fallback"):
            sub = band_score(val, *OPLEV_FALLBACK_ANCHORS)
        else:
            sub = band_score(val, spec.floor, spec.target, spec.elite)

        reason = raw.get(f"{spec.key}_reason")
        if reason is None and val is None:
            reason = "missing"

        metrics[spec.key] = MetricDetail(key=spec.key, raw=val, subscore=sub, reason=reason)

        if sub is not None:
            weighted_confidence += CONFIDENCE_WEIGHTS.get(spec.key, 1.0 / len(registry))
            w = spec.weight if spec.weight is not None else (1.0 / BUCKET_CARDINALITY[spec.bucket])
            bucket_sums[spec.bucket] = bucket_sums.get(spec.bucket, 0.0) + (sub * w)
            bucket_weights_used[spec.bucket] = bucket_weights_used.get(spec.bucket, 0.0) + w
            bucket_counts[spec.bucket] = bucket_counts.get(spec.bucket, 0) + 1

    # S8: Min-Bucket Rule -> count_present >= ceil(cardinality / 2)
    buckets: Dict[str, Optional[float]] = {}
    present_buckets_w = 0.0
    pre_gate_sum = 0.0

    for b_key, w_b in bucket_weights.items():
        cnt = bucket_counts.get(b_key, 0)
        min_required = math.ceil(BUCKET_CARDINALITY[b_key] / 2.0)
        if cnt >= min_required and b_key in bucket_sums and bucket_weights_used[b_key] > 0:
            b_score = bucket_sums[b_key] / bucket_weights_used[b_key]
            buckets[b_key] = b_score
            pre_gate_sum += b_score * w_b
            present_buckets_w += w_b
        else:
            buckets[b_key] = None

    composite_pre_gate = (pre_gate_sum / present_buckets_w) if present_buckets_w > 0 else None

    # S1/S4: Graduated Gate Cap & Unverified State Machine
    gates_fired = []
    gates_unverified = []
    gate_caps = []

    for spec in registry:
        if spec.is_gate:
            sub = metrics[spec.key].subscore
            if sub is None:
                gates_unverified.append(spec.key)
            else:
                if sub <= gate_trigger:
                    gates_fired.append(spec.key)
                    c_i = gate_cap
                elif sub < gate_ramp_ceil:
                    c_i = gate_cap + gate_ramp_slope * (sub - gate_trigger)
                else:
                    c_i = float("inf")
                gate_caps.append(c_i)

    effective_cap = float("inf")
    if gate_caps:
        effective_cap = min(effective_cap, min(gate_caps))
    if gates_unverified:
        effective_cap = min(effective_cap, unverified_cap)

    composite = composite_pre_gate
    gate_cap_applied = None

    if composite is not None and math.isfinite(effective_cap):
        if composite > effective_cap:
            composite = effective_cap
            gate_cap_applied = effective_cap

    # S7: Shrunk Composite
    confidence = min(1.0, max(0.0, weighted_confidence))
    composite_shrunk = None
    if composite is not None:
        composite_shrunk = confidence * composite + (1.0 - confidence) * confidence_prior

    return ScoreResult(
        composite=composite,
        composite_pre_gate=composite_pre_gate,
        composite_shrunk=composite_shrunk,
        gate_cap_applied=gate_cap_applied,
        buckets=buckets,
        metrics=metrics,
        gates_fired=tuple(gates_fired),
        gates_unverified=tuple(gates_unverified),
        confidence=confidence,
        engine_version=ENGINE_VERSION,
        registry_id=_rid,
        registry_hash=_rhash,
    )
