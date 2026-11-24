# filepath: compounderos_app/trajectory_engine.py
"""CompounderOS v2 Trajectory Engine (v1.1).
Deterministic, pure scalar math for Trajectory A/G/B components and confidence."""

import statistics
from typing import Dict, Any, Tuple, List, Optional
from .scoring_engine import band_score

TRAJ_ENGINE_VERSION = "traj-1.1.0"

# ==============================================================================
# PROVISIONAL CALIBRATED ANCHORS & CONSTANTS (v1.1)
# Calibrated against empirical probe; adjust here for full-universe recalibration.
# ==============================================================================
DEN_FLOOR = 0.10             # Pathological-zero denominator floor ($/share)
REV_FLOOR = -0.06            # Asymmetric negative revision anchor (target 0.0)
REV_ELITE = 0.11             # Asymmetric positive revision anchor (ties at 10.0)
BREADTH_FLOOR = -0.5         # Self revision breadth floor
BREADTH_ELITE = 0.5          # Self revision breadth elite
G_FLOOR = 0.02               # Forward revenue growth (1y) floor
G_TARGET = 0.08              # Forward revenue growth (1y) target
G_ELITE = 0.22               # Forward revenue growth (1y) elite
B_FLOOR = -0.04              # Peer-relative revision floor
B_ELITE = 0.04               # Peer-relative revision elite
MIN_PEERS = 5                # Minimum peer count for relative comparison


class EstimateInputs:
    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def self_rev_0y(self) -> Optional[float]:
        val = self.raw.get("self_rev_0y")
        if val is not None:
            return float(val)
        cur = self.raw.get("eps_cur_0y")
        d90 = self.raw.get("eps_90d_0y")
        if cur is None or d90 is None:
            return None
        return max(-1.0, min(1.0, (float(cur) - float(d90)) / max(abs(float(d90)), DEN_FLOOR)))

    @property
    def self_rev_1y(self) -> Optional[float]:
        val = self.raw.get("self_rev_1y")
        if val is not None:
            return float(val)
        cur = self.raw.get("eps_cur_1y")
        d90 = self.raw.get("eps_90d_1y")
        if cur is None or d90 is None:
            return None
        return max(-1.0, min(1.0, (float(cur) - float(d90)) / max(abs(float(d90)), DEN_FLOOR)))

    @property
    def rev_growth_1y(self) -> Optional[float]:
        val = self.raw.get("rev_growth_1y")
        if val is not None:
            return float(val)
        return None

    @property
    def n_analysts(self) -> Optional[float]:
        val = self.raw.get("self_n_analysts")
        if val is not None:
            return float(val)
        vals = [
            float(v) for v in (
                self.raw.get("n_analysts_0y"),
                self.raw.get("n_analysts_1y"),
            ) if v is not None
        ]
        return max(vals) if vals else None

    @property
    def up30(self) -> float:
        val = self.raw.get("self_up30")
        if val is not None:
            return float(val)
        u0 = self.raw.get("up30_0y") or 0.0
        u1 = self.raw.get("up30_1y") or 0.0
        return float(u0) + float(u1)

    @property
    def down30(self) -> float:
        val = self.raw.get("self_down30")
        if val is not None:
            return float(val)
        d0 = self.raw.get("down30_0y") or 0.0
        d1 = self.raw.get("down30_1y") or 0.0
        return float(d0) + float(d1)


def compute_A(inputs: EstimateInputs) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (A_Score, self_revision, self_rev_breadth) or None if withheld."""
    r0 = inputs.self_rev_0y
    r1 = inputs.self_rev_1y

    self_revision: Optional[float] = None
    if r0 is not None and r1 is not None:
        self_revision = 0.35 * r0 + 0.65 * r1
    elif r0 is not None:
        self_revision = r0
    elif r1 is not None:
        self_revision = r1

    votes = inputs.up30 + inputs.down30
    self_rev_breadth: Optional[float] = None
    if votes >= 4.0:
        self_rev_breadth = (inputs.up30 - inputs.down30) / float(votes + 2.0)

    if self_revision is None:
        return None, None, self_rev_breadth

    rev_score = band_score(self_revision, REV_FLOOR, 0.0, REV_ELITE)
    if self_rev_breadth is not None:
        br_score = band_score(self_rev_breadth, BREADTH_FLOOR, 0.0, BREADTH_ELITE)
        a_score = (0.6 * rev_score) + (0.4 * br_score)
    else:
        a_score = rev_score

    return a_score, self_revision, self_rev_breadth


def peer_aggregates(peer_inputs: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float], int, int]:
    """Returns (peer_med_rev, breadth, median_peer_growth, peer_n, peer_pos_n)."""
    usable_revs = []
    usable_growth = []
    positive_revs = 0

    for pi in peer_inputs:
        r = pi.get("self_revision")
        if r is not None:
            usable_revs.append(float(r))
            if float(r) > 0.0:
                positive_revs += 1

        g = pi.get("rev_growth_1y")
        if g is None:
            g = pi.get("growth_h")
        if g is not None:
            usable_growth.append(float(g))

    peer_n = len(usable_revs)
    if peer_n == 0:
        return None, None, None, 0, 0

    med_rev = float(statistics.median(usable_revs))
    breadth = float(positive_revs) / float(peer_n)
    med_growth = float(statistics.median(usable_growth)) if usable_growth else None

    return med_rev, breadth, med_growth, peer_n, positive_revs


def compute_trajectory(raw_inputs: Dict[str, Any]) -> Tuple[Optional[float], float, str, Dict[str, Any]]:
    """Pure function mapping raw_inputs to (trajectory, confidence, composition, derivations)."""
    inputs = EstimateInputs(raw_inputs)
    a_score, self_rev, self_breadth = compute_A(inputs)

    g_score: Optional[float] = None
    if inputs.rev_growth_1y is not None:
        g_score = band_score(inputs.rev_growth_1y, G_FLOOR, G_TARGET, G_ELITE)

    med_rev = raw_inputs.get("median_revision")
    peer_n = raw_inputs.get("peer_n")
    if peer_n is None:
        peer_n = raw_inputs.get("n_eff", 0)

    b_score: Optional[float] = None
    b_delta: Optional[float] = None
    if self_rev is not None and med_rev is not None:
        b_delta = float(self_rev) - float(med_rev)

    if int(peer_n or 0) >= MIN_PEERS and b_delta is not None:
        b_score = band_score(b_delta, B_FLOOR, 0.0, B_ELITE)

    if a_score is None:
        derivations = {
            "self_revision": self_rev,
            "self_rev_breadth": self_breadth,
            "self_breadth": self_breadth,
            "a_score": None,
            "g_score": g_score,
            "b_score": b_score,
            "b_delta": b_delta,
        }
        return None, 0.0, "WITHHELD", derivations

    weights = {"A": 0.45}
    scores = {"A": a_score}
    if g_score is not None:
        weights["G"] = 0.35
        scores["G"] = g_score
    if b_score is not None:
        weights["B"] = 0.20
        scores["B"] = b_score

    total_weight = sum(weights.values())
    trajectory = float(sum(scores[k] * (weights[k] / total_weight) for k in weights))
    composition = "".join(k for k in ("A", "G", "B") if k in weights)

    usable_horizons = sum(1 for x in [inputs.self_rev_0y, inputs.self_rev_1y] if x is not None)
    votes = inputs.up30 + inputs.down30
    dA = 0.7 * (float(usable_horizons) / 2.0) + 0.3 * min(1.0, float(votes) / 8.0)
    dG = 1.0 if g_score is not None else 0.0
    dB = min(1.0, max(0.0, float(peer_n or 0)) / 10.0) if b_score is not None else 0.0

    conf = 0.45 * dA + 0.35 * dG + 0.20 * dB
    conf = max(0.0, min(1.0, float(conf)))

    derivations = {
        "self_revision": self_rev,
        "self_rev_breadth": self_breadth,
        "self_breadth": self_breadth,
        "a_score": a_score,
        "g_score": g_score,
        "b_score": b_score,
        "b_delta": b_delta,
    }

    return trajectory, conf, composition, derivations
