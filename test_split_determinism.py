# filepath: compounderos_app/test_split_determinism.py
import json
import math
import os
import struct
import numpy as np
import pandas as pd
from compounderos_app.scoring_engine import (
    DEFAULT_REGISTRY,
    score_metrics,
    CANONICAL_REGISTRY_HASH,
    band_score,
)
from compounderos_app.analytics_core import RoicMomentumAnalyzer


def float_to_bytes(f: float) -> bytes:
    """Canonicalize float representations cross-architecture."""
    if f is None or math.isnan(f):
        return b""
    return struct.pack("<d", f)


def assert_byte_equal(a: float, b: float, name: str):
    assert float_to_bytes(a) == float_to_bytes(b), f"Mismatch in {name}: {a} != {b}"


def test_tier_1_determinism():
    """E: Prove bit-identical replication via golden reference parsing."""
    gv_path = os.path.join(os.path.dirname(__file__), "golden_vector.json")
    assert os.path.exists(
        gv_path
    ), f"FAIL: {gv_path} not found. Run generate_golden_vector.py first."

    with open(gv_path, "r") as f:
        gv = json.load(f)

    assert (
        CANONICAL_REGISTRY_HASH == gv["canonical_hash"]
    ), "Engine registry has drifted from the frozen golden vector"

    for i, case in enumerate(gv["cases"]):
        parsed = case["inputs"]
        expected = case["outputs"]

        res = score_metrics(parsed, registry=DEFAULT_REGISTRY)

        assert_byte_equal(res.composite, expected["composite"], f"composite (Case {i})")
        assert_byte_equal(
            res.confidence, expected["confidence"], f"confidence (Case {i})"
        )
        assert list(res.gates_fired) == expected["gates_fired"], f"gates_fired (Case {i})"


def test_tier_2_determinism():
    """E: Prove tolerance-based equality for BLAS/np.polyfit paths."""
    gv_path = os.path.join(os.path.dirname(__file__), "golden_vector.json")
    assert os.path.exists(
        gv_path
    ), f"FAIL: {gv_path} not found. Run generate_golden_vector.py first."

    with open(gv_path, "r") as f:
        gv = json.load(f)

    tier2 = gv["tier2_case"]
    dates = pd.date_range("2020-03-31", periods=len(tier2["roic_series"]), freq="QE")
    s = pd.Series(tier2["roic_series"], index=dates)

    rma = RoicMomentumAnalyzer()
    slope = rma.rolling_slope(s)
    local_slope = float(slope.iloc[-1])

    expected_slope = float(tier2["expected_slope"])
    assert np.allclose(
        local_slope, expected_slope, atol=1e-9
    ), "Tier 2 tolerance failed (atol=1e-9)"

    local_sub = float(band_score(local_slope, -0.03, 0.00, 0.03))
    expected_sub = float(tier2["expected_subscore"])

    assert_byte_equal(local_sub, expected_sub, "tier_2_quantized_subscore")


if __name__ == "__main__":
    test_tier_1_determinism()
    test_tier_2_determinism()
    print("Tier 1 & Tier 2 Determinism Tests Passed.")
