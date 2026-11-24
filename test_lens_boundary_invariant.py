"""
test_lens_boundary_invariant.py
CompounderOS v2 — mechanical enforcement of the Style & Strategy Lens boundary.

The lens contract is an INVARIANT, not a coding convention. Four guards:

  1. GOLDEN (synthetic + real-DB): score_metrics(json.loads(raw_json), CANONICAL)
     must reproduce the stored (score, confidence, gates) for every history row.
     Divergence means the engine changed under a frozen engine_version (a
     provenance bug) OR raw_json is lossy (a capture bug). Both are corruption.

  2. CONTRACT: a valid lens moves the composite but leaves confidence and gate
     state invariant (L1/L4), and never mutates in-process global state (Q5 #3).

  3. ENFORCEMENT: score_metrics stamps hash-derived provenance;
     record_quality_result raises on any non-canonical ScoreResult (Q1 guard);
     BUCKET_WEIGHTS is frozen; validate_lens rejects every silent-failure class
     (L1–L6); upsert_lens recomputes lens_hash and rejects invalid lenses (L8).

  4. SUBSTRATE: trajectory_history is written only through its choke point
     (Prompt 3 §3) with same-day replace and composition validation.

Run from the repo root:
    pytest -q compounderos_app/test_lens_boundary_invariant.py
    python compounderos_app/test_lens_boundary_invariant.py   (synthetic only)
"""
import json
import math
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compounderos_app import scoring_engine as E
from compounderos_app.database import DataStore

TOL = 1e-6


# ----------------------------------------------------------------------------- helpers
def _canonical_rescore(raw_json: str):
    """Re-score a stored raw_json under the CANONICAL registry (engine defaults)."""
    return E.score_metrics(json.loads(raw_json))


def _row_gate_state(gates_fired, gates_unverified) -> str:
    """Mirror record_quality_result's as-built 4-value enum mapping (database.py)."""
    fired, unver = bool(gates_fired), bool(gates_unverified)
    if fired and unver:
        return "fired_unverified"
    if fired:
        return "fired"
    if unver:
        return "unverified"
    return "pass"


def _approx(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, abs_tol=TOL)


def _raises(exc_type, fn) -> bool:
    try:
        fn()
    except exc_type:
        return True
    return False


# ----------------------------------------------------------------------------- fixtures
_SAMPLE_RAW = {
    "gross_margin": 0.55, "revenue_growth": 0.12, "operating_leverage": 0.04,
    "operating_leverage_fallback": True,          # exercises OPLEV_FALLBACK_ANCHORS
    "roic": 0.22, "roic_momentum": 0.01,
    "net_debt_ebitda": 1.4, "current_ratio": 1.6, "gaap_gap": None,  # F-A2 gap
    "capex_revenue": 0.06, "earnings_quality": 1.10,
    "share_dilution": -0.01, "sbc_revenue": 0.04,
}


def _make_valid_lens():
    """A lens that only re-parameterizes (L1–L6): 12 keys, buckets, gates all fixed."""
    reg = tuple(
        replace(s, floor=s.floor * 0.9, target=s.target * 0.9, elite=s.elite * 0.9)
        if (not s.is_gate and s.key not in ("share_dilution", "operating_leverage", "gaap_gap"))
        else s
        for s in E.DEFAULT_REGISTRY
    )
    weights = {"profitability": 0.50, "efficiency": 0.30,
               "balance_sheet": 0.15, "capital_alloc": 0.05}
    return reg, weights


# ----------------------------------------------------------------------------- invariants
def test_golden_synthetic():
    raw_json = json.dumps(_SAMPLE_RAW)
    canon = E.score_metrics(_SAMPLE_RAW)
    got = _canonical_rescore(raw_json)
    assert _approx(canon.composite, got.composite)
    assert _approx(canon.confidence, got.confidence)
    assert canon.gates_fired == got.gates_fired
    assert canon.gates_unverified == got.gates_unverified


def test_lens_moves_score_but_confidence_and_gates_invariant():
    raw_json = json.dumps(_SAMPLE_RAW)
    canon = _canonical_rescore(raw_json)
    reg, weights = _make_valid_lens()
    E.validate_lens(reg, weights)                 # the fixture must itself be a legal lens
    lensed = E.score_metrics(json.loads(raw_json), reg, weights)
    assert not _approx(canon.composite, lensed.composite), "lens should change the composite"
    assert _approx(canon.confidence, lensed.confidence), "L1: confidence is registry-invariant"
    assert canon.gates_fired == lensed.gates_fired, "L4: gate identity fixed"
    assert canon.gates_unverified == lensed.gates_unverified, "L4: unverified set fixed"


def test_no_global_mutation_after_lens():
    raw_json = json.dumps(_SAMPLE_RAW)
    before = _canonical_rescore(raw_json)
    reg, weights = _make_valid_lens()
    _ = E.score_metrics(json.loads(raw_json), reg, weights)   # a lens call in between
    after = _canonical_rescore(raw_json)
    assert _approx(before.composite, after.composite), "canonical must be unchanged by lens calls"


# ----------------------------------------------------------------------------- enforcement
def test_provenance_stamping():
    canon = E.score_metrics(_SAMPLE_RAW)
    assert canon.registry_id == E.CANONICAL_REGISTRY_ID
    assert canon.registry_hash == E.CANONICAL_REGISTRY_HASH

    reg, weights = _make_valid_lens()
    lensed = E.score_metrics(dict(_SAMPLE_RAW), reg, weights)
    assert lensed.registry_id == E.LENS_REGISTRY_ID
    assert lensed.registry_hash != E.CANONICAL_REGISTRY_HASH
    assert lensed.registry_hash == E.registry_hash(reg, weights), "hash is a stable identity"


def test_bucket_weights_frozen():
    # mappingproxy: __setitem__ raises TypeError; .update doesn't even exist.
    assert _raises((TypeError, AttributeError),
                   lambda: E.BUCKET_WEIGHTS.update({"profitability": 0.9}))

    def _setitem():
        E.BUCKET_WEIGHTS["profitability"] = 0.9
    assert _raises(TypeError, _setitem), "Q5 #3: module weights must be immutable"


def test_write_guard_blocks_lens_scores():
    store = DataStore(":memory:")
    canon = E.score_metrics(_SAMPLE_RAW)
    store.record_quality_result("TEST", "IT", canon, {}, dict(_SAMPLE_RAW))

    row = store.conn.execute(
        "SELECT score, registry_hash FROM sp500_scores_history WHERE symbol='TEST'"
    ).fetchone()
    assert _approx(row[0], canon.composite)
    assert row[1] == E.CANONICAL_REGISTRY_HASH, "canonical writes carry the audit hash"

    reg, weights = _make_valid_lens()
    lensed = E.score_metrics(dict(_SAMPLE_RAW), reg, weights)
    assert _raises(ValueError,
                   lambda: store.record_quality_result("TEST", "IT", lensed, {}, dict(_SAMPLE_RAW))), \
        "Q1 guard: a lens-scored ScoreResult must never reach canonical tables"
    n = store.conn.execute("SELECT COUNT(*) FROM sp500_scores_history").fetchone()[0]
    assert n == 1, "the refused write must leave history untouched"
    store.close()


def test_validate_lens_contract():
    reg, weights = _make_valid_lens()
    E.validate_lens(reg, weights)  # must pass

    base = E.DEFAULT_REGISTRY
    cw = dict(E.BUCKET_WEIGHTS)

    # L1 — no additions, no removals.
    extra = base + (E.MetricSpec("thirteenth", "13th", "profitability", 0.0, 0.5, 1.0),)
    assert _raises(ValueError, lambda: E.validate_lens(extra, cw))
    assert _raises(ValueError, lambda: E.validate_lens(base[:-1], cw))

    # L2 — no re-bucketing; bucket set fixed.
    rebucketed = tuple(replace(s, bucket="efficiency") if s.key == "gross_margin" else s
                       for s in base)
    assert _raises(ValueError, lambda: E.validate_lens(rebucketed, cw))
    novel = {"profitability": 0.35, "efficiency": 0.30, "balance_sheet": 0.20, "momentum": 0.15}
    assert _raises(ValueError, lambda: E.validate_lens(base, novel))

    # L3 — direction fixed (still monotonic, so validate_registry alone would pass this).
    inverted = tuple(replace(s, floor=0.60, target=0.40, elite=0.20) if s.key == "gross_margin" else s
                     for s in base)
    assert _raises(ValueError, lambda: E.validate_lens(inverted, cw))

    # L4 — gate anchors, is_gate flags, and gate params canonical-fixed.
    soft_gate = tuple(replace(s, floor=6.0) if s.key == "net_debt_ebitda" else s for s in base)
    assert _raises(ValueError, lambda: E.validate_lens(soft_gate, cw))
    flag_flip = tuple(replace(s, is_gate=True) if s.key == "current_ratio" else s for s in base)
    assert _raises(ValueError, lambda: E.validate_lens(flag_flip, cw))
    assert _raises(ValueError, lambda: E.validate_lens(base, cw, gate_cap=5.0))

    # L5 — mixed None/explicit weights: validate_registry PASSES this (non-None
    # sum is 1.0) while the engine substitutes 1.0 for None and scores wrongly.
    mixed = tuple(replace(s, weight=1.0) if s.key == "capex_revenue" else s for s in base)
    assert _raises(ValueError, lambda: E.validate_lens(mixed, cw))

    # L6 — bucket weights must sum to 1.0.
    bad_sum = {"profitability": 0.35, "efficiency": 0.30, "balance_sheet": 0.20, "capital_alloc": 0.05}
    assert _raises(ValueError, lambda: E.validate_lens(base, bad_sum))


def test_lens_table_recomputes_hash_and_rejects_invalid():
    store = DataStore(":memory:")
    reg, weights = _make_valid_lens()
    defn = E.lens_to_definition(reg, weights)

    h1 = store.upsert_lens("test_lens", "Test Lens", defn)
    assert h1 == E.registry_hash(reg, weights)
    got = store.get_lens("test_lens")
    assert got["lens_hash"] == h1 and got["definition"] == json.loads(json.dumps(defn))

    # Edit → hash must be recomputed (L8), not trusted from the caller.
    defn2 = json.loads(json.dumps(defn))
    for d in defn2["specs"]:
        if d["key"] == "gross_margin":
            d["elite"] = 0.70
    h2 = store.upsert_lens("test_lens", "Test Lens", defn2)
    assert h2 != h1

    # An invalid lens (softened gate anchor) never reaches the table.
    bad = json.loads(json.dumps(defn2))
    for d in bad["specs"]:
        if d["key"] == "net_debt_ebitda":
            d["floor"] = 6.0
    assert _raises(ValueError, lambda: store.upsert_lens("test_lens", "Test Lens", bad))
    assert store.get_lens("test_lens")["lens_hash"] == h2, "rejected write must not clobber"
    store.close()


def test_trajectory_choke_point():
    store = DataStore(":memory:")
    peers = sorted(["0000320193", "0000789019", "0001045810", "0000012345", "0000067890"])
    inputs = {"median_revision": 0.031, "breadth": 0.66, "median_peer_growth": 0.12}
    store.record_trajectory_result("NVDA", 7.1, 0.8, "AB", "industry",
                                   peers, inputs, "traj-0.1.0")
    row = store.conn.execute(
        "SELECT trajectory, composition, peer_set_json, raw_inputs_json, engine_version "
        "FROM trajectory_history WHERE symbol='NVDA'").fetchone()
    assert _approx(row[0], 7.1) and row[1] == "AB" and row[4] == "traj-0.1.0"
    assert json.loads(row[2]) == peers and json.loads(row[3]) == inputs

    # Same-day replace (§H.1 convention), not a duplicate row.
    store.record_trajectory_result("NVDA", 6.9, 0.7, "A", "industry",
                                   peers[:4], inputs, "traj-0.1.0")
    rows = store.conn.execute(
        "SELECT trajectory, composition FROM trajectory_history WHERE symbol='NVDA'").fetchall()
    assert len(rows) == 1 and _approx(rows[0][0], 6.9) and rows[0][1] == "A"

    # Composition flag is a closed enum.
    assert _raises(ValueError,
                   lambda: store.record_trajectory_result("NVDA", 5.0, 0.5, "B_ONLY",
                                                          "industry", peers, inputs, "traj-0.1.0"))
    store.close()


# ----------------------------------------------------------------------------- real DB
# Point at the live store to guard the boundary against engine/raw_json drift.
# Skipped automatically if the DB or the table is absent.
DB_PATH = os.environ.get("COMPOUNDEROS_DB")  # e.g. "sp500_scanner.db"


def test_golden_over_real_history():
    import pytest
    if not DB_PATH or not os.path.exists(DB_PATH):
        pytest.skip("set COMPOUNDEROS_DB to run the real-history golden check")
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT symbol, scan_date, score, confidence, gate_state,
               gates_fired, gates_unverified, engine_version, raw_json
        FROM sp500_scores_history
        WHERE raw_json IS NOT NULL AND source = 'scan'
    """).fetchall()
    conn.close()

    mismatches = []
    for (sym, sd, score, conf, gate_state, gf, gu, ev, raw_json) in rows:
        # Only rows written under the CURRENT engine can be expected to reproduce.
        if ev != E.ENGINE_VERSION:
            continue
        r = _canonical_rescore(raw_json)
        r_gate = _row_gate_state(r.gates_fired, r.gates_unverified)
        if not (_approx(score, r.composite) and _approx(conf, r.confidence) and gate_state == r_gate):
            mismatches.append((sym, sd, (score, conf, gate_state),
                               (r.composite, r.confidence, r_gate)))

    assert not mismatches, (
        f"{len(mismatches)} history rows are not reproducible from raw_json under "
        f"the canonical engine — provenance or capture bug. First: {mismatches[:3]}"
    )


def test_write_guard_blocks_spoofed_registry_hash_and_persists_provenance():
    store = DataStore(":memory:")
    canon = E.score_metrics(_SAMPLE_RAW)
    store.record_quality_result("AAPL", "IT", canon, {}, dict(_SAMPLE_RAW))

    hist_row = store.conn.execute(
        "SELECT registry_id, registry_hash FROM sp500_scores_history WHERE symbol='AAPL'"
    ).fetchone()
    assert hist_row[0] == E.CANONICAL_REGISTRY_ID
    assert hist_row[1] == E.CANONICAL_REGISTRY_HASH

    snap_row = store.conn.execute(
        "SELECT registry_id, registry_hash FROM sp500_scores WHERE symbol='AAPL'"
    ).fetchone()
    assert snap_row[0] == E.CANONICAL_REGISTRY_ID
    assert snap_row[1] == E.CANONICAL_REGISTRY_HASH

    # Spoofed ScoreResult with canonical registry_id but non-canonical registry_hash must be rejected
    spoofed = replace(canon, registry_id=E.CANONICAL_REGISTRY_ID, registry_hash="deadbeef" * 8)
    assert _raises(
        ValueError,
        lambda: store.record_quality_result("AAPL", "IT", spoofed, {}, dict(_SAMPLE_RAW)),
    )
    store.close()


def test_snapshot_confidence_monotonicity_same_day():
    store = DataStore(":memory:")
    high_conf = E.score_metrics(_SAMPLE_RAW)
    # Partial raw dict with lower confidence
    low_raw = {"roic": 0.10, "gross_margin": 0.30}
    low_conf = E.score_metrics(low_raw)
    assert low_conf.confidence < high_conf.confidence

    store.record_quality_result("MSFT", "IT", high_conf, {}, dict(_SAMPLE_RAW))
    store.record_quality_result("MSFT", "IT", low_conf, {}, dict(low_raw))

    snap_score = store.conn.execute(
        "SELECT score FROM sp500_scores WHERE symbol='MSFT'"
    ).fetchone()[0]
    hist_score = store.conn.execute(
        "SELECT score FROM sp500_scores_history WHERE symbol='MSFT'"
    ).fetchone()[0]
    assert _approx(snap_score, high_conf.composite)
    assert _approx(hist_score, high_conf.composite)
    store.close()


def test_pipeline_trajectory_stage_preserves_pure_raw_inputs():
    import threading
    from unittest.mock import patch
    from scanner_service.pipeline import ScannerPipeline

    store = DataStore(":memory:", role="sor", check_same_thread=False)
    stop_event = threading.Event()
    pipeline = ScannerPipeline(store, {}, stop_event)
    peers = ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM"]
    pipeline.universe = [(sym, "Information Technology", "Systems Software") for sym in peers]

    pure_raw = {
        "extraction_version": "traj-ext-1.1",
        "quarters_available": 12,
        "roic_current": 0.25,
        "roic_4q_ago": 0.20,
        "roic_8q_ago": 0.18,
        "margin_current": 0.30,
        "margin_4q_ago": 0.27,
        "rev_growth_current": 0.15,
        "rev_growth_4q_ago": 0.12,
        "eps_fy1_current": 5.50,
        "eps_fy1_90d_ago": 5.00,
    }

    with patch.object(
        pipeline.gateway, "fetch_estimates", lambda sym: {}
    ), patch(
        "compounderos_app.trajectory_extraction.extract_trajectory_inputs",
        lambda *a, **kw: dict(pure_raw),
    ):
        ok = pipeline._run_trajectory_stage()
        assert ok is True

    row = store.conn.execute(
        "SELECT raw_inputs_json, derived_json FROM trajectory_history WHERE symbol='AAPL'"
    ).fetchone()
    assert row is not None
    stored_raw = json.loads(row[0])
    for derived_key in ("median_revision", "breadth", "median_peer_growth", "peer_n", "peer_pos_n"):
        assert derived_key not in stored_raw, f"Derived peer key {derived_key} polluted raw_inputs_json"
    store.close()


if __name__ == "__main__":
    test_golden_synthetic()
    test_lens_moves_score_but_confidence_and_gates_invariant()
    test_no_global_mutation_after_lens()
    test_provenance_stamping()
    test_bucket_weights_frozen()
    test_write_guard_blocks_lens_scores()
    test_validate_lens_contract()
    test_lens_table_recomputes_hash_and_rejects_invalid()
    test_trajectory_choke_point()
    test_write_guard_blocks_spoofed_registry_hash_and_persists_provenance()
    test_snapshot_confidence_monotonicity_same_day()
    test_pipeline_trajectory_stage_preserves_pure_raw_inputs()
    print("synthetic + enforcement invariants OK — set COMPOUNDEROS_DB and run under "
          "pytest for the real-history golden guard")

