# filepath: compounderos_app/test_trajectory_extraction.py
import copy
import json
import sqlite3
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from compounderos_app.trajectory_extraction import extract_trajectory_inputs, EXTRACTION_VERSION
from compounderos_app.trajectory_engine import (
    compute_trajectory,
    EstimateInputs,
    compute_A,
    peer_aggregates,
    TRAJ_ENGINE_VERSION,
)
from compounderos_app.gics_map import GICS_MAP_VERSION, GICS_SUBIND_TO_IND


@pytest.fixture
def sample_estimates():
    idx = ["0y", "+1y", "0q", "+1q"]
    eps_trend = pd.DataFrame(
        {
            "current": [10.0, 11.5, 2.5, 2.8],
            "7daysAgo": [9.9, 11.4, 2.48, 2.78],
            "30daysAgo": [9.8, 11.2, 2.45, 2.75],
            "60daysAgo": [9.7, 11.0, 2.40, 2.70],
            "90daysAgo": [9.5, 10.8, 2.38, 2.68],
        },
        index=idx,
    )
    eps_revisions = pd.DataFrame(
        {
            "upLast7days": [2.0, 3.0, 1.0, 1.0],
            "upLast30days": [5.0, 6.0, 2.0, 3.0],
            "downLast30days": [1.0, 1.0, 0.0, 0.0],
            "downLast7Days": [0.0, 0.0, 0.0, 0.0],  # test inconsistent casing
            "currency": ["USD", "USD", "USD", "USD"],
        },
        index=idx,
    )
    earnings_estimate = pd.DataFrame(
        {
            "numberOfAnalysts": [25.0, 26.0, 18.0, 19.0],
            "growth": [0.15, 0.15, 0.10, 0.12],
            "yearAgoEps": [8.7, 10.0, 2.2, 2.5],
        },
        index=idx,
    )
    revenue_estimate = pd.DataFrame(
        {
            "growth": [0.12, 0.14, 0.08, 0.09],
            "avg": [10000.0, 11400.0, 2500.0, 2800.0],
            "yearAgoRevenue": [8928.0, 10000.0, 2314.0, 2568.0],
        },
        index=idx,
    )
    return {
        "eps_trend": eps_trend,
        "eps_revisions": eps_revisions,
        "earnings_estimate": earnings_estimate,
        "revenue_estimate": revenue_estimate,
    }


def test_extract_trajectory_inputs_full(sample_estimates):
    res = extract_trajectory_inputs(sample_estimates)
    assert res["extraction_version"] == EXTRACTION_VERSION
    assert res["gics_map_version"] == GICS_MAP_VERSION
    assert res["rev_window_used"] == "90d"
    assert "fetched_at" in res

    assert res["eps_cur_0y"] == 10.0
    assert res["eps_90d_0y"] == 9.5
    assert res["up30_0y"] == 5.0
    assert res["down30_0y"] == 1.0
    assert res["up7_0y"] == 2.0
    assert res["down7_0y"] == 0.0
    assert res["n_analysts_0y"] == 25.0
    assert res["eps_growth_0y"] == 0.15
    assert res["year_ago_eps_0y"] == 8.7
    assert res["rev_growth_0y"] == 0.12
    assert res["rev_avg_0y"] == 10000.0
    assert res["year_ago_rev_0y"] == 8928.0

    assert res["eps_cur_1y"] == 11.5
    assert res["eps_90d_1y"] == 10.8
    assert res["up30_1y"] == 6.0
    assert res["down30_1y"] == 1.0


def test_extract_trajectory_inputs_missing_and_nan(sample_estimates):
    # Null out 90daysAgo column entirely and drop eps_revisions
    sample_estimates["eps_trend"]["90daysAgo"] = np.nan
    sample_estimates["eps_revisions"] = None
    res = extract_trajectory_inputs(sample_estimates)

    assert res["eps_90d_0y"] is None
    assert res["eps_90d_1y"] is None
    assert res["up30_0y"] is None
    assert res["down30_0y"] is None
    assert res["eps_cur_0y"] == 10.0


def test_extract_trajectory_inputs_casing(sample_estimates):
    sample_estimates["eps_revisions"].columns = [
        "UPLAST7DAYS",
        "UpLast30Days",
        "Downlast30days",
        "DownLast7days",
        "currency",
    ]
    res = extract_trajectory_inputs(sample_estimates)
    assert res["up30_0y"] == 5.0
    assert res["down30_0y"] == 1.0
    assert res["up7_0y"] == 2.0
    assert res["down7_0y"] == 0.0


def test_missing_never_zero_or_omitted():
    res = extract_trajectory_inputs({})
    expected_horizons = ["0y", "1y"]
    fields_per_horizon = [
        "eps_cur", "eps_7d", "eps_30d", "eps_60d", "eps_90d",
        "up30", "down30", "up7", "down7", "n_analysts",
        "eps_growth", "year_ago_eps", "rev_growth", "rev_avg", "year_ago_rev",
    ]
    for h in expected_horizons:
        for prefix in fields_per_horizon:
            key = f"{prefix}_{h}"
            assert key in res, f"Missing key {key} in output dict"
            assert res[key] is None, f"Expected None for missing field {key}, got {res[key]}"


def test_compute_trajectory_purity_and_derivations(sample_estimates):
    raw_dict = extract_trajectory_inputs(sample_estimates)
    raw_dict["median_revision"] = 0.02
    raw_dict["breadth"] = 0.60
    raw_dict["median_peer_growth"] = 0.12
    raw_dict["peer_n"] = 8
    raw_dict["n_eff"] = 8

    before = copy.deepcopy(raw_dict)
    traj, conf, comp, derivations = compute_trajectory(raw_dict)

    # Assert purity
    assert raw_dict == before, "compute_trajectory mutated input raw_inputs dict!"

    # Assert derived values never appear in raw_inputs
    for derived_key in ("self_revision", "self_breadth", "a_score", "b_score", "g_score", "trajectory"):
        assert derived_key not in raw_dict

    # Assert peer aggregates survive in raw_inputs
    assert raw_dict["median_revision"] == 0.02
    assert raw_dict["breadth"] == 0.60
    assert raw_dict["median_peer_growth"] == 0.12

    # Assert derivations dict contains outputs
    assert derivations["a_score"] is not None
    assert derivations["g_score"] is not None
    assert derivations["b_score"] is not None
    assert comp == "AGB"


def test_migration_v8_composition_and_backfill():
    from compounderos_app.database import DataStore

    store = DataStore(":memory:")
    cur = store.conn.cursor()

    # Drop v8 table created during DataStore.__init__ so we can build schema v7
    cur.execute("DROP TABLE IF EXISTS trajectory_history")

    # Build schema v7 trajectory_history
    cur.execute("""
        CREATE TABLE trajectory_history (
            symbol TEXT,
            scan_date TEXT,
            trajectory REAL,
            confidence REAL,
            composition TEXT CHECK(composition IN ('A_PLUS_B', 'A_ONLY', 'WITHHELD')),
            granularity_used TEXT,
            peer_set_json TEXT,
            raw_inputs_json TEXT,
            engine_version TEXT,
            evaluated_at TEXT,
            PRIMARY KEY (symbol, scan_date)
        )
    """)
    cur.execute("PRAGMA user_version = 7")

    cur.execute("""
        INSERT INTO trajectory_history VALUES
        ('NVDA', '2026-08-01', 7.5, 0.8, 'A_PLUS_B', 'industry', '[]', '{}', 'traj-0.1.0', '2026-08-01T12:00:00'),
        ('AAPL', '2026-08-01', 6.0, 0.5, 'A_ONLY', 'industry', '[]', '{}', 'traj-0.1.0', '2026-08-01T12:00:00'),
        ('FOO', '2026-08-01', NULL, 0.0, 'WITHHELD', 'industry', '[]', '{}', 'traj-0.1.0', '2026-08-01T12:00:00')
    """)
    store.conn.commit()

    # Execute migration v8
    store._migrate_v8()

    # Check PRAGMA user_version
    uv = cur.execute("PRAGMA user_version").fetchone()[0]
    assert uv == 8

    # Check values after migration (convert sqlite3.Row to standard tuples)
    rows = [
        tuple(r)
        for r in cur.execute("""
            SELECT symbol, composition, peer_state, derived_json, extraction_version
            FROM trajectory_history ORDER BY symbol ASC
        """).fetchall()
    ]

    assert rows[0] == ("AAPL", "A", "NONE", None, None)
    assert rows[1] == ("FOO", "WITHHELD", "NONE", None, None)
    assert rows[2] == ("NVDA", "AB", "OK", None, None)

    # Idempotency check
    store._migrate_v8()
    assert cur.execute("PRAGMA user_version").fetchone()[0] == 8
    store.close()


def test_ddl_sites_agree():
    from compounderos_app.database import DataStore

    # 1. Get schema of a fresh DB (which ran _init_tables + migrations)
    fresh_store = DataStore(":memory:")
    fresh_cols = fresh_store.conn.execute("PRAGMA table_info(trajectory_history)").fetchall()
    fresh_col_names = [(c[1], c[2]) for c in fresh_cols]  # (name, type)
    fresh_store.close()

    # 2. Get schema of a migrated DB (starting from v7 -> v8)
    migrated_store = DataStore(":memory:")
    cur = migrated_store.conn.cursor()
    cur.execute("DROP TABLE IF EXISTS trajectory_history")
    cur.execute("""
        CREATE TABLE trajectory_history (
            symbol TEXT,
            scan_date TEXT,
            trajectory REAL,
            confidence REAL,
            composition TEXT CHECK(composition IN ('A_PLUS_B', 'A_ONLY', 'WITHHELD')),
            granularity_used TEXT,
            peer_set_json TEXT,
            raw_inputs_json TEXT,
            engine_version TEXT,
            evaluated_at TEXT,
            PRIMARY KEY (symbol, scan_date)
        )
    """)
    cur.execute("PRAGMA user_version = 7")
    migrated_store.conn.commit()
    migrated_store._migrate_v8()

    migrated_cols = migrated_store.conn.execute("PRAGMA table_info(trajectory_history)").fetchall()
    migrated_col_names = [(c[1], c[2]) for c in migrated_cols]
    migrated_store.close()

    assert fresh_col_names == migrated_col_names, "Fresh DB schema and migrated DB schema must have identical columns and types!"


# ==============================================================================
# TRAJECTORY v1.1 PASS B ENFORCEMENT & ENGINE TESTS (SECTION 10)
# ==============================================================================

def test_rev_h_formula_and_no_30d_fallback():
    # 1. Normal positive revision
    raw = {"eps_cur_0y": 2.20, "eps_90d_0y": 2.00}
    inputs = EstimateInputs(raw)
    assert inputs.self_rev_0y == pytest.approx(0.10, abs=1e-9)

    # 2. Negative-EPS base (sign correctness)
    raw_neg = {"eps_cur_0y": -2.00, "eps_90d_0y": -2.50}
    inputs_neg = EstimateInputs(raw_neg)
    assert inputs_neg.self_rev_0y == pytest.approx(0.20, abs=1e-9)

    # 3. 0.10 denominator floor triggering only at pathological values
    raw_tiny = {"eps_cur_0y": 0.05, "eps_90d_0y": 0.02}
    inputs_tiny = EstimateInputs(raw_tiny)
    assert inputs_tiny.self_rev_0y == pytest.approx(0.30, abs=1e-9)

    # 4. None when eps_90d is None (no 30d fallback!)
    raw_no90 = {"eps_cur_0y": 2.20, "eps_30d_0y": 2.00, "eps_90d_0y": None}
    inputs_no90 = EstimateInputs(raw_no90)
    assert inputs_no90.self_rev_0y is None


def test_horizon_weighting_and_drop_reweight():
    # 0.35/0.65 weighting when both present
    raw_both = {
        "eps_cur_0y": 10.0, "eps_90d_0y": 10.0,   # rev_0y = 0.0
        "eps_cur_1y": 11.0, "eps_90d_1y": 10.0,   # rev_1y = 0.10
    }
    inputs_both = EstimateInputs(raw_both)
    _, self_rev, _ = compute_A(inputs_both)
    assert self_rev == pytest.approx(0.65 * 0.10, abs=1e-9)

    # Single-horizon drop-and-reweight (weight 1.0 on available horizon)
    raw_single = {
        "eps_cur_0y": None, "eps_90d_0y": None,
        "eps_cur_1y": 11.0, "eps_90d_1y": 10.0,   # rev_1y = 0.10
    }
    inputs_single = EstimateInputs(raw_single)
    _, self_rev_single, _ = compute_A(inputs_single)
    assert self_rev_single == pytest.approx(0.10, abs=1e-9)


def test_laplace_breadth_smoothing():
    # 3-up / 1-down on 4 votes -> (3 - 1)/(4 + 2) = 2/6 = 0.333333 -> subscore ~8.333
    raw_3u1d = {
        "eps_cur_0y": 10.0, "eps_90d_0y": 10.0,
        "up30_0y": 3, "down30_0y": 1,
    }
    inputs_3u1d = EstimateInputs(raw_3u1d)
    a_score_3u1d, _, br_3u1d = compute_A(inputs_3u1d)
    assert br_3u1d == pytest.approx(1.0 / 3.0, abs=1e-9)
    # Band score for breadth 0.3333 over [-0.5, 0.0, 0.5] -> 5.0 + (0.3333/0.5)*5.0 = 8.333
    expected_br_score = 5.0 + (1.0 / 3.0) / 0.5 * 5.0
    expected_a = 0.6 * 5.0 + 0.4 * expected_br_score
    assert a_score_3u1d == pytest.approx(expected_a, abs=1e-9)

    # Unanimous 4u/0d -> (4 - 0)/(4 + 2) = 4/6 = 0.6667 -> saturates at 10.0
    raw_4u0d = {
        "eps_cur_0y": 10.0, "eps_90d_0y": 10.0,
        "up30_0y": 4, "down30_0y": 0,
    }
    a_score_4u0d, _, _ = compute_A(EstimateInputs(raw_4u0d))
    assert a_score_4u0d == pytest.approx(0.6 * 5.0 + 0.4 * 10.0, abs=1e-9)

    # votes < 4 -> breadth is None
    raw_low = {
        "eps_cur_0y": 10.0, "eps_90d_0y": 10.0,
        "up30_0y": 2, "down30_0y": 1,
    }
    _, _, br_low = compute_A(EstimateInputs(raw_low))
    assert br_low is None


def test_band_asymmetry_and_saturation():
    # Floor -0.06 -> 0.0; Target 0.0 -> 5.0; Elite +0.11 -> 10.0
    raw_floor = {"eps_cur_0y": 9.40, "eps_90d_0y": 10.0, "up30_0y": 0, "down30_0y": 0}
    a_floor, _, _ = compute_A(EstimateInputs(raw_floor))
    assert a_floor == pytest.approx(0.0, abs=1e-9)

    raw_elite = {"eps_cur_0y": 11.10, "eps_90d_0y": 10.0, "up30_0y": 0, "down30_0y": 0}
    a_elite, _, _ = compute_A(EstimateInputs(raw_elite))
    assert a_elite == pytest.approx(10.0, abs=1e-9)

    # Beyond elite (+0.20, +0.50) ties at exactly 10.0
    for val in (12.0, 15.0, 20.0):
        raw_sat = {"eps_cur_0y": val, "eps_90d_0y": 10.0, "up30_0y": 0, "down30_0y": 0}
        a_sat, _, _ = compute_A(EstimateInputs(raw_sat))
        assert a_sat == pytest.approx(10.0, abs=1e-9)


def test_composite_renormalization():
    # All four combinations: AGB, AG, AB, A
    raw_agb = {
        "eps_cur_0y": 11.10, "eps_90d_0y": 10.0,          # A_score = 10.0
        "rev_growth_1y": 0.22,                             # G_score = 10.0
        "median_revision": 0.0, "peer_n": 5,              # B_score = 10.0 (self=0.11 - peer=0.0 > 0.04)
    }
    traj_agb, _, comp_agb, _ = compute_trajectory(raw_agb)
    assert comp_agb == "AGB"
    assert traj_agb == pytest.approx(10.0, abs=1e-9)

    raw_ag = {
        "eps_cur_0y": 11.10, "eps_90d_0y": 10.0,
        "rev_growth_1y": 0.08,                             # G_score = 5.0 (target)
        "peer_n": 2,                                       # B absent (< 5 peers)
    }
    traj_ag, _, comp_ag, _ = compute_trajectory(raw_ag)
    assert comp_ag == "AG"
    expected_ag = (0.45 * 10.0 + 0.35 * 5.0) / (0.45 + 0.35)
    assert traj_ag == pytest.approx(expected_ag, abs=1e-9)

    raw_ab = {
        "eps_cur_0y": 11.10, "eps_90d_0y": 10.0,
        "rev_growth_1y": None,                             # G absent
        "median_revision": 0.11, "peer_n": 5,              # B_score = 5.0 (diff = 0.0)
    }
    traj_ab, _, comp_ab, _ = compute_trajectory(raw_ab)
    assert comp_ab == "AB"
    expected_ab = (0.45 * 10.0 + 0.20 * 5.0) / (0.45 + 0.20)
    assert traj_ab == pytest.approx(expected_ab, abs=1e-9)

    raw_a = {
        "eps_cur_0y": 11.10, "eps_90d_0y": 10.0,
        "rev_growth_1y": None,
        "peer_n": 0,
    }
    traj_a, _, comp_a, _ = compute_trajectory(raw_a)
    assert comp_a == "A"
    assert traj_a == pytest.approx(10.0, abs=1e-9)


def test_confidence_formula_and_a_only_ceiling():
    # A-only ceiling is exactly 0.45 (usable_horizons=2, votes=8 -> dA=1.0; dG=0; dB=0)
    raw_a_max = {
        "eps_cur_0y": 10.0, "eps_90d_0y": 10.0,
        "eps_cur_1y": 10.0, "eps_90d_1y": 10.0,
        "up30_0y": 8, "down30_0y": 0,
    }
    _, conf_a, comp_a, _ = compute_trajectory(raw_a_max)
    assert comp_a == "A"
    assert conf_a == pytest.approx(0.45, abs=1e-9)


def test_peer_sets_and_boundaries():
    # peer_state OK/THIN/NONE boundaries at 0, 1, 4, 5 peers
    peers_0 = []
    med_rev_0, br_0, med_g_0, n_0, _ = peer_aggregates(peers_0)
    assert n_0 == 0 and med_rev_0 is None

    peers_1 = [{"self_revision": 0.02, "rev_growth_1y": 0.05}]
    med_rev_1, br_1, med_g_1, n_1, _ = peer_aggregates(peers_1)
    assert n_1 == 1 and med_rev_1 == pytest.approx(0.02, abs=1e-9)

    peers_4 = [{"self_revision": 0.02, "rev_growth_1y": 0.05}] * 4
    med_rev_4, _, _, n_4, _ = peer_aggregates(peers_4)
    assert n_4 == 4

    peers_5 = [{"self_revision": 0.02, "rev_growth_1y": 0.05}] * 5
    med_rev_5, _, _, n_5, _ = peer_aggregates(peers_5)
    assert n_5 == 5


def test_composition_label_and_raw_derived_json():
    raw = {
        "eps_cur_0y": 10.5, "eps_90d_0y": 10.0,
        "rev_growth_1y": 0.10,
        "median_revision": 0.0, "peer_n": 5,
    }
    traj, conf, comp, derived = compute_trajectory(raw)
    assert comp == "AGB"
    assert "self_revision" in derived and "a_score" in derived and "g_score" in derived and "b_score" in derived
    for k in ("self_revision", "a_score", "g_score", "b_score", "trajectory"):
        assert k not in raw


def test_determinism():
    raw = {
        "eps_cur_0y": 10.5, "eps_90d_0y": 10.0,
        "up30_0y": 4, "down30_0y": 1,
        "rev_growth_1y": 0.10,
        "median_revision": 0.02, "peer_n": 6,
    }
    t1, c1, comp1, d1 = compute_trajectory(raw)
    t2, c2, comp2, d2 = compute_trajectory(raw)
    assert t1 == t2 and c1 == c2 and comp1 == comp2 and d1 == d2


def test_gics_subind_to_ind_completeness():
    from compounderos_app.gics_map import GICS_SUBIND_TO_IND, universe_verdict

    # 1. Assert all ~163 sub-industries resolve to non-empty industry strings
    assert len(GICS_SUBIND_TO_IND) >= 150
    for sub, ind in GICS_SUBIND_TO_IND.items():
        assert isinstance(sub, str) and sub.strip()
        assert isinstance(ind, str) and ind.strip()

    # 2. Verify Financials & Real Estate existing verdicts remain exact
    assert universe_verdict("Regional Banks", "Financials") == "NOT_APPLICABLE"
    assert universe_verdict("Industrial REITs", "Real Estate") == "NOT_APPLICABLE"
    assert universe_verdict("Transaction & Payment Processing Services", "Financials") == "SCORED"
    assert universe_verdict("Systems Software", "Information Technology") == "SCORED"
