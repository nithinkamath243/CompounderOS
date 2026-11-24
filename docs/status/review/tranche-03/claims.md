# Tranche 03 Claims & Evidence (`60f37ad`)

1. **Claim (`E-011`)**: Same-day snapshot writes to `sp500_scores` are confidence-monotonic and do not overwrite a higher-confidence same-day scan.
   - **File/Line**: `compounderos_app/database.py:553-577`.
   - **Test**: `test_lens_boundary_invariant.py::test_snapshot_confidence_monotonicity_same_day`.
2. **Claim (`E-012`)**: `ScannerPipeline._run_trajectory_stage` passes the unmutated `raw_dict` to `record_trajectory_result` so `trajectory_history.raw_inputs_json` never contains derived peer-aggregate keys (`median_revision`, `breadth`, `median_peer_growth`, `peer_n`, `peer_pos_n`).
   - **File/Line**: `scanner_service/pipeline.py:427-440`.
   - **Test**: `test_lens_boundary_invariant.py::test_pipeline_trajectory_stage_preserves_pure_raw_inputs`.
3. **Claim (`E-013`, `E-027`, `E-028`)**: `DataStore.record_quality_result` enforces both `registry_id == CANONICAL_REGISTRY_ID` and `registry_hash == CANONICAL_REGISTRY_HASH`, and persists `registry_id` and `registry_hash` in both `sp500_scores` and `sp500_scores_history`.
   - **File/Line**: `compounderos_app/database.py:23-88`, `compounderos_app/database.py:474-483`, `compounderos_app/database.py:548-634`.
   - **Test**: `test_lens_boundary_invariant.py::test_write_guard_blocks_spoofed_registry_hash_and_persists_provenance`.
