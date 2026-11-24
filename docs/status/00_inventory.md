# Phase 1 — Discovery & Repository Inventory (`00_inventory.md`)

## 1. Canonical Repository & Git State
- **Canonical Repository Path**: `/Users/nithinkamath/CompounderOSv2`
- **Other Directories Discovered (Out-of-Scope per User Instruction)**:
  - `/Users/nithinkamath/CompounderOS` (older desktop-only checkout, no `addons/` or `scanner_service/`)
  - `/Users/nithinkamath/CompounderOSv3` (prototype dir, no `addons/` or `scanner_service/`)
- **Initial Branch**: `main`
- **Initial HEAD Commit**: `2c30d9890509d9f06110f0316358f56c1eac760e` (`Initial commit: CompounderOS`, author `Nithin <nithinkamat@gmail.com>`, 2026-02-11)
- **Pre-Audit WIP Snapshot Branch**: `gemini/pre-audit-snapshot`
- **Pre-Audit Snapshot Commit (`BASE`)**: `d1b6206deb07462d9cec3e9f9743bea4b9aefd97` (`WIP snapshot of user's uncommitted work (pre-audit)`)
- **Implementation Branch**: `gemini/compounderos-impl` (branched from `BASE`)
- **Backup Tarball**: `/tmp/CompounderOS_Backups/compounderos_20261027_snapshot.tar.gz` (`21,716,308` bytes / `20.71 MB`)
- **Git Remotes**: None configured (`git remote -v` empty)
- **Git Tags / Stashes**: None (`git tag --list` and `git stash list` empty)

### Pre-Audit `git status --short` Summary (68 tracked/untracked source files committed to `BASE`)
- **Modified Tracked Files (27)**:
  - `compounderos.py`, `requirements.txt`, `compounderos_app/ai_engine.py`, `compounderos_app/charts.py`, `compounderos_app/config.py`, `compounderos_app/database.py`, `compounderos_app/main.py`
  - `compounderos_app/gui/app.py`, `compounderos_app/gui/components/common.py`, `compounderos_app/gui/components/theme.py`
  - `compounderos_app/gui/services/data_service.py`, `compounderos_app/gui/services/portfolio_service.py`
  - `compounderos_app/gui/tabs/{about,ai_report,charts_tab,earnings,insights,kpis,portfolio,quality,transcripts,valuation,watchlist}.py`
  - `legacy/{app_before_tabs_split,gui_legacy_split_backup,opentrim_v4_backup,opentrim_v4_legacy}.py`
- **Untracked Source/Config/Test Files Added to `BASE` (41)**:
  - `CompounderOS.spec`, `diag_gemini.py`, `engine_diagnostics.py`, `generate_golden_vector.py`, `requirements-scanner.txt`, `rescore_history.py`, `run_app.py`, `sp500_scanner.py`, `traj_probe.py`, `traj_probe_results.csv`
  - `addons/compounderos_scanner/{config.yaml,Dockerfile}`
  - `compounderos_app/{analytics_core,analytics_v2,gics_map,golden_vector.json,metric_extraction,scoring_engine,trajectory_engine,trajectory_extraction}.py`
  - `compounderos_app/{test_imports,test_lens_boundary_invariant,test_scoring_engine,test_split_determinism,test_trajectory_extraction,test_v2_core}.py`
  - `compounderos_app/gui/services/scanner_gateway.py`, `compounderos_app/gui/tabs/{due_diligence,sp500_ranking}.py`
  - `scanner_service/{__init__,__main__,alerts,api,pipeline,yf_gateway}.py`
  - `scanner_service/web/{__init__,views}.py`, `scanner_service/web/static/app.css`, `scanner_service/web/templates/{base,ranking,system}.html`
- **Excluded Local/Binary/Secret Artifacts (Never Committed per R10)**:
  - `.compounderos_config.json` (contains keys: `ALPHAVANTAGE_API_KEY`, `API_NINJAS_KEY`, `FINNHUB_API_KEY`, `GEMINI_API_KEY`, `SEC_USER_AGENT` — values redacted per R10)
  - `compounderos_data.db` (`4,722,688` bytes; copied to `/Users/nithinkamath/CompounderOSv2/docs/status/scratch/compounderos_data_copy.db` per R2)
  - `venv/`, `build/`, `dist/`, `.pytest_cache/`, `.hypothesis/`, `repomix-output.xml`

---

## 2. Module Map & Entrypoints

| Path | Lines | Purpose / Key Symbols |
| :--- | ---: | :--- |
| `compounderos.py` | 30 | Desktop CLI entrypoint (`from compounderos_app.main import main`) |
| `run_app.py` | 26 | PyInstaller bundle launcher (`from compounderos_app.main import main`) |
| `sp500_scanner.py` | 312 | Standalone & library S&P 500 scanner (`fetch_sp500_universe`, `pick_next_symbol`, `evaluate_symbol`, `_write_digest`) |
| `rescore_history.py` | 127 | Historical rescore utility (`rescore_all`) recomputing `sp500_scores_history` and `sp500_scores` from `raw_json` |
| `engine_diagnostics.py` | 585 | Diagnostic suite (`e1`–`e8`) auditing score distributions, anchor saturation, marginal exposure, gates, IC census, reinvestment, dilution guard, fallbacks |
| `traj_probe.py` | 140 | Trajectory extraction probe script over 21 archetype symbols (`traj_probe_results.csv`) |
| `generate_golden_vector.py` | 120 | Generator for `compounderos_app/golden_vector.json` cross-platform determinism vectors |
| `diag_gemini.py` | 141 | Diagnostic script for Gemini API key and model ladder verification |
| `compounderos_app/config.py` | 92 | Paths (`COMPOUNDEROS_BASE_DIR`, `DB_PATH`, `SCANNER_DB_PATH`), service key resolution (`get_service_key`) |
| `compounderos_app/scoring_engine.py` | 472 | Deterministic scoring engine `det-1.1.0` (`ENGINE_VERSION`, `DEFAULT_REGISTRY`, `BUCKET_WEIGHTS`, `CONFIDENCE_WEIGHTS`, `CANONICAL_REGISTRY_HASH`, `score_metrics`, `validate_lens`, `lens_to_definition`, `lens_from_definition`) |
| `compounderos_app/metric_extraction.py` | 485 | Raw financial extraction (`FinancialBundle`, `universe_verdict`, `extract_raw_metrics`, `compute_owner_earnings`, `compute_normalized_fcf`) |
| `compounderos_app/trajectory_engine.py` | 315 | Deterministic trajectory engine `traj-1.1.0` (`TRAJ_ENGINE_VERSION`, `EstimateInputs`, `compute_A`, `compute_G`, `compute_B`, `peer_aggregates`, `compute_trajectory`) |
| `compounderos_app/trajectory_extraction.py` | 164 | Yahoo Finance analyst estimate extraction `trajx-1.0.0` (`EXTRACTION_VERSION`, `extract_trajectory_inputs`) |
| `compounderos_app/gics_map.py` | 195 | Complete 2023 GICS sub-industry to industry mapping (`GICS_MAP_VERSION = "gics-map-2023-full"`, `GICS_SUBIND_TO_IND`) |
| `compounderos_app/analytics_core.py` | 868 | `ReverseDCFSolver`, `TailRiskScenarioTester`, `RoicMomentumAnalyzer`, `HeadlessDataReporter` |
| `compounderos_app/analytics_v2.py` | 205 | `DipTriggerAnalyzer`, `IncrementalRoicAnalyzer`, `ScoreBacktester` |
| `compounderos_app/database.py` | 1115 | SQLite `DataStore` (`user_version = 8`, migrations `_migrate_v6`..`_migrate_v8`, `record_quality_result`, `record_trajectory_result`, sync queries, alerts) |
| `compounderos_app/charts.py` | 1000 | `ChartRenderer`, `redraw_chart_now`, series alignment/TTM helpers |
| `compounderos_app/gui/app.py` | 860 | `CompounderOSApp` 3-column AlphaSpace-styled CustomTkinter desktop application |
| `compounderos_app/gui/services/scanner_gateway.py` | 255 | `ScannerSyncClient` background replica sync client for Pi `/v1/sync/*` and `/v1/control/*` |
| `scanner_service/__main__.py` | 68 | Headless Pi add-on entrypoint (`load_options`, `seed_database`, `ScannerPipeline`, `create_app`, `uvicorn.run`) |
| `scanner_service/api.py` | 227 | FastAPI application (`IngressMiddleware`, `/v1/health`, `/v1/status`, `/v1/alerts`, `/v1/sync/*`, `/v1/control/*`) |
| `scanner_service/pipeline.py` | 443 | `ScannerPipeline` background scheduler & worker threads (quality scan, daily prices, 2-pass trajectory stage, digest) |
| `scanner_service/yf_gateway.py` | 159 | `TokenBucket` rate limiter and `YFGateway` (`fetch_bundle`, `fetch_estimates`, `fetch_price_batch`, `is_throttle`) |
| `scanner_service/alerts.py` | 153 | `AlertEngine` (`dip_trigger`, `score_drop`, `gate_fired` state machine with auto-resolution) |
| `scanner_service/web/views.py` | 83 | Read-only HA Ingress web routes (`/ui/`, `/ui/ranking`, `/ui/system`) |

---

## 3. Call-Graph Sketch

1. **Headless Pi Add-on (`python -m scanner_service`)**:
   - `scanner_service/__main__.py` -> `load_options()` -> `seed_database()` -> `DataStore(db_path, role="sor")` -> `ScannerPipeline.start()`:
     - `_scheduler_loop()` -> `_run_price_stage()` (`YFGateway.fetch_price_batch` -> `DataStore.upsert_price_history` -> `AlertEngine.on_price_stage_done`)
     - `_scheduler_loop()` -> `_spawn_trajectory_stage()` -> `_run_trajectory_stage()` (`YFGateway.fetch_estimates` -> `extract_trajectory_inputs` -> `compute_A` -> `peer_aggregates` -> `compute_trajectory` -> `DataStore.record_trajectory_result`)
     - `_scheduler_loop()` -> `sp500_scanner.fetch_sp500_universe()` -> `_pick_next_symbol()` -> `task_queue`
     - `_worker_loop()` -> `sp500_scanner.evaluate_symbol()` (`YFGateway.fetch_bundle` -> `universe_verdict` -> `RoicMomentumAnalyzer.analyze` -> `extract_raw_metrics` -> `score_metrics` -> `DataStore.record_quality_result`) -> `AlertEngine.on_quality_written()`
   - `create_app(store, pipeline)` + `create_ui_router(pipeline)` -> `uvicorn.run(app, host="0.0.0.0", port=8321)`
2. **macOS Desktop Terminal (`python compounderos.py` / `run_app.py`)**:
   - `compounderos_app/main.py:main()` -> `CompounderOSApp`:
     - Opens local `DataStore(DB_PATH)` and `DataStore(SCANNER_DB_PATH, role="replica")`.
     - Starts `ScannerSyncClient` (if `SCANNER_URL` and `SCANNER_API_TOKEN` configured) polling `/v1/status` and `/v1/sync/{scores,history,trajectory,prices,ticker-map,alerts}`.
     - Renders 3-column AlphaSpace dashboard (`WatchlistTabMixin` left rail, 11-tab workbench center, `AiReportTabMixin` right rail).

---

## 4. Test Inventory
- `compounderos_app/test_imports.py` (1 test): Module import smoke test across `compounderos_app` and `scanner_service`.
- `compounderos_app/test_scoring_engine.py` (31 tests): Band scoring, confidence weights, drop-and-reweight, denominator guards, S1 unverified cap, S4 graduated gate ramp crossovers, S8 min-bucket rule, universe verdict, registry validation.
- `compounderos_app/test_trajectory_extraction.py` (16 tests): Extraction casing/NaN handling, revision horizon formula, no 30d-to-90d fallback, asymmetric bands, Laplace breadth smoothing, peer sets, A-only confidence ceiling, v8 migration.
- `compounderos_app/test_lens_boundary_invariant.py` (10 tests): L1–L6 lens contract validation, canonical write-guard, trajectory choke-point, provenance stamping, synthetic & real-history golden vector verification.
- `compounderos_app/test_split_determinism.py` (2 tests): Tier 1 (`score_metrics`, `compute_trajectory`) and Tier 2 (`RoicMomentumAnalyzer`, `ReverseDCFSolver`) cross-architecture determinism against `golden_vector.json`.
- `compounderos_app/test_v2_core.py` (8 tests): `DipTriggerAnalyzer` (guardrails, z-score firing, quality gate, score drop) and `IncrementalRoicAnalyzer` (normal, negative reinvestment, negative base).

---

## 5. Home Assistant Add-on Manifest (`addons/compounderos_scanner/config.yaml`)
- `name`: `CompounderOS Scanner`
- `version`: `"2.1.0"`
- `slug`: `compounderos_scanner`
- `arch`: `[aarch64]`
- `startup`: `application`, `boot`: `auto`, `init`: `true`
- `ingress`: `true`, `ingress_port`: `8321`, `panel_icon`: `mdi:finance`, `panel_title`: `CompounderOS`
- `ports`: `8321/tcp: 8321`
- `map`: `["share"]`
- `watchdog`: `http://[HOST]:[PORT:8321]/v1/health`
- `Dockerfile`: `FROM python:3.11-slim-bookworm`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `COMPOUNDEROS_BASE_DIR=/data`, `COMPOUNDEROS_SCANNER_DB=/data/sp500_scanner.db`, `CMD ["python", "-m", "scanner_service"]`.

---

## 6. Assumption Log (Rule R14)
1. **A-01 (Workspace Sandbox Boundary)**: Per explicit user instruction (`"continue, but you only have access to the this particular project directory"`), all commands run inside the standard sandbox (`BypassSandbox: false`) accessing only `/Users/nithinkamath/CompounderOSv2` and `/tmp`. External backup and audit artifacts are written to `/tmp/CompounderOS_Backups/` and `/Users/nithinkamath/CompounderOSv2/docs/status/` and mirrored in Phase 7 to `/Users/nithinkamath/CompounderOSv2/docs/status/`.
2. **A-02 (Sandboxed Git & Python 3.12 Execution)**: Because macOS seatbelt marks `.git/` read-only inside the sandbox and `venv/bin/python3` symlinks to `~/.local/bin/python3.12` outside the workspace, git operations use `/tmp/CompounderOS_git/.git` with `--work-tree=/Users/nithinkamath/CompounderOSv2` (and export a complete git bundle to `docs/status/compounderos-impl.bundle` in Phase 7), and Python 3.12 execution uses `/tmp/py312/bin/python3` linked against the repo's bundled `dist/CompounderOS/_internal/libpython3.12.dylib` and `venv/lib/python3.12/site-packages`.
3. **A-03 (External Fable 5 Docs Unavailable)**: Spotlight searches (`mdfind`) and repo scans confirmed no standalone Fable 5 specification files exist on disk. Per Phase 2 instructions, the Enhancement Source Register and Master Ledger are constructed from the Section 4 seed checklist (`WS-A` through `WS-I`) cross-referenced against in-repo docstrings, tests, diagnostic scripts (`engine_diagnostics.py`, `traj_probe.py`), and comments.
