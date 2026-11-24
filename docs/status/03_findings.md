# Phase 4 — Verification Results & Architectural Invariant Audit (`03_findings.md`)

## 1. Static Checks
- **Bytecode Compilation (`compileall`)**:
  - Command: `/tmp/py312/bin/python3 -c "import compileall; print(compileall.compile_dir('compounderos_app', quiet=1), compileall.compile_dir('scanner_service', quiet=1))"`
  - Result: `True True` (all `.py` files in `compounderos_app/`, `scanner_service/`, and repo root compile with zero syntax errors).
- **AST Quality Audit (`TODO` / `FIXME` / `XXX`, `NotImplementedError`, `pass`-only stubs)**:
  - Command: AST walk across all 66 Python files in `compounderos_app/`, `scanner_service/`, and repo root.
  - Result:
    - `TODO/FIXME/XXX`: `[]` (0 occurrences)
    - `Pass/Ellipsis functions`: `[]` (0 occurrences; only `ExtendedDataProvider` `Protocol` methods in `data_service.py:11-15` use `...`)
    - `NotImplementedError`: `[]` (0 occurrences)

---

## 2. Dynamic Test Suite Execution (Pre-Fix Baseline at `BASE = d1b6206`)

### 2.1 Passing Suites (60 tests)
Command:
```sh
MPLCONFIGDIR=/tmp/mpl /tmp/py312/bin/python3 -m pytest   compounderos_app/test_scoring_engine.py   compounderos_app/test_trajectory_extraction.py   compounderos_app/test_lens_boundary_invariant.py   compounderos_app/test_split_determinism.py
```
Verbatim Summary:
- `compounderos_app/test_scoring_engine.py`: **31 passed**
- `compounderos_app/test_trajectory_extraction.py`: **16 passed**
- `compounderos_app/test_lens_boundary_invariant.py`: **9 passed, 1 skipped** (`test_golden_over_real_history` when `COMPOUNDEROS_DB` unset; passes when pointed at migrated scratch DB)
- `compounderos_app/test_split_determinism.py`: **2 passed** (`test_tier_1_determinism`, `test_tier_2_determinism`)

### 2.2 Failing Suites at `BASE` (Pre-Fix)
Command:
```sh
MPLCONFIGDIR=/tmp/mpl /tmp/py312/bin/python3 -m pytest   compounderos_app/test_imports.py   compounderos_app/test_v2_core.py
```
Verbatim Failure Output:
```text
--- compounderos_app/test_imports.py::test_imports ---
  File "/Users/nithinkamath/CompounderOSv2/compounderos_app/test_imports.py", line 8, in test_imports
    import scanner_service.pipeline
  File "/Users/nithinkamath/CompounderOSv2/scanner_service/pipeline.py", line 13, in <module>
    from scanner_service.alerts import AlertEngine
  File "/Users/nithinkamath/CompounderOSv2/scanner_service/alerts.py", line 10, in <module>
    from compounderos_app.analytics_v2 import DipTriggerAnalyzer
  File "/Users/nithinkamath/CompounderOSv2/compounderos_app/analytics_v2.py", line 7, in <module>
    import statsmodels.api as sm
ModuleNotFoundError: No module named 'statsmodels'
```

---

## 3. Targeted Dynamic Probes (`/Users/nithinkamath/CompounderOSv2/docs/status/scratch/run_probes.py`)

### Probe A3: Starlette >= 0.33 `IngressMiddleware` & `Mount("/ui/static")` 404 Bug (`E-002`, `E-003`)
```text
=== PROBE A3: Starlette routing / templating & IngressMiddleware ===
Starlette version: 1.3.1
TemplateResponse signature: (self, request: 'Request', name: 'str', context: 'dict[str, Any] | None' = None, status_code: 'int' = 200, headers: 'Mapping[str, str] | None' = None, media_type: 'str | None' = None, background: 'BackgroundTask | None' = None) -> '_TemplateResponse'
  GET /ui/ranking (no header): 200 ranking ok root_path=
  GET /ui/ranking (with X-Ingress-Path): 200 ranking ok root_path=/api/hassio_ingress/tok123
  GET /ui/static/app.css (with X-Ingress-Path): 404 -> BROKEN 404 under Starlette >=0.33!
  GET /api/hassio_ingress/tok123/ui/static/app.css (with X-Ingress-Path): 200 2611
```
- **Root Cause**:
  1. Home Assistant Ingress strips `/api/hassio_ingress/<token>` before forwarding `GET /ui/static/app.css` to the container with header `X-Ingress-Path: /api/hassio_ingress/<token>`.
  2. `IngressMiddleware` (`scanner_service/api.py:46-51`) sets `request.scope["root_path"] = ingress_path` while `request.scope["path"]` remains `"/ui/static/app.css"`.
  3. In Starlette >= 0.33, `Mount.matches` (`starlette/routing.py`) sets `child_scope["root_path"] = root_path + "/ui/static"` (`"/api/hassio_ingress/<token>/ui/static"`). When `StaticFiles.get_path(child_scope)` calls `get_route_path(child_scope)`, `child_scope["path"]` (`"/ui/static/app.css"`) does not start with `child_scope["root_path"]`, so `get_route_path` fails to strip `"/ui/static"` and looks for `<static_dir>/ui/static/app.css`, returning **HTTP 404**.
  4. Additionally, `scanner_service/web/views.py:40,71` calls `templates.TemplateResponse("ranking.html", {"request": request, ...})` using the removed Starlette <0.29 positional signature instead of `TemplateResponse(request=request, name="ranking.html", context={...})`.

### Probe B4/B5: `ScannerPipeline` Worker Signature Mismatch & Starvation Livelocks (`E-009`, `E-010`)
```text
=== PROBE B5: Watchlist / Inactive Symbol Starvation in ScannerPipeline ===
  Tick 1 picked: ('DELIST', 'Unknown', 'Unknown', False)
  Worker skips evaluate_symbol because is_active=False! sp500_scores.evaluated_at never updated!
  Tick 2 picked: ('DELIST', 'Unknown', 'Unknown', False) -> IDENTICAL NON-SP500 WATCHLIST SYMBOL PICKED INFINITELY (STARVATION LIVELOCK)
```
- **Root Causes**:
  1. **Signature Mismatch (`E-009`)**: `scanner_service/pipeline.py:231` calls `sp500_scanner.evaluate_symbol(self.store, self.gateway, symbol, sector, sub_ind, is_active)` (6 positional args), whereas `sp500_scanner.py:156` defines `def evaluate_symbol(symbol: str, sector: str, store: DataStore, *, gateway: Optional[Any] = None) -> bool`. Every worker call raises `TypeError`.
  2. **Inactive / Watchlist Starvation Livelock (`E-010`)**: `_pick_next_symbol` (`pipeline.py:206-219`) includes `historical` symbols (`is_active = False`), while `_worker_loop` (`pipeline.py:230`) skips `evaluate_symbol` when `not is_active`, never updating `evaluated_at` or `self.failures`. Furthermore, when `evaluate_symbol` returns `False` (`sp500_scanner.py:175,196`), `_worker_loop` ignores the return value and does not add the symbol to `self.failures`, re-picking it on every scheduler tick.

### Probe C1: Confidence-Monotonic History Writes vs Snapshot Overwrite (`E-011`)
```text
=== PROBE C1: Confidence-Monotonic History Writes vs sp500_scores Snapshot ===
  High-conf scan: composite=7.9259, shrunk=6.7434, conf=0.5958
  Low-conf scan:  composite=2.8571, shrunk=4.2857, conf=0.3333
  sp500_scores_history (monotonic): score=7.9259, shrunk=6.7434, confidence=0.5958
  sp500_scores snapshot (OVERWRITTEN!): score=2.8571, composite_shrunk=4.2857
```
- **Root Cause**: `compounderos_app/database.py:521-549` enforces `WHERE excluded.confidence >= sp500_scores_history.confidence` on `sp500_scores_history`, but `database.py:500-512` executes an unconditional `REPLACE INTO sp500_scores`, overwriting the live snapshot table with lower-confidence same-day data.

### Probe C2: `raw_inputs_json` Pollution in Trajectory Stage (`E-012`)
- **Finding**: In `scanner_service/pipeline.py:417-434`, `_run_trajectory_stage` copies `raw_dict` to `symbol_raw`, injects 5 derived peer-aggregation keys (`median_revision`, `breadth`, `median_peer_growth`, `peer_n`, `peer_pos_n`), and passes `raw_inputs=symbol_raw` to `self.store.record_trajectory_result(...)`.

### Probe C3/F3/F4: Provenance & Lens Persistence Boundary (`E-013`, `E-027`, `E-028`)
```text
=== PROBE C3/F3/F4: Provenance & Lens Persistence Boundary ===
  Lens write blocked as expected: Write-guard error: Cannot persist custom lens result to canonical tables.
  sp500_scores_history columns: ['symbol', 'scan_date', 'evaluated_at', 'sector', 'score', 'score_pre_gate', 'composite_shrunk', 'b_profitability', 'b_efficiency', 'b_balance_sheet', 'b_capital_alloc', 'confidence', 'gate_state', 'gates_fired', 'gates_unverified', 'gate_cap_applied', 'verdict', 'pe_trailing', 'pe_forward', 'fcf_yield', 'market_cap', 'engine_version', 'source', 'raw_json', 'subscores_json', 'registry_hash']
  sp500_scores columns: ['symbol', 'sector', 'score', 'composite_shrunk', 'verdict', 'model', 'evaluated_at']
```

### Probe D & E: 100x Determinism, Boundary Probes & `analytics_core` Smoke Test
```text
=== PROBE D & E: 100x Determinism & Key-Order Invariance ===
  100x score_metrics key-order & run-to-run identical: True
  100x compute_trajectory key-order & run-to-run identical: True

=== PROBE E: Trajectory Engine Boundary Cases ===
  empty_dict         -> traj=None, conf=0.0, comp=WITHHELD
  all_none           -> traj=None, conf=0.0, comp=WITHHELD
  zero_peers         -> traj=None, conf=0.0, comp=WITHHELD
  one_peer_a_only    -> traj=None, conf=0.0, comp=WITHHELD
  negative_eps       -> traj=None, conf=0.0, comp=WITHHELD
  extreme_values     -> traj=None, conf=0.0, comp=WITHHELD

[ReverseDCF] implied growth: 16.20% (converged=True, iters=25, baseline=4.98B via TTM sum of last 4 quarters)
[Obsolescence] applied=5 skipped=[] | latest GM 37.7% (was 47.1%)
[CapEx shock]  applied=['CapitalExpenditure: x2 (sign preserved)', "FreeCashFlow: rebuilt as OCF - |CapEx'|"] | latest FCF 1.60B (was 2.50B)
[ROIC] latest=40.0% slope=+0.0352/yr verdict=COMPOUNDING IMPROVEMENT ic_method='TotalAssets - Cash - CurrentLiabilities' tax=20.0% (effective (from statements, clipped 0-50%))
```

### Probe H2: `ChartRenderer` & `ChartPayload` Decoupling (`E-035`)
```text
=== PROBE H2 / Invariant 8: ChartRenderer & ChartPayload check ===
  Classes in charts.py: ['ChartRenderer']
  ChartPayload present: False
```

---

## 4. Section 7 Architectural Invariants Evaluation (Pre-Implementation Baseline)

| # | Invariant | Baseline Verdict | Evidence & Citation | Remediation Item(s) |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Determinism**: Scoring and trajectory engines are pure functions of their input dicts/series and registry config. | **HOLDS** | `scoring_engine.py:347-471`, `trajectory_engine.py:279-315`; verified by `test_split_determinism.py` and 100x key-shuffled determinism probe (`True`). | — |
| **2** | **Version discipline**: Every persisted score/trajectory row carries the exact `engine_version`, `registry_id`, and `registry_hash` that produced it, matching `det-1.1.0` / `traj-1.1.0`. | **VIOLATED (minor)** | `ENGINE_VERSION = "det-1.1.0"` and `TRAJ_ENGINE_VERSION = "traj-1.1.0"` match, and `sp500_scores_history` stores `engine_version` + `registry_hash`, but `registry_id` is not persisted in `sp500_scores_history` or `sp500_scores` (`database.py:41-85`). | `E-013`, `E-028` (Tranche 03) |
| **3** | **Persistence boundary**: Only canonical-registry (`registry_id == CANONICAL_REGISTRY_ID` and matching `registry_hash`) results may be written to canonical history tables. | **VIOLATED (minor)** | `database.py:451` checks `getattr(result, "registry_id", "canonical") != CANONICAL_REGISTRY_ID` but omits checking `getattr(result, "registry_hash", None) != CANONICAL_REGISTRY_HASH`. | `E-027` (Tranche 03) |
| **4** | **Monotonic history**: A history write for `(symbol, scan_date)` never overwrites an existing row with lower confidence. | **VIOLATED (on snapshot)** | `sp500_scores_history` (`database.py:546`) enforces `excluded.confidence >= sp500_scores_history.confidence`, but `sp500_scores` (`database.py:502`) unconditionally overwrites the snapshot row with lower-confidence same-day data. | `E-011` (Tranche 03) |
| **5** | **Raw input immutability**: `raw_inputs_json` is written once from extracted raw inputs and never mutated by scoring or trajectory code. | **VIOLATED** | `scanner_service/pipeline.py:417-434` injects 5 derived peer fields into `symbol_raw` and passes `raw_inputs=symbol_raw` to `record_trajectory_result`. | `E-012` (Tranche 03) |
| **6** | **Rate limiting**: Every outbound call to Yahoo Finance or market-data endpoints passes through `TokenBucket`. | **VIOLATED** | `sp500_scanner.py:70` (`fetch_sp500_universe`) and `sp500_scanner.py:170-193` (`evaluate_symbol` when `gateway=None`) bypass `TokenBucket`. | `E-007` (Tranche 02) |
| **7** | **Ingress safety**: Every route, redirect, static asset URL, and template link in the web service works both at `/` and behind an arbitrary `X-Ingress-Path` prefix under Starlette >= 0.33. | **VIOLATED** | `IngressMiddleware` (`scanner_service/api.py:46-51`) sets `scope["root_path"] = ingress_path` without prefixing `scope["path"]`, causing Starlette >= 0.33 `Mount("/ui/static")` to return HTTP 404 on static assets (`Probe A3`). | `E-002`, `E-003` (Tranche 02) |
| **8** | **Decoupled rendering**: Rendering code consumes `ChartPayload` only; it does not reach into app/service state directly. | **VIOLATED** | `compounderos_app/charts.py:119-1000` has no `ChartPayload` dataclass and accesses `self.app.*` directly. | `E-035` (Tranche 04) |
