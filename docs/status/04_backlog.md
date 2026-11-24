# Phase 5 — Prioritized Implementation Backlog & Tranche Plan (`04_backlog.md`)

## 1. Priority Tiers (All Non-`IMPLEMENTED-VERIFIED` Ledger Items)

### P0 — Correctness & Runtime Blockers (4 items)
- **`E-030` (`WS-G3`)**: Remove unconditional top-level `statsmodels`/`scipy` imports in `compounderos_app/analytics_v2.py:7-8` that crash `scanner_service`, `DipTriggerAnalyzer`, `IncrementalRoicAnalyzer`, `test_imports.py`, and `test_v2_core.py`.
- **`E-009` (`WS-B4`)**: Fix `ScannerPipeline._worker_loop` call to `sp500_scanner.evaluate_symbol` at `scanner_service/pipeline.py:231` (passes 6 positional args to a 3-positional + 1-kwarg function, raising `TypeError` on every symbol).
- **`E-010` (`WS-B5`)**: Fix `ScannerPipeline` starvation livelocks: (a) exclude inactive historical-only symbols from `_pick_next_symbol` fundamental queue while allowing active/watchlist symbols to be evaluated, and (b) record failure cooldown in `self.failures` when `evaluate_symbol` returns `False`.
- **`E-003` & `E-002` (`WS-A3`, `WS-A2`)**: Fix `IngressMiddleware` in `scanner_service/api.py` so `Mount("/ui/static", StaticFiles(...))` returns `200 OK` under `X-Ingress-Path` in Starlette >= 0.33, and update `scanner_service/web/views.py` to use `TemplateResponse(request=request, name=..., context=...)` with fallback HTML rendering when `jinja2` is absent.

### P1 — Architectural Invariants & Engine/Analytics Completeness (7 items)
- **`E-032` (`WS-G2`)**: Implement deterministic pure-NumPy/math fallback in `ScoreBacktester` (`compounderos_app/analytics_v2.py`) for Spearman rank IC and Newey-West (`maxlags=1` Bartlett kernel) HAC t-statistics and p-values when `statsmodels`/`scipy` are not installed, plus unit tests in `compounderos_app/test_v2_core.py`.
- **`E-007` (`WS-B2`)**: Route `sp500_scanner.fetch_sp500_universe` and standalone `sp500_scanner.evaluate_symbol` (`gateway=None`) through `TokenBucket`/`YFGateway` (Invariant 6).
- **`E-008` (`WS-B3`)**: Ensure `sp500_scanner.fetch_sp500_universe` falls back to `store.get_universe_from_ticker_map()` when `http_get_text` returns empty string or `pd.read_html` raises an exception.
- **`E-011` (`WS-C1`)**: Enforce same-day confidence monotonicity on `sp500_scores` snapshot writes in `DataStore.record_quality_result` (Invariant 4).
- **`E-012` (`WS-C2`)**: Pass unmutated `raw_inputs=raw_dict` to `record_trajectory_result` in `scanner_service/pipeline.py:433` so `trajectory_history.raw_inputs_json` is never polluted with derived peer-aggregate keys (Invariant 5).
- **`E-013`, `E-027`, `E-028` (`WS-C3`, `WS-F3`, `WS-F4`)**: Enforce `registry_hash == CANONICAL_REGISTRY_HASH` in `DataStore.record_quality_result` write-guard (Invariant 3) and persist `registry_id` in `sp500_scores_history` and `registry_id` + `registry_hash` in `sp500_scores` via additive schema columns (Invariant 2).
- **`E-035` (`WS-H2`)**: Define `@dataclass(frozen=True) class ChartPayload` in `compounderos_app/charts.py` and refactor `ChartRenderer` to consume `ChartPayload` only without reaching into `self.app` directly (Invariant 8).

### P2 — Hygiene, Config & Packaging (3 items: 2 Ungated, 1 Gated G3)
- **`E-037` (`WS-I2`, Ungated)**: Add `venv/`, `build/`, `dist/`, `.pytest_cache/`, `.hypothesis/`, and `repomix-output.xml` to `.gitignore`.
- **`E-038` (`WS-I3`, Ungated)**: Update `diag_gemini.py:15-16` to resolve `.compounderos_config.json` via `compounderos_app.config.CONFIG_PATH` (with fallback to repo-local `.compounderos_config.json`).
- **`E-036` (`WS-I1`, Gated — Gate G3 `DEFERRED-APPROVAL`)**: Pin dependencies in `requirements.txt` and `requirements-scanner.txt` and add explicit `starlette` and `jinja2` pins. Proposal written to `proposals/E-036.patch` and `proposals/E-036.md`.

---

## 2. Implementation Tranches (Ungated Items Only)

| Tranche | Ledger IDs | Target Files | Est. Diff | Gate Check |
| :--- | :--- | :--- | ---: | :--- |
| **Tranche 01** | `E-030`, `E-032` | `compounderos_app/analytics_v2.py`, `compounderos_app/test_v2_core.py` | ~110 lines | Ungated (no G1–G6) |
| **Tranche 02** | `E-002`, `E-003`, `E-007`, `E-008`, `E-009`, `E-010` | `scanner_service/api.py`, `scanner_service/web/views.py`, `scanner_service/pipeline.py`, `sp500_scanner.py`, `compounderos_app/test_scanner_service.py` | ~220 lines | Ungated (no G1–G6) |
| **Tranche 03** | `E-011`, `E-012`, `E-013`, `E-027`, `E-028` | `compounderos_app/database.py`, `scanner_service/pipeline.py`, `compounderos_app/test_lens_boundary_invariant.py` | ~140 lines | Ungated (additive columns only; no G1–G6) |
| **Tranche 04** | `E-035`, `E-037`, `E-038` | `compounderos_app/charts.py`, `.gitignore`, `diag_gemini.py`, `compounderos_app/test_charts_payload.py` | ~260 lines | Ungated (no G1–G6) |

---

## 3. Gated Proposals (Rule R8 — Do Not Apply Until User Replies `approved <ID>`)

| Ledger ID | Gate | Summary | Proposal Files |
| :--- | :---: | :--- | :--- |
| **`E-036`** | **G3** | Pin exact versions in `requirements.txt` and `requirements-scanner.txt` and add missing `jinja2` and `starlette` pins for HAOS container reproducibility. | `proposals/E-036.patch`, `proposals/E-036.md` |
