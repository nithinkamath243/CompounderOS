# Tranche 02 Ledger Delta (Commit `9e48960`)

| ID | Before | After | Proof |
| :--- | :--- | :--- | :--- |
| `E-002` | `PARTIAL` | `IMPLEMENTED-VERIFIED` | `scanner_service/web/views.py:124-213`, `test_scanner_service.py::test_ingress_static_and_ui_routes_under_starlette` |
| `E-003` | `REGRESSED` | `IMPLEMENTED-VERIFIED` | `scanner_service/api.py:45-60`, `test_scanner_service.py::test_ingress_static_and_ui_routes_under_starlette` |
| `E-007` | `PARTIAL` | `IMPLEMENTED-VERIFIED` | `sp500_scanner.py:68-82,152-164`, `test_scanner_service.py::test_evaluate_symbol_standalone_routes_through_yf_gateway` |
| `E-008` | `PARTIAL` | `IMPLEMENTED-VERIFIED` | `sp500_scanner.py:107-134`, `test_scanner_service.py::test_fetch_sp500_universe_uses_token_bucket_and_db_fallback` |
| `E-009` | `PARTIAL` | `IMPLEMENTED-VERIFIED` | `sp500_scanner.py:155-263`, `scanner_service/pipeline.py:224-238`, `test_scanner_service.py::test_scanner_pipeline_worker_records_failure_when_evaluate_returns_false` |
| `E-010` | `PARTIAL` | `IMPLEMENTED-VERIFIED` | `scanner_service/pipeline.py:183-238`, `test_scanner_service.py::test_scanner_pipeline_pick_next_symbol_no_historical_starvation_and_watchlist` |
