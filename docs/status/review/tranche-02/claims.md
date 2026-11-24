# Tranche 02 Claims & Evidence (`9e48960`)

1. **Claim (`E-002`, `E-003`)**: `IngressMiddleware` normalizes `X-Ingress-Path` and prefixes `request.scope["path"]` so Starlette >= 0.33 `Mount("/ui/static", StaticFiles(...))` strips `child_scope["root_path"]` and serves `/ui/static/app.css` with `200 OK`, and `/ui/ranking` and `/ui/system` render via `TemplateResponse(request=..., name=..., context=...)`.
   - **File/Line**: `scanner_service/api.py:45-60`, `scanner_service/web/views.py:124-213`.
   - **Test**: `compounderos_app/test_scanner_service.py::test_ingress_static_and_ui_routes_under_starlette`.
2. **Claim (`E-007`, `E-008`)**: `sp500_scanner.fetch_sp500_universe` acquires a `TokenBucket` token before HTTP scraping and falls back to `ticker_map` rows in `store` before `FALLBACK_UNIVERSE`; standalone `evaluate_symbol(..., gateway=None)` routes through `_DEFAULT_GATEWAY` (`YFGateway`).
   - **File/Line**: `sp500_scanner.py:68-134`, `sp500_scanner.py:152-164`.
   - **Test**: `test_fetch_sp500_universe_uses_token_bucket_and_db_fallback`, `test_evaluate_symbol_standalone_routes_through_yf_gateway`.
3. **Claim (`E-009`, `E-010`)**: `ScannerPipeline._pick_next_symbol` excludes delisted historical-only symbols from the fundamental scoring queue and includes pushed watchlist symbols (`__mac_push__`), and `_worker_loop` records a failure cooldown when `evaluate_symbol` returns `False`.
   - **File/Line**: `scanner_service/pipeline.py:183-238`.
   - **Test**: `test_scanner_pipeline_pick_next_symbol_no_historical_starvation_and_watchlist`, `test_scanner_pipeline_worker_records_failure_when_evaluate_returns_false`.
