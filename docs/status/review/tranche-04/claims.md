# Tranche 04 — Verifiable Claims (`claims.md`)

1. **`ChartPayload` Dataclass Exists and is Frozen (`E-035`)**:
   - **File & Lines**: `compounderos_app/charts.py:211-277`
   - **Proof**: `@dataclass(frozen=True) class ChartPayload` encapsulates `fig`, `ax`, `mode`, `freq`, `price_range`, `anim_enabled`, `hist_period`, `hist_data`, `hist_data_long`, `df_fin_q`, `df_fin_a`, `df_bal_q`, `df_bal_a`, `df_cf_q`, `df_cf_a`, `dividends_series`, `info_dividend_rate`, `info_regular_price`, `info_previous_close`, and optional callbacks (`on_analysis`, `on_price_change`, `on_ensure_history_period`, `on_draw_idle`, `on_set_ax2`, `on_set_anim`).
   - **Test**: `compounderos_app/test_charts_payload.py::test_chart_payload_dataclass_and_no_self_app_in_renderer` (`PASSED`).

2. **`ChartRenderer` Has Zero `self.app` Accesses and Renders Standalone (`E-035`)**:
   - **File & Lines**: `compounderos_app/charts.py:280-1103`
   - **Proof**: AST inspection of `class ChartRenderer` confirms zero `self.app` attribute accesses; headless Matplotlib rendering of `ChartPayload` in `Revenue & EBITDA` and `Price` modes succeeds without a `CustomTkinter` `App` instance.
   - **Test**: `compounderos_app/test_charts_payload.py::test_chart_payload_dataclass_and_no_self_app_in_renderer` (`PASSED`).

3. **`.gitignore` Excludes Packaging/Test Artifacts (`E-037`)**:
   - **File & Lines**: `.gitignore:18-24`
   - **Proof**: `build/`, `dist/`, `.pytest_cache/`, `*.egg-info/`, `.coverage`, `docs/status/*.bundle` are present in `.gitignore`.
   - **Test**: `compounderos_app/test_charts_payload.py::test_gitignore_covers_build_artifacts` (`PASSED`).

4. **`diag_gemini.py` Uses Canonical `CONFIG_FILE` (`E-038`)**:
   - **File & Lines**: `diag_gemini.py:4-8`, `compounderos_app/config.py:13-21`
   - **Proof**: `diag_gemini.py` imports `CONFIG_FILE` from `compounderos_app.config` and has zero `Path.home()` calls.
   - **Test**: `compounderos_app/test_charts_payload.py::test_diag_gemini_uses_canonical_config_path` (`PASSED`).
