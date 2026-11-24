# Tranche 01 Claims & Evidence (`13d37e5`)

1. **Claim**: `compounderos_app/analytics_v2.py` no longer fails at import time when `statsmodels` or `scipy` are absent (`E-030`).
   - **File/Line**: `compounderos_app/analytics_v2.py:1-9`, `compounderos_app/analytics_v2.py:371-378`.
   - **Test**: `compounderos_app/test_v2_core.py` (all 10 tests pass).
2. **Claim**: `ScoreBacktester.rank_ic_newey_west` computes Spearman rank IC from average-tied ranks and Newey-West Bartlett HAC `t_stat` via a deterministic pure-NumPy fallback when `statsmodels` is absent (`E-032`).
   - **File/Line**: `compounderos_app/analytics_v2.py:358-401`.
   - **Test**: `compounderos_app/test_v2_core.py::test_score_backtester_rank_ic_newey_west` and `test_score_backtester_empty_store_returns_insufficient_sample`.
