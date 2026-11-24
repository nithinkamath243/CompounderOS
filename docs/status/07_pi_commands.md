# Raspberry Pi 5 / Home Assistant OS Diagnostic Commands (`07_pi_commands.md`)

Run these read-only commands on your Raspberry Pi 5 (via the Home Assistant SSH & Web Terminal add-on or direct SSH on port 22222) and paste the output back into a future session if you want to verify live production database state (`UNVERIFIABLE-LOCAL` items).

```bash
# 1. Check add-on status, version, and Ingress config
ha addons info local_compounderos_scanner

# 2. Inspect last 100 lines of scanner service logs
ha addons logs local_compounderos_scanner | tail -n 100

# 3. Check live health and pipeline status
curl -fsS http://127.0.0.1:8321/v1/health

# 4. Inspect SQLite row counts and engine version distribution (read-only copy)
cp /data/addons/data/local_compounderos_scanner/compounderos.db /tmp/compounderos_diag.db
sqlite3 /tmp/compounderos_diag.db "
SELECT 'sp500_scores', COUNT(*), MIN(evaluated_at), MAX(evaluated_at) FROM sp500_scores
UNION ALL
SELECT 'sp500_scores_history', COUNT(*), MIN(evaluated_at), MAX(evaluated_at) FROM sp500_scores_history
UNION ALL
SELECT 'trajectory_scores', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM trajectory_scores
UNION ALL
SELECT 'sp500_lens_scores', COUNT(*), MIN(evaluated_at), MAX(evaluated_at) FROM sp500_lens_scores;
"
sqlite3 /tmp/compounderos_diag.db "
SELECT score_version, registry_id, registry_hash, COUNT(*)
FROM sp500_scores
GROUP BY score_version, registry_id, registry_hash;
"
rm -f /tmp/compounderos_diag.db
```
