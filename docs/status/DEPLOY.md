# CompounderOS v2.1.1 — Raspberry Pi 5 / Home Assistant OS Deployment Runbook (`DEPLOY.md`)

> [!IMPORTANT]
> Per **Rule R2**, none of these deployment commands were executed by the agent. Execute them manually on your MacBook and Raspberry Pi 5 when ready to deploy release `2.1.1`.

---

## 1. Pre-Deploy SQLite Backup on Raspberry Pi 5 (HAOS)

SSH into your Home Assistant OS host (or use the Terminal & SSH add-on) and create a timestamped backup of `/data/compounderos.db` inside `/share`:

```bash
ha addons stop local_compounderos_scanner
mkdir -p /share/compounderos_backups
TS=$(date +%Y%m%d_%H%M%S)
cp -p /data/addons/data/local_compounderos_scanner/compounderos.db \
      /share/compounderos_backups/compounderos_${TS}.db
ls -lh /share/compounderos_backups/compounderos_${TS}.db
```

---

## 2. Stage Updated Add-on Payload & Sync to Raspberry Pi 5

From your MacBook in `/Users/nithinkamath/CompounderOSv2` (on branch `gemini/compounderos-impl`):

```bash
# 1. Stage the runtime code into addons/compounderos_scanner/app/
./scripts/stage_addon.sh

# 2. Sync the staged add-on directory to the Pi's /addons/local/compounderos_scanner/
rsync -avz --delete \
  /Users/nithinkamath/CompounderOSv2/addons/compounderos_scanner/ \
  root@homeassistant.local:/addons/local/compounderos_scanner/
```

---

## 3. Rebuild & Restart the Add-on on Home Assistant OS

On the Raspberry Pi 5 (SSH session):

```bash
ha store reload
ha addons rebuild local_compounderos_scanner
ha addons start local_compounderos_scanner
ha addons logs local_compounderos_scanner
```

---

## 4. Post-Deploy Health, Schema & Ingress Smoke Checks

On the Raspberry Pi 5 (or from your MacBook substituting `homeassistant.local`):

```bash
# 1. Health endpoint check (unauthenticated)
curl -fsS http://127.0.0.1:8321/v1/health | jq .

# 2. Verify schema columns (registry_id and registry_hash on sp500_scores & sp500_scores_history)
sqlite3 /data/addons/data/local_compounderos_scanner/compounderos.db \
  "PRAGMA table_info(sp500_scores);" | grep -E "registry_id|registry_hash"
sqlite3 /data/addons/data/local_compounderos_scanner/compounderos.db \
  "PRAGMA table_info(sp500_scores_history);" | grep -E "registry_id|registry_hash"

# 3. Verify static asset serving via dual mount (/ui/static/styles.css and /static/styles.css)
curl -fsSI http://127.0.0.1:8321/ui/static/styles.css | head -n 5
curl -fsSI http://127.0.0.1:8321/static/styles.css | head -n 5

# 4. Open Home Assistant on iPad -> CompounderOS panel -> confirm CSS/JS load with 200 OK
```

---

## 5. Rollback Procedure

If any issue occurs after deploying `2.1.1`:

```bash
# 1. Stop the add-on on the Pi
ha addons stop local_compounderos_scanner

# 2. Restore the pre-deploy SQLite database backup
cp -p /share/compounderos_backups/compounderos_${TS}.db \
      /data/addons/data/local_compounderos_scanner/compounderos.db

# 3. On MacBook, checkout gemini/pre-audit-snapshot, re-stage, and rsync back
git checkout gemini/pre-audit-snapshot
./scripts/stage_addon.sh
rsync -avz --delete \
  /Users/nithinkamath/CompounderOSv2/addons/compounderos_scanner/ \
  root@homeassistant.local:/addons/local/compounderos_scanner/

# 4. Rebuild and start the previous add-on version on the Pi
ha addons rebuild local_compounderos_scanner
ha addons start local_compounderos_scanner
```
