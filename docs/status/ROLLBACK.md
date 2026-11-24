# CompounderOS Audit & Implementation — Rollback Instructions

## 1. Snapshot Metadata
- **Canonical Repository**: `/Users/nithinkamath/CompounderOSv2`
- **Sandboxed Git Directory**: `/tmp/CompounderOS_git/.git`
- **Original Branch**: `main` (`651dc9d9dcbf4cb28d40cbcd6dfebfd0fd44bb9c`)
- **Pre-Audit Snapshot Branch**: `gemini/pre-audit-snapshot` (`BASE = d1b6206deb07462d9cec3e9f9743bea4b9aefd97`)
- **Implementation Branch**: `gemini/compounderos-impl` (branched from `BASE`)
- **Full Tarball Backup**: `/tmp/CompounderOS_Backups/compounderos_20261027_snapshot.tar.gz` (`21,716,308` bytes)

## 2. Rollback Commands

### Option A: Revert a single tranche commit on `gemini/compounderos-impl`
```zsh
/Applications/Xcode.app/Contents/Developer/usr/bin/git \
  --git-dir=/tmp/CompounderOS_git/.git \
  --work-tree=/Users/nithinkamath/CompounderOSv2 \
  revert --no-edit <commit-hash>
```

### Option B: Restore working tree to exact Pre-Audit Snapshot (`BASE`)
```zsh
/Applications/Xcode.app/Contents/Developer/usr/bin/git \
  --git-dir=/tmp/CompounderOS_git/.git \
  --work-tree=/Users/nithinkamath/CompounderOSv2 \
  checkout gemini/pre-audit-snapshot -- .
```

### Option C: Full restore from pre-audit tarball backup
```zsh
tar -xzf /tmp/CompounderOS_Backups/compounderos_20261027_snapshot.tar.gz -C /Users/nithinkamath/
```
