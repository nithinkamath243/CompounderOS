import json
import os
import sys
from pathlib import Path

# Base directories
COMPOUNDEROS_BASE_DIR = Path(os.environ.get("COMPOUNDEROS_BASE_DIR", Path.home() / "CompounderOS"))
try:
    COMPOUNDEROS_BASE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

BASE_DIR = str(COMPOUNDEROS_BASE_DIR)
LOG_DIR = str(COMPOUNDEROS_BASE_DIR / "logs")
EXPORT_DIR = str(COMPOUNDEROS_BASE_DIR / "exports")
TRANSCRIPTS_DIR = str(COMPOUNDEROS_BASE_DIR / "transcripts")
ASSETS_DIR = str(Path(__file__).parent / "assets")

DATA_DB = str(COMPOUNDEROS_BASE_DIR / "compounderos_data.db")
SCANNER_REPLICA_DB = str(COMPOUNDEROS_BASE_DIR / "scanner_replica.db")
WATCHLIST_FILE = str(COMPOUNDEROS_BASE_DIR / "watchlist.json")
LEGACY_WATCHLIST_FILE = str(COMPOUNDEROS_BASE_DIR / ".compounderos_watchlist.json")

CONFIG_FILE = str(COMPOUNDEROS_BASE_DIR / ".compounderos_config.json")

ICON_PATH = str(Path(ASSETS_DIR) / "icon.png") if os.path.exists(Path(ASSETS_DIR) / "icon.png") else None
ICON_ICNS_PATH = str(Path(ASSETS_DIR) / "icon.icns") if os.path.exists(Path(ASSETS_DIR) / "icon.icns") else None


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


CONFIG = load_config()


def save_config(cfg: dict):
    ensure_dirs()
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def get_setting(key: str, default: str = "") -> str:
    val = os.environ.get(key)
    if val:
        return val
    return CONFIG.get(key, default)


def set_setting(key: str, value: str):
    global CONFIG
    CONFIG[key] = value
    save_config(CONFIG)


def get_sec_user_agent() -> str:
    ua = get_setting("SEC_USER_AGENT", "")
    if ua:
        return ua
    return "CompounderOS/1.0 (contact: user@example.com)"


def sec_user_agent_is_valid(ua: str) -> bool:
    if not ua or "user@example.com" in ua:
        return False
    return "@" in ua and "(" in ua and ")" in ua


def get_service_key(key_name: str) -> str:
    return get_setting(key_name, "")


# A2: Network & Scanner Gateway Config read via get_setting() AFTER definition
# so PyInstaller packaged Mac apps successfully load credentials from JSON config.
SCANNER_API_URL = get_setting("SCANNER_API_URL", "")
SCANNER_API_TOKEN = get_setting("SCANNER_API_TOKEN", "")
SCANNER_SYNC_INTERVAL_SEC = int(get_setting("SCANNER_SYNC_INTERVAL_SEC", "300"))
