import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
import os
import ssl
import sys
import webbrowser
import numpy as np
import re
import time
import sqlite3
import io
import html
import textwrap
try:
    import requests
except Exception:
    requests = None
import urllib.request
import urllib.error
from datetime import datetime, UTC
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import logging

# --- GEMINI AI IMPORT ---
import google.generativeai as genai

# --- SSL CERTIFICATE FIX (For MacOS) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- CONFIGURATION ---
USER_HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(USER_HOME, "OpenTrim")
LOG_DIR = os.path.join(BASE_DIR, "logs")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
WATCHLIST_FILE = os.path.join(BASE_DIR, "opentrim_watchlist.json")
LEGACY_WATCHLIST_FILE = os.path.join(USER_HOME, "opentrim_watchlist.json")
API_KEY_FILE = os.path.join(BASE_DIR, ".gemini_key")
LEGACY_API_KEY_FILE = os.path.join(USER_HOME, ".gemini_key")
DATA_DB = os.path.join(BASE_DIR, "opentrim_data.db")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "opentrim_transcripts")
CONFIG_FILE = os.path.join(BASE_DIR, ".opentrim_config.json")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "opentrim_errors.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def log_exception(context: str, exc: Exception):
    logging.error("%s: %s", context, exc, exc_info=True)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            log_exception("load_config", e)
            return {}
    return {}

def save_config(data: Dict[str, Any]):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_exception("save_config", e)

CONFIG = load_config()

def migrate_legacy_gemini_key():
    if "GEMINI_API_KEY" in CONFIG:
        return
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r") as f:
                key = f.read().strip()
            if key:
                CONFIG["GEMINI_API_KEY"] = key
                save_config(CONFIG)
        except Exception as e:
            log_exception("migrate_legacy_gemini_key.primary", e)
    elif os.path.exists(LEGACY_API_KEY_FILE):
        try:
            with open(LEGACY_API_KEY_FILE, "r") as f:
                key = f.read().strip()
            if key:
                CONFIG["GEMINI_API_KEY"] = key
                save_config(CONFIG)
        except Exception as e:
            log_exception("migrate_legacy_gemini_key.legacy", e)

migrate_legacy_gemini_key()
SEC_USER_AGENT = CONFIG.get("SEC_USER_AGENT") or os.environ.get("SEC_USER_AGENT", "OpenTrim/4.2 (contact: user@example.com)")
FINNHUB_API_KEY = CONFIG.get("FINNHUB_API_KEY")
ALPHAVANTAGE_API_KEY = CONFIG.get("ALPHAVANTAGE_API_KEY")
API_NINJAS_KEY = CONFIG.get("API_NINJAS_KEY")

# --- THEME PALETTE ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

C_BG_MAIN     = "#0B0F14"  # App background
C_BG_CARD     = "#141B2A"  # Card background
C_BG_INPUT    = "#1B2436"
C_ACCENT      = "#4DA3FF"
C_ACCENT_HOVER = "#3A8DD6"
C_TEXT_MAIN   = "#E6EAF0"
C_TEXT_SUB    = "#9AA4B2"
C_GREEN       = "#3AD29F"
C_RED         = "#FF6B6B"
C_YELLOW      = "#F6C453"
C_PINK        = "#FF7AA2"
C_BORDER      = "#1F2A3D"

# --- WATCHLIST COLORS ---
COLOR_UP_BG   = "#0E3B2E"
COLOR_DIP_BG  = "#3B1414"

# --- TYPOGRAPHY ---
FONT_BRAND   = ("Avenir Next", 26, "bold")
FONT_DISPLAY = ("Avenir Next", 40, "bold")
FONT_SCORE   = ("Avenir Next", 30, "bold")
FONT_H1      = ("Avenir Next", 18, "bold")
FONT_H2      = ("Avenir Next", 14, "bold")
FONT_BODY    = ("Avenir Next", 12)
FONT_SMALL   = ("Avenir Next", 11)
FONT_TINY    = ("Avenir Next", 10)

ENTRY_KW = dict(font=FONT_BODY, fg_color=C_BG_INPUT, border_color=C_BORDER, corner_radius=10, height=30)
COMBO_KW = dict(
    font=FONT_BODY,
    fg_color=C_BG_INPUT,
    border_color=C_BORDER,
    button_color=C_BG_INPUT,
    button_hover_color="#2B364A",
    dropdown_fg_color=C_BG_CARD,
    dropdown_hover_color=C_BG_INPUT,
    text_color=C_TEXT_MAIN,
    corner_radius=10
)
CHECK_KW = dict(fg_color=C_ACCENT, hover_color=C_ACCENT_HOVER, text_color=C_TEXT_MAIN)
BTN_PRIMARY = dict(fg_color=C_ACCENT, hover_color=C_ACCENT_HOVER, text_color=C_TEXT_MAIN, corner_radius=10)
BTN_SECONDARY = dict(fg_color=C_BG_INPUT, hover_color="#223047", text_color=C_TEXT_MAIN, corner_radius=10)
BTN_DANGER = dict(fg_color=C_RED, hover_color="#E45555", text_color=C_TEXT_MAIN, corner_radius=10)

# Matplotlib Dark Theme
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': C_BG_CARD,
    'figure.facecolor': C_BG_CARD,
    'grid.color': '#2A364A',
    'text.color': C_TEXT_MAIN,
    'axes.labelcolor': C_TEXT_SUB,
    'xtick.color': C_TEXT_SUB,
    'ytick.color': C_TEXT_SUB,
    'axes.edgecolor': C_BG_CARD,
    'axes.linewidth': 0
})

ANIM_FRAMES = 25
PAST_EARNINGS_COUNT = 4
AV_TRANSCRIPTS_QUARTERS = 4
CALENDAR_CACHE_TTL_SEC = 12 * 60 * 60
TRANSCRIPT_FETCH_TTL_SEC = 6 * 60 * 60
SEARCH_DEBOUNCE_MS = 350
REDRAW_DEBOUNCE_MS = 120

class DataStore:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._migrate()

    def _migrate(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ticker_map (
            ticker TEXT PRIMARY KEY,
            cik TEXT NOT NULL,
            name TEXT,
            updated_at TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT NOT NULL,
            PRIMARY KEY (symbol, date, source)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            symbol TEXT NOT NULL,
            date TEXT,
            title TEXT,
            content TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_analysis (
            symbol TEXT NOT NULL,
            date TEXT,
            title TEXT,
            summary TEXT,
            qa TEXT,
            updated_at TEXT,
            PRIMARY KEY (symbol, date, title)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS kpis (
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            period TEXT,
            value TEXT,
            unit TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlists (
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            PRIMARY KEY (name, symbol)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            shares REAL,
            cost_basis REAL,
            PRIMARY KEY (name, symbol)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dcf_inputs (
            symbol TEXT PRIMARY KEY,
            data_type TEXT,
            fcf_or_eps REAL,
            growth REAL,
            discount REAL,
            terminal_mult REAL,
            target_return REAL,
            updated_at TEXT
        )
        """)
        self.conn.commit()

    def get_ticker_map(self, ticker: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT cik, name, updated_at FROM ticker_map WHERE ticker = ?", (ticker,))
            return cur.fetchone()

    def upsert_ticker_map(self, ticker: str, cik: str, name: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO ticker_map (ticker, cik, name, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik, name=excluded.name, updated_at=excluded.updated_at",
                (ticker, cik, name, datetime.now(UTC).isoformat())
            )
            self.conn.commit()

    def upsert_price_history(self, symbol: str, df: pd.DataFrame, source: str):
        if df is None or df.empty:
            return
        rows = []
        for idx, r in df.iterrows():
            rows.append((
                symbol, idx.strftime("%Y-%m-%d"),
                float(r.get("Open", np.nan)), float(r.get("High", np.nan)),
                float(r.get("Low", np.nan)), float(r.get("Close", np.nan)),
                float(r.get("Volume", np.nan)), source
            ))
        with self.lock:
            cur = self.conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO price_history (symbol, date, open, high, low, close, volume, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows
            )
            self.conn.commit()

    def load_price_history(self, symbol: str, source: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT date, open, high, low, close, volume FROM price_history WHERE symbol = ? AND source = ? ORDER BY date ASC",
                (symbol, source)
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df

    def upsert_transcript(self, symbol: str, date: str, title: str, content: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO transcripts (symbol, date, title, content) VALUES (?, ?, ?, ?)",
                (symbol, date, title, content)
            )
            self.conn.commit()

    def search_transcripts(self, symbol: str, query: str):
        with self.lock:
            cur = self.conn.cursor()
            q = f"%{query}%"
            cur.execute(
                "SELECT date, title, content FROM transcripts WHERE symbol = ? AND content LIKE ? ORDER BY date DESC",
                (symbol, q)
            )
            return cur.fetchall()

    def list_transcripts(self, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT date, title, content FROM transcripts WHERE symbol = ? ORDER BY date DESC",
                (symbol,)
            )
            return cur.fetchall()

    def transcript_exists(self, symbol: str, date: str, title: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT 1 FROM transcripts WHERE symbol = ? AND date = ? AND title = ? LIMIT 1",
                (symbol, date, title)
            )
            return cur.fetchone() is not None

    def get_transcript_analysis(self, symbol: str, date: str, title: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT summary, qa FROM transcript_analysis WHERE symbol = ? AND date = ? AND title = ?",
                (symbol, date, title)
            )
            return cur.fetchone()

    def upsert_transcript_analysis(self, symbol: str, date: str, title: str, summary: str, qa: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO transcript_analysis (symbol, date, title, summary, qa, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, date, title) DO UPDATE SET summary=excluded.summary, qa=excluded.qa, updated_at=excluded.updated_at",
                (symbol, date, title, summary, qa, datetime.now(UTC).isoformat())
            )
            self.conn.commit()

    def list_watchlists(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT name FROM watchlists ORDER BY name ASC")
            return [r[0] for r in cur.fetchall()]

    def get_watchlist(self, name: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT symbol FROM watchlists WHERE name = ? ORDER BY symbol ASC", (name,))
            return [r[0] for r in cur.fetchall()]

    def add_watchlist_symbol(self, name: str, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO watchlists (name, symbol) VALUES (?, ?)",
                (name, symbol)
            )
            self.conn.commit()

    def remove_watchlist_symbol(self, name: str, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM watchlists WHERE name = ? AND symbol = ?", (name, symbol))
            self.conn.commit()

    def list_portfolios(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT name FROM portfolios ORDER BY name ASC")
            return [r[0] for r in cur.fetchall()]

    def get_portfolio(self, name: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT symbol, shares, cost_basis FROM portfolios WHERE name = ? ORDER BY symbol ASC",
                (name,)
            )
            return cur.fetchall()

    def upsert_portfolio_symbol(self, name: str, symbol: str, shares: float, cost_basis: float):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO portfolios (name, symbol, shares, cost_basis) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name, symbol) DO UPDATE SET shares=excluded.shares, cost_basis=excluded.cost_basis",
                (name, symbol, shares, cost_basis)
            )
            self.conn.commit()

    def remove_portfolio_symbol(self, name: str, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM portfolios WHERE name = ? AND symbol = ?", (name, symbol))
            self.conn.commit()

    def list_kpis(self, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT name, period, value, unit FROM kpis WHERE symbol = ? ORDER BY name ASC",
                (symbol,)
            )
            return cur.fetchall()

    def add_kpi(self, symbol: str, name: str, period: str, value: str, unit: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO kpis (symbol, name, period, value, unit) VALUES (?, ?, ?, ?, ?)",
                (symbol, name, period, value, unit)
            )
            self.conn.commit()

    def clear_kpis(self, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM kpis WHERE symbol = ?", (symbol,))
            self.conn.commit()

    def clear_kpis_for_period(self, symbol: str, period: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM kpis WHERE symbol = ? AND period = ?", (symbol, period))
            self.conn.commit()

    def get_dcf_inputs(self, symbol: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT data_type, fcf_or_eps, growth, discount, terminal_mult, target_return FROM dcf_inputs WHERE symbol = ?",
                (symbol,)
            )
            return cur.fetchone()

    def upsert_dcf_inputs(self, symbol: str, data_type: str, fcf_or_eps: float, growth: float, discount: float, terminal_mult: float, target_return: float):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO dcf_inputs (symbol, data_type, fcf_or_eps, growth, discount, terminal_mult, target_return, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET data_type=excluded.data_type, fcf_or_eps=excluded.fcf_or_eps, "
                "growth=excluded.growth, discount=excluded.discount, terminal_mult=excluded.terminal_mult, target_return=excluded.target_return, updated_at=excluded.updated_at",
                (symbol, data_type, fcf_or_eps, growth, discount, terminal_mult, target_return, datetime.now(UTC).isoformat())
            )
            self.conn.commit()

class OpenTrimApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("OpenTrim v4.1 - Enhanced & Robust")
        
        # Safer default geometry for laptops
        self.root.geometry("1280x800")
        try:
            self.root.configure(fg_color=C_BG_MAIN)
            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
        except Exception:
            pass
        
        # Data Storage
        self.gemini_api_key = self.load_api_key()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.current_ticker: Optional[yf.Ticker] = None
        self.current_symbol: str = ""
        self.df_fin_q: Optional[pd.DataFrame] = None
        self.df_bal_q: Optional[pd.DataFrame] = None
        self.df_cf_q: Optional[pd.DataFrame] = None
        self.df_dividends: Optional[pd.Series] = None
        self.hist_data: Optional[pd.DataFrame] = None
        
        self.anim = None 
        self.metric_explanations = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        self.store = DataStore(DATA_DB)
        self.current_watchlist = "Main"
        self.current_portfolio = "Main"
        self.watchlist_data: List[str] = self.load_watchlist()
        self.gemini_models = None
        self.transcript_fetch_status = ""
        self.calendar_cache = {"ts": 0.0, "rows": [], "source": "None", "error": ""}
        self.transcript_fetch_cache: Dict[str, Dict[str, float]] = {}
        self.search_job = None
        self.pending_search_symbol = None
        self._redraw_job = None
        self._dcf_job = None
        self._ax2 = None
        self.chart_ai_cache = {}
        self.chart_ai_inflight = False
        self.chart_last_key = None
        self.chart_last_summary = None
        self.executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="OpenTrim")
        self._busy_lock = threading.Lock()
        self._busy_count = 0
        self.metric_payloads: Dict[str, Dict[str, str]] = {}
        self.quality_ai_cache: Dict[str, Dict[str, str]] = {}
        self.quality_ai_inflight = False
        self.quality_ai_key = ""
        
        # Threading Flags
        self.watchlist_loading = False
        self.request_id = 0
        self.hist_period = "1y"
        self.hist_data_long = None
        self.df_fin_a: Optional[pd.DataFrame] = None
        self.df_bal_a: Optional[pd.DataFrame] = None
        self.df_cf_a: Optional[pd.DataFrame] = None
        

        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        if not self.store.list_watchlists():
            self.store.add_watchlist_symbol("Main", "__INIT__")
            self.store.remove_watchlist_symbol("Main", "__INIT__")
        # Migrate legacy JSON watchlist if DB is empty
        try:
            if not self.store.get_watchlist("Main") and os.path.exists(WATCHLIST_FILE):
                with open(WATCHLIST_FILE, "r") as f:
                    legacy = json.load(f)
                if isinstance(legacy, list):
                    for sym in legacy:
                        if isinstance(sym, str) and sym.strip():
                            self.store.add_watchlist_symbol("Main", sym.strip().upper())
            elif not self.store.get_watchlist("Main") and os.path.exists(LEGACY_WATCHLIST_FILE):
                with open(LEGACY_WATCHLIST_FILE, "r") as f:
                    legacy = json.load(f)
                if isinstance(legacy, list):
                    for sym in legacy:
                        if isinstance(sym, str) and sym.strip():
                            self.store.add_watchlist_symbol("Main", sym.strip().upper())
        except Exception as e:
            log_exception("init.watchlist_migration", e)
        if not self.store.list_portfolios():
            self.store.upsert_portfolio_symbol("Main", "__INIT__", 0, 0)
            self.store.remove_portfolio_symbol("Main", "__INIT__")

        self.setup_styles()
        self.create_layout()

        # Try to maximize window on start
        try:
            self.root.after(100, lambda: self.root.state('zoomed'))
        except Exception as e:
            log_exception("init.zoomed", e)

    def setup_styles(self):
        # We still need ttk.Style for the Treeviews as CTK doesn't have a native Table widget
        s = ttk.Style()
        s.theme_use('clam')
        
        # Configure Treeview to look like CustomTkinter
        s.configure("Treeview", 
                    background=C_BG_CARD, 
                    foreground=C_TEXT_MAIN, 
                    fieldbackground=C_BG_CARD, 
                    borderwidth=0, 
                    rowheight=30, 
                    font=FONT_BODY)
        
        s.configure("Treeview.Heading", 
                    background=C_BG_INPUT, 
                    foreground="white", 
                    font=FONT_H2, 
                    borderwidth=0,
                    relief="flat")
        
        s.map("Treeview", background=[('selected', C_ACCENT)])
        s.map("Treeview.Heading", background=[('active', '#3a3a3a')])

    def load_api_key(self):
        # Config file first
        cfg_key = CONFIG.get("GEMINI_API_KEY")
        if cfg_key: return cfg_key
        
        # 2. Check File
        if os.path.exists(API_KEY_FILE):
            try:
                with open(API_KEY_FILE, "r") as f:
                    return f.read().strip()
            except Exception as e:
                log_exception("load_api_key", e)
        return None

    def save_api_key(self, key):
        CONFIG["GEMINI_API_KEY"] = key
        save_config(CONFIG)
        self.gemini_api_key = key

    # --- MATH HELPER ---
    def safe_div(self, n, d, default=0.0):
        """Safely divides n by d, handling zero, NaN, and Inf."""
        try:
            if d == 0 or pd.isna(d) or pd.isna(n) or np.isinf(d) or np.isinf(n):
                return default
            return n / d
        except Exception:
            return default

    def submit_task(self, fn, *args, context: str = "task"):
        self._busy_inc()

        def runner():
            try:
                return fn(*args)
            except Exception as e:
                log_exception(context, e)
                return None
            finally:
                self.root.after(0, self._busy_dec)

        try:
            return self.executor.submit(runner)
        except Exception as e:
            log_exception(f"submit_task:{context}", e)
            self._busy_dec()
            return None

    def _busy_inc(self):
        with self._busy_lock:
            self._busy_count += 1
        self.root.after(0, self._update_busy_ui)

    def _busy_dec(self):
        with self._busy_lock:
            self._busy_count = max(0, self._busy_count - 1)
        self._update_busy_ui()

    def _update_busy_ui(self):
        busy = self._busy_count > 0
        try:
            if busy:
                self.prog_busy.start()
                self.root.configure(cursor="watch")
            else:
                self.prog_busy.stop()
                self.root.configure(cursor="")
        except Exception:
            pass

    def make_combo(self, *args, **kwargs):
        cb = ctk.CTkComboBox(*args, **kwargs)
        try:
            cb.configure(state="readonly")
        except Exception:
            try:
                cb.configure(entry_state="readonly")
            except Exception:
                pass
        return cb

    def _series_stats(self, values, label: str, unit: str = ""):
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return f"{label}: no data."
        start = arr[0]
        end = arr[-1]
        minv = arr.min()
        maxv = arr.max()
        pct = (end - start) / abs(start) * 100 if start not in (0, 0.0) else 0.0
        return f"{label}: start {start:.2f}{unit}, end {end:.2f}{unit}, change {pct:+.1f}%, min {minv:.2f}{unit}, max {maxv:.2f}{unit}."

    def _set_chart_analysis(self, key, summary):
        sym = self.current_symbol or "N/A"
        self.chart_last_key = f"{sym}|{key}"
        self.chart_last_summary = summary
        self.schedule_chart_analysis()

    def schedule_chart_analysis(self, force: bool = False):
        if not hasattr(self, "txt_chart_ai"):
            return
        if not self.gemini_api_key:
            self._update_chart_ai("Gemini API key missing. Add it to the config file to enable chart insights.")
            return
        key = self.chart_last_key
        if not key or not self.chart_last_summary:
            return
        if not force and key in self.chart_ai_cache:
            self._update_chart_ai(self.chart_ai_cache[key])
            return
        if self.chart_ai_inflight:
            return
        self.chart_ai_inflight = True
        self._update_chart_ai("Analyzing chart...")

        def worker():
            try:
                prompt = (
                    f"You are a financial analyst. Analyze the chart data for {self.current_symbol or 'N/A'}.\n"
                    f"Chart: {key}\n"
                    f"Data summary:\n{self.chart_last_summary}\n\n"
                    "Provide 4-6 concise bullet insights about trends, inflections, and notable changes. "
                    "Avoid financial advice."
                )
                text = self.gemini_generate(prompt) or "Analysis unavailable."
                self.chart_ai_cache[key] = text
                self._update_chart_ai(text)
            except Exception as e:
                log_exception("schedule_chart_analysis.worker", e)
                self._update_chart_ai("Analysis unavailable.")
            finally:
                self.chart_ai_inflight = False
        self.submit_task(worker, context="chart_ai")

    def _update_chart_ai(self, text: str):
        if not hasattr(self, "txt_chart_ai"):
            return
        self.root.after(0, lambda: self._update_chart_ai_safe(text))

    def _update_chart_ai_safe(self, text: str):
        try:
            self.txt_chart_ai.configure(state="normal")
            self.txt_chart_ai.delete(1.0, tk.END)
            self.txt_chart_ai.insert(tk.END, text)
            self.txt_chart_ai.configure(state="disabled")
        except Exception:
            pass

    def create_layout(self):
        # Main container
        container = ctk.CTkFrame(self.root, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=24, pady=24)

        # --- HEADER ---
        top_bar = ctk.CTkFrame(container, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        top_bar.pack(fill=tk.X, pady=(0, 18))
        
        ctk.CTkLabel(top_bar, text="OPENTRIM", font=FONT_BRAND, text_color=C_TEXT_MAIN).pack(side=tk.LEFT, padx=(16, 20), pady=10)
        
        # Search Area
        self.search_var = tk.StringVar()
        self.entry_search = ctk.CTkEntry(
            top_bar, textvariable=self.search_var, width=220, height=34, font=FONT_BODY,
            placeholder_text="Ticker (e.g. AAPL)", fg_color=C_BG_INPUT, border_color=C_BORDER, corner_radius=12
        )
        self.entry_search.pack(side=tk.LEFT, padx=8, pady=10)
        self.entry_search.bind('<Return>', lambda e: self.start_search())

        ctk.CTkButton(top_bar, text="SEARCH", command=self.start_search, width=90, height=34, font=FONT_SMALL, **BTN_PRIMARY).pack(side=tk.LEFT, padx=8, pady=10)
        
        self.lbl_status = ctk.CTkLabel(top_bar, text="Ready", text_color=C_TEXT_SUB, font=FONT_SMALL)
        self.lbl_status.pack(side=tk.LEFT, padx=15, pady=10)

        self.prog_busy = ctk.CTkProgressBar(top_bar, width=120, mode="indeterminate", progress_color=C_ACCENT)
        self.prog_busy.pack(side=tk.RIGHT, padx=8, pady=10)
        self.prog_busy.stop()
        ctk.CTkButton(top_bar, text="EXPORT", command=self.export_current_view, width=90, height=32, font=FONT_SMALL, **BTN_SECONDARY).pack(side=tk.RIGHT, padx=6, pady=10)
        ctk.CTkButton(top_bar, text="+ WATCHLIST", command=self.add_to_watchlist, width=110, height=32, font=FONT_SMALL, **BTN_SECONDARY).pack(side=tk.RIGHT, padx=6, pady=10)

        # --- TABS ---
        self.tabview = ctk.CTkTabview(container)
        self.tabview.configure(
            fg_color="transparent",
            segmented_button_fg_color=C_BG_CARD,
            segmented_button_selected_color=C_ACCENT,
            segmented_button_selected_hover_color="#3A8DD6",
            segmented_button_unselected_color=C_BG_INPUT,
            segmented_button_unselected_hover_color="#263047",
            text_color=C_TEXT_MAIN,
            text_color_disabled=C_TEXT_SUB
        )
        self.tabview.pack(fill=tk.BOTH, expand=True)
        try:
            self.tabview._segmented_button.configure(font=FONT_SMALL)
        except:
            pass

        # Create Tabs
        self.tab_insights = self.tabview.add("INSIGHTS")
        self.tab_carlson = self.tabview.add("QUALITY SCORE")
        self.tab_charts = self.tabview.add("CHARTS")
        self.tab_val = self.tabview.add("VALUATION")
        self.tab_insider = self.tabview.add("INSIDER")
        self.tab_ai = self.tabview.add("AI REPORT")
        self.tab_watchlist = self.tabview.add("WATCHLIST")
        self.tab_earnings = self.tabview.add("EARNINGS")
        self.tab_transcripts = self.tabview.add("TRANSCRIPTS")
        self.tab_portfolio = self.tabview.add("PORTFOLIO")
        self.tab_kpis = self.tabview.add("KPIS")

        # Build Tab Content
        self.build_insights()
        self.build_carlson_tab()
        self.build_charts()
        self.build_valuation()
        self.build_insider_calendar()
        self.build_ai()
        self.build_watchlist()
        self.build_earnings()
        self.build_transcripts()
        self.build_portfolio()
        self.build_kpis()
        self.root.after(300, self.refresh_earnings)

    def ask_api_key(self):
        dialog = ctk.CTkInputDialog(text="Enter Gemini API Key:", title="API Key")
        k = dialog.get_input()
        if k: self.save_api_key(k)

    def export_current_view(self):
        try:
            tab = self.tabview.get()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(EXPORT_DIR, f"{tab.lower().replace(' ', '_')}_{stamp}.csv")
            rows = []
            headers = []

            if tab == "WATCHLIST":
                headers = ["Symbol", "Price", "SMA200", "Status"]
                rows = [self.tr_watch.item(i).get("values", []) for i in self.tr_watch.get_children()]
            elif tab == "EARNINGS":
                headers = ["Symbol", "Date", "Time", "EPS Est", "EPS Act", "Surprise"]
                rows = [self.tr_earn.item(i).get("values", []) for i in self.tr_earn.get_children()]
            elif tab == "QUALITY SCORE":
                headers = ["Metric", "Target", "Actual", "Result", "Context"]
                rows = [self.tr_check.item(i).get("values", []) for i in self.tr_check.get_children()]
            elif tab == "PORTFOLIO":
                headers = ["Symbol", "Shares", "Cost Basis", "Price", "Value"]
                rows = [self.tr_port.item(i).get("values", []) for i in self.tr_port.get_children()]
            elif tab == "KPIS":
                headers = ["Name", "Period", "Value", "Unit"]
                rows = [self.tr_kpi.item(i).get("values", []) for i in self.tr_kpi.get_children()]
            else:
                messagebox.showinfo("Export", f"Export is not configured for {tab} yet.")
                return

            if not rows:
                messagebox.showinfo("Export", f"No rows to export for {tab}.")
                return

            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(path, index=False)
            self.lbl_status.configure(text=f"Exported: {os.path.basename(path)}", text_color=C_GREEN)
            messagebox.showinfo("Export Complete", f"Saved to:\n{path}")
        except Exception as e:
            log_exception("export_current_view", e)
            messagebox.showerror("Export Error", str(e))

    # --- LOGIC ---
    def start_search(self):
        s = self.search_var.get().upper().strip()
        if not s: return
        self.pending_search_symbol = s
        if self.search_job:
            try: self.root.after_cancel(self.search_job)
            except: pass
        self.search_job = self.root.after(SEARCH_DEBOUNCE_MS, self._run_search)

    def _run_search(self):
        s = self.pending_search_symbol
        self.search_job = None
        if not s: 
            return
        self.lbl_status.configure(text=f"Loading {s}...", text_color=C_ACCENT)
        # Disable search during load
        self.entry_search.configure(state="disabled")
        self.request_id += 1
        req_id = self.request_id
        self.submit_task(self.fetch_data, s, req_id, context="fetch_data")

    def fetch_data(self, symbol, req_id):
        try:
            with self.cache_lock:
                cached = self.cache.get(symbol, {})
            t = cached.get("ticker") or yf.Ticker(symbol)

            # Fetch info separately to catch errors early
            try:
                i = cached.get("info") or t.info
                # Basic validation: if no regular market price, symbol might be invalid
                if 'currentPrice' not in i and 'regularMarketPrice' not in i:
                     raise ValueError("Symbol not found or no data.")
            except Exception as e:
                raise ValueError(f"Info fetch failed: {e}")
            
            def cached_df(key):
                df = cached.get(key)
                if df is None:
                    return None
                return df

            # Use safe fetching for dataframes
            h = cached.get("history_1y")
            if h is None:
                h = t.history(period="1y")
            h = self.calculate_technical_indicators(h)
            
            fin_q = cached_df("fin_q")
            if fin_q is None:
                fin_q = t.quarterly_financials.T if t.quarterly_financials is not None else pd.DataFrame()
            bs_q = cached_df("bs_q")
            if bs_q is None:
                bs_q = t.quarterly_balance_sheet.T if t.quarterly_balance_sheet is not None else pd.DataFrame()
            cf_q = cached_df("cf_q")
            if cf_q is None:
                cf_q = t.quarterly_cashflow.T if t.quarterly_cashflow is not None else pd.DataFrame()
            divs = cached.get("divs")
            if divs is None:
                divs = t.dividends
            
            cal = cached.get("cal")
            if cal is None:
                cal = pd.DataFrame()
            try:
                if cal is None or cal.empty:
                    if t.calendar is not None and isinstance(t.calendar, dict): 
                        cal = pd.DataFrame(list(t.calendar.items()), columns=['Label', 'Value'])
                    elif t.earnings_dates is not None: 
                        cal = t.earnings_dates.head(12)
            except Exception as e:
                log_exception("fetch_data.calendar", e)

            insider = cached.get("insider")
            if insider is None:
                insider = pd.DataFrame()
            try:
                if insider is None or insider.empty:
                    insider = t.insider_transactions.head(20)
            except Exception as e:
                log_exception("fetch_data.insider", e)

            with self.cache_lock:
                self.cache[symbol] = {
                    "ticker": t,
                    "info": i,
                    "history_1y": h,
                    "fin_q": fin_q,
                    "bs_q": bs_q,
                    "cf_q": cf_q,
                    "divs": divs,
                    "cal": cal,
                    "insider": insider,
                }
            
            self.root.after(0, lambda: self.update_ui(symbol, t, i, h, fin_q, bs_q, cf_q, insider, cal, divs, req_id))

            # Kick off long-history fetch via Stooq (non-blocking)
            self.submit_task(self.fetch_stooq_history, symbol, req_id, context="fetch_stooq_history")
            # Kick off SEC financials (non-blocking)
            self.submit_task(self.fetch_sec_financials, symbol, req_id, context="fetch_sec_financials")
        except Exception as e:
            log_exception("fetch_data", e)
            err_msg = str(e)
            self.root.after(0, lambda err_msg=err_msg: self.handle_error(err_msg, req_id))

    def handle_error(self, msg, req_id):
        if req_id != self.request_id:
            return
        self.lbl_status.configure(text=f"Error: {msg}", text_color=C_RED)
        self.entry_search.configure(state="normal")

    def update_ui(self, s, t, i, h, fin_q, bs_q, cf_q, insider, cal, divs, req_id):
        if req_id != self.request_id:
            return
        if not self.root.winfo_exists():
            return
        # Re-enable search
        self.entry_search.configure(state="normal")
        
        if h is not None and not h.empty: h.index = h.index.tz_localize(None)
        if divs is not None and not divs.empty: divs.index = divs.index.tz_localize(None)
        
        def clean_df(df):
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.sort_index()
            return df

        self.current_ticker = t
        self.current_symbol = s
        self.df_fin_q = clean_df(fin_q)
        self.df_bal_q = clean_df(bs_q)
        self.df_cf_q = clean_df(cf_q)
        self.hist_data = h
        self.hist_period = "1y"
        self.hist_data_long = None
        self.df_dividends = divs
        
        self.fill_insights(i, t)
        self.fill_charts()
        self.fill_insider_calendar(insider, cal)
        self.fill_quality_checklist(i, self.df_fin_q, self.df_bal_q, self.df_cf_q)
        self.reset_dcf(i)
        self.load_transcripts()
        self.load_kpis()
        self.auto_refresh_earnings(s)
        self.submit_task(self.fetch_sec_earnings_release, s, req_id, False, context="fetch_sec_earnings_release")
        self.submit_task(self.fetch_av_transcripts, s, req_id, False, context="fetch_av_transcripts")
        self.submit_task(self.fetch_api_ninjas_transcripts, s, req_id, False, context="fetch_api_ninjas_transcripts")
        
        # Safe AI Update via Thread
        self.txt_ai.configure(state="normal")
        self.txt_ai.delete(1.0, tk.END)
        self.txt_ai.insert(tk.END, f"Analyzing (Smart Mode)...\n")
        self.txt_ai.configure(state="disabled")
        self.submit_task(self.run_gemini_25_logic, s, i, self.df_fin_q, context="run_gemini_25_logic")
        
        self.lbl_status.configure(text=f"Active: {s}", text_color=C_GREEN)

    def calculate_technical_indicators(self, df: Optional[pd.DataFrame]):
        if df is None or df.empty or "Close" not in df.columns:
            return df
        try:
            out = df.copy()
            close = pd.to_numeric(out["Close"], errors="coerce")
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
            rs = gain / loss.replace(0, np.nan)
            out["RSI"] = 100 - (100 / (1 + rs))
            out["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
            out["SMA_200"] = close.rolling(window=200, min_periods=1).mean()
            return out
        except Exception as e:
            log_exception("calculate_technical_indicators", e)
            return df

    def get_series(self, df, possible_keys):
        if df is None or df.empty: return None
        for col in df.columns:
            norm_col = "".join(str(col).split()).lower()
            for k in possible_keys:
                if "".join(k.split()).lower() in norm_col:
                    return pd.to_numeric(df[col], errors='coerce').dropna().sort_index()
        return None

    # --- TAB: QUALITY CHECKLIST ---
    def build_carlson_tab(self):
        self.tab_carlson.grid_columnconfigure(0, weight=2)
        self.tab_carlson.grid_columnconfigure(1, weight=1)
        self.tab_carlson.grid_rowconfigure(0, weight=1)

        # Left: Table
        self.f_check = ctk.CTkFrame(self.tab_carlson, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        self.f_check.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        
        ctk.CTkLabel(self.f_check, text="COMPOUNDER CHECKLIST", font=FONT_H1, text_color=C_ACCENT).pack(anchor='w', pady=15, padx=15)
        
        # Frame for Treeview + Scrollbar
        tv_frame = ctk.CTkFrame(self.f_check, fg_color="transparent")
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        cols = ("Metric", "Target", "Actual", "Result", "Context")
        self.tr_check = ttk.Treeview(tv_frame, columns=cols, show='headings', height=10)
        
        # SCROLLBAR
        vsb = ctk.CTkScrollbar(tv_frame, orientation="vertical", command=self.tr_check.yview)
        self.tr_check.configure(yscrollcommand=vsb.set)
        
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_check.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tr_check.bind("<<TreeviewSelect>>", self.show_metric_details)
        
        # Right: Verdict & Details
        self.f_verdict = ctk.CTkFrame(self.tab_carlson, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        self.f_verdict.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)

        self.lbl_score = ctk.CTkLabel(self.f_verdict, text="SCORE: --/10", font=FONT_SCORE)
        self.lbl_score.pack(pady=20)
        
        self.txt_verdict = ctk.CTkTextbox(self.f_verdict, height=150, fg_color=C_BG_INPUT, font=FONT_BODY)
        self.txt_verdict.pack(fill=tk.X, padx=15, pady=5)
        
        ctk.CTkLabel(self.f_verdict, text="METRIC DEEP DIVE", font=FONT_H2, text_color=C_ACCENT).pack(anchor='w', pady=(20, 5), padx=15)
        self.txt_deep = ctk.CTkTextbox(self.f_verdict, fg_color=C_BG_INPUT, font=FONT_BODY)
        self.txt_deep.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    def fill_quality_checklist(self, info, fin_df, bal_df, cf_df):
        self.tr_check.delete(*self.tr_check.get_children())
        for c in ("Metric", "Target", "Actual", "Result", "Context"): 
            self.tr_check.heading(c, text=c)
        
        self.tr_check.column("Metric", width=150); self.tr_check.column("Target", width=80)
        self.tr_check.column("Actual", width=80); self.tr_check.column("Result", width=60)
        self.tr_check.column("Context", width=200)
        
        score = 0
        self.metric_explanations = {}
        self.metric_payloads = {}
        symbol = (self.current_symbol or "").upper()

        def add_m(name, target_display, actual_val, passed, short_ctx, logic, calc, source):
            res = "PASS" if passed else "FAIL"
            item_id = self.tr_check.insert("", tk.END, values=(name, target_display, actual_val, res, short_ctx))
            if passed: self.tr_check.item(item_id, tags=("pass",))
            else: self.tr_check.item(item_id, tags=("fail",))
            payload = {
                "name": str(name),
                "target": str(target_display),
                "actual": str(actual_val),
                "result": res,
                "context": str(short_ctx),
                "logic": str(logic),
                "calc": str(calc),
                "source": str(source),
            }
            self.metric_payloads[item_id] = payload
            explanation = self._compose_metric_explanation(payload)
            self.metric_explanations[item_id] = explanation
            return 1 if passed else 0

        def finite_num(v):
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except Exception:
                pass
            return None

        def clean_series(s: Optional[pd.Series]):
            if s is None:
                return None
            s = pd.to_numeric(s, errors='coerce').dropna()
            if s.empty:
                return None
            try:
                s.index = pd.to_datetime(s.index).tz_localize(None)
            except Exception:
                pass
            return s.sort_index()

        # -- Metric Calculations (CompounderOS Quality Framework + Enhancements) --
        rev_q = clean_series(self.get_series(fin_df, ["TotalRevenue", "Revenue"]))
        gp_q = clean_series(self.get_series(fin_df, ["GrossProfit"]))
        op_inc_q = clean_series(self.get_series(fin_df, ["OperatingIncome", "OperatingIncomeLoss", "EBIT"]))
        ni_q = clean_series(self.get_series(fin_df, ["NetIncome"]))
        capex_q = clean_series(self.get_series(cf_df, ["CapitalExpenditure", "CapEx", "PaymentsToAcquirePropertyPlantAndEquipment"]))
        fcf_q = clean_series(self.get_series(cf_df, ["FreeCashFlow", "Free Cash Flow"]))
        ocf_q = clean_series(self.get_series(cf_df, ["OperatingCashFlow", "NetCashProvidedByUsedInOperatingActivities"]))
        shares_q = clean_series(self.get_series(bal_df, ["CommonStockSharesOutstanding", "ShareIssued"]))

        rev_a = clean_series(self.get_series(self.df_fin_a, ["TotalRevenue", "Revenue"]))
        ni_a = clean_series(self.get_series(self.df_fin_a, ["NetIncome"]))
        capex_a = clean_series(self.get_series(self.df_cf_a, ["CapitalExpenditure", "CapEx", "PaymentsToAcquirePropertyPlantAndEquipment"]))
        fcf_a = clean_series(self.get_series(self.df_cf_a, ["FreeCashFlow", "Free Cash Flow"]))
        ocf_a = clean_series(self.get_series(self.df_cf_a, ["OperatingCashFlow", "NetCashProvidedByUsedInOperatingActivities"]))
        shares_a = clean_series(self.get_series(self.df_bal_a, ["CommonStockSharesOutstanding", "ShareIssued"]))

        def yoy_growth(s: Optional[pd.Series]):
            if s is None or len(s) < 5:
                return None
            base = float(s.iloc[-5])
            if base == 0:
                return None
            return self.safe_div(float(s.iloc[-1]) - base, abs(base), None)

        def ttm_ratio(num: Optional[pd.Series], den: Optional[pd.Series]):
            if num is None or den is None:
                return None
            df = pd.DataFrame({"n": num, "d": den}).dropna()
            if len(df) < 4:
                return None
            return self.safe_div(float(df["n"].tail(4).sum()), float(df["d"].tail(4).sum()), None)

        # Core 1: Monopoly
        gm = finite_num(info.get('grossMargins'))
        gm_source = "Yahoo Info (TTM)"
        gm_series = None
        if gp_q is not None and rev_q is not None:
            gm_df = pd.DataFrame({"gp": gp_q, "rev": rev_q}).dropna()
            gm_df = gm_df[gm_df["rev"] != 0]
            if not gm_df.empty:
                gm_series = (gm_df["gp"] / gm_df["rev"]).sort_index()
            if gm is None and len(gm_df) >= 4:
                gm = self.safe_div(float(gm_df["gp"].tail(4).sum()), float(gm_df["rev"].tail(4).sum()), None)
                gm_source = "Quarterly Financials (TTM)"
        gm = gm if gm is not None else 0.0
        gm_vol = None
        if gm_series is not None and len(gm_series.dropna()) >= 8:
            gm_vol = float(gm_series.tail(8).std())
        monopoly_pass = gm >= 0.40 and (gm_vol is None or gm_vol <= 0.07)
        monopoly_actual = f"{gm:.1%}" if gm_vol is None else f"{gm:.1%} | vol {gm_vol*100:.1f}pp"
        monopoly_ctx = "Durable moat signals" if monopoly_pass else "Moat not clearly demonstrated"
        score += add_m(
            "Monopoly",
            "GM>40% + stable",
            monopoly_actual,
            monopoly_pass,
            monopoly_ctx,
            "High and stable gross margins can signal competitive moat and limited substitution risk.",
            "TTM Gross Margin and 8-quarter margin stability (std dev).",
            gm_source
        )

        # Core 2: Pricing Power
        rev_yoy = finite_num(info.get('revenueGrowth'))
        rev_yoy_source = "Yahoo Info (MRQ)"
        if rev_yoy is None:
            rev_yoy = yoy_growth(rev_q)
            rev_yoy_source = "Quarterly Financials (YoY)"
        gm_delta_pp = None
        if gm_series is not None and len(gm_series.dropna()) >= 5:
            gm_delta_pp = (float(gm_series.iloc[-1]) - float(gm_series.iloc[-5])) * 100.0
        pricing_pass = (rev_yoy is not None and rev_yoy > 0.05) and (gm_delta_pp is None or gm_delta_pp >= -1.0)
        if rev_yoy is None:
            pricing_actual = "N/A"
        else:
            pricing_actual = f"Rev {rev_yoy:+.1%}" + (f" | GM Δ {gm_delta_pp:+.1f}pp" if gm_delta_pp is not None else "")
        pricing_ctx = "Demand and margin resilience" if pricing_pass else "Potential pricing pressure"
        score += add_m(
            "Pricing Power",
            "Rev>5% + GM stable",
            pricing_actual,
            pricing_pass,
            pricing_ctx,
            "A company with pricing power can grow sales without giving up margin quality.",
            "Revenue YoY growth and gross margin change vs prior year quarter.",
            f"{rev_yoy_source} + {gm_source}"
        )

        # Core 3: Operating Leverage
        op_lev_pass = False
        op_lev_txt = "N/A"
        op_ctx = "Margins contracting"
        if op_inc_q is not None and rev_q is not None:
            qdf = pd.DataFrame({"op": op_inc_q, "rev": rev_q}).dropna()
            if len(qdf) > 4 and qdf['op'].iloc[-5] != 0 and qdf['rev'].iloc[-5] != 0:
                op_growth = self.safe_div(float(qdf['op'].iloc[-1]) - float(qdf['op'].iloc[-5]), abs(float(qdf['op'].iloc[-5])), 0.0)
                rev_growth = self.safe_div(float(qdf['rev'].iloc[-1]) - float(qdf['rev'].iloc[-5]), abs(float(qdf['rev'].iloc[-5])), 0.0)
                op_lev_pass = op_growth > rev_growth
                op_lev_txt = f"Op {op_growth:+.1%} vs Rev {rev_growth:+.1%}"
                op_ctx = "Profit growing faster than sales" if op_lev_pass else "Cost structure not scaling"
        score += add_m(
            "Operating Leverage",
            "Op > Rev",
            op_lev_txt,
            op_lev_pass,
            op_ctx,
            "Compounding businesses should convert growth into faster operating profit growth.",
            "Operating income YoY growth vs revenue YoY growth.",
            "Quarterly Financials"
        )

        # Core 4: Organic Growth
        org_growth = rev_yoy if rev_yoy is not None else 0.0
        org_pass = org_growth > 0.07
        org_ctx = "Healthy expansion" if org_pass else "Subscale growth"
        score += add_m(
            "Organic Growth",
            "> 7%",
            f"{org_growth:.1%}" if rev_yoy is not None else "N/A",
            org_pass,
            org_ctx,
            "Sustained top-line growth is required to fuel long-duration compounding.",
            "Revenue growth (YoY).",
            rev_yoy_source
        )

        # Core 5: Capital Light
        cap_pass = False
        cap_txt = "N/A"
        ctx_cap = "Capital heavy"
        cap_source = "Quarterly Data (TTM)"
        intensity = None
        if capex_a is not None and rev_a is not None and not capex_a.empty and not rev_a.empty:
            intensity = self.safe_div(abs(float(capex_a.iloc[-1])), float(rev_a.iloc[-1]), None)
            cap_source = "Annual Data (Latest FY)"
        elif capex_q is not None and rev_q is not None:
            intensity = ttm_ratio(abs(capex_q), rev_q)
        if intensity is not None and np.isfinite(intensity):
            cap_pass = intensity < 0.10
            cap_txt = f"{intensity:.1%}"
            ctx_cap = "Asset light model" if cap_pass else "High reinvestment burden"
        score += add_m(
            "Capital Light",
            "< 10% Rev",
            cap_txt,
            cap_pass,
            ctx_cap,
            "Businesses requiring less reinvestment can convert more revenue into owner cash flow.",
            "Capex / Revenue (annual latest or TTM fallback).",
            cap_source
        )

        # Core 6: Smart Capital Allocation
        if fcf_a is None and ocf_a is not None and capex_a is not None:
            fcf_a = ocf_a + capex_a
        if fcf_q is None and ocf_q is not None and capex_q is not None:
            fcf_q = ocf_q + capex_q
        conv = None
        conv_source = "Quarterly Data (TTM)"
        if fcf_a is not None and ni_a is not None and not fcf_a.empty and not ni_a.empty:
            conv = self.safe_div(float(fcf_a.iloc[-1]), float(ni_a.iloc[-1]), None)
            conv_source = "Annual Data (Latest FY)"
        elif fcf_q is not None and ni_q is not None:
            conv = ttm_ratio(fcf_q, ni_q)

        share_change = None
        bb_source = "Balance Sheet (Quarterly)"
        curr = prev = None
        if shares_a is not None and len(shares_a) >= 2:
            curr = float(shares_a.iloc[-1])
            prev = float(shares_a.iloc[-2])
            bb_source = "Balance Sheet (Annual)"
        elif shares_q is not None and len(shares_q) >= 5:
            curr = float(shares_q.iloc[-1])
            prev = float(shares_q.iloc[-5])
        if curr is not None and prev is not None and prev > 0:
            share_change = self.safe_div(curr - prev, prev, None)

        div_yield = finite_num(info.get("dividendYield"))
        conv_ok = (conv is not None and np.isfinite(conv) and conv >= 0.80)
        share_ok = (share_change is not None and share_change <= 0.005)  # flat to down
        payout_ok = ((share_change is not None and share_change < -0.01) or (div_yield is not None and div_yield >= 0.005))
        smart_alloc_pass = conv_ok and (share_ok or payout_ok)
        conv_txt = f"{conv:.1%}" if conv is not None and np.isfinite(conv) else "N/A"
        sh_txt = f"{share_change:+.1%}" if share_change is not None and np.isfinite(share_change) else "N/A"
        dv_txt = f"{div_yield:.1%}" if div_yield is not None and np.isfinite(div_yield) else "N/A"
        smart_actual = f"FCF/NI {conv_txt} | Shares {sh_txt} | Div {dv_txt}"
        smart_ctx = "Cash returns look disciplined" if smart_alloc_pass else "Capital allocation needs improvement"
        score += add_m(
            "Smart Capital Allocation",
            "FCF + shareholder friendly",
            smart_actual,
            smart_alloc_pass,
            smart_ctx,
            "Quality capital allocation means converting earnings to cash and returning/compounding it intelligently.",
            "FCF/Net Income, share count trend, and dividend support.",
            f"{conv_source} + {bb_source} + Yahoo Info"
        )

        # Enhancements
        roe = finite_num(info.get('returnOnEquity'))
        roe_source = "Yahoo Info (TTM)"
        if roe is None:
            eq_a = clean_series(self.get_series(self.df_bal_a, ["TotalStockholderEquity", "StockholdersEquity"]))
            if ni_a is not None and eq_a is not None and not ni_a.empty and not eq_a.empty:
                roe = self.safe_div(float(ni_a.iloc[-1]), float(eq_a.iloc[-1]), 0.0)
                roe_source = "Annual Financials"
        roe = roe if roe is not None else 0.0
        res_roe = roe > 0.15
        ctx_roe = "Elite efficiency" if res_roe else "Weak capital efficiency"
        score += add_m(
            "Efficiency (ROE)",
            "> 15%",
            f"{roe:.1%}",
            res_roe,
            ctx_roe,
            "ROE helps validate if management converts equity capital into profit efficiently.",
            "Net Income / Shareholder Equity.",
            roe_source
        )

        cr = finite_num(info.get('currentRatio'))
        cr_source = "Yahoo Info (MRQ)"
        if cr is None:
            cur_assets = clean_series(self.get_series(bal_df, ["CurrentAssets"]))
            cur_liab = clean_series(self.get_series(bal_df, ["CurrentLiabilities"]))
            if cur_assets is not None and cur_liab is not None and not cur_assets.empty and not cur_liab.empty:
                cr = self.safe_div(float(cur_assets.iloc[-1]), float(cur_liab.iloc[-1]), 0.0)
                cr_source = "Quarterly Balance Sheet"
        cr = cr if cr is not None else 0.0
        res_cr = cr > 1.0
        ctx_cr = "Liquid / safe" if res_cr else "Liquidity risk"
        score += add_m(
            "Financial Health",
            "> 1.0",
            f"{cr:.2f}",
            res_cr,
            ctx_cr,
            "Short-term balance sheet resilience reduces forced capital raises during downturns.",
            "Current Assets / Current Liabilities.",
            cr_source
        )

        peg = finite_num(info.get('pegRatio'))
        pe = finite_num(info.get('forwardPE')) or finite_num(info.get('trailingPE'))
        g = finite_num(info.get('earningsGrowth'))
        if g is None or g <= 0:
            g = finite_num(info.get('revenueGrowth'))
        val_source = "Yahoo Info"
        if peg is None or peg <= 0:
            if pe is not None and g is not None and g > 0:
                peg = self.safe_div(pe, g * 100.0, None)
                val_source = "Derived (PE / growth%)"
        if peg is not None and np.isfinite(peg) and peg > 0:
            val_pass = peg < 2.0
            val_txt = f"PEG {peg:.2f}"
            val_ctx = "Reasonable vs growth" if val_pass else "Growth-adjusted premium"
            target_txt = "< 2.0 PEG"
        elif pe is not None and np.isfinite(pe):
            val_pass = pe < 30.0
            val_txt = f"P/E {pe:.1f}"
            val_ctx = "Fallback valuation check" if val_pass else "Expensive on earnings multiple"
            target_txt = "< 2 PEG or < 30 P/E"
            val_source = "Yahoo Info (P/E fallback)"
        else:
            val_pass = False
            val_txt = "N/A"
            val_ctx = "Valuation data unavailable"
            target_txt = "< 2 PEG or < 30 P/E"
        score += add_m(
            "Valuation Discipline",
            target_txt,
            val_txt,
            val_pass,
            val_ctx,
            "Quality still needs a sensible entry multiple to preserve long-term return potential.",
            "PEG preferred; P/E fallback when growth inputs are missing.",
            val_source
        )

        curr_p = finite_num(info.get('currentPrice'))
        target = finite_num(info.get('targetMeanPrice'))
        is_safe = False
        margin_txt = "N/A"
        ctx_safe = "No analyst target data"
        if target is not None and curr_p is not None and target > 0 and curr_p > 0:
            upside = self.safe_div(target - curr_p, curr_p, None)
            if upside is not None and np.isfinite(upside):
                is_safe = upside > 0.15
                margin_txt = f"{upside:.1%}"
                ctx_safe = "Undervalued vs consensus" if is_safe else "Limited margin vs consensus"
        score += add_m(
            "Margin of Safety",
            "> 15%",
            margin_txt,
            is_safe,
            ctx_safe,
            "A valuation cushion reduces downside if execution weakens or macro conditions deteriorate.",
            "(Target Mean Price - Current Price) / Current Price.",
            "Analyst Estimates"
        )

        self.tr_check.tag_configure("pass", foreground=C_GREEN)
        self.tr_check.tag_configure("fail", foreground=C_RED)
        self.lbl_score.configure(text=f"SCORE: {score}/10", text_color=C_GREEN if score >= 7 else C_YELLOW)
        
        verdict = "VERDICT: HOLD / WATCH\n\n"
        if score >= 8: verdict = "VERDICT: ELITE COMPOUNDER (STRONG BUY)\n\n"
        elif score <= 4: verdict = "VERDICT: LOW QUALITY / SPECULATIVE\n\n"
        base_note = "Core framework follows CompounderOS Quality Framework: Monopoly, Pricing Power, Operating Leverage, Organic Growth, Capital Light, Smart Capital Allocation.\n\n"
        self.txt_verdict.delete(1.0, tk.END); self.txt_verdict.insert(tk.END, verdict + base_note + "Select any metric in the table above to see the Deep Dive explanation here.")
        self._enhance_quality_deep_dive(symbol)

    def show_metric_details(self, event):
        sel = self.tr_check.selection()
        if not sel: return
        item_id = sel[0]
        text = self.metric_explanations.get(item_id, "No details available.")
        self.txt_deep.configure(state="normal")
        self.txt_deep.delete(1.0, tk.END)
        self.txt_deep.insert(tk.END, text)
        self.txt_deep.configure(state="disabled")

    def _compose_metric_explanation(self, payload: Dict[str, str], ai_extra: Optional[str] = None):
        text = (
            f"METRIC: {payload.get('name', 'N/A')}\n\n"
            f"THE LOGIC:\n{payload.get('logic', 'N/A')}\n\n"
            f"CALCULATION:\n{payload.get('calc', 'N/A')}\n\n"
            f"SOURCE:\n{payload.get('source', 'N/A')}\n\n"
            f"CONTEXT:\n"
            f"Your Value: {payload.get('actual', 'N/A')}\n"
            f"Target: {payload.get('target', 'N/A')} -> {payload.get('result', 'N/A')}\n"
            f"{payload.get('context', 'N/A')}"
        )
        if ai_extra:
            text += f"\n\nGEMINI INSIGHT:\n{ai_extra}"
        return text

    def _quality_ai_cache_key(self, symbol: str):
        parts = []
        for payload in self.metric_payloads.values():
            parts.append(f"{payload.get('name')}|{payload.get('actual')}|{payload.get('result')}")
        parts.sort()
        return f"{symbol}|{'#'.join(parts)}"

    def _enhance_quality_deep_dive(self, symbol: str):
        if not symbol or not self.metric_payloads:
            return
        if not self.gemini_api_key:
            return
        key = self._quality_ai_cache_key(symbol)
        self.quality_ai_key = key
        cached = self.quality_ai_cache.get(key)
        if cached:
            self._apply_quality_ai_insights(symbol, key, cached)
            return
        if self.quality_ai_inflight:
            return
        self.quality_ai_inflight = True
        metrics = [dict(v) for v in self.metric_payloads.values()]
        self.submit_task(self._quality_ai_worker, symbol, key, metrics, context="quality_ai_worker")

    def _quality_ai_worker(self, symbol: str, key: str, metrics: List[Dict[str, str]]):
        insights = {}
        try:
            lines = []
            for m in metrics:
                lines.append(
                    f"- {m.get('name')}: target={m.get('target')}, actual={m.get('actual')}, result={m.get('result')}, context={m.get('context')}"
                )
            prompt = (
                f"You are a senior equity analyst.\n"
                f"Company: {symbol}\n"
                "For each metric below, write one high-signal deep-dive insight (2-4 sentences) that explains:\n"
                "1) why this metric matters,\n"
                "2) what this specific value implies,\n"
                "3) one concrete thing to monitor in the next quarter.\n"
                "Be precise, avoid generic language, and do not give investment advice.\n\n"
                "Return EXACTLY one line per metric in this format:\n"
                "METRIC_NAME || INSIGHT\n\n"
                "Metrics:\n"
                + "\n".join(lines)
            )
            raw = self.gemini_generate(prompt) or ""
            for line in raw.splitlines():
                if "||" not in line:
                    continue
                left, right = line.split("||", 1)
                name = left.strip().lstrip("-").strip()
                insight = right.strip()
                if name and insight:
                    insights[name] = insight
        except Exception:
            insights = {}
        self.quality_ai_inflight = False
        if insights:
            self.quality_ai_cache[key] = insights
            self.root.after(0, lambda s=symbol, k=key, d=insights: self._apply_quality_ai_insights(s, k, d))

    def _apply_quality_ai_insights(self, symbol: str, key: str, insights: Dict[str, str]):
        if symbol != (self.current_symbol or "").upper():
            return
        if key != self.quality_ai_key:
            return
        for item_id, payload in self.metric_payloads.items():
            if not self.tr_check.exists(item_id):
                continue
            ai_txt = insights.get(payload.get("name", ""), "")
            self.metric_explanations[item_id] = self._compose_metric_explanation(payload, ai_txt if ai_txt else None)
        if self.tr_check.selection():
            self.show_metric_details(None)

    # --- INSIGHTS ---
    def build_insights(self):
        pad = ctk.CTkFrame(self.tab_insights, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        pad.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        row1 = ctk.CTkFrame(pad, fg_color="transparent")
        row1.pack(fill=tk.X, pady=(15, 30), padx=20)
        
        self.lbl_price = ctk.CTkLabel(row1, text="$0.00", font=FONT_DISPLAY)
        self.lbl_price.pack(side=tk.LEFT)
        
        self.lbl_chg = ctk.CTkLabel(row1, text="0.00%", font=FONT_H1, text_color=C_ACCENT)
        self.lbl_chg.pack(side=tk.LEFT, padx=20, pady=(15,0))

        self.lbl_after = ctk.CTkLabel(row1, text="After Hours: N/A", font=FONT_SMALL, text_color=C_TEXT_SUB)
        self.lbl_after.pack(side=tk.LEFT, padx=10, pady=(15,0))
        
        self.lbl_targets = ctk.CTkLabel(pad, text="Waiting...", font=FONT_BODY, anchor="w")
        self.lbl_targets.pack(fill=tk.X, padx=20)
        self.lbl_top_news = ctk.CTkLabel(pad, text="Top news: N/A", font=FONT_SMALL, text_color=C_TEXT_SUB, anchor="w")
        self.lbl_top_news.pack(fill=tk.X, padx=20, pady=(5,0))

        # Key Metrics Panel
        metrics = ctk.CTkFrame(pad, fg_color=C_BG_INPUT, corner_radius=12, border_width=1, border_color=C_BORDER)
        metrics.pack(fill=tk.X, padx=20, pady=(10, 15))
        self.metric_labels = {}
        keys = ["P/E", "P/S", "FCF Yield", "Gross %", "Op %", "Net %", "Debt/Equity", "Div Yield"]
        for k in keys:
            box = ctk.CTkFrame(metrics, fg_color="transparent")
            box.pack(side=tk.LEFT, padx=10, pady=8)
            ctk.CTkLabel(box, text=k, font=FONT_TINY, text_color=C_TEXT_SUB).pack()
            val = ctk.CTkLabel(box, text="--", font=FONT_H2)
            val.pack()
            self.metric_labels[k] = val

        # Company Profile + Analyst Estimates
        info_row = ctk.CTkFrame(pad, fg_color="transparent")
        info_row.pack(fill=tk.BOTH, expand=False, padx=20, pady=(10, 15))
        prof = ctk.CTkFrame(info_row, fg_color=C_BG_INPUT, corner_radius=12, border_width=1, border_color=C_BORDER)
        prof.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        ctk.CTkLabel(prof, text="COMPANY PROFILE", font=FONT_H2, text_color=C_ACCENT).pack(anchor="w", padx=10, pady=(10,5))
        self.txt_profile = ctk.CTkTextbox(prof, height=120, fg_color=C_BG_INPUT, font=FONT_BODY)
        self.txt_profile.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        est = ctk.CTkFrame(info_row, fg_color=C_BG_INPUT, corner_radius=12, border_width=1, border_color=C_BORDER)
        est.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctk.CTkLabel(est, text="ANALYST ESTIMATES", font=FONT_H2, text_color=C_ACCENT).pack(anchor="w", padx=10, pady=(10,5))
        self.txt_estimates = ctk.CTkTextbox(est, height=120, fg_color=C_BG_INPUT, font=FONT_BODY)
        self.txt_estimates.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        ctk.CTkLabel(pad, text="LATEST NEWS", font=FONT_H1, text_color=C_ACCENT).pack(anchor="w", padx=20, pady=(30, 10))
        
        ctk.CTkButton(pad, text="Open Google News", command=self.open_google_news, width=150, height=28, font=FONT_SMALL, **BTN_SECONDARY).pack(anchor="w", padx=20, pady=(0,10))
        
        self.txt_news = ctk.CTkTextbox(pad, font=FONT_BODY, fg_color=C_BG_INPUT)
        self.txt_news.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))

    def open_google_news(self):
        if self.current_symbol: webbrowser.open(f"https://www.google.com/search?q={self.current_symbol}+stock+news&tbm=nws")

    def _news_title(self, item):
        return item.get('title') or item.get('headline') or item.get('content', {}).get('title', '')

    def _news_summary(self, item):
        return item.get('summary') or item.get('content', {}).get('summary', '')

    def _news_provider(self, item):
        provider = item.get('publisher') or item.get('provider') or item.get('source')
        if provider:
            return str(provider)
        content = item.get('content', {})
        if isinstance(content, dict):
            p = content.get('provider', {})
            if isinstance(p, dict):
                return p.get('displayName') or p.get('name') or "Unknown"
        return "Unknown"

    def _news_timestamp(self, item):
        ts = item.get('providerPublishTime', 0)
        try:
            ts = int(ts)
            if ts > 0:
                return ts
        except Exception:
            pass
        content = item.get('content', {})
        if isinstance(content, dict):
            raw = content.get('pubDate') or content.get('publishedAt') or content.get('displayTime')
            if raw:
                try:
                    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    pass
        return 0

    def _collect_week_news(self, ticker, symbol: str, short_name: str):
        news_data = ticker.news if ticker else []
        if not news_data:
            try:
                news_data = yf.Search(symbol).news
            except Exception:
                news_data = []
        now_ts = time.time()
        week_ago = now_ts - (7 * 86400)
        filter_keywords = {symbol.upper(), short_name.upper()} if short_name else {symbol.upper()}
        items = []
        seen_titles = set()
        for item in news_data or []:
            title = self._news_title(item)
            summary = self._news_summary(item)
            if not title:
                continue
            blob = f"{title} {summary}".upper()
            if filter_keywords and not any(k in blob for k in filter_keywords if k):
                continue
            ts = self._news_timestamp(item)
            if ts and ts < week_ago:
                continue
            key = title.strip().upper()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            items.append({
                "ts": ts,
                "title": title.strip(),
                "provider": self._news_provider(item),
            })
        items.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return items

    def fill_insights(self, i, t):
        p = i.get('currentPrice', 0)
        if p is None: p = 0
        prev = i.get('previousClose', p)
        if prev is None: prev = p
        # Safe calc
        c = self.safe_div(p - prev, prev) * 100
        
        self.lbl_price.configure(text=f"${p:,.2f}")
        self.lbl_chg.configure(text=f"{c:+.2f}%", text_color=C_GREEN if c>=0 else C_RED)

        # After-hours
        post_p = i.get('postMarketPrice')
        post_chg = i.get('postMarketChangePercent')
        if post_p is not None and post_chg is not None:
            self.lbl_after.configure(text=f"After Hours: ${post_p:,.2f} ({post_chg:+.2f}%)")
        else:
            self.lbl_after.configure(text="After Hours: N/A")
        
        recom = i.get('recommendationKey', 'N/A').replace('_', ' ').upper()
        mcap = i.get('marketCap', 0)
        if mcap is None: mcap = 0
        self.lbl_targets.configure(text=f"Market Cap: ${mcap/1e9:,.2f} B   |   Consensus: {recom}   |   Target: ${i.get('targetMeanPrice', 'N/A')}   |   P/E: {i.get('trailingPE', 'N/A')}")

        # Key Metrics Panel
        pe = i.get('trailingPE')
        ps = None
        total_rev = i.get('totalRevenue')
        if total_rev and mcap:
            ps = self.safe_div(mcap, total_rev)
        fcf = i.get('freeCashflow')
        fcf_yield = self.safe_div(fcf, mcap) * 100 if fcf and mcap else None
        gm = i.get('grossMargins')
        om = i.get('operatingMargins')
        nm = i.get('profitMargins')
        de = i.get('debtToEquity')
        if de is not None:
            de = de / 100.0
        div_yield = i.get('dividendYield')

        def set_metric(key, val, fmt="{:.2f}"):
            lbl = self.metric_labels.get(key)
            if not lbl: return
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                lbl.configure(text="--")
            else:
                lbl.configure(text=fmt.format(val))

        set_metric("P/E", pe)
        set_metric("P/S", ps)
        set_metric("FCF Yield", fcf_yield, "{:.2f}%")
        set_metric("Gross %", gm * 100 if gm is not None else None, "{:.1f}%")
        set_metric("Op %", om * 100 if om is not None else None, "{:.1f}%")
        set_metric("Net %", nm * 100 if nm is not None else None, "{:.1f}%")
        set_metric("Debt/Equity", de, "{:.2f}")
        set_metric("Div Yield", div_yield * 100 if div_yield is not None else None, "{:.2f}%")

        # Company Profile
        prof = []
        prof.append(i.get("longName", i.get("shortName", "")))
        prof.append(f"Sector: {i.get('sector', 'N/A')} | Industry: {i.get('industry', 'N/A')}")
        prof.append(f"Employees: {i.get('fullTimeEmployees', 'N/A')}")
        prof.append(f"Website: {i.get('website', 'N/A')}")
        prof.append("")
        prof.append(i.get("longBusinessSummary", ""))
        self.txt_profile.configure(state="normal")
        self.txt_profile.delete(1.0, tk.END)
        self.txt_profile.insert(tk.END, "\n".join([p for p in prof if p is not None]))
        self.txt_profile.configure(state="disabled")

        # Analyst Estimates
        est = []
        est.append(f"Target Low / Mean / High: {i.get('targetLowPrice','N/A')} / {i.get('targetMeanPrice','N/A')} / {i.get('targetHighPrice','N/A')}")
        est.append(f"Analyst Count: {i.get('numberOfAnalystOpinions','N/A')}")
        est.append(f"Recommendation: {recom}")
        est.append(f"Forward P/E: {i.get('forwardPE','N/A')}")
        est.append(f"Earnings Growth: {i.get('earningsGrowth','N/A')}")
        est.append(f"Revenue Growth: {i.get('revenueGrowth','N/A')}")
        self.txt_estimates.configure(state="normal")
        self.txt_estimates.delete(1.0, tk.END)
        self.txt_estimates.insert(tk.END, "\n".join(est))
        self.txt_estimates.configure(state="disabled")
        
        self.txt_news.configure(state="normal")
        self.txt_news.delete(1.0, tk.END)
        try:
            symbol = self.current_symbol.upper()
            short_name = i.get('shortName', '').split(" ")[0].upper()
            if not short_name:
                short_name = symbol
            news_items = self._collect_week_news(t, symbol, short_name)
            if news_items:
                self.txt_news.insert(tk.END, "News from last 7 days:\n\n")
                for item in news_items[:20]:
                    ts = item.get("ts", 0)
                    if ts > 0:
                        date_txt = datetime.fromtimestamp(ts).strftime("%a, %Y-%m-%d %H:%M")
                    else:
                        date_txt = "Date unavailable"
                    provider = item.get("provider", "Unknown")
                    title = item.get("title", "")
                    self.txt_news.insert(tk.END, f"•  {date_txt}  |  {provider}  |  {title}\n\n")

                top_item = None
                now_ts = time.time()
                for item in news_items:
                    ts = item.get("ts", 0)
                    if ts and (now_ts - ts) <= 86400:
                        top_item = item
                        break
                if top_item is None:
                    top_item = news_items[0]
                ts = top_item.get("ts", 0)
                if ts > 0:
                    d = datetime.fromtimestamp(ts).strftime("%a, %Y-%m-%d")
                    self.lbl_top_news.configure(text=f"Top news: {d} | {top_item.get('title', 'N/A')}")
                else:
                    self.lbl_top_news.configure(text=f"Top news: {top_item.get('title', 'N/A')}")
            else:
                self.txt_news.insert(tk.END, f"No direct news found for {short_name} in the last 7 days.")
                self.lbl_top_news.configure(text="Top news: N/A")
        except Exception:
            self.txt_news.insert(tk.END, "News unavailable.")
            self.lbl_top_news.configure(text="Top news: N/A")
        self.txt_news.configure(state="disabled")

    # --- TAB 2: CHARTS ---
    def build_charts(self):
        bar = ctk.CTkFrame(self.tab_charts, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        bar.pack(fill=tk.X, padx=20, pady=20)
        
        ctk.CTkLabel(bar, text="VISUALIZATION (QUARTERLY):", font=FONT_H2, text_color=C_ACCENT).pack(side=tk.LEFT, padx=15, pady=10)
        
        self.chart_var = tk.StringVar(value="Price")
        OPTIONS = [
            "Price",
            "Revenue & EBITDA",
            "EPS & Dividends",
            "Cash vs Debt",
            "Margins (Gross/Op/Net)",
            "Valuation (P/E)",
            "Valuation (FCF Yield)",
            "Valuation (P/S)",
            "Cash Flow Detail (FCF vs SBC)",
            "FCF Per Share"
        ]
        
        cb = self.make_combo(bar, variable=self.chart_var, values=OPTIONS, width=250, command=self.redraw_chart, **COMBO_KW)
        cb.pack(side=tk.LEFT, pady=10)

        self.price_range_var = tk.StringVar(value="1Y")
        price_cb = self.make_combo(bar, variable=self.price_range_var, values=["1D","1W","1M","3M","6M","1Y","5Y","10Y","YTD","ALL"], width=90, command=self.redraw_chart, **COMBO_KW)
        price_cb.pack(side=tk.LEFT, padx=8)
        self.lbl_price_change = ctk.CTkLabel(bar, text="", font=FONT_SMALL, text_color=C_TEXT_SUB)
        self.lbl_price_change.pack(side=tk.RIGHT, padx=10)

        self.freq_var = tk.StringVar(value="Quarterly")
        freq_cb = self.make_combo(bar, variable=self.freq_var, values=["Quarterly", "TTM", "Annual"], width=130, command=self.redraw_chart, **COMBO_KW)
        freq_cb.pack(side=tk.LEFT, padx=10)
        
        self.anim_enabled = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar, text="Animate", variable=self.anim_enabled, command=self.redraw_chart, **CHECK_KW).pack(side=tk.LEFT, padx=15)
        
        chart_body = ctk.CTkFrame(self.tab_charts, fg_color="transparent")
        chart_body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))
        chart_body.grid_rowconfigure(0, weight=1)
        chart_body.grid_rowconfigure(1, weight=0)
        chart_body.grid_columnconfigure(0, weight=1)

        # Frame for plot
        plot_frame = ctk.CTkFrame(chart_body, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0,10))

        self.fig, self.ax = plt.subplots(figsize=(10,6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Chart Insights
        insight_frame = ctk.CTkFrame(chart_body, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        insight_frame.grid(row=1, column=0, sticky="ew")
        top_row = ctk.CTkFrame(insight_frame, fg_color="transparent")
        top_row.pack(fill=tk.X, pady=(8,4))
        ctk.CTkLabel(top_row, text="CHART INSIGHTS (GEMINI)", font=FONT_H2, text_color=C_ACCENT).pack(side=tk.LEFT, padx=10)
        self.txt_chart_ai = ctk.CTkTextbox(insight_frame, height=120, fg_color=C_BG_INPUT, font=FONT_BODY)
        self.txt_chart_ai.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

    def fill_charts(self): self.redraw_chart()

    def normalize_stooq_symbol(self, symbol: str):
        s = symbol.strip().lower()
        if "." in s:
            return s
        return f"{s}.us"

    def fetch_stooq_history(self, symbol: str, req_id: int):
        try:
            stooq_symbol = self.normalize_stooq_symbol(symbol)
            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            text = self.http_get_text(url, headers=None, timeout=15)
            if not text:
                return
            df = pd.read_csv(io.StringIO(text))
            if df.empty or "Date" not in df.columns:
                return
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            self.store.upsert_price_history(symbol, df, "stooq")
            if req_id != self.request_id:
                return
            if not self.root.winfo_exists():
                return
            # If the user is on the Price chart, refresh with long history
            self.hist_data_long = df
            self.root.after(0, lambda: self.redraw_chart())
        except Exception as e:
            log_exception("fetch_stooq_history", e)

# --- HELPER: ROBUST DATA ALIGNMENT ---
    def align_data(self, primary_series, secondary_series=None, fill_value=0.0, freq: Optional[str] = None):
        if primary_series is None or primary_series.empty:
            return [], np.array([]), np.array([])
        
        freq = freq or self.freq_var.get()
        freq_code = self._freq_code(freq)

        def clean_series(s, col_name):
            s = s.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            df = s.to_frame(name=col_name)
            df['period'] = df.index.to_period(freq_code)
            return df.groupby('period').last()

        df_p = clean_series(primary_series, 'p')
        
        if secondary_series is not None and not secondary_series.empty:
            df_s = clean_series(secondary_series, 's')
            df_merged = pd.merge(df_p, df_s, on='period', how='left')
        else:
            df_merged = df_p.copy()
            df_merged['s'] = np.nan

        df_merged['s'] = df_merged['s'].fillna(fill_value)
        df_merged = df_merged.sort_index()
        
        periods = df_merged.index
        labels = self._format_period_labels(periods, freq)
        return labels, df_merged['p'].values, df_merged['s'].values

    def get_freq_df(self, kind: str):
        # kind: "fin", "bal", "cf"
        if kind == "fin":
            q = self.df_fin_q
            a = self.df_fin_a
        elif kind == "bal":
            q = self.df_bal_q
            a = self.df_bal_a
        else:
            q = self.df_cf_q
            a = self.df_cf_a
        freq = self.freq_var.get()
        if freq == "Annual" and a is not None and not a.empty:
            return a
        return q

    def to_ttm(self, s: pd.Series):
        if s is None or s.empty:
            return s
        s = s.sort_index()
        return s.rolling(4).sum()

    def _style_axes(self, title: Optional[str] = None, y_label: Optional[str] = None):
        self.ax.set_facecolor(C_BG_CARD)
        self.ax.grid(True, linestyle="--", alpha=0.18)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.tick_params(axis='both', colors=C_TEXT_SUB, labelsize=9)
        if title:
            self.ax.set_title(title, fontweight="bold", color=C_TEXT_MAIN, pad=12)
        if y_label:
            self.ax.set_ylabel(y_label, color=C_TEXT_SUB)

    def _set_period_ticks(self, x_idx, labels, max_ticks=8):
        if len(labels) == 0:
            return
        step = max(1, int(len(labels) / max_ticks))
        ticks = x_idx[::step]
        tick_labels = [labels[i] for i in range(0, len(labels), step)]
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    def _set_date_ticks(self, x_index):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def _apply_layout(self):
        try:
            # Extra top/bottom room prevents title and x-label clipping on high-DPI displays.
            self.fig.subplots_adjust(bottom=0.24, left=0.08, right=0.97, top=0.84)
        except Exception:
            pass

    def _add_vertical_guides(self, x_positions):
        if len(x_positions) > 40:
            return
        for x in x_positions:
            self.ax.axvline(x, color=C_BORDER, lw=0.5, alpha=0.12, zorder=0, linestyle="--")

    def _freq_code(self, freq: str):
        return "Y" if freq == "Annual" else "Q"

    def _format_period_labels(self, periods: pd.PeriodIndex, freq: str):
        if freq == "Annual":
            return [str(p.year) for p in periods]
        return [f"{p.year}-Q{p.quarter}" for p in periods]

    def _to_period_index(self, idx, freq: str):
        freq_code = self._freq_code(freq)
        return pd.PeriodIndex(pd.to_datetime(idx).tz_localize(None), freq=freq_code)

    def _trim_ttm(self, s: Optional[pd.Series], n: int = 4):
        if s is None or s.empty:
            return s
        s = s.dropna()
        return s.tail(n)

    def _last_n_periods(self, s: Optional[pd.Series], n: int = 4, freq: str = "Quarterly"):
        if s is None or s.empty:
            return s
        freq_code = self._freq_code(freq)
        pidx = pd.PeriodIndex(pd.to_datetime(s.index).tz_localize(None), freq=freq_code)
        last_p = pidx.max()
        if last_p is pd.NaT:
            return s
        periods = pd.period_range(end=last_p, periods=n, freq=freq_code)
        s2 = pd.Series(s.values, index=pidx).groupby(level=0).last()
        s2 = s2.reindex(periods)
        s2.index = s2.index.to_timestamp()
        return s2

    def _ttm_series(self, s_q: Optional[pd.Series]):
        s_q = self._group_series(s_q, "Quarterly", agg="last")
        if s_q is None or s_q.empty:
            return s_q
        pidx = pd.PeriodIndex(pd.to_datetime(s_q.index).tz_localize(None), freq="Q")
        full = pd.period_range(start=pidx.min(), end=pidx.max(), freq="Q")
        s_full = pd.Series(s_q.values, index=pidx).reindex(full)
        ttm = s_full.rolling(4, min_periods=4).sum().dropna()
        if ttm.empty:
            return ttm
        ttm = ttm.tail(4)
        ttm.index = ttm.index.to_timestamp()
        return ttm

    def _group_series(self, s: Optional[pd.Series], freq: str, agg: str = "last"):
        if s is None or s.empty:
            return s
        idx = pd.to_datetime(s.index).tz_localize(None)
        pidx = pd.PeriodIndex(idx, freq=self._freq_code(freq))
        df = pd.DataFrame({"v": s.values}, index=pidx)
        if agg == "sum":
            out = df.groupby(level=0).sum()["v"]
        else:
            out = df.groupby(level=0).last()["v"]
        # convert to timestamp for consistent downstream handling
        out.index = out.index.to_timestamp()
        return out

    def _series_for_flow(self, s_q: Optional[pd.Series], s_a: Optional[pd.Series], freq: str):
        if freq == "Annual":
            if s_a is not None and not s_a.empty:
                return self._group_series(s_a, "Annual", agg="sum")
            return self._group_series(s_q, "Annual", agg="sum")
        if freq == "TTM":
            return self._ttm_series(s_q)
        return self._group_series(s_q, "Quarterly", agg="last")

    def _series_for_balance(self, s_q: Optional[pd.Series], s_a: Optional[pd.Series], freq: str):
        if freq == "Annual":
            if s_a is not None and not s_a.empty:
                return self._group_series(s_a, "Annual", agg="last")
            return self._group_series(s_q, "Annual", agg="last")
        s = self._group_series(s_q, "Quarterly", agg="last")
        if freq == "TTM":
            return self._last_n_periods(s, n=4, freq="Quarterly")
        return s

    def _dividend_series(self, freq: str):
        if self.df_dividends is None or self.df_dividends.empty:
            return None
        base_freq = "Quarterly" if freq == "TTM" else freq
        idx = pd.to_datetime(self.df_dividends.index).tz_localize(None)
        pidx = pd.PeriodIndex(idx, freq=self._freq_code(base_freq))
        s = pd.Series(self.df_dividends.values, index=pidx).groupby(level=0).sum()
        if freq == "TTM":
            s = s.rolling(4, min_periods=4).sum().dropna()
            if s.empty:
                return s
            s = s.tail(4)
        s.index = s.index.to_timestamp()
        return s

    def _price_period_to_yf(self, label: str):
        mapping = {
            "1D": "1d",
            "1W": "5d",
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
            "5Y": "5y",
            "10Y": "10y",
            "YTD": "ytd",
            "ALL": "max",
        }
        return mapping.get(label, "1y")

    def _safe_ylim(self, y_min, y_max, pad=0.08, zero_floor=False):
        if not np.isfinite(y_min):
            y_min = 0.0
        if not np.isfinite(y_max):
            y_max = 1.0
        if zero_floor:
            y_min = min(0.0, y_min)
        if y_max == y_min:
            y_max = y_min + 1.0
        span = y_max - y_min
        self.ax.set_ylim(y_min - span * pad, y_max + span * pad)

    def _draw_no_data(self, msg="No data available"):
        self._style_axes()
        self.ax.text(0.5, 0.5, msg, color=C_TEXT_SUB, ha='center', va='center', transform=self.ax.transAxes)

    # --- GRAPHING ---
    def redraw_chart(self, _=None):
        # Debounce rapid redraws
        if self._redraw_job:
            try: self.root.after_cancel(self._redraw_job)
            except: pass
        self._redraw_job = self.root.after(REDRAW_DEBOUNCE_MS, self._redraw_chart_now)

    def _redraw_chart_now(self):
        self._redraw_job = None
        if hasattr(self, "txt_chart_ai"):
            self._update_chart_ai("Refreshing chart insights...")
        # Stop existing animation safely to prevent leaks
        if self.anim and self.anim.event_source:
            try: self.anim.event_source.stop()
            except: pass
        self.anim = None
        if self._ax2 is not None:
            try:
                self._ax2.remove()
            except Exception:
                pass
            self._ax2 = None
        # Defensive cleanup for any lingering twin axes
        try:
            for extra_ax in list(self.fig.axes):
                if extra_ax is not self.ax:
                    extra_ax.remove()
        except Exception:
            pass
        
        self.ax.clear()
        mode = self.chart_var.get()
        freq = self.freq_var.get()
        if hasattr(self, "lbl_price_change"):
            self.lbl_price_change.configure(text="")
        
        # --- 1. PRICE CHART (Line) ---
        if mode == "Price" and self.hist_data is not None:
            self.lbl_price_change.configure(text="")
            desired = self._price_period_to_yf(self.price_range_var.get())
            if self.hist_period != desired:
                self.ensure_history_period(desired)
                self._draw_no_data(f"Loading {self.price_range_var.get()} data...")
                self.canvas.draw_idle()
                return
            # Prefer yfinance period data for consistency; stooq is only fallback for ALL.
            base = self.hist_data
            if (base is None or base.empty) and desired == "max":
                hist = getattr(self, "hist_data_long", None)
                if hist is not None and not hist.empty:
                    base = hist
            if base is None or base.empty:
                self._draw_no_data("No Price Data")
                self.canvas.draw_idle()
                return
            base = self.calculate_technical_indicators(base)
            valid_data = base[base['Close'] > 0].copy()
            if not valid_data.empty:
                x = valid_data.index
                y = valid_data['Close'].values.astype(float)
                y = np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan)
                mask = np.isfinite(y)
                if mask.sum() == 0:
                    self._draw_no_data("No Price Data")
                    self.canvas.draw_idle()
                    return
                x = x[mask]
                y = y[mask]
                if y.max() <= 0:
                    self._draw_no_data("No Positive Price Data")
                    self.canvas.draw_idle()
                    return

                self._style_axes("PRICE HISTORY", "Price ($)")
                self._set_date_ticks(x)
                line, = self.ax.plot([], [], color=C_ACCENT, linewidth=1.8)
                line.set_data(x[:1], y[:1])
                sma50_line = None
                sma200_line = None

                if "SMA_50" in valid_data.columns:
                    s50 = pd.to_numeric(valid_data["SMA_50"], errors="coerce").reindex(x)
                    if np.isfinite(s50.values).any():
                        sma50_line, = self.ax.plot(x, s50.values, color=C_GREEN, linewidth=1.0, alpha=0.8, label="SMA 50")
                if "SMA_200" in valid_data.columns:
                    s200 = pd.to_numeric(valid_data["SMA_200"], errors="coerce").reindex(x)
                    if np.isfinite(s200.values).any():
                        sma200_line, = self.ax.plot(x, s200.values, color=C_YELLOW, linewidth=1.0, alpha=0.8, label="SMA 200")

                y_min = np.nanmin(y)
                y_max = np.nanmax(y)
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > 0 and (y_max / y_min) > 50:
                    # Protect against split/outlier distortions from inconsistent history sources.
                    q1, q99 = np.nanpercentile(y, [1, 99])
                    if np.isfinite(q1) and np.isfinite(q99) and q99 > q1:
                        y_min, y_max = q1, q99
                self._safe_ylim(y_min, y_max, pad=0.06, zero_floor=False)
                self.ax.set_xlim(x.min(), x.max())
                if len(y) >= 2 and y[0] != 0:
                    pct = (y[-1] - y[0]) / abs(y[0]) * 100
                    self.lbl_price_change.configure(
                        text=f"{self.price_range_var.get()} Change: {pct:+.2f}%",
                        text_color=C_GREEN if pct >= 0 else C_RED
                    )

                if self.anim_enabled.get():
                    def update(frame):
                        if len(x) == 0: return line,
                        prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                        limit = int(prog * len(x))
                        limit = max(1, min(len(x), limit))
                        line.set_data(x[:limit], y[:limit])
                        return line,
                    self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES+5, interval=20, blit=False, repeat=False)
                else:
                    line.set_data(x, y)
                    self.ax.fill_between(x, y, color=C_ACCENT, alpha=0.08)
                handles, labels = self.ax.get_legend_handles_labels()
                if self._ax2 is not None:
                    h2, l2 = self._ax2.get_legend_handles_labels()
                    handles += h2
                    labels += l2
                if handles:
                    self.ax.legend(handles, labels, loc="upper left")
                summary = self._series_stats(y, "Price", "$")
                self._set_chart_analysis(f"Price|{self.price_range_var.get()}", summary)
            else:
                self._draw_no_data("No Price Data")
            self._apply_layout()
        # --- 2. REVENUE & EBITDA ---
        elif mode == "Revenue & EBITDA":
            rev_q = self.get_series(self.df_fin_q, ["TotalRevenue", "OperatingRevenue", "Revenue"])
            rev_a = self.get_series(self.df_fin_a, ["TotalRevenue", "OperatingRevenue", "Revenue"])
            ebitda_q = self.get_series(self.df_fin_q, ["EBITDA", "NormalizedEBITDA"])
            ebitda_a = self.get_series(self.df_fin_a, ["EBITDA", "NormalizedEBITDA"])
            ebitda_label = "EBITDA"
            if ebitda_q is None or ebitda_q.empty:
                ebitda_q = self.get_series(self.df_fin_q, ["OperatingIncome", "OperatingIncomeLoss", "EBIT"])
            if ebitda_a is None or ebitda_a.empty:
                ebitda_a = self.get_series(self.df_fin_a, ["OperatingIncome", "OperatingIncomeLoss", "EBIT"])
            if (ebitda_q is not None and not ebitda_q.empty) or (ebitda_a is not None and not ebitda_a.empty):
                if ebitda_a is None and ebitda_q is not None:
                    ebitda_label = "Operating Income"
            rev = self._series_for_flow(rev_q, rev_a, freq)
            ebitda = self._series_for_flow(ebitda_q, ebitda_a, freq)
            
            if rev is not None:
                dates, vals_rev, vals_ebitda = self.align_data(
                    rev / 1e9,
                    ebitda / 1e9 if ebitda is not None else None,
                    fill_value=np.nan,
                    freq=freq
                )
                has_ebitda = ebitda is not None and np.isfinite(vals_ebitda).any()
                vals_rev = np.nan_to_num(vals_rev, nan=0.0, posinf=0.0, neginf=0.0)
                vals_ebitda = np.nan_to_num(vals_ebitda, nan=0.0, posinf=0.0, neginf=0.0)
                if len(dates) > 0:
                    x_idx = np.arange(len(dates))
                    self._style_axes("REVENUE & EBITDA", "USD ($B)")
                    self._set_period_ticks(x_idx, dates)
                    self.ax.set_xlim(-0.5, len(dates) - 0.5)
                    self._add_vertical_guides(x_idx)
                    show_ebitda = has_ebitda and np.nanmax(np.abs(vals_ebitda)) > 0
                    bars = self.ax.bar(x_idx, [0]*len(dates), color="#3B3F4A", label="Revenue")
                    ebitda_line = None
                    use_twin = False
                    if show_ebitda:
                        rev_max = np.nanmax(np.abs(vals_rev)) if len(vals_rev) else 0
                        ebitda_max = np.nanmax(np.abs(vals_ebitda)) if len(vals_ebitda) else 0
                        if rev_max > 0 and ebitda_max > 0:
                            ratio = max(rev_max, ebitda_max) / min(rev_max, ebitda_max)
                            use_twin = ratio >= 6
                        if use_twin:
                            self._ax2 = self.ax.twinx()
                            self._ax2.set_facecolor("none")
                            for spine in self._ax2.spines.values():
                                spine.set_visible(False)
                            self._ax2.tick_params(axis='y', colors=C_GREEN, labelsize=9)
                            self._ax2.set_ylabel(f"{ebitda_label} ($B)", color=C_GREEN)
                            ebitda_line, = self._ax2.plot([], [], color=C_GREEN, marker='o', label=ebitda_label, linewidth=2)
                        else:
                            ebitda_line, = self.ax.plot([], [], color=C_GREEN, marker='o', label=ebitda_label, linewidth=2)

                    y_max = max(vals_rev.max() if len(vals_rev) else 0, vals_ebitda.max() if show_ebitda else 0)
                    y_min = min(0, vals_rev.min() if len(vals_rev) else 0, vals_ebitda.min() if show_ebitda else 0)
                    self._safe_ylim(y_min, y_max, pad=0.15, zero_floor=True)
                    self.ax.axhline(0, color=C_BORDER, lw=1, alpha=0.6)
                    if self._ax2 is not None:
                        emax = np.nanmax(np.abs(vals_ebitda)) if len(vals_ebitda) else 0
                        if not np.isfinite(emax) or emax <= 0:
                            emax = 1.0
                        self._ax2.set_ylim(-emax * 1.2 if np.nanmin(vals_ebitda) < 0 else 0, emax * 1.2)

                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            for rect, h in zip(bars, vals_rev):
                                rect.set_height(h * prog)
                            if show_ebitda and ebitda_line is not None:
                                limit = int(prog * len(dates))
                                limit = max(1, min(len(dates), limit))
                                ebitda_line.set_data(x_idx[:limit], vals_ebitda[:limit])
                            return bars
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        for rect, h in zip(bars, vals_rev):
                            rect.set_height(h)
                        if show_ebitda and ebitda_line is not None:
                            ebitda_line.set_data(x_idx, vals_ebitda)
                    handles, labels = self.ax.get_legend_handles_labels()
                    if self._ax2 is not None:
                        h2, l2 = self._ax2.get_legend_handles_labels()
                        handles += h2
                        labels += l2
                    self.ax.legend(handles, labels, loc='upper left')
                    summary = "\n".join([
                        self._series_stats(vals_rev, "Revenue", "B"),
                        self._series_stats(vals_ebitda, ebitda_label, "B") if show_ebitda else f"{ebitda_label}: no data."
                    ])
                    self._set_chart_analysis(f"Revenue & EBITDA|{freq}", summary)
                else:
                    self._draw_no_data("Revenue data unavailable")
            else:
                self._draw_no_data("Revenue data unavailable")
            self._apply_layout()

        # --- 3. EPS & DIVIDENDS ---
        elif mode == "EPS & Dividends":
            eps_q = self.get_series(self.df_fin_q, ["BasicEPS", "DilutedEPS"])
            eps_a = self.get_series(self.df_fin_a, ["BasicEPS", "DilutedEPS"])
            eps = self._series_for_flow(eps_q, eps_a, freq)
            if eps is not None:
                df = pd.DataFrame({'eps': eps}).sort_index()
                if not df['eps'].dropna().empty:
                    div_s = self._dividend_series(freq)
                    # Align to proper period range
                    if freq == "Annual":
                        pidx = pd.PeriodIndex(pd.to_datetime(df.index), freq="Y")
                        eps_series = pd.Series(df['eps'].values, index=pidx).groupby(level=0).sum()
                        df = pd.DataFrame({'eps': eps_series.values}, index=eps_series.index.to_timestamp())
                        if div_s is not None and not div_s.empty:
                            d_pidx = pd.PeriodIndex(pd.to_datetime(div_s.index), freq="Y")
                            d_series = pd.Series(div_s.values, index=d_pidx).groupby(level=0).sum()
                            df['div'] = d_series.reindex(eps_series.index).fillna(0).values
                    elif freq == "TTM":
                        pidx = pd.PeriodIndex(pd.to_datetime(df.index), freq="Q")
                        if not pidx.empty:
                            periods = pd.period_range(end=pidx.max(), periods=4, freq="Q")
                            eps_series = pd.Series(df['eps'].values, index=pidx).reindex(periods)
                            df = pd.DataFrame({'eps': eps_series.values}, index=periods.to_timestamp())
                            if div_s is not None and not div_s.empty:
                                d_pidx = pd.PeriodIndex(pd.to_datetime(div_s.index), freq="Q")
                                d_series = pd.Series(div_s.values, index=d_pidx).reindex(periods).fillna(0)
                                df['div'] = d_series.values
                    else:
                        if div_s is not None and not div_s.empty:
                            df['div'] = div_s.reindex(df.index).fillna(0).values
                    periods = self._to_period_index(df.index, freq)
                    dates = self._format_period_labels(periods, freq)
                    x_idx = np.arange(len(dates))
                    y_eps = df['eps'].values.astype(float)
                    y_div = np.nan_to_num(df['div'].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0) if 'div' in df.columns else None

                    self._style_axes("EPS & DIVIDENDS", "Per Share ($)")
                    self._set_period_ticks(x_idx, dates)
                    self.ax.set_xlim(-0.5, len(dates) - 0.5)
                    self.ax.axhline(0, color=C_BORDER, lw=1, alpha=0.6)
                    self._add_vertical_guides(x_idx)

                    eps_line, = self.ax.plot([], [], color=C_ACCENT, marker='o', label="EPS", linewidth=2)
                    div_bars = None
                    if y_div is not None and np.nanmax(y_div) > 0:
                        self._ax2 = self.ax.twinx()
                        self._ax2.set_facecolor("none")
                        for spine in self._ax2.spines.values():
                            spine.set_visible(False)
                        self._ax2.tick_params(axis='y', colors=C_YELLOW, labelsize=9)
                        self._ax2.set_ylabel("Dividends ($)", color=C_YELLOW)
                        div_bars = self._ax2.bar(x_idx, [0]*len(dates), color=C_YELLOW, alpha=0.35, label="Dividends")

                    if np.isfinite(y_eps).any():
                        y_max = np.nanmax(y_eps)
                        y_min = np.nanmin(y_eps)
                    else:
                        y_max = 1.0
                        y_min = 0.0
                    self._safe_ylim(min(0, y_min), y_max, pad=0.2, zero_floor=False)
                    if self._ax2 is not None:
                        div_max = y_div.max() if len(y_div) else 0
                        if not np.isfinite(div_max) or div_max <= 0:
                            div_max = 1.0
                        self._ax2.set_ylim(0, div_max * 1.25)

                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            eps_line.set_data(x_idx[:limit], y_eps[:limit])
                            if div_bars is not None:
                                for rect, h in zip(div_bars, y_div):
                                    rect.set_height(h * prog)
                            return eps_line,
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        eps_line.set_data(x_idx, y_eps)
                        if div_bars is not None:
                            for rect, h in zip(div_bars, y_div):
                                rect.set_height(h)
                    handles, labels = self.ax.get_legend_handles_labels()
                    if self._ax2 is not None:
                        h2, l2 = self._ax2.get_legend_handles_labels()
                        handles += h2
                        labels += l2
                    self.ax.legend(handles, labels)
                    summary = self._series_stats(y_eps, "EPS", "")
                    if y_div is not None:
                        summary = summary + "\n" + self._series_stats(y_div, "Dividends", "")
                    self._set_chart_analysis(f"EPS & Dividends|{freq}", summary)
                else:
                    self._draw_no_data("EPS data unavailable")
            else:
                self._draw_no_data("EPS data unavailable")
            self._apply_layout()

        # --- 4. CASH VS DEBT ---
        elif mode == "Cash vs Debt":
            cash_q = self.get_series(self.df_bal_q, ["CashAndCashEquivalents", "Cash"])
            cash_a = self.get_series(self.df_bal_a, ["CashAndCashEquivalents", "Cash"])
            debt_q = self.get_series(self.df_bal_q, ["TotalDebt", "LongTermDebt"])
            debt_a = self.get_series(self.df_bal_a, ["TotalDebt", "LongTermDebt"])
            cash = self._series_for_balance(cash_q, cash_a, freq)
            debt = self._series_for_balance(debt_q, debt_a, freq)
            
            if cash is not None:
                dates, v_cash, v_debt = self.align_data(cash/1e9, debt/1e9 if debt is not None else None, freq=freq)
                if len(dates) > 0:
                    x_idx = np.arange(len(dates))
                    v_cash = np.nan_to_num(v_cash.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    v_debt = np.nan_to_num(v_debt.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

                    self._style_axes("CASH vs DEBT", "USD ($B)")
                    self._set_period_ticks(x_idx, dates)
                    self.ax.set_xlim(-0.5, len(dates) - 0.5)
                    self._add_vertical_guides(x_idx)
                    l1, = self.ax.plot([], [], color=C_GREEN, marker='o', label="Cash")
                    l2 = None
                    use_twin = False
                    if debt is not None:
                        cash_max = np.nanmax(np.abs(v_cash)) if len(v_cash) else 0
                        debt_max = np.nanmax(np.abs(v_debt)) if len(v_debt) else 0
                        if cash_max > 0 and debt_max > 0:
                            ratio = max(cash_max, debt_max) / min(cash_max, debt_max)
                            use_twin = ratio >= 6
                        if use_twin:
                            self._ax2 = self.ax.twinx()
                            self._ax2.set_facecolor("none")
                            for spine in self._ax2.spines.values():
                                spine.set_visible(False)
                            self._ax2.tick_params(axis='y', colors=C_RED, labelsize=9)
                            self._ax2.set_ylabel("Debt ($B)", color=C_RED)
                            l2, = self._ax2.plot([], [], color=C_RED, marker='x', label="Debt")
                        else:
                            l2, = self.ax.plot([], [], color=C_RED, marker='x', label="Debt")

                    max_val = max(v_cash.max() if len(v_cash)>0 else 0, v_debt.max() if len(v_debt)>0 else 0)
                    self._safe_ylim(0, max_val, pad=0.2, zero_floor=True)
                    if self._ax2 is not None:
                        dmax = np.nanmax(np.abs(v_debt)) if len(v_debt) else 0
                        if not np.isfinite(dmax) or dmax <= 0:
                            dmax = 1.0
                        self._ax2.set_ylim(0, dmax * 1.25)

                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            l1.set_data(x_idx[:limit], v_cash[:limit])
                            if l2 is not None:
                                l2.set_data(x_idx[:limit], v_debt[:limit])
                            return l1,
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        l1.set_data(x_idx, v_cash)
                        if l2 is not None:
                            l2.set_data(x_idx, v_debt)
                    handles, labels = self.ax.get_legend_handles_labels()
                    if self._ax2 is not None:
                        h2, l2 = self._ax2.get_legend_handles_labels()
                        handles += h2
                        labels += l2
                    self.ax.legend(handles, labels)
                    summary = "\n".join([
                        self._series_stats(v_cash, "Cash", "B"),
                        self._series_stats(v_debt, "Debt", "B") if debt is not None else "Debt: no data."
                    ])
                    self._set_chart_analysis(f"Cash vs Debt|{freq}", summary)
                else:
                    self._draw_no_data("Cash/Debt data unavailable")
            else:
                self._draw_no_data("Cash/Debt data unavailable")
            self._apply_layout()

        # --- 5. MARGINS ---
        elif mode == "Margins (Gross/Op/Net)":
            rev_q = self.get_series(self.df_fin_q, ["TotalRevenue"])
            rev_a = self.get_series(self.df_fin_a, ["TotalRevenue"])
            rev = self._series_for_flow(rev_q, rev_a, freq)
            if rev is not None:
                gp_q = self.get_series(self.df_fin_q, ["GrossProfit"])
                gp_a = self.get_series(self.df_fin_a, ["GrossProfit"])
                op_q = self.get_series(self.df_fin_q, ["OperatingIncome", "OperatingIncomeLoss"])
                op_a = self.get_series(self.df_fin_a, ["OperatingIncome", "OperatingIncomeLoss"])
                ni_q = self.get_series(self.df_fin_q, ["NetIncome"])
                ni_a = self.get_series(self.df_fin_a, ["NetIncome"])
                gp = self._series_for_flow(gp_q, gp_a, freq)
                op = self._series_for_flow(op_q, op_a, freq)
                ni = self._series_for_flow(ni_q, ni_a, freq)
                
                df = pd.DataFrame({'r': rev, 'g': gp, 'o': op, 'n': ni}).dropna(subset=['r'])
                df['g'] = df['g'].fillna(0)
                df['o'] = df['o'].fillna(0)
                df['n'] = df['n'].fillna(0)
                df = df.sort_index()
                
                if not df.empty:
                    periods = self._to_period_index(df.index, freq)
                    dates = self._format_period_labels(periods, freq)
                    v_gross = np.nan_to_num(((df['g'] / df['r']) * 100).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    v_op = np.nan_to_num(((df['o'] / df['r']) * 100).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    v_net = np.nan_to_num(((df['n'] / df['r']) * 100).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    x_idx = np.arange(len(dates))

                    self._style_axes("MARGINS", "Percent (%)")
                    self._set_period_ticks(x_idx, dates)
                    self.ax.set_xlim(-0.5, len(dates) - 0.5)
                    self.ax.axhline(0, color=C_BORDER, lw=1, alpha=0.6)
                    self._add_vertical_guides(x_idx)
                    l1, = self.ax.plot([], [], color=C_YELLOW, label="Gross %")
                    l2, = self.ax.plot([], [], color=C_ACCENT, label="Op %")
                    l3, = self.ax.plot([], [], color=C_GREEN, label="Net %")
                    y_min = min(0, v_net.min(), v_op.min(), v_gross.min())
                    y_max = max(5, v_gross.max(), v_op.max(), v_net.max())
                    self._safe_ylim(y_min, y_max, pad=0.2, zero_floor=False)

                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            l1.set_data(x_idx[:limit], v_gross[:limit])
                            l2.set_data(x_idx[:limit], v_op[:limit])
                            l3.set_data(x_idx[:limit], v_net[:limit])
                            return l1, l2, l3
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        l1.set_data(x_idx, v_gross)
                        l2.set_data(x_idx, v_op)
                        l3.set_data(x_idx, v_net)
                    self.ax.legend()
                    summary = "\n".join([
                        self._series_stats(v_gross, "Gross Margin", "%"),
                        self._series_stats(v_op, "Operating Margin", "%"),
                        self._series_stats(v_net, "Net Margin", "%")
                    ])
                    self._set_chart_analysis(f"Margins|{freq}", summary)
                else:
                    self._draw_no_data("Margins data unavailable")
            else:
                self._draw_no_data("Margins data unavailable")
            self._apply_layout()

        # --- 6. CASH FLOW DETAIL ---
        elif mode == "Cash Flow Detail (FCF vs SBC)":
            fcf_q = self.get_series(self.df_cf_q, ["FreeCashFlow", "Free Cash Flow"])
            fcf_a = self.get_series(self.df_cf_a, ["FreeCashFlow", "Free Cash Flow"])
            fcf = self._series_for_flow(fcf_q, fcf_a, freq)
            
            if fcf is None:
                ocf_q = self.get_series(self.df_cf_q, ["OperatingCashFlow"])
                ocf_a = self.get_series(self.df_cf_a, ["OperatingCashFlow"])
                cap_q = self.get_series(self.df_cf_q, ["CapitalExpenditure"])
                cap_a = self.get_series(self.df_cf_a, ["CapitalExpenditure"])
                ocf = self._series_for_flow(ocf_q, ocf_a, freq)
                cap = self._series_for_flow(cap_q, cap_a, freq)
                if ocf is not None and cap is not None: 
                    fcf = ocf + cap
            
            sbc_q = self.get_series(self.df_cf_q, ["StockBasedCompensation", "ShareBasedCompensation"])
            sbc_a = self.get_series(self.df_cf_a, ["StockBasedCompensation", "ShareBasedCompensation"])
            sbc = self._series_for_flow(sbc_q, sbc_a, freq)

            if fcf is not None:
                dates, v_fcf, v_sbc = self.align_data(fcf/1e9, sbc/1e9 if sbc is not None else None, fill_value=0.0, freq=freq)
                
                if len(dates) > 0:
                    v_fcf = np.nan_to_num(v_fcf.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    v_sbc = np.nan_to_num(v_sbc.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    x_idx = np.arange(len(dates))

                    self._style_axes("CASH FLOW DETAIL", "USD ($B)")
                    self._set_period_ticks(x_idx, dates)
                    self.ax.set_xlim(-0.5, len(dates) - 0.5)
                    self._add_vertical_guides(x_idx)
                    self.ax.axhline(0, color=C_BORDER, lw=1, alpha=0.6)
                    bars = self.ax.bar(x_idx, [0]*len(dates), color=C_GREEN, label="FCF", alpha=0.6)
                    sbc_line = None
                    use_twin = False
                    if sbc is not None:
                        fcf_max = np.nanmax(np.abs(v_fcf)) if len(v_fcf) else 0
                        sbc_max = np.nanmax(np.abs(v_sbc)) if len(v_sbc) else 0
                        if fcf_max > 0 and sbc_max > 0:
                            ratio = max(fcf_max, sbc_max) / min(fcf_max, sbc_max)
                            use_twin = ratio >= 6
                        if use_twin:
                            self._ax2 = self.ax.twinx()
                            self._ax2.set_facecolor("none")
                            for spine in self._ax2.spines.values():
                                spine.set_visible(False)
                            self._ax2.tick_params(axis='y', colors=C_PINK, labelsize=9)
                            self._ax2.set_ylabel("SBC ($B)", color=C_PINK)
                            sbc_line, = self._ax2.plot([], [], color=C_PINK, marker='o', label="SBC", linewidth=2)
                        else:
                            sbc_line, = self.ax.plot([], [], color=C_PINK, marker='o', label="SBC", linewidth=2)
                    
                    max_val = max(v_fcf.max() if len(v_fcf) > 0 else 0, v_sbc.max() if len(v_sbc) > 0 else 0)
                    min_val = min(0, v_fcf.min() if len(v_fcf) > 0 else 0, v_sbc.min() if len(v_sbc) > 0 else 0)
                    self._safe_ylim(min_val, max_val, pad=0.2, zero_floor=False)
                    if self._ax2 is not None:
                        smax = np.nanmax(np.abs(v_sbc)) if len(v_sbc) else 0
                        if not np.isfinite(smax) or smax <= 0:
                            smax = 1.0
                        self._ax2.set_ylim(-smax * 1.2 if np.nanmin(v_sbc) < 0 else 0, smax * 1.2)

                    if self.anim_enabled.get():
                        def update(frame):
                            if len(dates) == 0: return bars
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            for rect, h in zip(bars, v_fcf): 
                                rect.set_height(h * prog)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            if sbc_line is not None:
                                sbc_line.set_data(x_idx[:limit], v_sbc[:limit])
                            return bars
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        for rect, h in zip(bars, v_fcf):
                            rect.set_height(h)
                        if sbc_line is not None:
                            sbc_line.set_data(x_idx, v_sbc)
                    handles, labels = self.ax.get_legend_handles_labels()
                    if self._ax2 is not None:
                        h2, l2 = self._ax2.get_legend_handles_labels()
                        handles += h2
                        labels += l2
                    self.ax.legend(handles, labels)
                    summary = "\n".join([
                        self._series_stats(v_fcf, "FCF", "B"),
                        self._series_stats(v_sbc, "SBC", "B") if sbc is not None else "SBC: no data."
                    ])
                    self._set_chart_analysis(f"Cash Flow Detail|{freq}", summary)
                else:
                    self._draw_no_data("Cash flow data unavailable")
            else:
                self._draw_no_data("Cash flow data unavailable")
            self._apply_layout()

        # --- 7. VALUATION ---
        elif mode in ("Valuation (P/E)", "Valuation (FCF Yield)", "Valuation (P/S)"):
            if self.hist_data is not None and self.df_fin_q is not None:
                fin_df = self.get_freq_df("fin")
                eps = self.get_series(fin_df, ["BasicEPS"])
                rev = self.get_series(fin_df, ["TotalRevenue"])
                fcf = self.get_series(self.get_freq_df("cf"), ["FreeCashFlow", "Free Cash Flow"])
                shares = self.get_series(self.get_freq_df("bal"), ["ShareIssued", "CommonStockSharesOutstanding"])
                price = pd.to_numeric(self.hist_data.get("Close"), errors="coerce")
                if freq == "TTM":
                    eps = self.to_ttm(eps)
                    rev = self.to_ttm(rev)
                    fcf = self.to_ttm(fcf)
                if mode == "Valuation (P/E)" and eps is not None:
                    aligned_eps = pd.to_numeric(eps.reindex(self.hist_data.index, method='ffill'), errors="coerce")
                    pe = price / (aligned_eps * 4)
                    pe = pe.replace([np.inf, -np.inf], np.nan)
                    pe = pe[(pe > 0) & (pe < 250)].dropna()
                    series = pe
                    label = "P/E Ratio"
                    color = "#D165E8"
                elif mode == "Valuation (P/S)" and rev is not None and shares is not None:
                    aligned_rev = pd.to_numeric(rev.reindex(self.hist_data.index, method='ffill'), errors="coerce")
                    aligned_shares = pd.to_numeric(shares.reindex(self.hist_data.index, method='ffill'), errors="coerce")
                    ps = (price * aligned_shares) / (aligned_rev * 4)
                    ps = ps.replace([np.inf, -np.inf], np.nan)
                    ps = ps[(ps > 0) & (ps < 50)].dropna()
                    series = ps
                    label = "P/S Ratio"
                    color = C_ACCENT
                elif mode == "Valuation (FCF Yield)" and fcf is not None and shares is not None:
                    aligned_fcf = pd.to_numeric(fcf.reindex(self.hist_data.index, method='ffill'), errors="coerce")
                    aligned_shares = pd.to_numeric(shares.reindex(self.hist_data.index, method='ffill'), errors="coerce")
                    mcap = price * aligned_shares
                    fy = ((aligned_fcf * 4) / mcap.replace(0, np.nan)) * 100
                    fy = fy.replace([np.inf, -np.inf], np.nan).dropna()
                    series = fy
                    label = "FCF Yield %"
                    color = C_GREEN
                else:
                    series = None
                if series is not None and not series.empty:
                    x = series.index
                    y = np.nan_to_num(series.values.astype(float), nan=np.nan, posinf=np.nan, neginf=np.nan)
                    self._style_axes(label.upper(), label)
                    self._set_date_ticks(x)
                    line, = self.ax.plot([], [], color=color, label=label, linewidth=1.8)
                    self.ax.set_xlim(x.min(), x.max())
                    self._safe_ylim(np.nanmin(y), np.nanmax(y), pad=0.15, zero_floor=False)
                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            limit = int(prog * len(x))
                            limit = max(1, min(len(x), limit))
                            line.set_data(x[:limit], y[:limit])
                            return line,
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        line.set_data(x, y)
                    self.ax.legend()
                    summary = self._series_stats(y, label, "")
                    self._set_chart_analysis(f"{label}|{freq}", summary)
                else:
                    self._draw_no_data("Insufficient data for valuation")
            else:
                self._draw_no_data("Insufficient data for valuation")
            self._apply_layout()

        # --- 8. FCF PER SHARE ---
        elif mode == "FCF Per Share":
            fcf_q = self.get_series(self.df_cf_q, ["FreeCashFlow", "Free Cash Flow"])
            fcf_a = self.get_series(self.df_cf_a, ["FreeCashFlow", "Free Cash Flow"])
            shares_q = self.get_series(self.df_bal_q, ["ShareIssued", "CommonStockSharesOutstanding"])
            shares_a = self.get_series(self.df_bal_a, ["ShareIssued", "CommonStockSharesOutstanding"])
            fcf = self._series_for_flow(fcf_q, fcf_a, freq)
            shares = self._series_for_balance(shares_q, shares_a, freq)
            if fcf is not None and shares is not None:
                try:
                    fps = (fcf / shares).replace([np.inf, -np.inf], np.nan)
                except Exception:
                    fps = self.safe_div(fcf, shares)
                if not isinstance(fps, pd.Series):
                    try:
                        fps = pd.Series(fps)
                    except Exception:
                        fps = None
                df = pd.DataFrame({'fps': fps}).dropna().sort_index() if fps is not None else pd.DataFrame()
                if not df.empty:
                    periods = self._to_period_index(df.index, freq)
                    dates = self._format_period_labels(periods, freq)
                    x_idx = np.arange(len(dates))
                    y = np.nan_to_num(df['fps'].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    self._style_axes("FCF PER SHARE", "Per Share ($)")
                    self._set_period_ticks(x_idx, dates)
                    self.ax.set_xlim(-0.5, len(dates) - 0.5)
                    self._add_vertical_guides(x_idx)
                    line, = self.ax.plot([], [], color=C_GREEN, marker='o', label="FCF/Share", linewidth=2)
                    self._safe_ylim(min(0, y.min()), max(0.1, y.max()), pad=0.2, zero_floor=False)
                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            limit = int(prog * len(x_idx))
                            limit = max(1, min(len(x_idx), limit))
                            line.set_data(x_idx[:limit], y[:limit])
                            return line,
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        line.set_data(x_idx, y)
                    self.ax.legend()
                    summary = self._series_stats(y, "FCF/Share", "")
                    self._set_chart_analysis(f"FCF/Share|{freq}", summary)
                else:
                    self._draw_no_data("FCF/share unavailable")
            else:
                self._draw_no_data("FCF/share unavailable")
            self._apply_layout()

        self.canvas.draw_idle()


    # --- TAB 3: DCF ---
    def build_valuation(self):
        center = ctk.CTkFrame(self.tab_val, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        center.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        dcf_tabs = ctk.CTkTabview(center, width=800, height=500)
        dcf_tabs.configure(
            fg_color="transparent",
            segmented_button_fg_color=C_BG_CARD,
            segmented_button_selected_color=C_ACCENT,
            segmented_button_selected_hover_color=C_ACCENT_HOVER,
            segmented_button_unselected_color=C_BG_INPUT,
            segmented_button_unselected_hover_color="#263047",
            text_color=C_TEXT_MAIN,
            text_color_disabled=C_TEXT_SUB
        )
        try:
            dcf_tabs._segmented_button.configure(font=FONT_SMALL)
        except:
            pass
        dcf_tabs.pack(expand=True, fill=tk.BOTH)
        
        fwd = dcf_tabs.add("FORWARD DCF")
        rev = dcf_tabs.add("REVERSE DCF")
        
        self.setup_fwd_dcf(fwd)
        self.setup_rev_dcf(rev)

    def setup_fwd_dcf(self, p):
        grid_frame = ctk.CTkFrame(p, fg_color="transparent")
        grid_frame.pack(expand=True)
        
        ctk.CTkLabel(grid_frame, text="Basis", font=FONT_BODY).grid(row=0, column=0, sticky='e', padx=20, pady=10)
        self.dcf_type = tk.StringVar(value="FCF")
        self.make_combo(grid_frame, variable=self.dcf_type, values=["FCF", "EPS"], width=120, command=self.on_dcf_type_change, **COMBO_KW).grid(row=0, column=1, sticky='w', padx=20, pady=10)

        self.lbl_fcf = ctk.CTkLabel(grid_frame, text="FCF / Share ($)", font=FONT_BODY)
        self.lbl_fcf.grid(row=1, column=0, sticky='e', padx=20, pady=10)
        self.f_fcf = ctk.CTkEntry(grid_frame, width=150, justify='center', **ENTRY_KW)
        self.f_fcf.insert(0, "0")
        self.f_fcf.grid(row=1, column=1, sticky='w', padx=20, pady=10)
        self.f_gr = self.mk_inp(grid_frame, "Growth Rate (%)", 2, "10")
        self.f_dr = self.mk_inp(grid_frame, "Discount Rate (%)", 3, "10"); self.f_tm = self.mk_inp(grid_frame, "Terminal Mult", 4, "15")
        for w in (self.f_fcf, self.f_gr, self.f_dr, self.f_tm):
            self._bind_dcf_autocalc(w)
        
        ctk.CTkButton(grid_frame, text="CALCULATE FAIR VALUE", command=self.calc_fwd, width=200, height=40, font=FONT_H2, **BTN_PRIMARY).grid(row=5, column=0, columnspan=2, pady=30)
        self.lbl_fwd = ctk.CTkLabel(grid_frame, text="$ --", font=FONT_DISPLAY)
        self.lbl_fwd.grid(row=6, column=0, columnspan=2)

    def setup_rev_dcf(self, p):
        grid_frame = ctk.CTkFrame(p, fg_color="transparent")
        grid_frame.pack(expand=True)

        self.lbl_r_fcf = ctk.CTkLabel(grid_frame, text="FCF / Share ($)", font=FONT_BODY)
        self.lbl_r_fcf.grid(row=0, column=0, sticky='e', padx=20, pady=10)
        self.r_fcf = ctk.CTkEntry(grid_frame, width=150, justify='center', **ENTRY_KW)
        self.r_fcf.insert(0, "0")
        self.r_fcf.grid(row=0, column=1, sticky='w', padx=20, pady=10)
        self.r_gr = self.mk_inp(grid_frame, "Growth Rate (%)", 1, "10")
        self.r_ret = self.mk_inp(grid_frame, "Target Return (%)", 2, "15"); self.r_tm = self.mk_inp(grid_frame, "Terminal Mult", 3, "15")
        for w in (self.r_fcf, self.r_gr, self.r_ret, self.r_tm):
            self._bind_dcf_autocalc(w)
        
        ctk.CTkButton(grid_frame, text="CALC MAX BUY PRICE", command=self.calc_rev, width=200, height=40, font=FONT_H2, **BTN_PRIMARY).grid(row=4, column=0, columnspan=2, pady=30)
        self.lbl_rev = ctk.CTkLabel(grid_frame, text="$ --", font=FONT_DISPLAY)
        self.lbl_rev.grid(row=5, column=0, columnspan=2)

    def mk_inp(self, p, txt, r, val="0"):
        ctk.CTkLabel(p, text=txt, font=FONT_BODY).grid(row=r, column=0, sticky='e', padx=20, pady=10)
        e = ctk.CTkEntry(p, width=150, justify='center', **ENTRY_KW)
        e.insert(0, val)
        e.grid(row=r, column=1, sticky='w', padx=20, pady=10)
        return e

    def _bind_dcf_autocalc(self, widget):
        try:
            widget.bind("<KeyRelease>", lambda _e: self._schedule_dcf_autocalc())
            widget.bind("<FocusOut>", lambda _e: self._schedule_dcf_autocalc())
        except Exception:
            pass

    def _schedule_dcf_autocalc(self):
        if self._dcf_job:
            try:
                self.root.after_cancel(self._dcf_job)
            except Exception:
                pass
        self._dcf_job = self.root.after(250, self._run_dcf_autocalc)

    def _run_dcf_autocalc(self):
        self._dcf_job = None
        self.calc_fwd()
        self.calc_rev()

    def on_dcf_type_change(self, _=None):
        lbl = "FCF / Share ($)" if self.dcf_type.get() == "FCF" else "EPS ($)"
        if hasattr(self, "lbl_fcf"):
            self.lbl_fcf.configure(text=lbl)
        if hasattr(self, "lbl_r_fcf"):
            self.lbl_r_fcf.configure(text=lbl)
        self._schedule_dcf_autocalc()

    def reset_dcf(self, i):
        try:
            saved = self.store.get_dcf_inputs(self.current_symbol)
            if saved:
                data_type, val, g, d, tm, tr = saved
                if data_type in ("FCF", "EPS"):
                    self.dcf_type.set(data_type)
                    self.on_dcf_type_change()
                if val is not None:
                    self.f_fcf.delete(0,tk.END); self.f_fcf.insert(0,f"{val:.2f}")
                    self.r_fcf.delete(0,tk.END); self.r_fcf.insert(0,f"{val:.2f}")
                if g is not None: self.f_gr.delete(0,tk.END); self.f_gr.insert(0,f"{g*100:.2f}")
                if d is not None: self.f_dr.delete(0,tk.END); self.f_dr.insert(0,f"{d*100:.2f}")
                if tm is not None: self.f_tm.delete(0,tk.END); self.f_tm.insert(0,f"{tm:.2f}")
                if tr is not None: self.r_ret.delete(0,tk.END); self.r_ret.insert(0,f"{tr*100:.2f}")
            else:
                val = self.safe_div(i.get('freeCashflow',0), i.get('sharesOutstanding',1))
                self.f_fcf.delete(0,tk.END); self.f_fcf.insert(0,f"{val:.2f}")
                self.r_fcf.delete(0,tk.END); self.r_fcf.insert(0,f"{val:.2f}")
        except: pass
        self._schedule_dcf_autocalc()

    def calc_fwd(self):
        try:
            f,g,r,m = float(self.f_fcf.get()), float(self.f_gr.get())/100, float(self.f_dr.get())/100, float(self.f_tm.get())
            vals = [f*((1+g)**i) for i in range(1,6)]; term = vals[-1]*m
            pv = sum([self.safe_div(v, ((1+r)**(i+1))) for i,v in enumerate(vals)]) + self.safe_div(term, ((1+r)**5))
            self.lbl_fwd.configure(text=f"${pv:.2f}")
            if self.current_symbol:
                self.store.upsert_dcf_inputs(self.current_symbol, self.dcf_type.get(), f, g, r, m, float(self.r_ret.get())/100 if self.r_ret.get() else 0.0)
        except: pass

    def calc_rev(self):
        try:
            f,g,ret,m = float(self.r_fcf.get()), float(self.r_gr.get())/100, float(self.r_ret.get())/100, float(self.r_tm.get())
            vals = [f*((1+g)**i) for i in range(1,6)]; term = vals[-1]*m
            max_price = sum([self.safe_div(v, ((1+ret)**(i+1))) for i,v in enumerate(vals)]) + self.safe_div(term, ((1+ret)**5))
            self.lbl_rev.configure(text=f"${max_price:.2f}")
            if self.current_symbol:
                self.store.upsert_dcf_inputs(self.current_symbol, self.dcf_type.get(), f, g, float(self.f_dr.get())/100 if self.f_dr.get() else 0.0, m, ret)
        except: pass

    # --- TAB 4: INSIDER & CALENDAR ---
    def build_insider_calendar(self):
        self.tab_insider.grid_columnconfigure(0, weight=1)
        self.tab_insider.grid_columnconfigure(1, weight=1)
        self.tab_insider.grid_rowconfigure(0, weight=1)

        # INSIDER FRAME
        f1 = ctk.CTkFrame(self.tab_insider, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        f1.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        
        ctk.CTkLabel(f1, text="INSIDER TRADING", font=FONT_H2, text_color=C_ACCENT).pack(anchor='w', pady=10, padx=10)
        
        f1_tv = ctk.CTkFrame(f1, fg_color="transparent")
        f1_tv.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        cols = ("Date", "Insider", "Type", "Value")
        self.tr_insider = ttk.Treeview(f1_tv, columns=cols, show='headings')
        for c in cols: self.tr_insider.heading(c, text=c.upper()); self.tr_insider.column(c, width=90)
        
        # SCROLLBAR INSIDER
        vsb1 = ctk.CTkScrollbar(f1_tv, orientation="vertical", command=self.tr_insider.yview)
        self.tr_insider.configure(yscrollcommand=vsb1.set)
        vsb1.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_insider.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # CALENDAR FRAME
        f2 = ctk.CTkFrame(self.tab_insider, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        f2.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        
        ctk.CTkLabel(f2, text="EARNINGS CALENDAR", font=FONT_H2, text_color=C_ACCENT).pack(anchor='w', pady=10, padx=10)
        
        f2_tv = ctk.CTkFrame(f2, fg_color="transparent")
        f2_tv.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        c2 = ("Date", "Event")
        self.tr_cal = ttk.Treeview(f2_tv, columns=c2, show='headings')
        for c in c2: self.tr_cal.heading(c, text=c.upper()); self.tr_cal.column(c, width=150)
        
        # SCROLLBAR CALENDAR
        vsb2 = ctk.CTkScrollbar(f2_tv, orientation="vertical", command=self.tr_cal.yview)
        self.tr_cal.configure(yscrollcommand=vsb2.set)
        vsb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_cal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def fill_insider_calendar(self, ins, cal):
        def _to_date_str(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                if len(v) == 0:
                    return None
                return _to_date_str(v[0])
            if isinstance(v, pd.Timestamp):
                return v.tz_localize(None).strftime("%Y-%m-%d")
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d")
            if isinstance(v, (int, float, np.integer, np.floating)):
                if not np.isfinite(v):
                    return None
                # Epoch support (seconds / ms)
                if v > 10_000_000_000:
                    ts = pd.to_datetime(int(v), unit="ms", errors="coerce")
                elif v > 1_000_000_000:
                    ts = pd.to_datetime(int(v), unit="s", errors="coerce")
                else:
                    return None
                return ts.strftime("%Y-%m-%d") if pd.notna(ts) else None
            s = str(v).strip()
            if not s or s.lower() in ("nan", "nat", "none"):
                return None
            ts = pd.to_datetime(s, errors="coerce")
            if pd.notna(ts):
                return ts.strftime("%Y-%m-%d")
            m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
            if m:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            return None

        def _find_col(df: pd.DataFrame, names: List[str]):
            if df is None or df.empty:
                return None
            norm = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
            for n in names:
                key = n.lower().replace("_", " ")
                if key in norm:
                    return norm[key]
            return None

        # --- Insider table ---
        for i in self.tr_insider.get_children():
            self.tr_insider.delete(i)
        try:
            if ins is not None and not ins.empty:
                ins_df = ins.copy().reset_index(drop=False)
                date_col = _find_col(ins_df, ["date", "start date", "filing date", "transaction date"])
                who_col = _find_col(ins_df, ["insider", "name", "insider name", "text"])
                type_col = _find_col(ins_df, ["transaction", "transaction type", "text"])
                value_col = _find_col(ins_df, ["value", "transaction value"])
                shares_col = _find_col(ins_df, ["shares", "shares traded", "shares owned"])
                price_col = _find_col(ins_df, ["price", "transaction price"])

                rows = []
                for _, r in ins_df.iterrows():
                    raw_date = r.get(date_col) if date_col else r.get("index")
                    d = _to_date_str(raw_date) or "N/A"
                    who = str(r.get(who_col, "")).strip() if who_col else ""
                    if not who:
                        who = "N/A"
                    raw_text = ""
                    if type_col:
                        raw_text = str(r.get(type_col, "")).lower()
                    if not raw_text:
                        raw_text = " ".join(str(x).lower() for x in r.values)
                    if any(k in raw_text for k in ("buy", "purchase", "acquired", "option exercise", "p ")):
                        typ = "BUY"
                    elif any(k in raw_text for k in ("sell", "sale", "disposed", "s ")):
                        typ = "SELL"
                    else:
                        typ = "-"

                    val_num = None
                    if value_col is not None:
                        v = pd.to_numeric(r.get(value_col), errors="coerce")
                        if pd.notna(v):
                            val_num = float(v)
                    if val_num is None and shares_col is not None and price_col is not None:
                        sh = pd.to_numeric(r.get(shares_col), errors="coerce")
                        pr = pd.to_numeric(r.get(price_col), errors="coerce")
                        if pd.notna(sh) and pd.notna(pr):
                            val_num = abs(float(sh) * float(pr))
                    val_txt = f"{val_num:,.0f}" if val_num is not None and np.isfinite(val_num) else "N/A"
                    dt_sort = pd.to_datetime(d, errors="coerce")
                    rows.append((dt_sort, d, who, typ, val_txt))

                dated = [r for r in rows if pd.notna(r[0])]
                undated = [r for r in rows if pd.isna(r[0])]
                dated.sort(key=lambda x: x[0], reverse=True)
                rows = dated + undated
                for _, d, who, typ, val_txt in rows[:50]:
                    self.tr_insider.insert("", tk.END, values=(d, who, typ, val_txt))
        except Exception as e:
            log_exception("fill_insider_calendar.insider", e)

        # --- Earnings calendar table ---
        for i in self.tr_cal.get_children():
            self.tr_cal.delete(i)
        try:
            cal_rows = []
            if cal is not None and not cal.empty:
                if "Label" in cal.columns and "Value" in cal.columns:
                    # yfinance dict-style calendar: map to Date/Event correctly.
                    for _, r in cal.iterrows():
                        event = str(r.get("Label", "")).strip()
                        value = r.get("Value")
                        d = _to_date_str(value)
                        if d:
                            cal_rows.append((d, event))
                        elif "date" in event.lower():
                            cal_rows.append(("N/A", f"{event}: {value}"))
                elif isinstance(cal.index, pd.DatetimeIndex):
                    # earnings_dates dataframe
                    for idx, r in cal.head(12).iterrows():
                        d = _to_date_str(idx) or "N/A"
                        est = r.get("EPS Estimate", None)
                        act = r.get("Reported EPS", None)
                        sur = r.get("Surprise(%)", None)
                        extra = []
                        if pd.notna(est): extra.append(f"Est {est}")
                        if pd.notna(act): extra.append(f"Act {act}")
                        if pd.notna(sur): extra.append(f"Surprise {sur}%")
                        ev = "Earnings"
                        if extra:
                            ev = f"Earnings ({', '.join(extra)})"
                        cal_rows.append((d, ev))
                else:
                    # Generic fallback: first date-like value in row + label text
                    for _, r in cal.iterrows():
                        d = None
                        ev = None
                        for c in cal.columns:
                            v = r.get(c)
                            if d is None:
                                d = _to_date_str(v)
                            if ev is None and isinstance(v, str) and len(v) < 80 and not _to_date_str(v):
                                ev = v
                        if d and ev:
                            cal_rows.append((d, ev))
            cal_rows.sort(key=lambda x: (x[0] == "N/A", x[0]))
            for d, ev in cal_rows[:30]:
                self.tr_cal.insert("", tk.END, values=(d, ev))
        except Exception as e:
            log_exception("fill_insider_calendar.calendar", e)

    # --- TAB 5: AI ---
    def build_ai(self):
        pad = ctk.CTkFrame(self.tab_ai, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        pad.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        btns = ctk.CTkFrame(pad, fg_color="transparent")
        btns.pack(fill=tk.X, pady=(0,10))
        ctk.CTkLabel(btns, text="Single Holistic AI Report", font=FONT_H2, text_color=C_ACCENT).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(btns, text="Auto-generated on symbol load", font=FONT_SMALL, text_color=C_TEXT_SUB).pack(side=tk.RIGHT, padx=5)
        
        self.txt_ai = ctk.CTkTextbox(pad, font=FONT_BODY, fg_color=C_BG_INPUT, text_color=C_TEXT_MAIN)
        self.txt_ai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def build_ai_report_context(self, sym, info):
        parts = []
        try:
            name = info.get("longName", info.get("shortName", sym))
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            mcap = info.get("marketCap", 0) or 0
            price = info.get("currentPrice", None)
            pe = info.get("trailingPE", "N/A")
            ps = info.get("priceToSalesTrailing12Months", "N/A")
            pb = info.get("priceToBook", "N/A")
            roe = info.get("returnOnEquity", None)
            roa = info.get("returnOnAssets", None)
            parts.append(f"Company: {name} ({sym})")
            parts.append(f"Sector/Industry: {sector} / {industry}")
            if price is not None:
                parts.append(f"Current Price: {price}")
            if mcap:
                parts.append(f"Market Cap: ${mcap/1e9:,.1f}B")
            parts.append(f"Valuation Multiples: P/E {pe}, P/S {ps}, P/B {pb}")
            if roe is not None or roa is not None:
                parts.append(f"Returns: ROE {roe}, ROA {roa}")
        except Exception:
            pass

        # Financial snapshot
        try:
            snapshot = self.get_financial_snapshot()
            if snapshot:
                parts.append("Financial Snapshot:")
                parts.append(snapshot)
        except Exception:
            pass

        # Price performance (multi-horizon)
        try:
            perf = []
            with self.cache_lock:
                c = self.cache.get(sym, {}) or {}
            for lbl, yf_period in [("1W", "5d"), ("1M", "1mo"), ("3M", "3mo"), ("6M", "6mo"), ("1Y", "1y"), ("5Y", "5y"), ("10Y", "10y")]:
                h = c.get(f"history_{yf_period}")
                if h is not None and not h.empty and "Close" in h.columns:
                    arr = h["Close"].astype(float).dropna().values
                    if len(arr) >= 2 and arr[0] != 0:
                        ch = (arr[-1] - arr[0]) / abs(arr[0]) * 100
                        perf.append(f"{lbl}: {ch:+.2f}%")
            if perf:
                parts.append("Price Performance:")
                parts.append(", ".join(perf))
        except Exception:
            pass

        # Chart metrics (Quarterly trend)
        try:
            freq = "Quarterly"
            rev_q = self.get_series(self.df_fin_q, ["TotalRevenue", "OperatingRevenue", "Revenue"])
            rev_a = self.get_series(self.df_fin_a, ["TotalRevenue", "OperatingRevenue", "Revenue"])
            ebitda_q = self.get_series(self.df_fin_q, ["EBITDA", "NormalizedEBITDA"])
            ebitda_a = self.get_series(self.df_fin_a, ["EBITDA", "NormalizedEBITDA"])
            eps_q = self.get_series(self.df_fin_q, ["BasicEPS", "DilutedEPS"])
            eps_a = self.get_series(self.df_fin_a, ["BasicEPS", "DilutedEPS"])
            fcf_q = self.get_series(self.df_cf_q, ["FreeCashFlow", "Free Cash Flow"])
            fcf_a = self.get_series(self.df_cf_a, ["FreeCashFlow", "Free Cash Flow"])
            cash_q = self.get_series(self.df_bal_q, ["CashAndCashEquivalents", "Cash"])
            cash_a = self.get_series(self.df_bal_a, ["CashAndCashEquivalents", "Cash"])
            debt_q = self.get_series(self.df_bal_q, ["TotalDebt", "LongTermDebt"])
            debt_a = self.get_series(self.df_bal_a, ["TotalDebt", "LongTermDebt"])

            rev = self._series_for_flow(rev_q, rev_a, freq)
            ebitda = self._series_for_flow(ebitda_q, ebitda_a, freq)
            eps = self._series_for_flow(eps_q, eps_a, freq)
            fcf = self._series_for_flow(fcf_q, fcf_a, freq)
            cash = self._series_for_balance(cash_q, cash_a, freq)
            debt = self._series_for_balance(debt_q, debt_a, freq)
            divs = self._dividend_series(freq)

            lines = []
            if rev is not None and not rev.empty:
                lines.append(self._series_stats(rev.tail(8) / 1e9, "Revenue (last 8Q)", "B"))
            if ebitda is not None and not ebitda.empty:
                lines.append(self._series_stats(ebitda.tail(8) / 1e9, "EBITDA (last 8Q)", "B"))
            if eps is not None and not eps.empty:
                lines.append(self._series_stats(eps.tail(8), "EPS (last 8Q)", ""))
            if fcf is not None and not fcf.empty:
                lines.append(self._series_stats(fcf.tail(8) / 1e9, "FCF (last 8Q)", "B"))
            if cash is not None and not cash.empty:
                lines.append(self._series_stats(cash.tail(8) / 1e9, "Cash (last 8Q)", "B"))
            if debt is not None and not debt.empty:
                lines.append(self._series_stats(debt.tail(8) / 1e9, "Debt (last 8Q)", "B"))
            if divs is not None and not divs.empty:
                lines.append(self._series_stats(divs.tail(8), "Dividends (last 8Q)", ""))
            if lines:
                parts.append("Chart Metrics (Quarterly):")
                parts.extend(lines)
        except Exception:
            pass

        # Cached chart insights
        try:
            if self.chart_ai_cache:
                parts.append("Chart Insight Snippets:")
                for k in list(self.chart_ai_cache.keys())[-4:]:
                    txt = self.chart_ai_cache.get(k, "")
                    if txt:
                        cleaned = txt.replace("\n", " ").strip()
                        parts.append(f"- {k}: {cleaned[:240]}")
        except Exception:
            pass

        # Recent earnings releases/transcripts (titles only)
        try:
            rows = self.store.list_transcripts(sym)[:3]
            if rows:
                parts.append("Recent Transcripts / Releases:")
                for d, t, _ in rows:
                    parts.append(f"- {d}: {t}")
        except Exception:
            pass

        # Earnings calendar and insider activity from current tab data
        try:
            cal_rows = []
            for iid in self.tr_cal.get_children()[:4]:
                vals = self.tr_cal.item(iid).get("values", [])
                if vals:
                    cal_rows.append(f"- {vals[0]}: {vals[1]}")
            if cal_rows:
                parts.append("Upcoming/Recent Earnings Calendar Entries:")
                parts.extend(cal_rows)
        except Exception:
            pass
        try:
            ins_rows = []
            for iid in self.tr_insider.get_children()[:4]:
                vals = self.tr_insider.item(iid).get("values", [])
                if vals:
                    ins_rows.append(f"- {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]}")
            if ins_rows:
                parts.append("Recent Insider Activity:")
                parts.extend(ins_rows)
        except Exception:
            pass

        # Top recent headlines
        try:
            if self.current_ticker:
                news = self.current_ticker.news or []
                if news:
                    parts.append("Recent Headlines:")
                    for it in news[:5]:
                        ttl = it.get("title") or it.get("headline") or it.get("content", {}).get("title", "")
                        ts = it.get("providerPublishTime")
                        if ttl:
                            if ts:
                                d = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                                parts.append(f"- {d}: {ttl}")
                            else:
                                parts.append(f"- {ttl}")
        except Exception:
            pass

        return "\n".join(parts)

    def _gemini_model_candidates(self):
        preferred = [
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-1.5-flash",
            "models/gemini-pro",
        ]
        if self.gemini_models is None:
            try:
                self.gemini_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
            except Exception:
                self.gemini_models = []
        seen = set()
        out = []
        for m in preferred + self.gemini_models:
            if m and m not in seen:
                out.append(m)
                seen.add(m)
        return out

    def _gemini_generate_with_model(self, prompt: str):
        if not self.gemini_api_key:
            return None, None
        genai.configure(api_key=self.gemini_api_key)
        for m in self._gemini_model_candidates():
            try:
                model = genai.GenerativeModel(m)
                resp = model.generate_content(prompt)
                txt = getattr(resp, "text", None)
                if txt:
                    return txt, m
            except Exception:
                continue
        return None, None

    def run_gemini_25_logic(self, sym, info, f):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        
        context = self.build_ai_report_context(sym, info)
        prompt = (
            f"Role: Senior Equity Analyst.\n"
            f"Task: Provide a holistic analysis for {sym} based on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            "Output sections:\n"
            "1) Business Overview\n"
            "2) Recent Performance (price/metrics)\n"
            "3) Profitability & Cash Generation\n"
            "4) Balance Sheet & Liquidity\n"
            "5) Shareholder Returns (Dividends/Buybacks)\n"
            "6) Key Risks & Watch Items\n"
            "7) Summary (3-5 bullet takeaways)\n"
            "Do not give financial advice."
        )

        final_response, used_model = self._gemini_generate_with_model(prompt)
        if final_response:
            self.update_ai(f"MODEL: {used_model}\n\n{final_response}")
        else:
            self.update_ai("Analysis Unavailable.")

    def run_ai_report(self):
        if not self.current_symbol or not self.current_ticker:
            return
        self.submit_task(self.run_gemini_25_logic, self.current_symbol, self.current_ticker.info, self.df_fin_q, context="run_ai_report")

    def run_ai_company(self):
        # Backward-compatible alias
        self.run_ai_report()

    def run_ai_competitive(self):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        sym = self.current_symbol
        if not sym:
            return
        summary = ""
        if self.current_ticker:
            try:
                summary = self.current_ticker.info.get("longBusinessSummary", "")
            except:
                summary = ""
        prompt = f"Role: Senior Equity Analyst. Task: Provide Competitive Advantages and Investment Risks for {sym}. Use this company summary: {summary}. Output: Competitive Advantages, Key Risks, Moat Durability."
        self.submit_task(self.run_gemini_prompt, prompt, context="run_ai_competitive")

    def run_ai_earnings(self):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        if not self.current_symbol:
            return
        rows = self.store.list_transcripts(self.current_symbol)
        if not rows:
            self.update_ai("No transcripts found. Import transcripts into the transcripts folder.")
            return
        date, title, content = rows[0]
        prompt = f"Summarize earnings transcript for {self.current_symbol} ({date}). Provide sections: Revenue, Margins, Guidance, Risks, Q&A Highlights. Transcript: {content[:8000]}"
        self.submit_task(self.run_gemini_prompt, prompt, context="run_ai_earnings")

    def run_ai_portfolio(self):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        holdings = self.store.get_portfolio(self.current_portfolio)
        if not holdings:
            self.update_ai("Portfolio is empty.")
            return
        parts = []
        for sym, shares, cost in holdings:
            parts.append(f"{sym}: {shares} shares, cost basis {cost}")
        prompt = "Role: Portfolio Analyst. Task: Analyze portfolio qualities and risks. Holdings:\n" + "\n".join(parts)
        self.submit_task(self.run_gemini_prompt, prompt, context="run_ai_portfolio")

    def run_gemini_prompt(self, prompt: str):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        final_response, used_model = self._gemini_generate_with_model(prompt)
        if final_response:
            self.update_ai(f"MODEL: {used_model}\n\n{final_response}")
        else:
            self.update_ai("Analysis Unavailable.")

    def gemini_generate(self, prompt: str):
        text, _model = self._gemini_generate_with_model(prompt)
        return text

    def update_ai(self, t):
        # THREAD SAFETY FIX: Ensure UI update happens on main thread
        self.root.after(0, lambda: self._update_ai_safe(t))

    def _update_ai_safe(self, t):
        if not self.root.winfo_exists():
            return
        self.txt_ai.configure(state="normal")
        self.txt_ai.delete(1.0, tk.END)
        self.txt_ai.insert(tk.END, t)
        self.txt_ai.configure(state="disabled")

    # --- TAB 6: WATCHLIST ---
    def build_watchlist(self):
        cols = ("Symbol", "Price", "SMA200", "Status")
        
        container = ctk.CTkFrame(self.tab_watchlist, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20,10))

        top = ctk.CTkFrame(container, fg_color="transparent")
        top.pack(fill=tk.X, pady=(0,10))
        ctk.CTkLabel(top, text="Watchlist:", font=FONT_H2).pack(side=tk.LEFT, padx=(0,8))
        self.watchlist_name_var = tk.StringVar(value=self.current_watchlist)
        self.watchlist_menu = self.make_combo(top, variable=self.watchlist_name_var, values=self.store.list_watchlists() or ["Main"], width=140, command=self.on_watchlist_change, **COMBO_KW)
        self.watchlist_menu.pack(side=tk.LEFT)
        ctk.CTkButton(top, text="NEW LIST", command=self.create_watchlist, width=100, font=FONT_SMALL, **BTN_SECONDARY).pack(side=tk.LEFT, padx=8)
        
        tv_frame = ctk.CTkFrame(container, fg_color="transparent")
        tv_frame.pack(fill=tk.BOTH, expand=True)

        self.tr_watch = ttk.Treeview(tv_frame, columns=cols, show='headings')
        for c in cols: self.tr_watch.heading(c, text=c.upper()); self.tr_watch.column(c, anchor="center")
        
        # SCROLLBAR WATCHLIST
        vsb = ctk.CTkScrollbar(tv_frame, orientation="vertical", command=self.tr_watch.yview)
        self.tr_watch.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_watch.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.tr_watch.bind("<<TreeviewSelect>>", self.on_watchlist_select)
        
        b_frame = ctk.CTkFrame(self.tab_watchlist, fg_color="transparent")
        b_frame.pack(pady=(0,20))
        
        ctk.CTkButton(b_frame, text="REMOVE", command=self.remove_from_watchlist, width=100, font=FONT_SMALL, **BTN_DANGER).pack(side=tk.LEFT, padx=5)

        # Dip Finder Chart
        chart_frame = ctk.CTkFrame(self.tab_watchlist, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        chart_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0,20))
        self.fig_wl, self.ax_wl = plt.subplots(figsize=(6,3), dpi=100)
        self.canvas_wl = FigureCanvasTkAgg(self.fig_wl, master=chart_frame)
        self.canvas_wl.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.watchlist_data = self.load_watchlist()
        if self.watchlist_data: self.refresh_watchlist()

    # --- TAB 7: EARNINGS CALENDAR ---
    def build_earnings(self):
        container = ctk.CTkFrame(self.tab_earnings, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        top = ctk.CTkFrame(container, fg_color="transparent")
        top.pack(fill=tk.X, pady=(0,10))
        ctk.CTkLabel(top, text="Earnings Calendar (Active Symbol)", font=FONT_H2, text_color=C_ACCENT).pack(side=tk.LEFT)
        self.lbl_earn_status = ctk.CTkLabel(container, text="", text_color=C_TEXT_SUB)
        self.lbl_earn_status.pack(anchor="w", pady=(0,5))

        cols = ("Symbol", "Date", "Time", "EPS Est", "EPS Act", "Surprise")
        self.tr_earn = ttk.Treeview(container, columns=cols, show='headings')
        for c in cols: self.tr_earn.heading(c, text=c.upper()); self.tr_earn.column(c, width=110, anchor="center")
        vsb = ctk.CTkScrollbar(container, orientation="vertical", command=self.tr_earn.yview)
        self.tr_earn.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_earn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def refresh_earnings(self):
        for i in self.tr_earn.get_children(): self.tr_earn.delete(i)
        symbol = (self.current_symbol or "").strip().upper()
        if not symbol:
            self.earnings_source = "None"
            self.earnings_error = "No active symbol selected."
            self._render_earnings([])
            return
        self.submit_task(self._proc_earnings_for_symbol, symbol, context="_proc_earnings_for_symbol")

    def auto_refresh_earnings(self, symbol: str):
        for i in self.tr_earn.get_children(): self.tr_earn.delete(i)
        self.submit_task(self._proc_earnings_for_symbol, symbol, context="_proc_earnings_for_symbol")

    def _render_earnings(self, rows):
        if hasattr(self, "lbl_earn_status"):
            if rows:
                sym = (self.current_symbol or "N/A").upper()
                self.lbl_earn_status.configure(text=f"{sym} | Source: {getattr(self, 'earnings_source', 'Unknown')}")
            else:
                src = getattr(self, "earnings_source", "Unknown")
                err = getattr(self, "earnings_error", "")
                sym = (self.current_symbol or "N/A").upper()
                msg = f"No earnings entries found for {sym}. Source: {src}"
                if err:
                    msg = f"{msg} | Details: {err}"
                self.lbl_earn_status.configure(text=msg)
        for r in rows:
            self.tr_earn.insert("", tk.END, values=r)

    def _proc_earnings(self):
        symbol = (self.current_symbol or "").strip().upper()
        if not symbol:
            self.earnings_source = "None"
            self.earnings_error = "No active symbol selected."
            self.root.after(0, lambda: self._render_earnings([]))
            return
        self._proc_earnings_for_symbol(symbol)

    def _proc_earnings_for_symbol(self, symbol: str):
        rows = []
        sources = set()
        symbol = symbol.strip().upper()
        match_symbols = {symbol, symbol.split(".")[0]} if "." in symbol else {symbol}
        api_rows, api_source, api_error = self.fetch_calendar_api_rows()
        if api_rows:
            sources.add(api_source)
            rows.extend([r for r in api_rows if r[0] in match_symbols])
        if not rows:
            sym_rows, sym_source, sym_error = self.fetch_calendar_api_rows(symbol=symbol)
            if sym_rows:
                sources.add(sym_source)
                rows.extend(sym_rows)
            else:
                rows.extend(self.fetch_calendar_yfinance_rows(symbol))
                sources.add("yfinance")
        # Add past earnings for this symbol
        past_rows, past_source = self.fetch_past_earnings_rows(symbol)
        if past_rows:
            sources.add(past_source)
            rows.extend(past_rows)
        if rows:
            self.earnings_source = "+".join(sorted(sources))
            self.earnings_error = ""
        else:
            self.earnings_error = api_error or sym_error or f"No match for {symbol}"
        self.root.after(0, lambda rows=rows: self._render_earnings(rows))

    def fetch_calendar_api_rows(self, symbol: Optional[str] = None):
        # Finnhub first
        rows = []
        source = "None"
        error = ""
        now_ts = time.time()
        if symbol is None:
            cache = self.calendar_cache
            if cache and (now_ts - cache.get("ts", 0) < CALENDAR_CACHE_TTL_SEC):
                return cache.get("rows", []), cache.get("source", "None"), cache.get("error", "")
        else:
            cache = self.calendar_cache
            if cache and (now_ts - cache.get("ts", 0) < CALENDAR_CACHE_TTL_SEC):
                cached_rows = [r for r in cache.get("rows", []) if r[0] == symbol]
                if cached_rows:
                    return cached_rows, cache.get("source", "None"), cache.get("error", "")
        if FINNHUB_API_KEY:
            try:
                today = datetime.now().date()
                start = today.strftime("%Y-%m-%d")
                end = (today + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
                url = f"https://finnhub.io/api/v1/calendar/earnings?from={start}&to={end}&token={FINNHUB_API_KEY}"
                if symbol:
                    url = f"{url}&symbol={symbol}"
                text = self.http_get_text(url, headers=None, timeout=20)
                if not text:
                    error = "Finnhub request failed."
                else:
                    try:
                        fn = os.path.join(LOG_DIR, f"opentrim_finnhub_{symbol}.json") if symbol else os.path.join(LOG_DIR, "opentrim_finnhub.json")
                        with open(fn, "w") as f:
                            f.write(text)
                    except:
                        pass
                    data = json.loads(text)
                    if isinstance(data, dict) and data.get("error"):
                        error = f"Finnhub error: {data.get('error')}"
                    for r in data.get("earningsCalendar", []):
                        sym = (r.get("symbol", "") or "").upper()
                        date = r.get("date", "")
                        eps_est = r.get("epsEstimate", "N/A")
                        eps_act = r.get("epsActual", "N/A")
                        surprise = r.get("epsSurprise", "N/A")
                        rows.append((sym, date, "N/A", eps_est, eps_act, surprise))
                    if not rows and not error:
                        error = "Finnhub returned 0 rows."
            except:
                if not error:
                    error = "Finnhub parse error."
        if rows:
            source = "Finnhub"
            if symbol is None:
                self.calendar_cache = {"ts": now_ts, "rows": rows, "source": source, "error": error}
            return rows, source, error
        # Alpha Vantage fallback
        if ALPHAVANTAGE_API_KEY:
            try:
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={ALPHAVANTAGE_API_KEY}"
                text = self.http_get_text(url, headers=None, timeout=20)
                if not text:
                    if not error:
                        error = "Alpha Vantage request failed."
                else:
                    if text.strip().startswith("{"):
                        try:
                            msg = json.loads(text)
                            error = msg.get("Error Message") or msg.get("Note") or msg.get("Information") or "Alpha Vantage error."
                        except:
                            error = "Alpha Vantage error."
                    else:
                        df = pd.read_csv(io.StringIO(text))
                        if "symbol" in df.columns:
                            for _, r in df.iterrows():
                                sym = (r.get("symbol", "") or "").upper()
                                date = r.get("reportDate", "")
                                tm = r.get("reportTime", "N/A")
                                eps_est = r.get("estimate", "N/A")
                                eps_act = r.get("reportedEPS", "N/A")
                                rows.append((sym, date, tm, eps_est, eps_act, "N/A"))
                        else:
                            error = "Alpha Vantage returned unexpected format."
            except:
                if not error:
                    error = "Alpha Vantage parse error."
        if rows:
            source = "Alpha Vantage"
        if symbol is None and rows:
            self.calendar_cache = {"ts": now_ts, "rows": rows, "source": source, "error": error}
        return rows, source, error

    def fetch_calendar_yfinance_rows(self, symbol: str):
        rows = []
        try:
            symbol = symbol.upper()
            t = yf.Ticker(symbol)
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                ed = ed.head(6)
                for idx, r in ed.iterrows():
                    dt = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
                    tm = r.get("Time", "N/A") if hasattr(r, "get") else "N/A"
                    est = r.get("EPS Estimate", "N/A") if hasattr(r, "get") else "N/A"
                    act = r.get("Reported EPS", "N/A") if hasattr(r, "get") else "N/A"
                    surprise = r.get("Surprise(%)", "N/A") if hasattr(r, "get") else "N/A"
                    rows.append((symbol, dt, tm, est, act, surprise))
        except:
            pass
        if rows:
            self.earnings_source = "yfinance"
        return rows

    def fetch_past_earnings_rows(self, symbol: str):
        rows = []
        source = "FinnhubPast"
        if FINNHUB_API_KEY:
            try:
                url = f"https://finnhub.io/api/v1/stock/earnings?symbol={symbol}&limit={PAST_EARNINGS_COUNT}&token={FINNHUB_API_KEY}"
                text = self.http_get_text(url, headers=None, timeout=20)
                if text:
                    data = json.loads(text)
                    for r in data:
                        date = r.get("period", "")
                        eps_est = r.get("estimate", "N/A")
                        eps_act = r.get("actual", "N/A")
                        surprise = r.get("surprise", "N/A")
                        rows.append((symbol, date, "HIST", eps_est, eps_act, surprise))
            except:
                pass
        return rows, source

    def _collect_release_rows(self, symbol: str):
        rows = []
        try:
            rels = self.store.list_transcripts(symbol)
            for d, t, _ in rels:
                if isinstance(t, str) and t.startswith("Earnings Release"):
                    rows.append((symbol, d or "N/A", "SEC 8-K", t))
        except:
            pass
        return rows

    # --- TAB 8: TRANSCRIPTS ---
    def build_transcripts(self):
        container = ctk.CTkFrame(self.tab_transcripts, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        top = ctk.CTkFrame(container, fg_color="transparent")
        top.pack(fill=tk.X, pady=(0,10))
        ctk.CTkLabel(top, text="Transcripts", font=FONT_H2, text_color=C_ACCENT).pack(side=tk.LEFT)
        ctk.CTkButton(top, text="IMPORT", command=self.import_transcripts, width=100, font=FONT_SMALL, **BTN_SECONDARY).pack(side=tk.RIGHT)

        search_row = ctk.CTkFrame(container, fg_color="transparent")
        search_row.pack(fill=tk.X, pady=(0,10))
        self.transcript_query = tk.StringVar()
        ctk.CTkEntry(search_row, textvariable=self.transcript_query, width=200, placeholder_text="Search transcripts", **ENTRY_KW).pack(side=tk.LEFT)
        ctk.CTkButton(search_row, text="SEARCH", command=self.search_transcripts, width=80, font=FONT_SMALL, **BTN_PRIMARY).pack(side=tk.LEFT, padx=8)

        cols = ("Date", "Title")
        self.tr_trans = ttk.Treeview(container, columns=cols, show='headings', height=8)
        for c in cols: self.tr_trans.heading(c, text=c.upper()); self.tr_trans.column(c, width=150, anchor="center")
        self.tr_trans.pack(fill=tk.X, expand=False)
        self.tr_trans.bind("<<TreeviewSelect>>", lambda e: self.show_transcript())

        self.txt_transcript = ctk.CTkTextbox(container, fg_color=C_BG_INPUT, font=FONT_BODY, wrap="word")
        self.txt_transcript.pack(fill=tk.BOTH, expand=True, pady=(10,0))

        action_row = ctk.CTkFrame(container, fg_color="transparent")
        action_row.pack(fill=tk.X, pady=(10,5))
        self.lbl_transcript_status = ctk.CTkLabel(action_row, text="", text_color=C_TEXT_SUB)
        self.lbl_transcript_status.pack(side=tk.LEFT, padx=8)

        self.txt_transcript_ai = ctk.CTkTextbox(container, fg_color=C_BG_INPUT, font=FONT_BODY, wrap="word")
        self.txt_transcript_ai.pack(fill=tk.BOTH, expand=True, pady=(5,0))

    def _format_transcript_text(self, text: str):
        if not text:
            return ""
        t = str(text).replace("\r\n", "\n").replace("\r", "\n")
        t = html.unescape(t)
        # Preserve structure from HTML-like transcript content.
        t = re.sub(r"(?i)<br\\s*/?>", "\n", t)
        t = re.sub(r"(?i)</(p|div|tr|li|h1|h2|h3|h4|h5|h6)>", "\n", t)
        t = re.sub(r"(?i)<li[^>]*>", "• ", t)
        t = re.sub(r"(?i)<[^>]+>", " ", t)
        t = re.sub(r"[ \t]+", " ", t)

        # If transcript is effectively one paragraph, insert readable breaks.
        if t.count("\n") <= 2 and len(t) > 600:
            t = re.sub(r"(?<!\n)(\b(?:Operator|Q&A|Questions\\s+and\\s+Answers?|Prepared\\s+Remarks?|Forward-Looking\\s+Statements?)\\b)",
                       r"\n\n\1", t, flags=re.IGNORECASE)
            t = re.sub(r"(?<!\n)([A-Z][A-Za-z\\.'\\- ]{2,60}:)", r"\n\1", t)
            t = re.sub(r"([\\.!?])\\s+(?=[A-Z])", r"\1\n", t)

        # Normalize spacing and wrap long paragraphs for readability.
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        paragraphs = [p.strip() for p in t.split("\n") if p.strip()]
        wrapped = []
        for p in paragraphs:
            if len(p) > 180 and ":" not in p[:80]:
                wrapped.append(textwrap.fill(p, width=120))
            else:
                wrapped.append(p)
        return "\n\n".join(wrapped)

    def import_transcripts(self):
        if not os.path.exists(TRANSCRIPTS_DIR):
            return
        for fn in os.listdir(TRANSCRIPTS_DIR):
            if not (fn.endswith(".txt") or fn.endswith(".md")):
                continue
            path = os.path.join(TRANSCRIPTS_DIR, fn)
            try:
                with open(path, "r") as f:
                    content = f.read()
                parts = fn.split("_")
                symbol = parts[0].upper() if parts else ""
                date = ""
                title = fn
                if len(parts) >= 2:
                    date = parts[1]
                if not self.store.transcript_exists(symbol, date, title):
                    self.store.upsert_transcript(symbol, date, title, content)
            except:
                continue
        self.load_transcripts()

    def load_transcripts(self):
        for i in self.tr_trans.get_children(): self.tr_trans.delete(i)
        if not self.current_symbol:
            return
        rows = self.store.list_transcripts(self.current_symbol)
        for d, t, _ in rows:
            self.tr_trans.insert("", tk.END, values=(d, t))
        if not rows:
            self.txt_transcript.configure(state="normal")
            self.txt_transcript.delete(1.0, tk.END)
            msg = "No legal transcripts or earnings releases found for this stock."
            if self.transcript_fetch_status:
                msg = f"{msg}\n\nDetails: {self.transcript_fetch_status}"
            self.txt_transcript.insert(tk.END, msg)
            self.txt_transcript.configure(state="disabled")
            self.txt_transcript_ai.configure(state="normal")
            self.txt_transcript_ai.delete(1.0, tk.END)
            self.txt_transcript_ai.insert(tk.END, "No transcript analysis available.")
            self.txt_transcript_ai.configure(state="disabled")
        else:
            # Auto-select the most recent transcript
            first = self.tr_trans.get_children()[0]
            self.tr_trans.selection_set(first)
            self.show_transcript()
        if hasattr(self, "lbl_transcript_status"):
            self.lbl_transcript_status.configure(text="")

    def search_transcripts(self):
        for i in self.tr_trans.get_children(): self.tr_trans.delete(i)
        if not self.current_symbol:
            return
        q = self.transcript_query.get().strip()
        rows = self.store.search_transcripts(self.current_symbol, q) if q else self.store.list_transcripts(self.current_symbol)
        for d, t, _ in rows:
            self.tr_trans.insert("", tk.END, values=(d, t))

    def show_transcript(self):
        sel = self.tr_trans.selection()
        if not sel:
            return
        date, title = self.tr_trans.item(sel[0])['values']
        rows = self.store.list_transcripts(self.current_symbol)
        if rows:
            content = ""
            for d, t, c in rows:
                if t == title and d == date:
                    content = c
                    break
            display_content = self._format_transcript_text(content)
            self.txt_transcript.configure(state="normal")
            self.txt_transcript.delete(1.0, tk.END)
            self.txt_transcript.insert(tk.END, display_content)
            self.txt_transcript.configure(state="disabled")
            self.load_transcript_analysis(date, title)
            if not self.store.get_transcript_analysis(self.current_symbol, date, title):
                self.analyze_selected_transcript()

    def refresh_transcripts(self):
        if not self.current_symbol:
            return
        self.transcript_fetch_status = ""
        self.pending_transcript_fetches = 3
        self.lbl_transcript_status.configure(text="Refreshing...")
        self.submit_task(self.fetch_sec_earnings_release, self.current_symbol, self.request_id, True, context="refresh_sec_earnings_release")
        self.submit_task(self.fetch_av_transcripts, self.current_symbol, self.request_id, True, context="refresh_av_transcripts")
        self.submit_task(self.fetch_api_ninjas_transcripts, self.current_symbol, self.request_id, True, context="refresh_api_ninjas_transcripts")

    def _mark_transcript_fetch_done(self, source: str):
        try:
            self.pending_transcript_fetches = max(0, getattr(self, "pending_transcript_fetches", 1) - 1)
            if self.pending_transcript_fetches == 0:
                self.lbl_transcript_status.configure(text=f"Refresh complete ({source})")
        except:
            pass

    def write_api_log(self, name: str, symbol: str, text: str):
        try:
            safe = symbol.replace("/", "_")
            path = os.path.join(LOG_DIR, f"opentrim_{name}_{safe}.log")
            with open(path, "w") as f:
                f.write(text)
        except:
            pass

    def load_transcript_analysis(self, date: str, title: str):
        res = self.store.get_transcript_analysis(self.current_symbol, date, title)
        self.txt_transcript_ai.configure(state="normal")
        self.txt_transcript_ai.delete(1.0, tk.END)
        if res:
            summary, qa = res
            self.txt_transcript_ai.insert(tk.END, "SUMMARY\n\n")
            self.txt_transcript_ai.insert(tk.END, summary or "N/A")
            self.txt_transcript_ai.insert(tk.END, "\n\nQ&A\n\n")
            self.txt_transcript_ai.insert(tk.END, qa or "N/A")
        else:
            self.txt_transcript_ai.insert(tk.END, "No transcript analysis available.")
        self.txt_transcript_ai.configure(state="disabled")

    def analyze_selected_transcript(self):
        sel = self.tr_trans.selection()
        if not sel or not self.current_symbol:
            return
        date, title = self.tr_trans.item(sel[0])['values']
        rows = self.store.list_transcripts(self.current_symbol)
        content = None
        for d, t, c in rows:
            if t == title and d == date:
                content = c
                break
        if not content:
            return
        # Avoid re-analysis if already exists
        if self.store.get_transcript_analysis(self.current_symbol, date, title):
            self.load_transcript_analysis(date, title)
            return
        info = {}
        try:
            if self.current_ticker:
                info = self.current_ticker.info or {}
        except:
            info = {}
        news_ctx = self.get_recent_news_context(info)
        fin_ctx = self.get_financial_context()
        self.lbl_transcript_status.configure(text="Analyzing...")
        self.txt_transcript_ai.configure(state="normal")
        self.txt_transcript_ai.delete(1.0, tk.END)
        self.txt_transcript_ai.insert(tk.END, "Analyzing transcript + recent news + financials with Gemini...")
        self.txt_transcript_ai.configure(state="disabled")
        self.submit_task(self._analyze_transcript_worker, self.current_symbol, date, title, content, news_ctx, fin_ctx, info, context="_analyze_transcript_worker")

    def _analyze_transcript_worker(self, symbol: str, date: str, title: str, content: str, news_ctx: str, fin_ctx: str, info: Dict[str, Any]):
        if not self.gemini_api_key:
            self.root.after(0, lambda: self.lbl_transcript_status.configure(text="Gemini API key missing"))
            return
        qa_section = self.extract_qa_section(content)
        transcript_text = content
        if len(content) > 12000:
            transcript_text = content[:8000] + "\n\n[...]\n\n" + content[-4000:]
        company_summary = info.get("longBusinessSummary", "") if isinstance(info, dict) else ""
        prompt = (
            f"Analyze the following earnings transcript for {symbol} ({date}). "
            f"Provide a detailed summary and list every question with its answer in order. "
            f"If a question has multiple answers, combine them. "
            f"Summary should include revenue, margins, guidance, risks, and key themes. "
            f"Also incorporate recent news context and a concise financial snapshot.\n\n"
            f"COMPANY SUMMARY:\n{company_summary}\n\n"
            f"RECENT NEWS (last 30 days):\n{news_ctx}\n\n"
            f"FINANCIAL SNAPSHOT:\n{fin_ctx}\n\n"
            f"TRANSCRIPT SUMMARY SOURCE:\n{transcript_text}\n\n"
            f"Q&A SECTION (if available):\n{qa_section}\n\n"
            f"Output format:\nSUMMARY:\n...\n\nNEWS CONTEXT:\n...\n\nFINANCIAL CONTEXT:\n...\n\nQ&A:\n1) Q: ... A: ...\n"
        )
        result = self.gemini_generate(prompt)
        if not result:
            self.root.after(0, lambda: self.lbl_transcript_status.configure(text="Analysis failed"))
            return
        summary, qa = self.split_summary_qa(result)
        self.store.upsert_transcript_analysis(symbol, date, title, summary, qa)
        self.root.after(0, lambda: self._finish_transcript_analysis(date, title))

    def _finish_transcript_analysis(self, date: str, title: str):
        self.lbl_transcript_status.configure(text="Analysis complete")
        self.load_transcript_analysis(date, title)

    def extract_qa_section(self, text: str):
        markers = ["questions and answers", "question-and-answer", "q&a", "q & a"]
        lower = text.lower()
        for m in markers:
            idx = lower.find(m)
            if idx != -1:
                return text[idx:]
        return "No explicit Q&A section found."

    def split_summary_qa(self, text: str):
        if "Q&A" in text:
            parts = text.split("Q&A", 1)
            summary = parts[0].strip()
            qa = parts[1].strip()
            return summary, qa
        return text.strip(), "No Q&A section detected."

    def get_recent_news_context(self, info: Dict[str, Any]):
        items = []
        try:
            if self.current_ticker:
                news_data = self.current_ticker.news or []
            else:
                news_data = []
            symbol = (self.current_symbol or "").upper()
            short_name = (info.get("shortName", "") or "").split(" ")[0].upper()
            now_ts = time.time()
            for item in news_data:
                ts = item.get('providerPublishTime', 0)
                if ts and (now_ts - ts) > 30 * 86400:
                    continue
                title = item.get('title') or item.get('headline') or item.get('content', {}).get('title', '')
                summary = item.get('summary') or item.get('content', {}).get('summary', '')
                if symbol and symbol not in (title + " " + summary).upper() and short_name and short_name not in (title + " " + summary).upper():
                    continue
                d = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else "Recent"
                items.append(f"{d}: {title}")
                if len(items) >= 5:
                    break
        except:
            pass
        return "\n".join(items) if items else "No recent news found."

    def get_financial_context(self):
        parts = []
        try:
            fin = self.df_fin_q
            if fin is not None and not fin.empty:
                rev = self.get_series(fin, ["TotalRevenue", "Revenue"])
                ni = self.get_series(fin, ["NetIncome"])
                eps = self.get_series(fin, ["BasicEPS", "DilutedEPS"])
                if rev is not None:
                    parts.append(f"Latest Revenue: {rev.iloc[-1]:,.0f}")
                if ni is not None:
                    parts.append(f"Latest Net Income: {ni.iloc[-1]:,.0f}")
                if eps is not None:
                    parts.append(f"Latest EPS: {eps.iloc[-1]:.2f}")
            bal = self.df_bal_q
            if bal is not None and not bal.empty:
                cash = self.get_series(bal, ["CashAndCashEquivalents", "Cash"])
                debt = self.get_series(bal, ["TotalDebt", "LongTermDebt"])
                if cash is not None:
                    parts.append(f"Cash: {cash.iloc[-1]:,.0f}")
                if debt is not None:
                    parts.append(f"Debt: {debt.iloc[-1]:,.0f}")
            cf = self.df_cf_q
            if cf is not None and not cf.empty:
                fcf = self.get_series(cf, ["FreeCashFlow", "Free Cash Flow"])
                if fcf is not None:
                    parts.append(f"Free Cash Flow: {fcf.iloc[-1]:,.0f}")
        except:
            pass
        return "\n".join(parts) if parts else "Financial snapshot unavailable."

    # --- TAB 9: PORTFOLIO ---
    def build_portfolio(self):
        container = ctk.CTkFrame(self.tab_portfolio, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        top = ctk.CTkFrame(container, fg_color="transparent")
        top.pack(fill=tk.X, pady=(0,10))
        ctk.CTkLabel(top, text="Portfolio:", font=FONT_H2).pack(side=tk.LEFT, padx=(0,8))
        self.portfolio_name_var = tk.StringVar(value=self.current_portfolio)
        self.portfolio_menu = self.make_combo(top, variable=self.portfolio_name_var, values=self.store.list_portfolios() or ["Main"], width=140, command=self.on_portfolio_change, **COMBO_KW)
        self.portfolio_menu.pack(side=tk.LEFT)
        ctk.CTkButton(top, text="NEW PORTFOLIO", command=self.create_portfolio, width=120, font=FONT_SMALL, **BTN_SECONDARY).pack(side=tk.LEFT, padx=8)

        cols = ("Symbol", "Shares", "Cost Basis", "Price", "Value")
        self.tr_port = ttk.Treeview(container, columns=cols, show='headings')
        for c in cols: self.tr_port.heading(c, text=c.upper()); self.tr_port.column(c, width=110, anchor="center")
        vsb = ctk.CTkScrollbar(container, orientation="vertical", command=self.tr_port.yview)
        self.tr_port.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_port.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        form = ctk.CTkFrame(container, fg_color="transparent")
        form.pack(fill=tk.X, pady=(10,0))
        self.port_sym = tk.StringVar(); self.port_shares = tk.StringVar(); self.port_cost = tk.StringVar()
        ctk.CTkEntry(form, textvariable=self.port_sym, width=100, placeholder_text="Symbol", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.port_shares, width=120, placeholder_text="Shares", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.port_cost, width=120, placeholder_text="Cost Basis", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="ADD/UPDATE", command=self.add_portfolio, width=120, font=FONT_SMALL, **BTN_PRIMARY).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="REMOVE", command=self.remove_portfolio, width=100, font=FONT_SMALL, **BTN_DANGER).pack(side=tk.LEFT, padx=5)

        self.lbl_port_total = ctk.CTkLabel(container, text="Total: $0.00", font=FONT_H2)
        self.lbl_port_total.pack(anchor="e", pady=(10,0))
        self.refresh_portfolio()

    def on_portfolio_change(self, _=None):
        self.current_portfolio = self.portfolio_name_var.get().strip()
        self.refresh_portfolio()
        self.refresh_earnings()

    def create_portfolio(self):
        dlg = ctk.CTkInputDialog(title="New Portfolio", text="Portfolio name:")
        name = dlg.get_input()
        if not name:
            return
        name = name.strip()
        if not name:
            return
        lists = self.store.list_portfolios()
        if name not in lists:
            self.store.upsert_portfolio_symbol(name, "__INIT__", 0, 0)
            self.store.remove_portfolio_symbol(name, "__INIT__")
        self.portfolio_menu.configure(values=self.store.list_portfolios())
        self.portfolio_name_var.set(name)
        self.on_portfolio_change()

    def refresh_portfolio(self):
        self.submit_task(self._proc_portfolio, context="_proc_portfolio")

    def _proc_portfolio(self):
        rows = []
        total = 0.0
        for sym, shares, cost in self.store.get_portfolio(self.current_portfolio):
            price = None
            try:
                with self.cache_lock:
                    price = (self.cache.get(sym, {}) or {}).get("info", {}).get("currentPrice")
            except Exception as e:
                log_exception(f"_proc_portfolio.cache:{sym}", e)
                price = None
            if price is None:
                try:
                    info = yf.Ticker(sym).info
                    price = info.get("currentPrice")
                    with self.cache_lock:
                        cached = self.cache.get(sym, {})
                        self.cache[sym] = {**cached, "info": info}
                except Exception as e:
                    log_exception(f"_proc_portfolio.fetch:{sym}", e)
                    price = 0
            value = (shares or 0) * (price or 0)
            total += value
            rows.append((sym, shares, cost, price or 0, value))
        self.root.after(0, lambda rows=rows, total=total: self._render_portfolio(rows, total))

    def _render_portfolio(self, rows, total):
        for i in self.tr_port.get_children(): self.tr_port.delete(i)
        for sym, shares, cost, price, value in rows:
            self.tr_port.insert("", tk.END, values=(sym, shares, f"{cost:.2f}", f"{price:.2f}", f"{value:.2f}"))
        self.lbl_port_total.configure(text=f"Total: ${total:,.2f}")

    def add_portfolio(self):
        sym = self.port_sym.get().strip().upper()
        if not sym:
            return
        try:
            shares = float(self.port_shares.get())
            cost = float(self.port_cost.get())
        except Exception:
            return
        self.store.upsert_portfolio_symbol(self.current_portfolio, sym, shares, cost)
        self.refresh_portfolio()
        self.refresh_earnings()

    def remove_portfolio(self):
        sel = self.tr_port.selection()
        if not sel:
            return
        sym = self.tr_port.item(sel[0])['values'][0]
        self.store.remove_portfolio_symbol(self.current_portfolio, sym)
        self.refresh_portfolio()
        self.refresh_earnings()

    # --- TAB 10: KPIS ---
    def build_kpis(self):
        container = ctk.CTkFrame(self.tab_kpis, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        form = ctk.CTkFrame(container, fg_color="transparent")
        form.pack(fill=tk.X, pady=(0,10))
        self.kpi_name = tk.StringVar(); self.kpi_period = tk.StringVar(); self.kpi_value = tk.StringVar(); self.kpi_unit = tk.StringVar()
        ctk.CTkEntry(form, textvariable=self.kpi_name, width=140, placeholder_text="KPI Name", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.kpi_period, width=120, placeholder_text="Period", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.kpi_value, width=120, placeholder_text="Value", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.kpi_unit, width=80, placeholder_text="Unit", **ENTRY_KW).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="ADD KPI", command=self.add_kpi, width=100, font=FONT_SMALL, **BTN_PRIMARY).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="CLEAR", command=self.clear_kpis, width=80, font=FONT_SMALL, **BTN_SECONDARY).pack(side=tk.LEFT, padx=5)

        cols = ("Name", "Period", "Value", "Unit")
        self.tr_kpi = ttk.Treeview(container, columns=cols, show='headings')
        for c in cols: self.tr_kpi.heading(c, text=c.upper()); self.tr_kpi.column(c, width=140, anchor="center")
        self.tr_kpi.pack(fill=tk.BOTH, expand=True)

    def load_kpis(self):
        for i in self.tr_kpi.get_children(): self.tr_kpi.delete(i)
        if not self.current_symbol:
            return
        for name, period, value, unit in self.store.list_kpis(self.current_symbol):
            self.tr_kpi.insert("", tk.END, values=(name, period, value, unit))

    def add_kpi(self):
        if not self.current_symbol:
            return
        self.store.add_kpi(self.current_symbol, self.kpi_name.get(), self.kpi_period.get(), self.kpi_value.get(), self.kpi_unit.get())
        self.load_kpis()

    def clear_kpis(self):
        if not self.current_symbol:
            return
        self.store.clear_kpis(self.current_symbol)
        self.load_kpis()

    def update_kpis_from_sec(self, fin_q, bal_q, cf_q):
        if not self.current_symbol:
            return
        period = "Latest (SEC)"
        self.store.clear_kpis_for_period(self.current_symbol, period)
        def latest_val(series):
            if series is None or series.empty:
                return None
            return series.sort_index().iloc[-1]

        rev = self.get_series(fin_q, ["TotalRevenue", "Revenue"])
        eps = self.get_series(fin_q, ["BasicEPS", "DilutedEPS"])
        ni = self.get_series(fin_q, ["NetIncome"])
        op = self.get_series(fin_q, ["OperatingIncome", "OperatingIncomeLoss"])
        fcf = self.get_series(cf_q, ["FreeCashFlow", "Free Cash Flow"])
        cash = self.get_series(bal_q, ["CashAndCashEquivalents", "Cash"])
        debt = self.get_series(bal_q, ["TotalDebt", "LongTermDebt"])

        kv = [
            ("Revenue", latest_val(rev), "USD"),
            ("EPS", latest_val(eps), "USD/share"),
            ("Net Income", latest_val(ni), "USD"),
            ("Operating Income", latest_val(op), "USD"),
            ("Free Cash Flow", latest_val(fcf), "USD"),
            ("Cash", latest_val(cash), "USD"),
            ("Debt", latest_val(debt), "USD"),
        ]
        for name, val, unit in kv:
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                continue
            self.store.add_kpi(self.current_symbol, name, period, f"{val:.2f}", unit)

    def load_watchlist(self):
        return self.store.get_watchlist(self.current_watchlist)
    def save_watchlist(self):
        # no-op: stored in DB
        pass
    def add_to_watchlist(self):
        if not self.current_symbol:
            return
        symbol = self.current_symbol.strip().upper()
        self.store.add_watchlist_symbol(self.current_watchlist, symbol)
        self.watchlist_data = self.load_watchlist()
        self.refresh_watchlist()
        self.refresh_earnings()
        self.lbl_status.configure(text=f"Added {symbol} to watchlist", text_color=C_GREEN)
    def remove_from_watchlist(self):
        sel = self.tr_watch.selection(); 
        if not sel: return
        sym = self.tr_watch.item(sel[0])['values'][0]
        if sym in self.watchlist_data:
            self.watchlist_data.remove(sym)
            self.store.remove_watchlist_symbol(self.current_watchlist, sym)
            self.refresh_watchlist()
            self.refresh_earnings()
    def analyze_watchlist(self):
        sel = self.tr_watch.selection()
        if not sel: return
        self.search_var.set(self.tr_watch.item(sel[0])['values'][0]); self.start_search()

    def on_watchlist_select(self, _event=None):
        self.update_watchlist_chart()
        sel = self.tr_watch.selection()
        if not sel:
            return
        sym = str(self.tr_watch.item(sel[0])['values'][0]).upper().strip()
        if not sym:
            return
        if sym == (self.current_symbol or "").upper():
            return
        self.search_var.set(sym)
        self.start_search()

    def on_watchlist_change(self, _=None):
        name = self.watchlist_name_var.get().strip()
        if not name:
            return
        self.current_watchlist = name
        self.watchlist_data = self.load_watchlist()
        self.refresh_watchlist()
        self.refresh_earnings()

    def create_watchlist(self):
        dlg = ctk.CTkInputDialog(title="New Watchlist", text="Watchlist name:")
        name = dlg.get_input()
        if not name:
            return
        name = name.strip()
        if not name:
            return
        lists = self.store.list_watchlists()
        if name not in lists:
            self.store.add_watchlist_symbol(name, "__INIT__")
            self.store.remove_watchlist_symbol(name, "__INIT__")
        self.watchlist_menu.configure(values=self.store.list_watchlists())
        self.watchlist_name_var.set(name)
        self.on_watchlist_change()
    
    def refresh_watchlist(self):
        if self.watchlist_loading: return # Prevent double refresh
        self.watchlist_loading = True
        self.submit_task(self._proc_watch, context="_proc_watch")
    
    def _proc_watch(self):
        rows = []
        try:
            symbols = [str(s).upper().strip() for s in self.watchlist_data if isinstance(s, str) and s.strip()]
            if not symbols:
                self.root.after(0, lambda: self._render_watchlist([]))
                return
            yf_map = {s: (s.split(".")[0] if "." in s else s) for s in symbols}
            yf_symbols = [yf_map[s] for s in symbols]

            batch_hist: Dict[str, pd.DataFrame] = {}
            try:
                downloaded = yf.download(
                    tickers=" ".join(yf_symbols),
                    period="1y",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if downloaded is not None and not downloaded.empty:
                    if len(symbols) == 1:
                        df = downloaded.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            level0 = [str(x).upper() for x in df.columns.get_level_values(0)]
                            key = yf_map[symbols[0]].upper()
                            if key in level0:
                                df = df[key]
                        batch_hist[symbols[0]] = df
                    elif isinstance(downloaded.columns, pd.MultiIndex):
                        l0 = {str(x).upper() for x in downloaded.columns.get_level_values(0)}
                        for sym in symbols:
                            key = yf_map[sym].upper()
                            if key in l0:
                                batch_hist[sym] = downloaded[key].copy()
            except Exception as e:
                log_exception("_proc_watch.batch_download", e)

            for s in symbols:
                try:
                    h = batch_hist.get(s)
                    if h is None or h.empty:
                        with self.cache_lock:
                            cached = self.cache.get(s, {})
                        h = cached.get("history_1y")
                        if h is None or h.empty:
                            h = yf.Ticker(yf_map.get(s, s)).history(period="1y")
                    h = self.calculate_technical_indicators(h)
                    with self.cache_lock:
                        cached = self.cache.get(s, {})
                        self.cache[s] = {**cached, "history_1y": h}
                    if h is None or h.empty or "Close" not in h.columns:
                        rows.append((s, np.nan, np.nan, "N/A", "up"))
                        continue
                    close = pd.to_numeric(h["Close"], errors="coerce").dropna()
                    if close.empty:
                        rows.append((s, np.nan, np.nan, "N/A", "up"))
                        continue
                    c = float(close.iloc[-1])
                    sma = float(close.rolling(200, min_periods=1).mean().iloc[-1])
                    st, tag = ("DIP", "dip") if c < sma else ("UP", "up")
                    rows.append((s, c, sma, st, tag))
                except Exception as e:
                    log_exception(f"_proc_watch.symbol:{s}", e)
                    rows.append((s, np.nan, np.nan, "N/A", "up"))
            self.root.after(0, lambda rows=rows: self._render_watchlist(rows))
        except Exception as e:
            log_exception("_proc_watch", e)
        finally:
            self.watchlist_loading = False # Ensure flag is reset
            
    def _render_watchlist(self, rows):
        if not self.root.winfo_exists():
            return
        for i in self.tr_watch.get_children():
            self.tr_watch.delete(i)
        if not rows:
            return
        for s, c, sm, st, t in rows:
            self._add_watch_row(s, c, sm, st, t)
        first = self.tr_watch.get_children()
        if first:
            self.tr_watch.selection_set(first[0])
            self.on_watchlist_select()

    def _add_watch_row(self, s, c, sm, st, t):
        if not self.root.winfo_exists():
            return
        c_txt = f"${c:.2f}" if isinstance(c, (int, float, np.floating)) and np.isfinite(c) else "N/A"
        sm_txt = f"${sm:.2f}" if isinstance(sm, (int, float, np.floating)) and np.isfinite(sm) else "N/A"
        i = self.tr_watch.insert("", tk.END, values=(s, c_txt, sm_txt, st))
        bg = COLOR_DIP_BG if t=='dip' else COLOR_UP_BG
        self.tr_watch.tag_configure(t, background=bg, foreground="white")
        self.tr_watch.item(i, tags=(t,))
        if len(self.tr_watch.get_children()) == 1:
            self.update_watchlist_chart()

    def update_watchlist_chart(self):
        if not hasattr(self, "ax_wl"):
            return
        sel = self.tr_watch.selection()
        if not sel:
            return
        sym = self.tr_watch.item(sel[0])['values'][0]
        with self.cache_lock:
            h = (self.cache.get(sym, {}) or {}).get("history_1y")
        if h is None or h.empty:
            return
        self.ax_wl.clear()
        close = h['Close']
        sma = close.rolling(200).mean()
        color = C_GREEN if close.iloc[-1] >= sma.iloc[-1] else C_RED
        self.ax_wl.plot(close.index, close.values, color=color, linewidth=1.5, label="Price")
        self.ax_wl.plot(sma.index, sma.values, color=C_YELLOW, linewidth=1.2, label="SMA200")
        self.ax_wl.set_title(f"Dip Finder: {sym}", fontweight="bold")
        self.ax_wl.legend()
        self.canvas_wl.draw_idle()

    def on_close(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            log_exception("on_close.executor_shutdown", e)
        try:
            if hasattr(self, "fig"):
                plt.close(self.fig)
            if hasattr(self, "fig_wl"):
                plt.close(self.fig_wl)
        except Exception as e:
            log_exception("on_close.plt_close", e)
        self.root.destroy()
        sys.exit()

    def ensure_history_period(self, period):
        if not self.current_symbol:
            return
        if self.hist_period == period:
            return
        with self.cache_lock:
            cached = self.cache.get(self.current_symbol, {})
        if period == "5y" and "history_5y" in cached:
            self.hist_data = cached["history_5y"]
            self.hist_period = "5y"
            self.redraw_chart()
            return
        self.submit_task(self._fetch_history_period, self.current_symbol, period, self.request_id, context="_fetch_history_period")

    def _fetch_history_period(self, symbol, period, req_id):
        try:
            with self.cache_lock:
                t = (self.cache.get(symbol, {}) or {}).get("ticker")
            t = t or yf.Ticker(symbol)
            h = t.history(period=period)
            h = self.calculate_technical_indicators(h)
            with self.cache_lock:
                cached = self.cache.get(symbol, {})
            key = f"history_{period}"
            with self.cache_lock:
                self.cache[symbol] = {**cached, "ticker": t, key: h}
            self.root.after(0, lambda: self._update_history_period(h, period, req_id))
        except Exception as e:
            log_exception("_fetch_history_period", e)

    def _update_history_period(self, h, period, req_id):
        if req_id != self.request_id:
            return
        if not self.root.winfo_exists():
            return
        if h is not None and not h.empty:
            h.index = h.index.tz_localize(None)
            self.hist_data = h
            self.hist_period = period
            self.redraw_chart()

    # --- SEC XBRL (Free) ---
    def fetch_sec_financials(self, symbol: str, req_id: int):
        try:
            cik = self.get_cik(symbol)
            if not cik:
                return
            facts = self.fetch_sec_companyfacts(cik)
            if not facts:
                return
            fin_q, fin_a, bal_q, bal_a, cf_q, cf_a = self.build_sec_dataframes(facts)
            if req_id != self.request_id:
                return
            self.root.after(0, lambda: self._update_sec_data(fin_q, fin_a, bal_q, bal_a, cf_q, cf_a, req_id))
        except Exception as e:
            log_exception("fetch_sec_financials", e)

    def _update_sec_data(self, fin_q, fin_a, bal_q, bal_a, cf_q, cf_a, req_id):
        if req_id != self.request_id:
            return
        if not self.root.winfo_exists():
            return
        if fin_q is not None and not fin_q.empty:
            self.df_fin_q = fin_q
        if fin_a is not None and not fin_a.empty:
            self.df_fin_a = fin_a
        if bal_q is not None and not bal_q.empty:
            self.df_bal_q = bal_q
        if bal_a is not None and not bal_a.empty:
            self.df_bal_a = bal_a
        if cf_q is not None and not cf_q.empty:
            self.df_cf_q = cf_q
        if cf_a is not None and not cf_a.empty:
            self.df_cf_a = cf_a
        self.update_kpis_from_sec(self.df_fin_q, self.df_bal_q, self.df_cf_q)
        try:
            info = self.current_ticker.info if self.current_ticker else {}
        except Exception:
            info = {}
        self.fill_quality_checklist(info or {}, self.df_fin_q, self.df_bal_q, self.df_cf_q)
        self.redraw_chart()

    def get_cik(self, symbol: str):
        row = self.store.get_ticker_map(symbol)
        if row:
            return row[0]
        # Fetch SEC ticker map
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            text = self.http_get_text(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=15)
            if not text:
                return None
            data = json.loads(text)
            for _, v in data.items():
                if v.get("ticker", "").upper() == symbol.upper():
                    cik = str(v.get("cik_str", "")).zfill(10)
                    name = v.get("title", "")
                    self.store.upsert_ticker_map(symbol, cik, name)
                    return cik
        except Exception as e:
            log_exception("get_cik", e)
            return None
        return None

    def fetch_sec_companyfacts(self, cik: str):
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            text = self.http_get_text(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if not text:
                return None
            return json.loads(text)
        except Exception as e:
            log_exception("fetch_sec_companyfacts", e)
            return None

    def fetch_sec_submissions(self, cik: str):
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            text = self.http_get_text(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if not text:
                return None
            return json.loads(text)
        except Exception as e:
            log_exception("fetch_sec_submissions", e)
            return None

    def _should_skip_transcript_fetch(self, symbol: str, source: str, force: bool):
        if force:
            return False
        now_ts = time.time()
        cache = self.transcript_fetch_cache.get(symbol, {})
        last = cache.get(source, 0)
        if last and (now_ts - last) < TRANSCRIPT_FETCH_TTL_SEC:
            return True
        cache[source] = now_ts
        self.transcript_fetch_cache[symbol] = cache
        return False

    def fetch_sec_earnings_release(self, symbol: str, req_id: int, force: bool = False):
        if self._should_skip_transcript_fetch(symbol, "SEC", force):
            self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
            return
        try:
            cik = self.get_cik(symbol)
            if not cik:
                self.transcript_fetch_status = "SEC: No CIK found."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
                return
            sub = self.fetch_sec_submissions(cik)
            if not sub:
                self.transcript_fetch_status = "SEC: Submissions unavailable."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
                return
            filings = sub.get("filings", {}).get("recent", {})
            accession = None
            filing_date = None
            forms = filings.get("form", [])
            accession_numbers = filings.get("accessionNumber", [])
            filing_dates = filings.get("filingDate", [])
            for idx, form in enumerate(forms):
                if form == "8-K":
                    accession = accession_numbers[idx].replace("-", "")
                    filing_date = filing_dates[idx]
                    break
            if not accession:
                self.transcript_fetch_status = "SEC: No recent 8-K found."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
                return
            cik_no_lead = str(int(cik))
            index_url = f"https://data.sec.gov/Archives/edgar/data/{cik_no_lead}/{accession}/index.json"
            index_text = self.http_get_text(index_url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if not index_text:
                self.transcript_fetch_status = "SEC: 8-K index not accessible."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
                return
            index_data = json.loads(index_text)
            files = index_data.get("directory", {}).get("item", [])
            ex_file = None
            for f in files:
                name = f.get("name", "").lower()
                if "ex-99.1" in name or "ex99.1" in name:
                    ex_file = f.get("name")
                    break
            if not ex_file:
                self.transcript_fetch_status = "SEC: No EX-99.1 found in 8-K."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
                return
            doc_url = f"https://data.sec.gov/Archives/edgar/data/{cik_no_lead}/{accession}/{ex_file}"
            doc_text = self.http_get_text(doc_url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if not doc_text:
                self.transcript_fetch_status = "SEC: EX-99.1 not accessible."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
                return
            clean = self._format_transcript_text(doc_text)
            title = f"Earnings Release {filing_date}"
            if not self.store.transcript_exists(symbol, filing_date or "", title):
                self.store.upsert_transcript(symbol, filing_date or "", title, clean)
            if req_id != self.request_id:
                return
            self.root.after(0, lambda: (self.load_transcripts(), self._mark_transcript_fetch_done("SEC")))
        except Exception as e:
            log_exception("fetch_sec_earnings_release", e)
            self.transcript_fetch_status = "SEC: Unexpected error."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))

    def recent_quarters(self, n: int):
        today = datetime.now()
        year = today.year
        q = (today.month - 1) // 3 + 1
        qs = []
        for _ in range(n):
            qs.append(f"{year}Q{q}")
            q -= 1
            if q == 0:
                q = 4
                year -= 1
        return qs

    def fetch_av_transcripts(self, symbol: str, req_id: int, force: bool = False):
        if self._should_skip_transcript_fetch(symbol, "AlphaVantage", force):
            self.root.after(0, lambda: self._mark_transcript_fetch_done("AlphaVantage"))
            return
        if not ALPHAVANTAGE_API_KEY:
            self.transcript_fetch_status = "Alpha Vantage: API key missing."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("AlphaVantage"))
            return
        try:
            quarters = self.recent_quarters(AV_TRANSCRIPTS_QUARTERS)
            for q in quarters:
                title = f"Earnings Call Transcript {q} (Alpha Vantage)"
                if self.store.transcript_exists(symbol, q, title):
                    continue
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={symbol}&quarter={q}&apikey={ALPHAVANTAGE_API_KEY}"
                text = self.http_get_text(url, headers=None, timeout=20)
                if text:
                    self.write_api_log(f"alphavantage_{q}", symbol, text)
                if not text:
                    continue
                data = json.loads(text)
                if isinstance(data, dict) and (data.get("Error Message") or data.get("Note") or data.get("Information")):
                    self.transcript_fetch_status = f"Alpha Vantage: {data.get('Error Message') or data.get('Note') or data.get('Information')}"
                    break
                transcript = None
                if isinstance(data, dict):
                    transcript = data.get("transcript") or data.get("content") or data.get("text")
                if isinstance(transcript, list):
                    parts = []
                    for item in transcript:
                        if not isinstance(item, dict):
                            continue
                        speaker = item.get("speaker", "")
                        title_s = item.get("title", "")
                        content = item.get("content", "")
                        header = speaker
                        if title_s:
                            header = f"{header} ({title_s})" if header else title_s
                        if header:
                            parts.append(f"{header}: {content}")
                        else:
                            parts.append(content)
                    transcript = "\n".join([p for p in parts if p])
                elif isinstance(transcript, dict):
                    transcript = json.dumps(transcript, indent=2)
                if not transcript:
                    continue
                self.store.upsert_transcript(symbol, q, title, transcript)
            if req_id != self.request_id:
                return
            self.root.after(0, lambda: (self.load_transcripts(), self._mark_transcript_fetch_done("AlphaVantage")))
        except Exception as e:
            log_exception("fetch_av_transcripts", e)
            if not self.transcript_fetch_status:
                self.transcript_fetch_status = "Alpha Vantage: Unexpected error."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("AlphaVantage"))

    def fetch_api_ninjas_transcripts(self, symbol: str, req_id: int, force: bool = False):
        if self._should_skip_transcript_fetch(symbol, "APINinjas", force):
            self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))
            return
        if not API_NINJAS_KEY:
            self.transcript_fetch_status = "API Ninjas: API key missing."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))
            return
        try:
            url = f"https://api.api-ninjas.com/v1/earningstranscript?ticker={symbol}"
            headers = {"X-Api-Key": API_NINJAS_KEY}
            text = self.http_get_text(url, headers=headers, timeout=20)
            if text:
                self.write_api_log("api_ninjas", symbol, text)
            if not text:
                self.transcript_fetch_status = "API Ninjas: request failed."
                self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))
                return
            data = json.loads(text)
            transcripts = []
            if isinstance(data, list):
                transcripts = data
            elif isinstance(data, dict):
                if "transcripts" in data and isinstance(data["transcripts"], list):
                    transcripts = data["transcripts"]
                else:
                    transcripts = [data]
            else:
                self.transcript_fetch_status = "API Ninjas: unexpected response."
                return
            if not transcripts:
                self.transcript_fetch_status = "API Ninjas: no transcripts."
            for t in transcripts:
                if not isinstance(t, dict):
                    continue
                date = t.get("date") or t.get("period") or t.get("quarter") or ""
                content = t.get("transcript") or t.get("content") or t.get("text") or ""
                if not content:
                    continue
                title = f"Earnings Call Transcript {date} (API Ninjas)".strip()
                if self.store.transcript_exists(symbol, date, title):
                    continue
                self.store.upsert_transcript(symbol, date, title, content)
            if req_id != self.request_id:
                return
            self.root.after(0, lambda: (self.load_transcripts(), self._mark_transcript_fetch_done("APINinjas")))
        except Exception as e:
            log_exception("fetch_api_ninjas_transcripts", e)
            if not self.transcript_fetch_status:
                self.transcript_fetch_status = "API Ninjas: unexpected error."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))

    def http_get_text(self, url: str, headers: Optional[Dict[str, str]], timeout: int):
        try:
            if requests is not None:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code != 200:
                    return None
                return r.text
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            return None

    def extract_sec_series(self, facts, tag_opts: List[str], unit_pref: str = "USD", annual_only: bool = False, exclude_fy: bool = False):
        try:
            fact_root = facts.get("facts", {}).get("us-gaap", {})
            for tag in tag_opts:
                if tag not in fact_root:
                    continue
                units = fact_root[tag].get("units", {})
                unit = unit_pref if unit_pref in units else (list(units.keys())[0] if units else None)
                if not unit:
                    continue
                items = units[unit]
                rows = []
                for it in items:
                    end = it.get("end")
                    val = it.get("val")
                    fp = it.get("fp")
                    form = it.get("form")
                    if end is None or val is None:
                        continue
                    if annual_only:
                        if fp != "FY" and form != "10-K":
                            continue
                    elif exclude_fy:
                        if fp == "FY" or (form and "10-K" in form):
                            continue
                    rows.append((end, val))
                if rows:
                    s = pd.Series({pd.to_datetime(d): v for d, v in rows})
                    s = s.sort_index()
                    return s
        except Exception as e:
            log_exception("extract_sec_series", e)
            return None
        return None

    def align_to_quarter_end(self, series: Optional[pd.Series]):
        """Snap SEC fact dates to quarter end to avoid near-date misalignment (e.g., 09-29 vs 09-30)."""
        if series is None or series.empty:
            return series
        try:
            s = pd.to_numeric(series, errors="coerce").dropna().copy()
            idx = pd.to_datetime(s.index).tz_localize(None)
            s.index = idx + pd.tseries.offsets.QuarterEnd(0)
            return s.groupby(level=0).last().sort_index()
        except Exception as e:
            log_exception("align_to_quarter_end", e)
            return series

    def align_to_year_end(self, series: Optional[pd.Series]):
        if series is None or series.empty:
            return series
        try:
            s = pd.to_numeric(series, errors="coerce").dropna().copy()
            idx = pd.to_datetime(s.index).tz_localize(None)
            s.index = idx + pd.tseries.offsets.YearEnd(0)
            return s.groupby(level=0).last().sort_index()
        except Exception as e:
            log_exception("align_to_year_end", e)
            return series

    def build_sec_dataframes(self, facts):
        # Financials
        rev = self.extract_sec_series(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], exclude_fy=True)
        gp = self.extract_sec_series(facts, ["GrossProfit"], exclude_fy=True)
        op = self.extract_sec_series(facts, ["OperatingIncomeLoss"], exclude_fy=True)
        ni = self.extract_sec_series(facts, ["NetIncomeLoss"], exclude_fy=True)
        ebitda = self.extract_sec_series(facts, ["EarningsBeforeInterestTaxesDepreciationAmortization"], exclude_fy=True)
        eps = self.extract_sec_series(facts, ["EarningsPerShareBasic"], unit_pref="USD/shares", exclude_fy=True)
        eps_d = self.extract_sec_series(facts, ["EarningsPerShareDiluted"], unit_pref="USD/shares", exclude_fy=True)
        rev = self.align_to_quarter_end(rev)
        gp = self.align_to_quarter_end(gp)
        op = self.align_to_quarter_end(op)
        ni = self.align_to_quarter_end(ni)
        ebitda = self.align_to_quarter_end(ebitda)
        eps = self.align_to_quarter_end(eps)
        eps_d = self.align_to_quarter_end(eps_d)
        fin_q = pd.DataFrame({
            "TotalRevenue": rev,
            "GrossProfit": gp,
            "OperatingIncome": op,
            "NetIncome": ni,
            "EBITDA": ebitda,
            "BasicEPS": eps,
            "DilutedEPS": eps_d,
        }).dropna(how="all").sort_index()

        rev_a = self.extract_sec_series(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], annual_only=True)
        gp_a = self.extract_sec_series(facts, ["GrossProfit"], annual_only=True)
        op_a = self.extract_sec_series(facts, ["OperatingIncomeLoss"], annual_only=True)
        ni_a = self.extract_sec_series(facts, ["NetIncomeLoss"], annual_only=True)
        ebitda_a = self.extract_sec_series(facts, ["EarningsBeforeInterestTaxesDepreciationAmortization"], annual_only=True)
        eps_a = self.extract_sec_series(facts, ["EarningsPerShareBasic"], unit_pref="USD/shares", annual_only=True)
        epsd_a = self.extract_sec_series(facts, ["EarningsPerShareDiluted"], unit_pref="USD/shares", annual_only=True)
        rev_a = self.align_to_year_end(rev_a)
        gp_a = self.align_to_year_end(gp_a)
        op_a = self.align_to_year_end(op_a)
        ni_a = self.align_to_year_end(ni_a)
        ebitda_a = self.align_to_year_end(ebitda_a)
        eps_a = self.align_to_year_end(eps_a)
        epsd_a = self.align_to_year_end(epsd_a)
        fin_a = pd.DataFrame({
            "TotalRevenue": rev_a,
            "GrossProfit": gp_a,
            "OperatingIncome": op_a,
            "NetIncome": ni_a,
            "EBITDA": ebitda_a,
            "BasicEPS": eps_a,
            "DilutedEPS": epsd_a,
        }).dropna(how="all").sort_index()

        # Balance Sheet
        cash = self.extract_sec_series(facts, ["CashAndCashEquivalentsAtCarryingValue"])
        debt = self.extract_sec_series(facts, ["LongTermDebt", "Debt"])
        equity = self.extract_sec_series(facts, ["StockholdersEquity"])
        shares = self.extract_sec_series(facts, ["CommonStockSharesOutstanding"], unit_pref="shares")
        cash = self.align_to_quarter_end(cash)
        debt = self.align_to_quarter_end(debt)
        equity = self.align_to_quarter_end(equity)
        shares = self.align_to_quarter_end(shares)
        bal_q = pd.DataFrame({
            "CashAndCashEquivalents": cash,
            "TotalDebt": debt,
            "TotalStockholderEquity": equity,
            "ShareIssued": shares,
        }).dropna(how="all").sort_index()

        cash_a = self.extract_sec_series(facts, ["CashAndCashEquivalentsAtCarryingValue"], annual_only=True)
        debt_a = self.extract_sec_series(facts, ["LongTermDebt", "Debt"], annual_only=True)
        equity_a = self.extract_sec_series(facts, ["StockholdersEquity"], annual_only=True)
        shares_a = self.extract_sec_series(facts, ["CommonStockSharesOutstanding"], unit_pref="shares", annual_only=True)
        cash_a = self.align_to_year_end(cash_a)
        debt_a = self.align_to_year_end(debt_a)
        equity_a = self.align_to_year_end(equity_a)
        shares_a = self.align_to_year_end(shares_a)
        bal_a = pd.DataFrame({
            "CashAndCashEquivalents": cash_a,
            "TotalDebt": debt_a,
            "TotalStockholderEquity": equity_a,
            "ShareIssued": shares_a,
        }).dropna(how="all").sort_index()

        # Cash Flow
        ocf = self.extract_sec_series(facts, ["NetCashProvidedByUsedInOperatingActivities"], exclude_fy=True)
        capex = self.extract_sec_series(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"], exclude_fy=True)
        sbc = self.extract_sec_series(facts, ["ShareBasedCompensation", "StockBasedCompensation"], exclude_fy=True)
        ocf = self.align_to_quarter_end(ocf)
        capex = self.align_to_quarter_end(capex)
        sbc = self.align_to_quarter_end(sbc)
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf + capex
        cf_q = pd.DataFrame({
            "OperatingCashFlow": ocf,
            "CapitalExpenditure": capex,
            "FreeCashFlow": fcf,
            "StockBasedCompensation": sbc,
        }).dropna(how="all").sort_index()

        ocf_a = self.extract_sec_series(facts, ["NetCashProvidedByUsedInOperatingActivities"], annual_only=True)
        capex_a = self.extract_sec_series(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"], annual_only=True)
        sbc_a = self.extract_sec_series(facts, ["ShareBasedCompensation", "StockBasedCompensation"], annual_only=True)
        ocf_a = self.align_to_year_end(ocf_a)
        capex_a = self.align_to_year_end(capex_a)
        sbc_a = self.align_to_year_end(sbc_a)
        fcf_a = None
        if ocf_a is not None and capex_a is not None:
            fcf_a = ocf_a + capex_a
        cf_a = pd.DataFrame({
            "OperatingCashFlow": ocf_a,
            "CapitalExpenditure": capex_a,
            "FreeCashFlow": fcf_a,
            "StockBasedCompensation": sbc_a,
        }).dropna(how="all").sort_index()

        return fin_q, fin_a, bal_q, bal_a, cf_q, cf_a

if __name__ == "__main__":
    root = ctk.CTk()
    app = OpenTrimApp(root)
    root.mainloop()
