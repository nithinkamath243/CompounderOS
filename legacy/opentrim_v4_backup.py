import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import customtkinter as ctk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
try:
    import requests
except Exception:
    requests = None
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional, Dict, List, Any

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
WATCHLIST_FILE = os.path.join(BASE_DIR, "opentrim_watchlist.json")
LEGACY_WATCHLIST_FILE = os.path.join(USER_HOME, "opentrim_watchlist.json")
API_KEY_FILE = os.path.join(BASE_DIR, ".gemini_key")
LEGACY_API_KEY_FILE = os.path.join(USER_HOME, ".gemini_key")
DATA_DB = os.path.join(BASE_DIR, "opentrim_data.db")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "opentrim_transcripts")
CONFIG_FILE = os.path.join(BASE_DIR, ".opentrim_config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except:
            return {}
    return {}

def save_config(data: Dict[str, Any]):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

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
        except:
            pass
    elif os.path.exists(LEGACY_API_KEY_FILE):
        try:
            with open(LEGACY_API_KEY_FILE, "r") as f:
                key = f.read().strip()
            if key:
                CONFIG["GEMINI_API_KEY"] = key
                save_config(CONFIG)
        except:
            pass

migrate_legacy_gemini_key()
SEC_USER_AGENT = CONFIG.get("SEC_USER_AGENT") or os.environ.get("SEC_USER_AGENT", "OpenTrim/4.2 (contact: user@example.com)")
FINNHUB_API_KEY = CONFIG.get("FINNHUB_API_KEY")
ALPHAVANTAGE_API_KEY = CONFIG.get("ALPHAVANTAGE_API_KEY")
API_NINJAS_KEY = CONFIG.get("API_NINJAS_KEY")

# --- THEME PALETTE ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

C_BG_MAIN     = "#121212" # CTK Window Bg
C_BG_CARD     = "#1E1E2E" # CTK Frame Bg
C_BG_INPUT    = "#2A2A3C"
C_ACCENT      = "#3B8ED0"
C_TEXT_MAIN   = "#E0E0E0"
C_TEXT_SUB    = "#A0A0A0"
C_GREEN       = "#2ECC71"
C_RED         = "#E74C3C"
C_YELLOW      = "#F1C40F"
C_PINK        = "#FF69B4"

# --- WATCHLIST COLORS ---
COLOR_UP_BG   = "#004d00"
COLOR_DIP_BG  = "#4d0000"

# Matplotlib Dark Theme
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': C_BG_CARD,
    'figure.facecolor': C_BG_CARD,
    'grid.color': '#333333',
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
                (ticker, cik, name, datetime.now().isoformat())
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
                (symbol, date, title, summary, qa, datetime.now().isoformat())
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
                (symbol, data_type, fcf_or_eps, growth, discount, terminal_mult, target_return, datetime.now().isoformat())
            )
            self.conn.commit()

class OpenTrimApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("OpenTrim v4.1 - Enhanced & Robust")
        
        # Safer default geometry for laptops
        self.root.geometry("1280x800")
        
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
        except:
            pass
        if not self.store.list_portfolios():
            self.store.upsert_portfolio_symbol("Main", "__INIT__", 0, 0)
            self.store.remove_portfolio_symbol("Main", "__INIT__")

        self.setup_styles()
        self.create_layout()

        # Try to maximize window on start
        try:
            self.root.after(100, lambda: self.root.state('zoomed'))
        except:
            pass

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
                    rowheight=35, 
                    font=("Roboto", 11))
        
        s.configure("Treeview.Heading", 
                    background="#2b2b2b", 
                    foreground="white", 
                    font=("Roboto", 11, "bold"), 
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
            except:
                pass
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

    def create_layout(self):
        # Main container
        container = ctk.CTkFrame(self.root, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- HEADER ---
        top_bar = ctk.CTkFrame(container, fg_color="transparent")
        top_bar.pack(fill=tk.X, pady=(0, 15))
        
        ctk.CTkLabel(top_bar, text="OPENTRIM", font=("Impact", 28), text_color=C_TEXT_MAIN).pack(side=tk.LEFT, padx=(0, 20))
        
        # Search Area
        self.search_var = tk.StringVar()
        self.entry_search = ctk.CTkEntry(top_bar, textvariable=self.search_var, width=200, placeholder_text="Ticker (e.g. AAPL)")
        self.entry_search.pack(side=tk.LEFT, padx=10)
        self.entry_search.bind('<Return>', lambda e: self.start_search())

        ctk.CTkButton(top_bar, text="SEARCH", command=self.start_search, width=80).pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = ctk.CTkLabel(top_bar, text="Ready", text_color=C_TEXT_SUB)
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        ctk.CTkButton(top_bar, text="API KEY", command=self.ask_api_key, width=80, fg_color="#444444").pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(top_bar, text="+ WATCHLIST", command=self.add_to_watchlist, width=100, fg_color="#2b2b2b").pack(side=tk.RIGHT, padx=5)

        # --- TABS ---
        self.tabview = ctk.CTkTabview(container)
        self.tabview.pack(fill=tk.BOTH, expand=True)

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

    def ask_api_key(self):
        dialog = ctk.CTkInputDialog(text="Enter Gemini API Key:", title="API Key")
        k = dialog.get_input()
        if k: self.save_api_key(k)

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
        threading.Thread(target=self.fetch_data, args=(s, req_id), daemon=True).start()

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
            except:
                pass

            insider = cached.get("insider")
            if insider is None:
                insider = pd.DataFrame()
            try:
                if insider is None or insider.empty:
                    insider = t.insider_transactions.head(20)
            except:
                pass

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
            threading.Thread(target=self.fetch_stooq_history, args=(symbol, req_id), daemon=True).start()
            # Kick off SEC financials (non-blocking)
            threading.Thread(target=self.fetch_sec_financials, args=(symbol, req_id), daemon=True).start()
        except Exception as e:
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
        self.df_dividends = divs
        
        self.fill_insights(i, t)
        self.fill_charts()
        self.fill_insider_calendar(insider, cal)
        self.fill_quality_checklist(i, self.df_fin_q, self.df_bal_q, self.df_cf_q)
        self.reset_dcf(i)
        self.load_transcripts()
        self.load_kpis()
        self.auto_refresh_earnings(s)
        threading.Thread(target=self.fetch_sec_earnings_release, args=(s, req_id, False), daemon=True).start()
        threading.Thread(target=self.fetch_av_transcripts, args=(s, req_id, False), daemon=True).start()
        threading.Thread(target=self.fetch_api_ninjas_transcripts, args=(s, req_id, False), daemon=True).start()
        
        # Safe AI Update via Thread
        self.txt_ai.configure(state="normal")
        self.txt_ai.delete(1.0, tk.END)
        self.txt_ai.insert(tk.END, f"Analyzing (Smart Mode)...\n")
        self.txt_ai.configure(state="disabled")
        threading.Thread(target=self.run_gemini_25_logic, args=(s, i, self.df_fin_q), daemon=True).start()
        
        self.lbl_status.configure(text=f"Active: {s}", text_color=C_GREEN)

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
        self.f_check = ctk.CTkFrame(self.tab_carlson, fg_color=C_BG_CARD)
        self.f_check.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        
        ctk.CTkLabel(self.f_check, text="COMPOUNDER CHECKLIST", font=("Roboto", 16, "bold"), text_color=C_ACCENT).pack(anchor='w', pady=15, padx=15)
        
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
        self.f_verdict = ctk.CTkFrame(self.tab_carlson, fg_color=C_BG_CARD)
        self.f_verdict.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)

        self.lbl_score = ctk.CTkLabel(self.f_verdict, text="SCORE: --/10", font=("Roboto", 32, "bold"))
        self.lbl_score.pack(pady=20)
        
        self.txt_verdict = ctk.CTkTextbox(self.f_verdict, height=150, fg_color=C_BG_MAIN)
        self.txt_verdict.pack(fill=tk.X, padx=15, pady=5)
        
        ctk.CTkLabel(self.f_verdict, text="METRIC DEEP DIVE", font=("Roboto", 12, "bold"), text_color=C_ACCENT).pack(anchor='w', pady=(20, 5), padx=15)
        self.txt_deep = ctk.CTkTextbox(self.f_verdict, fg_color=C_BG_INPUT)
        self.txt_deep.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    def fill_quality_checklist(self, info, fin_df, bal_df, cf_df):
        self.tr_check.delete(*self.tr_check.get_children())
        for c in ("Metric", "Target", "Actual", "Result", "Context"): 
            self.tr_check.heading(c, text=c)
        
        self.tr_check.column("Metric", width=150); self.tr_check.column("Target", width=80)
        self.tr_check.column("Actual", width=80); self.tr_check.column("Result", width=60)
        self.tr_check.column("Context", width=200)
        
        score = 0; self.metric_explanations = {}

        def add_m(name, target_display, actual_val, passed, short_ctx, logic, calc, source):
            res = "PASS" if passed else "FAIL"
            item_id = self.tr_check.insert("", tk.END, values=(name, target_display, actual_val, res, short_ctx))
            if passed: self.tr_check.item(item_id, tags=("pass",))
            else: self.tr_check.item(item_id, tags=("fail",))
            
            explanation = f"""METRIC: {name}\n\nTHE LOGIC:\n{logic}\n\nCALCULATION:\n{calc}\n\nSOURCE:\n{source}\n\nCONTEXT:\nYour Value: {actual_val}\nTarget: {target_display} -> {res}\n{short_ctx}"""
            self.metric_explanations[item_id] = explanation
            return 1 if passed else 0

        # -- Metric Calculations using safe_div --
        gm = info.get('grossMargins', 0)
        if gm is None: gm = 0
        res_gm = gm > 0.40
        ctx_gm = "High margins = Pricing Power" if res_gm else "Low margins = Competitive"
        score += add_m("Moat (Gross Margin)", "> 40%", f"{gm:.1%}", res_gm, ctx_gm, "High gross margins indicate pricing power.", "Gross Profit / Total Revenue", "Yahoo Info (TTM)")

        rg = info.get('revenueGrowth', 0)
        if rg is None: rg = 0
        res_rg = rg > 0.07
        ctx_rg = "Healthy expansion" if res_rg else "Stagnant top-line"
        score += add_m("Organic Growth", "> 7%", f"{rg:.1%}", res_rg, ctx_rg, "Consistent growth shows demand.", "Quarterly Rev Growth (YoY)", "Yahoo Info (MRQ)")

        op_inc = self.get_series(fin_df, ["OperatingIncome", "EBIT"]); rev = self.get_series(fin_df, ["TotalRevenue"])
        op_lev_pass = False; op_lev_txt = "N/A"; op_ctx = "Margins Contracting"
        if op_inc is not None and rev is not None and len(op_inc) > 4 and len(rev) > 4:
            # Safe calculation for Operating Leverage
            op_growth = self.safe_div(op_inc.iloc[-1] - op_inc.iloc[-5], abs(op_inc.iloc[-5]))
            rev_growth = self.safe_div(rev.iloc[-1] - rev.iloc[-5], abs(rev.iloc[-5]))
            op_lev_pass = op_growth > rev_growth
            op_lev_txt = f"Op:{op_growth:.1%} > Rev:{rev_growth:.1%}"
            op_ctx = "Profit growing faster than sales" if op_lev_pass else "Costs growing faster than sales"
        score += add_m("Operating Leverage", "Op > Rev", op_lev_txt, op_lev_pass, op_ctx, "Profit grows faster than revenue.", "Op Inc vs Rev Growth (YoY)", "Quarterly Financials")

        roe = info.get('returnOnEquity', 0)
        if roe is None: roe = 0
        res_roe = roe > 0.15
        ctx_roe = "Elite efficiency" if res_roe else "Poor capital use"
        score += add_m("Efficiency (ROE)", "> 15%", f"{roe:.1%}", res_roe, ctx_roe, "ROE measures capital efficiency.", "Net Income / Shareholder Equity", "Yahoo Info (TTM)")

        cr = info.get('currentRatio', 0)
        if cr is None: cr = 0
        res_cr = cr > 1.0
        ctx_cr = "Liquid / Safe" if res_cr else "Liquidity Risk"
        score += add_m("Financial Health", "> 1.0", f"{cr:.2f}", res_cr, ctx_cr, "Can pay short-term debts?", "Current Assets / Current Liabilities", "Yahoo Info (MRQ)")

        capex = self.get_series(cf_df, ["CapitalExpenditure", "CapEx"]); rev_q = self.get_series(fin_df, ["TotalRevenue"])
        cap_pass = False; cap_txt = "N/A"; ctx_cap = "Capital Heavy"
        if capex is not None and rev_q is not None:
            c_ttm = abs(capex.tail(4).sum()); r_ttm = rev_q.tail(4).sum()
            intensity = self.safe_div(c_ttm, r_ttm, 999.0)
            cap_pass = intensity < 0.10; cap_txt = f"{intensity:.1%}"
            ctx_cap = "Asset Light Model" if cap_pass else "Asset Heavy"
        score += add_m("Capital Light", "< 10% Rev", cap_txt, cap_pass, ctx_cap, "Compounders need little capital.", "TTM Capex / TTM Revenue", "Quarterly Data")

        fcf = self.get_series(cf_df, ["FreeCashFlow"]); ni = self.get_series(fin_df, ["NetIncome"])
        conv_pass = False; conv_txt = "N/A"; ctx_conv = "Low Quality Earnings"
        if fcf is None: 
             ocf = self.get_series(cf_df, ["OperatingCashFlow"]); cap = self.get_series(cf_df, ["CapitalExpenditure"])
             if ocf is not None and cap is not None: fcf = ocf + cap
        if fcf is not None and ni is not None:
            f_ttm = fcf.tail(4).sum(); n_ttm = ni.tail(4).sum()
            conv = self.safe_div(f_ttm, n_ttm)
            conv_pass = conv > 0.80; conv_txt = f"{conv:.1%}"
            ctx_conv = "Real Cash Earnings" if conv_pass else "Accounting Profits"
        score += add_m("FCF Conversion", "> 80%", conv_txt, conv_pass, ctx_conv, "Are earnings real cash?", "TTM FCF / TTM Net Income", "Quarterly Data")

        shares = self.get_series(bal_df, ["ShareIssued", "CommonStockSharesOutstanding"])
        is_buyback = False; share_txt = "N/A"; ctx_bb = "Unknown"
        if shares is not None and len(shares) > 4:
            curr = shares.iloc[-1]; prev = shares.iloc[-5]
            is_buyback = curr < prev; share_txt = "Decreasing" if is_buyback else "Diluting"
            ctx_bb = "Value Created" if is_buyback else "Value Destroyed"
        score += add_m("Share Buybacks", "Decreasing", share_txt, is_buyback, ctx_bb, "Reducing share count.", "Share Count (Current vs 1Yr Ago)", "Balance Sheet")

        peg = info.get('pegRatio', 999); 
        if peg is None: peg = 999
        res_peg = 0 < peg < 2.0
        ctx_peg = "Fair Price" if res_peg else "Expensive"
        score += add_m("Valuation (PEG)", "< 2.0", f"{peg:.2f}", res_peg, ctx_peg, "Growth adjusted P/E.", "PE / Growth Rate", "Yahoo Info")

        curr_p = info.get('currentPrice', 0); target = info.get('targetMeanPrice', 0)
        is_safe = False; margin_txt = "N/A"; ctx_safe = "N/A"
        if target and target > 0 and curr_p > 0:
            upside = self.safe_div(target - curr_p, curr_p)
            is_safe = upside > 0.15; margin_txt = f"{upside:.1%}"
            ctx_safe = "Undervalued" if is_safe else "Overvalued"
        score += add_m("Margin of Safety", "> 15%", margin_txt, is_safe, ctx_safe, "Trading below analyst targets?", "(Target - Current) / Current", "Analyst Estimates")

        self.tr_check.tag_configure("pass", foreground=C_GREEN)
        self.tr_check.tag_configure("fail", foreground=C_RED)
        self.lbl_score.configure(text=f"SCORE: {score}/10", text_color=C_GREEN if score >= 7 else C_YELLOW)
        
        verdict = "VERDICT: HOLD / WATCH\n\n"
        if score >= 8: verdict = "VERDICT: ELITE COMPOUNDER (STRONG BUY)\n\n"
        elif score <= 4: verdict = "VERDICT: LOW QUALITY / SPECULATIVE\n\n"
        self.txt_verdict.delete(1.0, tk.END); self.txt_verdict.insert(tk.END, verdict + "Select any metric in the table above to see the Deep Dive explanation here.")

    def show_metric_details(self, event):
        sel = self.tr_check.selection()
        if not sel: return
        item_id = sel[0]
        text = self.metric_explanations.get(item_id, "No details available.")
        self.txt_deep.configure(state="normal")
        self.txt_deep.delete(1.0, tk.END)
        self.txt_deep.insert(tk.END, text)
        self.txt_deep.configure(state="disabled")

    # --- INSIGHTS ---
    def build_insights(self):
        pad = ctk.CTkFrame(self.tab_insights, fg_color=C_BG_CARD)
        pad.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        row1 = ctk.CTkFrame(pad, fg_color="transparent")
        row1.pack(fill=tk.X, pady=(15, 30), padx=20)
        
        self.lbl_price = ctk.CTkLabel(row1, text="$0.00", font=("Roboto", 42, "bold"))
        self.lbl_price.pack(side=tk.LEFT)
        
        self.lbl_chg = ctk.CTkLabel(row1, text="0.00%", font=("Roboto", 20, "bold"), text_color=C_ACCENT)
        self.lbl_chg.pack(side=tk.LEFT, padx=20, pady=(15,0))

        self.lbl_after = ctk.CTkLabel(row1, text="After Hours: N/A", font=("Roboto", 12), text_color=C_TEXT_SUB)
        self.lbl_after.pack(side=tk.LEFT, padx=10, pady=(15,0))
        
        self.lbl_targets = ctk.CTkLabel(pad, text="Waiting...", font=("Roboto", 14), anchor="w")
        self.lbl_targets.pack(fill=tk.X, padx=20)
        self.lbl_top_news = ctk.CTkLabel(pad, text="Top news: N/A", font=("Roboto", 12), text_color=C_TEXT_SUB, anchor="w")
        self.lbl_top_news.pack(fill=tk.X, padx=20, pady=(5,0))

        # Key Metrics Panel
        metrics = ctk.CTkFrame(pad, fg_color=C_BG_INPUT)
        metrics.pack(fill=tk.X, padx=20, pady=(10, 15))
        self.metric_labels = {}
        keys = ["P/E", "P/S", "FCF Yield", "Gross %", "Op %", "Net %", "Debt/Equity", "Div Yield"]
        for k in keys:
            box = ctk.CTkFrame(metrics, fg_color="transparent")
            box.pack(side=tk.LEFT, padx=10, pady=8)
            ctk.CTkLabel(box, text=k, font=("Roboto", 10), text_color=C_TEXT_SUB).pack()
            val = ctk.CTkLabel(box, text="--", font=("Roboto", 12, "bold"))
            val.pack()
            self.metric_labels[k] = val

        # Company Profile + Analyst Estimates
        info_row = ctk.CTkFrame(pad, fg_color="transparent")
        info_row.pack(fill=tk.BOTH, expand=False, padx=20, pady=(10, 15))
        prof = ctk.CTkFrame(info_row, fg_color=C_BG_INPUT)
        prof.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        ctk.CTkLabel(prof, text="COMPANY PROFILE", font=("Roboto", 12, "bold"), text_color=C_ACCENT).pack(anchor="w", padx=10, pady=(10,5))
        self.txt_profile = ctk.CTkTextbox(prof, height=120, fg_color=C_BG_MAIN)
        self.txt_profile.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        est = ctk.CTkFrame(info_row, fg_color=C_BG_INPUT)
        est.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctk.CTkLabel(est, text="ANALYST ESTIMATES", font=("Roboto", 12, "bold"), text_color=C_ACCENT).pack(anchor="w", padx=10, pady=(10,5))
        self.txt_estimates = ctk.CTkTextbox(est, height=120, fg_color=C_BG_MAIN)
        self.txt_estimates.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        ctk.CTkLabel(pad, text="LATEST NEWS", font=("Roboto", 16, "bold"), text_color=C_ACCENT).pack(anchor="w", padx=20, pady=(30, 10))
        
        ctk.CTkButton(pad, text="Open Google News", command=self.open_google_news, width=150, height=25).pack(anchor="w", padx=20, pady=(0,10))
        
        self.txt_news = ctk.CTkTextbox(pad, font=("Roboto", 12), fg_color=C_BG_INPUT)
        self.txt_news.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))

    def open_google_news(self):
        if self.current_symbol: webbrowser.open(f"https://www.google.com/search?q={self.current_symbol}+stock+news&tbm=nws")

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
            symbol = self.current_symbol.upper(); short_name = i.get('shortName', '').split(" ")[0].upper(); 
            if not short_name: short_name = symbol
            filter_keywords = {symbol, short_name}; news_data = t.news
            if not news_data: news_data = yf.Search(self.current_symbol).news
            valid_count = 0
            top_item = None
            if news_data:
                for item in news_data:
                    if valid_count >= 8: break
                    title = item.get('title') or item.get('headline') or item.get('content', {}).get('title', '')
                    summary = item.get('summary') or item.get('content', {}).get('summary', '')
                    if not any(k in (title + " " + summary).upper() for k in filter_keywords): continue 
                    ts = item.get('providerPublishTime', 0); d = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts > 0 else "Recent"
                    self.txt_news.insert(tk.END, f"•  {d}  |  {title}\n\n"); valid_count += 1
                    if top_item is None:
                        top_item = (ts, title, summary)
            if valid_count == 0: self.txt_news.insert(tk.END, f"No direct news found for {short_name}. Check Google News.")
            # Top news snippet (24h)
            if news_data:
                best = None
                now_ts = time.time()
                for item in news_data:
                    ts = item.get('providerPublishTime', 0)
                    title = item.get('title') or item.get('headline') or item.get('content', {}).get('title', '')
                    summary = item.get('summary') or item.get('content', {}).get('summary', '')
                    if ts and (now_ts - ts) <= 86400:
                        best = (ts, title, summary)
                        break
                if best is None and top_item is not None:
                    best = top_item
                if best:
                    self.lbl_top_news.configure(text=f"Top news: {best[1]}")
                else:
                    self.lbl_top_news.configure(text="Top news: N/A")
        except: self.txt_news.insert(tk.END, "News unavailable.")
        self.txt_news.configure(state="disabled")

    # --- TAB 2: CHARTS ---
    def build_charts(self):
        bar = ctk.CTkFrame(self.tab_charts, fg_color=C_BG_CARD)
        bar.pack(fill=tk.X, padx=20, pady=20)
        
        ctk.CTkLabel(bar, text="VISUALIZATION (QUARTERLY):", font=("Roboto", 12, "bold"), text_color=C_ACCENT).pack(side=tk.LEFT, padx=15, pady=10)
        
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
        
        cb = ctk.CTkComboBox(bar, variable=self.chart_var, values=OPTIONS, width=250, command=self.redraw_chart)
        cb.pack(side=tk.LEFT, pady=10)

        self.freq_var = tk.StringVar(value="Quarterly")
        freq_cb = ctk.CTkComboBox(bar, variable=self.freq_var, values=["Quarterly", "TTM", "Annual"], width=130, command=self.redraw_chart)
        freq_cb.pack(side=tk.LEFT, padx=10)
        
        self.anim_enabled = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar, text="Animate", variable=self.anim_enabled, command=self.redraw_chart).pack(side=tk.LEFT, padx=15)
        
        # Frame for plot
        plot_frame = ctk.CTkFrame(self.tab_charts, fg_color=C_BG_CARD)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))

        self.fig, self.ax = plt.subplots(figsize=(10,6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        except:
            pass

# --- HELPER: ROBUST DATA ALIGNMENT ---
    def align_data(self, primary_series, secondary_series=None, fill_value=0.0):
        if primary_series is None or primary_series.empty:
            return [], np.array([]), np.array([])
        
        def clean_series(s, col_name):
            s = s.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            df = s.to_frame(name=col_name)
            df['period'] = df.index.to_period('Q')
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
        
        dates = [p.strftime('%Y-%m') for p in df_merged.index]
        return dates, df_merged['p'].values, df_merged['s'].values

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

    # --- GRAPHING ---
    def redraw_chart(self, _=None):
        # Debounce rapid redraws
        if self._redraw_job:
            try: self.root.after_cancel(self._redraw_job)
            except: pass
        self._redraw_job = self.root.after(REDRAW_DEBOUNCE_MS, self._redraw_chart_now)

    def _redraw_chart_now(self):
        self._redraw_job = None
        # Stop existing animation safely to prevent leaks
        if self.anim and self.anim.event_source:
            try: self.anim.event_source.stop()
            except: pass
        self.anim = None
        
        self.ax.clear()
        mode = self.chart_var.get()
        freq = self.freq_var.get()
        
        # --- 1. PRICE CHART (Line) ---
        if mode == "Price" and self.hist_data is not None:
            if self.hist_period != "5y":
                self.ensure_history_period("5y")
                self.ax.text(0.5, 0.5, "Loading full history...", color="white", ha='center')
                self.canvas.draw_idle()
                return
            # Prefer long history from Stooq if present
            hist = getattr(self, "hist_data_long", None)
            if hist is not None and not hist.empty:
                base = hist
            else:
                base = self.hist_data
            valid_data = base[base['Close'] > 0].copy()
            if not valid_data.empty:
                x = valid_data.index
                y = valid_data['Close'].values
                if y.max() <= 0:
                    self.ax.text(0.5, 0.5, "No Positive Price Data", color="white", ha='center')
                    self.canvas.draw_idle()
                    return

                self.ax.set_title("PRICE HISTORY", fontweight="bold")
                line, = self.ax.plot([], [], color=C_ACCENT, linewidth=1.5)
                # Seed with the first point to avoid log-scale warnings on empty data
                line.set_data(x[:1], y[:1])
                y_min = y.min()
                
                self.ax.set_xlim(x.min(), x.max())
                self.ax.set_ylim(y_min * 0.95, y.max() * 1.05)
                
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
            else:
                self.ax.text(0.5, 0.5, "No Price Data", color="white", ha='center')

        # --- 2. REVENUE & EBITDA ---
        elif mode == "Revenue & EBITDA":
            fin_df = self.get_freq_df("fin")
            rev = self.get_series(fin_df, ["TotalRevenue", "OperatingRevenue", "Revenue"])
            ebitda = self.get_series(fin_df, ["EBITDA", "NormalizedEBITDA"])
            if freq == "TTM":
                rev = self.to_ttm(rev)
                ebitda = self.to_ttm(ebitda)
            
            if rev is not None:
                dates, vals_rev, vals_ebitda = self.align_data(rev / 1e9, ebitda / 1e9 if ebitda is not None else None)
                if len(dates) > 0:
                    x_idx = np.arange(len(dates))
                    bars = self.ax.bar(x_idx, [0]*len(dates), color='#444444', label="Rev ($B)")
                    ebitda_line, = self.ax.plot([], [], color=C_GREEN, marker='o', label="EBITDA ($B)", linewidth=2)
                    
                    self.ax.set_ylim(0, max(vals_rev.max(), vals_ebitda.max()) * 1.3)
                    self.ax.set_xticks(x_idx)
                    self.ax.set_xticklabels(dates, rotation=45, fontsize=8)

                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            for rect, h in zip(bars, vals_rev):
                                rect.set_height(h * prog)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            ebitda_line.set_data(x_idx[:limit], vals_ebitda[:limit])
                            return bars
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        for rect, h in zip(bars, vals_rev):
                            rect.set_height(h)
                        ebitda_line.set_data(x_idx, vals_ebitda)
                    self.ax.legend(loc='upper left')

        # --- 3. EPS & DIVIDENDS ---
        elif mode == "EPS & Dividends":
            fin_df = self.get_freq_df("fin")
            eps = self.get_series(fin_df, ["BasicEPS", "DilutedEPS"])
            if freq == "TTM":
                eps = self.to_ttm(eps)
            if eps is not None:
                self.ax.set_title("EPS vs DIVIDENDS", fontweight="bold")
                line, = self.ax.plot([], [], color=C_ACCENT, marker='o', label="EPS", linewidth=2)
                
                df = pd.DataFrame({'eps': eps}).dropna().sort_index()
                x = df.index
                y = df['eps'].values
                
                if len(x) > 0:
                    self.ax.set_xlim(x.min(), x.max())
                    y_min = min(0, y.min() * 1.1)
                    y_max = max(0.1, y.max() * 1.2)
                    self.ax.set_ylim(y_min, y_max)

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

                    if self.df_dividends is not None and not self.df_dividends.empty:
                        divs = self.df_dividends[self.df_dividends.index >= x.min()]
                        if not divs.empty:
                            self.ax.bar(divs.index, divs.values, color=C_YELLOW, alpha=0.5, label="Div ($)", width=10)
                
                self.ax.legend()

        # --- 4. CASH VS DEBT ---
        elif mode == "Cash vs Debt":
            bal_df = self.get_freq_df("bal")
            cash = self.get_series(bal_df, ["CashAndCashEquivalents", "Cash"])
            debt = self.get_series(bal_df, ["TotalDebt", "LongTermDebt"])
            
            if cash is not None:
                dates, v_cash, v_debt = self.align_data(cash/1e9, debt/1e9 if debt is not None else None)
                if len(dates) > 0:
                    x_idx = np.arange(len(dates))
                    l1, = self.ax.plot([], [], color=C_GREEN, marker='o', label="Cash ($B)")
                    l2, = self.ax.plot([], [], color=C_RED, marker='x', label="Debt ($B)")
                    
                    max_val = max(v_cash.max() if len(v_cash)>0 else 0, v_debt.max() if len(v_debt)>0 else 0)
                    self.ax.set_ylim(0, max_val * 1.5)
                    self.ax.set_xlim(0, len(dates)-1)
                    self.ax.set_xticks(x_idx)
                    self.ax.set_xticklabels(dates, rotation=45, fontsize=8)

                    if self.anim_enabled.get():
                        def update(frame):
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            l1.set_data(x_idx[:limit], v_cash[:limit])
                            l2.set_data(x_idx[:limit], v_debt[:limit])
                            return l1, l2
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        l1.set_data(x_idx, v_cash)
                        l2.set_data(x_idx, v_debt)
                    self.ax.legend()

        # --- 5. MARGINS ---
        elif mode == "Margins (Gross/Op/Net)":
            fin_df = self.get_freq_df("fin")
            rev = self.get_series(fin_df, ["TotalRevenue"])
            if rev is not None:
                gp = self.get_series(fin_df, ["GrossProfit"])
                op = self.get_series(fin_df, ["OperatingIncome", "OperatingIncomeLoss"])
                ni = self.get_series(fin_df, ["NetIncome"])
                if freq == "TTM":
                    rev = self.to_ttm(rev)
                    gp = self.to_ttm(gp)
                    op = self.to_ttm(op)
                    ni = self.to_ttm(ni)
                
                df = pd.DataFrame({'r': rev, 'g': gp, 'o': op, 'n': ni}).dropna(subset=['r'])
                df['g'] = df['g'].fillna(0)
                df['o'] = df['o'].fillna(0)
                df['n'] = df['n'].fillna(0)
                df = df.sort_index()
                
                if not df.empty:
                    dates = [d.strftime('%Y-%m') for d in df.index]
                    v_gross = ((df['g'] / df['r']) * 100).values
                    v_op = ((df['o'] / df['r']) * 100).values
                    v_net = ((df['n'] / df['r']) * 100).values
                    x_idx = np.arange(len(dates))

                    l1, = self.ax.plot([], [], color=C_YELLOW, label="Gross %")
                    l2, = self.ax.plot([], [], color=C_ACCENT, label="Op %")
                    l3, = self.ax.plot([], [], color=C_GREEN, label="Net %")
                    
                    self.ax.set_xlim(0, len(dates)-1)
                    y_min = min(0, v_net.min(), v_op.min())
                    y_max = max(10, v_gross.max() * 1.2)
                    self.ax.set_ylim(y_min, y_max)
                    
                    self.ax.set_xticks(x_idx)
                    self.ax.set_xticklabels(dates, rotation=45, fontsize=8)

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

        # --- 6. CASH FLOW DETAIL ---
        elif mode == "Cash Flow Detail (FCF vs SBC)":
            cf_df = self.get_freq_df("cf")
            fcf = self.get_series(cf_df, ["FreeCashFlow", "Free Cash Flow"])
            
            if fcf is None:
                ocf = self.get_series(cf_df, ["OperatingCashFlow"])
                cap = self.get_series(cf_df, ["CapitalExpenditure"])
                if ocf is not None and cap is not None: 
                    fcf = ocf + cap
            
            sbc = self.get_series(cf_df, ["StockBasedCompensation", "ShareBasedCompensation"])
            if freq == "TTM":
                fcf = self.to_ttm(fcf)
                sbc = self.to_ttm(sbc)

            if fcf is not None:
                dates, v_fcf, v_sbc = self.align_data(fcf/1e9, sbc/1e9 if sbc is not None else None, fill_value=0.0)
                
                if len(dates) > 0:
                    v_fcf = v_fcf.astype(float)
                    v_sbc = v_sbc.astype(float)
                    x_idx = np.arange(len(dates))

                    bars = self.ax.bar(x_idx, [0]*len(dates), color=C_GREEN, label="FCF ($B)", alpha=0.6)
                    sbc_line, = self.ax.plot([], [], color=C_PINK, marker='o', label="SBC ($B)", linewidth=2)
                    
                    max_val = max(v_fcf.max(), v_sbc.max()) if len(v_fcf) > 0 else 1
                    min_val = min(0, v_fcf.min(), v_sbc.min())
                    self.ax.set_ylim(min_val * 1.1, max_val * 1.3)
                    
                    self.ax.set_xticks(x_idx)
                    self.ax.set_xticklabels(dates, rotation=45, fontsize=8)

                    if self.anim_enabled.get():
                        def update(frame):
                            if len(dates) == 0: return bars
                            prog = min(1.0, (frame + 1) / ANIM_FRAMES)
                            for rect, h in zip(bars, v_fcf): 
                                rect.set_height(h * prog)
                            limit = int(prog * len(dates))
                            limit = max(1, min(len(dates), limit))
                            sbc_line.set_data(x_idx[:limit], v_sbc[:limit])
                            return bars
                        self.anim = animation.FuncAnimation(self.fig, update, frames=ANIM_FRAMES, interval=20, blit=False, repeat=False)
                    else:
                        for rect, h in zip(bars, v_fcf):
                            rect.set_height(h)
                        sbc_line.set_data(x_idx, v_sbc)
                    self.ax.legend()

        # --- 7. VALUATION ---
        elif mode in ("Valuation (P/E)", "Valuation (FCF Yield)", "Valuation (P/S)"):
            if self.hist_data is not None and self.df_fin_q is not None:
                fin_df = self.get_freq_df("fin")
                eps = self.get_series(fin_df, ["BasicEPS"])
                rev = self.get_series(fin_df, ["TotalRevenue"])
                fcf = self.get_series(self.get_freq_df("cf"), ["FreeCashFlow", "Free Cash Flow"])
                shares = self.get_series(self.get_freq_df("bal"), ["ShareIssued", "CommonStockSharesOutstanding"])
                if freq == "TTM":
                    eps = self.to_ttm(eps)
                    rev = self.to_ttm(rev)
                    fcf = self.to_ttm(fcf)
                if mode == "Valuation (P/E)" and eps is not None:
                    aligned_eps = eps.reindex(self.hist_data.index, method='ffill')
                    pe = self.hist_data['Close'] / (aligned_eps * 4)
                    pe = pe[(pe > 0) & (pe < 250)].dropna()
                    series = pe
                    label = "P/E Ratio"
                    color = "#D165E8"
                elif mode == "Valuation (P/S)" and rev is not None:
                    aligned_rev = rev.reindex(self.hist_data.index, method='ffill')
                    ps = self.hist_data['Close'] * (shares.reindex(self.hist_data.index, method='ffill') if shares is not None else 0) / (aligned_rev * 4)
                    ps = ps[(ps > 0) & (ps < 50)].dropna()
                    series = ps
                    label = "P/S Ratio"
                    color = C_ACCENT
                elif mode == "Valuation (FCF Yield)" and fcf is not None:
                    aligned_fcf = fcf.reindex(self.hist_data.index, method='ffill')
                    mcap = self.hist_data['Close'] * (shares.reindex(self.hist_data.index, method='ffill') if shares is not None else 0)
                    fy = self.safe_div(aligned_fcf * 4, mcap) * 100
                    fy = fy.replace([np.inf, -np.inf], np.nan).dropna()
                    series = fy
                    label = "FCF Yield %"
                    color = C_GREEN
                else:
                    series = None
                if series is not None and not series.empty:
                    x = series.index
                    y = series.values
                    line, = self.ax.plot([], [], color=color, label=label)
                    self.ax.set_xlim(x.min(), x.max())
                    self.ax.set_ylim(y.min(), y.max())
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

        # --- 8. FCF PER SHARE ---
        elif mode == "FCF Per Share":
            cf_df = self.get_freq_df("cf")
            fcf = self.get_series(cf_df, ["FreeCashFlow", "Free Cash Flow"])
            shares = self.get_series(self.get_freq_df("bal"), ["ShareIssued", "CommonStockSharesOutstanding"])
            if fcf is not None and shares is not None:
                if freq == "TTM":
                    fcf = self.to_ttm(fcf)
                fps = self.safe_div(fcf, shares)
                df = pd.DataFrame({'fps': fps}).dropna().sort_index()
                if not df.empty:
                    x = df.index
                    y = df['fps'].values
                    line, = self.ax.plot([], [], color=C_GREEN, marker='o', label="FCF/Share", linewidth=2)
                    self.ax.set_xlim(x.min(), x.max())
                    self.ax.set_ylim(min(0, y.min()*1.1), max(0.1, y.max()*1.2))
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

        self.canvas.draw_idle()


    # --- TAB 3: DCF ---
    def build_valuation(self):
        center = ctk.CTkFrame(self.tab_val, fg_color="transparent")
        center.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        dcf_tabs = ctk.CTkTabview(center, width=800, height=500)
        dcf_tabs.pack(expand=True, fill=tk.BOTH)
        
        fwd = dcf_tabs.add("FORWARD DCF")
        rev = dcf_tabs.add("REVERSE DCF")
        
        self.setup_fwd_dcf(fwd)
        self.setup_rev_dcf(rev)

    def setup_fwd_dcf(self, p):
        grid_frame = ctk.CTkFrame(p, fg_color="transparent")
        grid_frame.pack(expand=True)
        
        ctk.CTkLabel(grid_frame, text="Basis", font=("Roboto", 14)).grid(row=0, column=0, sticky='e', padx=20, pady=10)
        self.dcf_type = tk.StringVar(value="FCF")
        ctk.CTkComboBox(grid_frame, variable=self.dcf_type, values=["FCF", "EPS"], width=120, command=self.on_dcf_type_change).grid(row=0, column=1, sticky='w', padx=20, pady=10)

        self.lbl_fcf = ctk.CTkLabel(grid_frame, text="FCF / Share ($)", font=("Roboto", 14))
        self.lbl_fcf.grid(row=1, column=0, sticky='e', padx=20, pady=10)
        self.f_fcf = ctk.CTkEntry(grid_frame, width=150, justify='center')
        self.f_fcf.insert(0, "0")
        self.f_fcf.grid(row=1, column=1, sticky='w', padx=20, pady=10)
        self.f_gr = self.mk_inp(grid_frame, "Growth Rate (%)", 2, "10")
        self.f_dr = self.mk_inp(grid_frame, "Discount Rate (%)", 3, "10"); self.f_tm = self.mk_inp(grid_frame, "Terminal Mult", 4, "15")
        
        ctk.CTkButton(grid_frame, text="CALCULATE FAIR VALUE", command=self.calc_fwd, width=200, height=40, font=("Roboto", 14, "bold")).grid(row=5, column=0, columnspan=2, pady=30)
        self.lbl_fwd = ctk.CTkLabel(grid_frame, text="$ --", font=("Roboto", 36, "bold"))
        self.lbl_fwd.grid(row=6, column=0, columnspan=2)

    def setup_rev_dcf(self, p):
        grid_frame = ctk.CTkFrame(p, fg_color="transparent")
        grid_frame.pack(expand=True)

        self.lbl_r_fcf = ctk.CTkLabel(grid_frame, text="FCF / Share ($)", font=("Roboto", 14))
        self.lbl_r_fcf.grid(row=0, column=0, sticky='e', padx=20, pady=10)
        self.r_fcf = ctk.CTkEntry(grid_frame, width=150, justify='center')
        self.r_fcf.insert(0, "0")
        self.r_fcf.grid(row=0, column=1, sticky='w', padx=20, pady=10)
        self.r_gr = self.mk_inp(grid_frame, "Growth Rate (%)", 1, "10")
        self.r_ret = self.mk_inp(grid_frame, "Target Return (%)", 2, "15"); self.r_tm = self.mk_inp(grid_frame, "Terminal Mult", 3, "15")
        
        ctk.CTkButton(grid_frame, text="CALC MAX BUY PRICE", command=self.calc_rev, width=200, height=40, font=("Roboto", 14, "bold")).grid(row=4, column=0, columnspan=2, pady=30)
        self.lbl_rev = ctk.CTkLabel(grid_frame, text="$ --", font=("Roboto", 36, "bold"))
        self.lbl_rev.grid(row=5, column=0, columnspan=2)

    def mk_inp(self, p, txt, r, val="0"):
        ctk.CTkLabel(p, text=txt, font=("Roboto", 14)).grid(row=r, column=0, sticky='e', padx=20, pady=10)
        e = ctk.CTkEntry(p, width=150, justify='center')
        e.insert(0, val)
        e.grid(row=r, column=1, sticky='w', padx=20, pady=10)
        return e

    def on_dcf_type_change(self, _=None):
        lbl = "FCF / Share ($)" if self.dcf_type.get() == "FCF" else "EPS ($)"
        if hasattr(self, "lbl_fcf"):
            self.lbl_fcf.configure(text=lbl)
        if hasattr(self, "lbl_r_fcf"):
            self.lbl_r_fcf.configure(text=lbl)

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

    def calc_fwd(self):
        try:
            f,g,r,m = float(self.f_fcf.get()), float(self.f_gr.get())/100, float(self.f_dr.get())/100, float(self.f_tm.get())
            vals = [f*((1+g)**i) for i in range(1,6)]; term = vals[-1]*m
            pv = sum([self.safe_div(v, ((1+r)**(i+1))) for i,v in enumerate(vals)]) + self.safe_div(term, ((1+r)**5))
            self.lbl_fwd.configure(text=f"${pv:.2f}")
            self.store.upsert_dcf_inputs(self.current_symbol, self.dcf_type.get(), f, g, r, m, float(self.r_ret.get())/100 if self.r_ret.get() else 0.0)
        except: pass

    def calc_rev(self):
        try:
            f,g,ret,m = float(self.r_fcf.get()), float(self.r_gr.get())/100, float(self.r_ret.get())/100, float(self.r_tm.get())
            vals = [f*((1+g)**i) for i in range(1,6)]; term = vals[-1]*m
            max_price = sum([self.safe_div(v, ((1+ret)**(i+1))) for i,v in enumerate(vals)]) + self.safe_div(term, ((1+ret)**5))
            self.lbl_rev.configure(text=f"${max_price:.2f}")
            self.store.upsert_dcf_inputs(self.current_symbol, self.dcf_type.get(), f, g, float(self.f_dr.get())/100 if self.f_dr.get() else 0.0, m, ret)
        except: pass

    # --- TAB 4: INSIDER & CALENDAR ---
    def build_insider_calendar(self):
        self.tab_insider.grid_columnconfigure(0, weight=1)
        self.tab_insider.grid_columnconfigure(1, weight=1)
        self.tab_insider.grid_rowconfigure(0, weight=1)

        # INSIDER FRAME
        f1 = ctk.CTkFrame(self.tab_insider, fg_color=C_BG_CARD)
        f1.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        
        ctk.CTkLabel(f1, text="INSIDER TRADING", font=("Roboto", 14, "bold"), text_color=C_ACCENT).pack(anchor='w', pady=10, padx=10)
        
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
        f2 = ctk.CTkFrame(self.tab_insider, fg_color=C_BG_CARD)
        f2.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        
        ctk.CTkLabel(f2, text="EARNINGS CALENDAR", font=("Roboto", 14, "bold"), text_color=C_ACCENT).pack(anchor='w', pady=10, padx=10)
        
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
        for i in self.tr_insider.get_children(): self.tr_insider.delete(i)
        if not ins.empty:
            ins = ins.reset_index()
            for _, r in ins.iterrows():
                 d = str(r.get('Date', 'N/A')); 
                 if isinstance(r.get('Date'), pd.Timestamp): d = r.get('Date').strftime('%Y-%m-%d')
                 who = str(r.get('Insider', r.get('Text', 'N/A'))); val = f"{r.get('Value',0):,.0f}"
                 txt = str(r.values).lower(); typ = "BUY" if "buy" in txt else "SELL" if "sale" in txt or "sell" in txt else "-"
                 self.tr_insider.insert("", tk.END, values=(d, who, typ, val))
        
        for i in self.tr_cal.get_children(): self.tr_cal.delete(i)
        if not cal.empty:
            for _, r in cal.iterrows():
                raw_date = r.iloc[0]; raw_val = r.iloc[1]
                if isinstance(raw_val, (list, tuple)): raw_val = raw_val[0] if len(raw_val)>0 else ""
                d_str = str(raw_date)
                if hasattr(raw_date, 'strftime'): d_str = raw_date.strftime('%Y-%m-%d')
                if "datetime" in d_str: 
                    match = re.search(r"(\d{4}),\s*(\d{1,2}),\s*(\d{1,2})", d_str)
                    if match: d_str = f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
                self.tr_cal.insert("", tk.END, values=(d_str, str(raw_val)))

    # --- TAB 5: AI ---
    def build_ai(self):
        pad = ctk.CTkFrame(self.tab_ai, fg_color=C_BG_CARD)
        pad.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        btns = ctk.CTkFrame(pad, fg_color="transparent")
        btns.pack(fill=tk.X, pady=(0,10))
        ctk.CTkButton(btns, text="Company Analysis", command=self.run_ai_company, width=140).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btns, text="Competitive & Risks", command=self.run_ai_competitive, width=160).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btns, text="Earnings Summary", command=self.run_ai_earnings, width=140).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btns, text="Portfolio Analysis", command=self.run_ai_portfolio, width=140).pack(side=tk.LEFT, padx=5)
        
        self.txt_ai = ctk.CTkTextbox(pad, font=("Roboto", 14), fg_color=C_BG_MAIN, text_color=C_TEXT_MAIN)
        self.txt_ai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def run_gemini_25_logic(self, sym, info, f):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        # Expanded model list to include standard fallback
        target_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-pro"]
        genai.configure(api_key=self.gemini_api_key)
        
        fin_txt = "N/A"
        if f is not None and not f.empty:
            fin_txt = f.tail(8).iloc[:, :2].to_string()
        prompt = f"Role: Senior Wall Street Analyst. Task: Analyze {sym}. Data: {info.get('longBusinessSummary')} Financials: {fin_txt}. Output: # ANALYSIS REPORT: {sym}. Sections: Economic Moat, Key Risks, Financial Health, Valuation."
        
        success = False; final_response = ""; used_model = ""
        try:
            if self.gemini_models is None:
                self.gemini_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            avail = self.gemini_models
            prio = [a for t in target_models for a in avail if t in a]
            if not prio: prio = ["models/gemini-1.5-flash"] # Fallback
            
            for m in prio:
                try:
                    model = genai.GenerativeModel(m); resp = model.generate_content(prompt)
                    if resp.text: final_response = resp.text; used_model = m; success = True; break
                except: continue
        except Exception as e: self.update_ai(f"System Error: {e}"); return
        
        if success:
            self.update_ai(f"MODEL: {used_model}\n\n{final_response}")
        else:
            self.update_ai("Analysis Unavailable.")

    def run_ai_company(self):
        if not self.current_symbol or not self.current_ticker:
            return
        threading.Thread(target=self.run_gemini_25_logic, args=(self.current_symbol, self.current_ticker.info, self.df_fin_q), daemon=True).start()

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
        threading.Thread(target=self.run_gemini_prompt, args=(prompt,), daemon=True).start()

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
        threading.Thread(target=self.run_gemini_prompt, args=(prompt,), daemon=True).start()

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
        threading.Thread(target=self.run_gemini_prompt, args=(prompt,), daemon=True).start()

    def run_gemini_prompt(self, prompt: str):
        if not self.gemini_api_key: self.update_ai("Error: API Key missing."); return
        target_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-pro"]
        genai.configure(api_key=self.gemini_api_key)
        success = False; final_response = ""; used_model = ""
        try:
            if self.gemini_models is None:
                self.gemini_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            avail = self.gemini_models
            prio = [a for t in target_models for a in avail if t in a]
            if not prio: prio = ["models/gemini-1.5-flash"]
            for m in prio:
                try:
                    model = genai.GenerativeModel(m); resp = model.generate_content(prompt)
                    if resp.text: final_response = resp.text; used_model = m; success = True; break
                except: continue
        except Exception as e: self.update_ai(f"System Error: {e}"); return
        if success:
            self.update_ai(f"MODEL: {used_model}\n\n{final_response}")
        else:
            self.update_ai("Analysis Unavailable.")

    def gemini_generate(self, prompt: str):
        if not self.gemini_api_key:
            return None
        target_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-pro"]
        genai.configure(api_key=self.gemini_api_key)
        try:
            if self.gemini_models is None:
                self.gemini_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            avail = self.gemini_models
            prio = [a for t in target_models for a in avail if t in a]
            if not prio: prio = ["models/gemini-1.5-flash"]
            for m in prio:
                try:
                    model = genai.GenerativeModel(m)
                    resp = model.generate_content(prompt)
                    if resp.text:
                        return resp.text
                except:
                    continue
        except:
            return None
        return None

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
        ctk.CTkLabel(top, text="Watchlist:", font=("Roboto", 12, "bold")).pack(side=tk.LEFT, padx=(0,8))
        self.watchlist_name_var = tk.StringVar(value=self.current_watchlist)
        self.watchlist_menu = ctk.CTkComboBox(top, variable=self.watchlist_name_var, values=self.store.list_watchlists() or ["Main"], width=140, command=self.on_watchlist_change)
        self.watchlist_menu.pack(side=tk.LEFT)
        ctk.CTkButton(top, text="NEW LIST", command=self.create_watchlist, width=100).pack(side=tk.LEFT, padx=8)
        
        tv_frame = ctk.CTkFrame(container, fg_color="transparent")
        tv_frame.pack(fill=tk.BOTH, expand=True)

        self.tr_watch = ttk.Treeview(tv_frame, columns=cols, show='headings')
        for c in cols: self.tr_watch.heading(c, text=c.upper()); self.tr_watch.column(c, anchor="center")
        
        # SCROLLBAR WATCHLIST
        vsb = ctk.CTkScrollbar(tv_frame, orientation="vertical", command=self.tr_watch.yview)
        self.tr_watch.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tr_watch.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.tr_watch.bind("<Double-1>", lambda e: self.analyze_watchlist())
        self.tr_watch.bind("<<TreeviewSelect>>", lambda e: self.update_watchlist_chart())
        
        b_frame = ctk.CTkFrame(self.tab_watchlist, fg_color="transparent")
        b_frame.pack(pady=(0,20))
        
        ctk.CTkButton(b_frame, text="REFRESH", command=self.refresh_watchlist, width=100).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(b_frame, text="ANALYZE", command=self.analyze_watchlist, width=100).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(b_frame, text="REMOVE", command=self.remove_from_watchlist, width=100, fg_color=C_RED).pack(side=tk.LEFT, padx=5)

        # Dip Finder Chart
        chart_frame = ctk.CTkFrame(self.tab_watchlist, fg_color=C_BG_CARD)
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
        ctk.CTkLabel(top, text="Earnings Calendar (Watchlists + Portfolios)", font=("Roboto", 12, "bold"), text_color=C_ACCENT).pack(side=tk.LEFT)
        ctk.CTkButton(top, text="REFRESH", command=self.refresh_earnings, width=100).pack(side=tk.RIGHT)
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
        threading.Thread(target=self._proc_earnings, daemon=True).start()

    def auto_refresh_earnings(self, symbol: str):
        for i in self.tr_earn.get_children(): self.tr_earn.delete(i)
        threading.Thread(target=self._proc_earnings_for_symbol, args=(symbol,), daemon=True).start()

    def _render_earnings(self, rows):
        if hasattr(self, "lbl_earn_status"):
            if rows:
                self.lbl_earn_status.configure(text=f"Source: {getattr(self, 'earnings_source', 'Unknown')}")
            else:
                src = getattr(self, "earnings_source", "Unknown")
                err = getattr(self, "earnings_error", "")
                msg = f"No upcoming earnings in range for selected symbols. Source: {src}"
                if err:
                    msg = f"{msg} | Details: {err}"
                self.lbl_earn_status.configure(text=msg)
        for r in rows:
            self.tr_earn.insert("", tk.END, values=r)

    def _proc_earnings(self):
        symbols = set()
        for wl in self.store.list_watchlists():
            symbols.update(self.store.get_watchlist(wl))
        for p in self.store.list_portfolios():
            symbols.update([s for s, _, _ in self.store.get_portfolio(p)])
        symbols = {s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()}
        symbols = {s for s in symbols if re.match(r'^[A-Z0-9\\.\\-]+$', s)}
        # Allow matching against base ticker (e.g., AAPL for AAPL.US)
        match_symbols = set(symbols)
        for s in list(symbols):
            if "." in s:
                match_symbols.add(s.split(".")[0])
        rows = []
        sources = set()
        # Try API-based calendar first (Finnhub -> Alpha Vantage), then fall back to yfinance
        api_rows, api_source, api_error = self.fetch_calendar_api_rows()
        if api_rows:
            sources.add(api_source)
            rows.extend([r for r in api_rows if r[0] in match_symbols])
            matched = {r[0] for r in rows}
            missing = match_symbols - matched
        else:
            missing = match_symbols

        # If missing symbols, try Finnhub per-symbol (uses symbol param)
        for s in sorted(missing):
            sym_rows, sym_source, sym_error = self.fetch_calendar_api_rows(symbol=s)
            if sym_rows:
                sources.add(sym_source)
                rows.extend(sym_rows)
            else:
                rows.extend(self.fetch_calendar_yfinance_rows(s))
                sources.add("yfinance")

        # Add past earnings (Finnhub stock/earnings) for each symbol
        for s in sorted(match_symbols):
            past_rows, past_source = self.fetch_past_earnings_rows(s)
            if past_rows:
                sources.add(past_source)
                rows.extend(past_rows)

        if rows:
            self.earnings_source = "+".join(sorted(sources))
            self.earnings_error = ""
        else:
            if api_error:
                self.earnings_error = api_error
            else:
                self.earnings_error = f"No matches for symbols {sorted(list(match_symbols))[:10]}"
        self.root.after(0, lambda rows=rows: self._render_earnings(rows))

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
        ctk.CTkLabel(top, text="Transcripts", font=("Roboto", 12, "bold"), text_color=C_ACCENT).pack(side=tk.LEFT)
        ctk.CTkButton(top, text="IMPORT", command=self.import_transcripts, width=100).pack(side=tk.RIGHT)

        search_row = ctk.CTkFrame(container, fg_color="transparent")
        search_row.pack(fill=tk.X, pady=(0,10))
        self.transcript_query = tk.StringVar()
        ctk.CTkEntry(search_row, textvariable=self.transcript_query, width=200, placeholder_text="Search transcripts").pack(side=tk.LEFT)
        ctk.CTkButton(search_row, text="SEARCH", command=self.search_transcripts, width=80).pack(side=tk.LEFT, padx=8)

        cols = ("Date", "Title")
        self.tr_trans = ttk.Treeview(container, columns=cols, show='headings', height=8)
        for c in cols: self.tr_trans.heading(c, text=c.upper()); self.tr_trans.column(c, width=150, anchor="center")
        self.tr_trans.pack(fill=tk.X, expand=False)
        self.tr_trans.bind("<<TreeviewSelect>>", lambda e: self.show_transcript())

        self.txt_transcript = ctk.CTkTextbox(container, fg_color=C_BG_INPUT)
        self.txt_transcript.pack(fill=tk.BOTH, expand=True, pady=(10,0))

        action_row = ctk.CTkFrame(container, fg_color="transparent")
        action_row.pack(fill=tk.X, pady=(10,5))
        ctk.CTkButton(action_row, text="ANALYZE HOLISTIC", command=self.analyze_selected_transcript, width=160).pack(side=tk.LEFT)
        ctk.CTkButton(action_row, text="REFRESH", command=self.refresh_transcripts, width=100).pack(side=tk.LEFT, padx=8)
        self.lbl_transcript_status = ctk.CTkLabel(action_row, text="", text_color=C_TEXT_SUB)
        self.lbl_transcript_status.pack(side=tk.LEFT, padx=10)

        self.txt_transcript_ai = ctk.CTkTextbox(container, fg_color=C_BG_MAIN)
        self.txt_transcript_ai.pack(fill=tk.BOTH, expand=True, pady=(5,0))

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
            self.txt_transcript.configure(state="normal")
            self.txt_transcript.delete(1.0, tk.END)
            self.txt_transcript.insert(tk.END, content)
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
        threading.Thread(target=self.fetch_sec_earnings_release, args=(self.current_symbol, self.request_id, True), daemon=True).start()
        threading.Thread(target=self.fetch_av_transcripts, args=(self.current_symbol, self.request_id, True), daemon=True).start()
        threading.Thread(target=self.fetch_api_ninjas_transcripts, args=(self.current_symbol, self.request_id, True), daemon=True).start()

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
        threading.Thread(target=self._analyze_transcript_worker, args=(self.current_symbol, date, title, content, news_ctx, fin_ctx, info), daemon=True).start()

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
        ctk.CTkLabel(top, text="Portfolio:", font=("Roboto", 12, "bold")).pack(side=tk.LEFT, padx=(0,8))
        self.portfolio_name_var = tk.StringVar(value=self.current_portfolio)
        self.portfolio_menu = ctk.CTkComboBox(top, variable=self.portfolio_name_var, values=self.store.list_portfolios() or ["Main"], width=140, command=self.on_portfolio_change)
        self.portfolio_menu.pack(side=tk.LEFT)
        ctk.CTkButton(top, text="NEW PORTFOLIO", command=self.create_portfolio, width=120).pack(side=tk.LEFT, padx=8)

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
        ctk.CTkEntry(form, textvariable=self.port_sym, width=100, placeholder_text="Symbol").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.port_shares, width=120, placeholder_text="Shares").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.port_cost, width=120, placeholder_text="Cost Basis").pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="ADD/UPDATE", command=self.add_portfolio, width=120).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="REMOVE", command=self.remove_portfolio, width=100).pack(side=tk.LEFT, padx=5)

        self.lbl_port_total = ctk.CTkLabel(container, text="Total: $0.00", font=("Roboto", 14, "bold"))
        self.lbl_port_total.pack(anchor="e", pady=(10,0))
        self.refresh_portfolio()

    def on_portfolio_change(self, _=None):
        self.current_portfolio = self.portfolio_name_var.get().strip()
        self.refresh_portfolio()

    def create_portfolio(self):
        name = simpledialog.askstring("New Portfolio", "Portfolio name:")
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
        threading.Thread(target=self._proc_portfolio, daemon=True).start()

    def _proc_portfolio(self):
        rows = []
        total = 0.0
        for sym, shares, cost in self.store.get_portfolio(self.current_portfolio):
            price = None
            try:
                with self.cache_lock:
                    price = (self.cache.get(sym, {}) or {}).get("info", {}).get("currentPrice")
            except:
                price = None
            if price is None:
                try:
                    info = yf.Ticker(sym).info
                    price = info.get("currentPrice")
                    with self.cache_lock:
                        cached = self.cache.get(sym, {})
                        self.cache[sym] = {**cached, "info": info}
                except:
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
        except:
            return
        self.store.upsert_portfolio_symbol(self.current_portfolio, sym, shares, cost)
        self.refresh_portfolio()

    def remove_portfolio(self):
        sel = self.tr_port.selection()
        if not sel:
            return
        sym = self.tr_port.item(sel[0])['values'][0]
        self.store.remove_portfolio_symbol(self.current_portfolio, sym)
        self.refresh_portfolio()

    # --- TAB 10: KPIS ---
    def build_kpis(self):
        container = ctk.CTkFrame(self.tab_kpis, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        form = ctk.CTkFrame(container, fg_color="transparent")
        form.pack(fill=tk.X, pady=(0,10))
        self.kpi_name = tk.StringVar(); self.kpi_period = tk.StringVar(); self.kpi_value = tk.StringVar(); self.kpi_unit = tk.StringVar()
        ctk.CTkEntry(form, textvariable=self.kpi_name, width=140, placeholder_text="KPI Name").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.kpi_period, width=120, placeholder_text="Period").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.kpi_value, width=120, placeholder_text="Value").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(form, textvariable=self.kpi_unit, width=80, placeholder_text="Unit").pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="ADD KPI", command=self.add_kpi, width=100).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(form, text="CLEAR", command=self.clear_kpis, width=80).pack(side=tk.LEFT, padx=5)

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
        if self.current_symbol and self.current_symbol not in self.watchlist_data:
            self.watchlist_data.append(self.current_symbol)
            self.store.add_watchlist_symbol(self.current_watchlist, self.current_symbol)
            self.refresh_watchlist()
            self.lbl_status.configure(text=f"Added {self.current_symbol} to watchlist", text_color=C_GREEN)
    def remove_from_watchlist(self):
        sel = self.tr_watch.selection(); 
        if not sel: return
        sym = self.tr_watch.item(sel[0])['values'][0]
        if sym in self.watchlist_data:
            self.watchlist_data.remove(sym)
            self.store.remove_watchlist_symbol(self.current_watchlist, sym)
            self.refresh_watchlist()
    def analyze_watchlist(self):
        sel = self.tr_watch.selection()
        if not sel: return
        self.search_var.set(self.tr_watch.item(sel[0])['values'][0]); self.start_search(); self.tabview.set("INSIGHTS")

    def on_watchlist_change(self, _=None):
        name = self.watchlist_name_var.get().strip()
        if not name:
            return
        self.current_watchlist = name
        self.watchlist_data = self.load_watchlist()
        self.refresh_watchlist()

    def create_watchlist(self):
        name = simpledialog.askstring("New Watchlist", "Watchlist name:")
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
        threading.Thread(target=self._proc_watch, daemon=True).start()
    
    def _proc_watch(self):
        rows = []
        try:
            for s in self.watchlist_data:
                try:
                    with self.cache_lock:
                        cached = self.cache.get(s, {})
                    h = cached.get("history_1y") or yf.Ticker(s).history(period="1y")
                    with self.cache_lock:
                        self.cache[s] = {**cached, "history_1y": h}
                    if h is None or h.empty:
                        continue
                    c = h['Close'].iloc[-1]
                    sma = h['Close'].rolling(200).mean().iloc[-1]
                    st, tag = ("DIP", "dip") if c < sma else ("UP", "up")
                    rows.append((s, c, sma, st, tag))
                except: pass
            self.root.after(0, lambda rows=rows: self._render_watchlist(rows))
        finally:
            self.watchlist_loading = False # Ensure flag is reset
            
    def _render_watchlist(self, rows):
        if not self.root.winfo_exists():
            return
        if not rows:
            return
        for i in self.tr_watch.get_children():
            self.tr_watch.delete(i)
        for s, c, sm, st, t in rows:
            self._add_watch_row(s, c, sm, st, t)

    def _add_watch_row(self, s, c, sm, st, t):
        if not self.root.winfo_exists():
            return
        i = self.tr_watch.insert("", tk.END, values=(s, f"${c:.2f}", f"${sm:.2f}", st))
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
    def on_close(self): self.root.destroy(); sys.exit()

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
        threading.Thread(target=self._fetch_history_period, args=(self.current_symbol, period, self.request_id), daemon=True).start()

    def _fetch_history_period(self, symbol, period, req_id):
        try:
            with self.cache_lock:
                t = (self.cache.get(symbol, {}) or {}).get("ticker")
            t = t or yf.Ticker(symbol)
            h = t.history(period=period)
            with self.cache_lock:
                cached = self.cache.get(symbol, {})
            key = f"history_{period}"
            with self.cache_lock:
                self.cache[symbol] = {**cached, "ticker": t, key: h}
            self.root.after(0, lambda: self._update_history_period(h, period, req_id))
        except:
            pass

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
        except:
            pass

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
        except:
            return None
        return None

    def fetch_sec_companyfacts(self, cik: str):
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            text = self.http_get_text(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if not text:
                return None
            return json.loads(text)
        except:
            return None

    def fetch_sec_submissions(self, cik: str):
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            text = self.http_get_text(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if not text:
                return None
            return json.loads(text)
        except:
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
            clean = re.sub(r"<[^>]+>", " ", doc_text)
            clean = re.sub(r"\s+", " ", clean).strip()
            title = f"Earnings Release {filing_date}"
            if not self.store.transcript_exists(symbol, filing_date or "", title):
                self.store.upsert_transcript(symbol, filing_date or "", title, clean)
            if req_id != self.request_id:
                return
            self.root.after(0, lambda: (self.load_transcripts(), self._mark_transcript_fetch_done("SEC")))
        except:
            self.transcript_fetch_status = "SEC: Unexpected error."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("SEC"))
            pass

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
                time.sleep(12)  # Respect API rate limits
            if req_id != self.request_id:
                return
            self.root.after(0, lambda: (self.load_transcripts(), self._mark_transcript_fetch_done("AlphaVantage")))
        except:
            if not self.transcript_fetch_status:
                self.transcript_fetch_status = "Alpha Vantage: Unexpected error."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("AlphaVantage"))
            pass

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
        except:
            if not self.transcript_fetch_status:
                self.transcript_fetch_status = "API Ninjas: unexpected error."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))
            pass

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

    def extract_sec_series(self, facts, tag_opts: List[str], unit_pref: str = "USD", annual_only: bool = False):
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
                    rows.append((end, val))
                if rows:
                    s = pd.Series({pd.to_datetime(d): v for d, v in rows})
                    s = s.sort_index()
                    return s
        except:
            return None
        return None

    def build_sec_dataframes(self, facts):
        # Financials
        rev = self.extract_sec_series(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
        gp = self.extract_sec_series(facts, ["GrossProfit"])
        op = self.extract_sec_series(facts, ["OperatingIncomeLoss"])
        ni = self.extract_sec_series(facts, ["NetIncomeLoss"])
        ebitda = self.extract_sec_series(facts, ["EarningsBeforeInterestTaxesDepreciationAmortization"])
        eps = self.extract_sec_series(facts, ["EarningsPerShareBasic"], unit_pref="USD/shares")
        eps_d = self.extract_sec_series(facts, ["EarningsPerShareDiluted"], unit_pref="USD/shares")
        fin_q = pd.DataFrame({
            "TotalRevenue": rev,
            "GrossProfit": gp,
            "OperatingIncome": op,
            "NetIncome": ni,
            "EBITDA": ebitda,
            "BasicEPS": eps,
            "DilutedEPS": eps_d,
        }).dropna(how="all")

        rev_a = self.extract_sec_series(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], annual_only=True)
        gp_a = self.extract_sec_series(facts, ["GrossProfit"], annual_only=True)
        op_a = self.extract_sec_series(facts, ["OperatingIncomeLoss"], annual_only=True)
        ni_a = self.extract_sec_series(facts, ["NetIncomeLoss"], annual_only=True)
        ebitda_a = self.extract_sec_series(facts, ["EarningsBeforeInterestTaxesDepreciationAmortization"], annual_only=True)
        eps_a = self.extract_sec_series(facts, ["EarningsPerShareBasic"], unit_pref="USD/shares", annual_only=True)
        epsd_a = self.extract_sec_series(facts, ["EarningsPerShareDiluted"], unit_pref="USD/shares", annual_only=True)
        fin_a = pd.DataFrame({
            "TotalRevenue": rev_a,
            "GrossProfit": gp_a,
            "OperatingIncome": op_a,
            "NetIncome": ni_a,
            "EBITDA": ebitda_a,
            "BasicEPS": eps_a,
            "DilutedEPS": epsd_a,
        }).dropna(how="all")

        # Balance Sheet
        cash = self.extract_sec_series(facts, ["CashAndCashEquivalentsAtCarryingValue"])
        debt = self.extract_sec_series(facts, ["LongTermDebt", "Debt"])
        equity = self.extract_sec_series(facts, ["StockholdersEquity"])
        shares = self.extract_sec_series(facts, ["CommonStockSharesOutstanding"], unit_pref="shares")
        bal_q = pd.DataFrame({
            "CashAndCashEquivalents": cash,
            "TotalDebt": debt,
            "TotalStockholderEquity": equity,
            "ShareIssued": shares,
        }).dropna(how="all")

        cash_a = self.extract_sec_series(facts, ["CashAndCashEquivalentsAtCarryingValue"], annual_only=True)
        debt_a = self.extract_sec_series(facts, ["LongTermDebt", "Debt"], annual_only=True)
        equity_a = self.extract_sec_series(facts, ["StockholdersEquity"], annual_only=True)
        shares_a = self.extract_sec_series(facts, ["CommonStockSharesOutstanding"], unit_pref="shares", annual_only=True)
        bal_a = pd.DataFrame({
            "CashAndCashEquivalents": cash_a,
            "TotalDebt": debt_a,
            "TotalStockholderEquity": equity_a,
            "ShareIssued": shares_a,
        }).dropna(how="all")

        # Cash Flow
        ocf = self.extract_sec_series(facts, ["NetCashProvidedByUsedInOperatingActivities"])
        capex = self.extract_sec_series(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"])
        sbc = self.extract_sec_series(facts, ["ShareBasedCompensation", "StockBasedCompensation"])
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf + capex
        cf_q = pd.DataFrame({
            "OperatingCashFlow": ocf,
            "CapitalExpenditure": capex,
            "FreeCashFlow": fcf,
            "StockBasedCompensation": sbc,
        }).dropna(how="all")

        ocf_a = self.extract_sec_series(facts, ["NetCashProvidedByUsedInOperatingActivities"], annual_only=True)
        capex_a = self.extract_sec_series(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"], annual_only=True)
        sbc_a = self.extract_sec_series(facts, ["ShareBasedCompensation", "StockBasedCompensation"], annual_only=True)
        fcf_a = None
        if ocf_a is not None and capex_a is not None:
            fcf_a = ocf_a + capex_a
        cf_a = pd.DataFrame({
            "OperatingCashFlow": ocf_a,
            "CapitalExpenditure": capex_a,
            "FreeCashFlow": fcf_a,
            "StockBasedCompensation": sbc_a,
        }).dropna(how="all")

        return fin_q, fin_a, bal_q, bal_a, cf_q, cf_a

if __name__ == "__main__":
    root = ctk.CTk()
    app = OpenTrimApp(root)
    root.mainloop()
