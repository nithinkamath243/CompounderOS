import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
import io
import html
import textwrap
from datetime import datetime
from typing import Optional, Dict, List, Any

from .database import DataStore
from .ai_engine import GeminiEngine
from .analysis import (
    safe_div as analysis_safe_div,
    clean_numeric_series as analysis_clean_numeric_series,
    clean_datetime_index_df as analysis_clean_datetime_index_df,
    calculate_technical_indicators as analysis_calculate_technical_indicators,
    align_to_quarter_end as analysis_align_to_quarter_end,
    align_to_year_end as analysis_align_to_year_end,
    calculate_piotroski_f_score as analysis_calculate_piotroski_f_score,
)
from .data_fetcher import http_get_text as shared_http_get_text
from .charts import CHART_OPTIONS, redraw_chart_now
from .config import (
    CONFIG,
    LOG_DIR,
    EXPORT_DIR,
    WATCHLIST_FILE,
    LEGACY_WATCHLIST_FILE,
    DATA_DB,
    TRANSCRIPTS_DIR,
    ensure_dirs,
    get_service_key,
    get_sec_user_agent,
    set_setting,
    sec_user_agent_is_valid,
)
from concurrent.futures import ThreadPoolExecutor
import logging
try:
    from cachetools import TTLCache
except Exception:
    class TTLCache(dict):
        def __init__(self, maxsize=256, ttl=300):
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl
            self._exp = {}

        def _purge(self):
            now = time.time()
            expired = [k for k, v in self._exp.items() if v <= now]
            for k in expired:
                self.pop(k, None)
                self._exp.pop(k, None)
            while len(self) > self.maxsize:
                k = next(iter(self))
                self.pop(k, None)
                self._exp.pop(k, None)

        def __setitem__(self, key, value):
            self._purge()
            super().__setitem__(key, value)
            self._exp[key] = time.time() + self.ttl

        def get(self, key, default=None):
            self._purge()
            return super().get(key, default)

# --- SSL CERTIFICATE FIX (For MacOS) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- CONFIGURATION ---
ensure_dirs()

LOG_FILE = os.path.join(LOG_DIR, "opentrim_errors.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def log_exception(context: str, exc: Exception):
    logging.error("%s: %s", context, exc, exc_info=True)


SEC_USER_AGENT = get_sec_user_agent()

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
C_PLOT_FRAME  = "#2A364A"

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
        self.ai_engine = GeminiEngine(self.gemini_api_key)
        self.sec_user_agent = SEC_USER_AGENT
        self._ensure_sec_user_agent()
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
        self._view_cache_lock = threading.Lock()
        self.metric_payloads: Dict[str, Dict[str, str]] = {}
        self.quality_ai_cache: Dict[str, Dict[str, str]] = {}
        self.quality_ai_inflight = False
        self.quality_ai_key = ""
        self._shown_tips = set()
        self.watchlist_rows_cache = TTLCache(maxsize=32, ttl=180)
        self.portfolio_rows_cache = TTLCache(maxsize=32, ttl=180)
        
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

    def _ensure_sec_user_agent(self):
        if sec_user_agent_is_valid(self.sec_user_agent):
            return
        try:
            dlg = ctk.CTkInputDialog(
                text="SEC API needs a valid contact email. Enter your email:",
                title="SEC User-Agent Setup"
            )
            email = (dlg.get_input() or "").strip()
            if email and "@" in email:
                self.sec_user_agent = f"OpenTrim/4.2 (contact: {email})"
                set_setting("SEC_USER_AGENT", self.sec_user_agent)
        except Exception as e:
            log_exception("_ensure_sec_user_agent", e)

    def load_api_key(self):
        cfg_key = CONFIG.get("GEMINI_API_KEY")
        return cfg_key.strip() if isinstance(cfg_key, str) and cfg_key.strip() else None

    def save_api_key(self, key):
        set_setting("GEMINI_API_KEY", key)
        self.gemini_api_key = key
        self.ai_engine.set_api_key(key)

    # --- MATH HELPER ---
    def safe_div(self, n, d, default=0.0):
        return analysis_safe_div(n, d, default)

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

    def _show_toast(self, message: str, duration_ms: int = 3200):
        try:
            if not self.root.winfo_exists():
                return
            toast = ctk.CTkToplevel(self.root)
            toast.overrideredirect(True)
            try:
                toast.attributes("-topmost", True)
            except Exception:
                pass
            try:
                toast.attributes("-alpha", 0.96)
            except Exception:
                pass

            frame = ctk.CTkFrame(
                toast,
                fg_color=C_BG_CARD,
                corner_radius=12,
                border_width=1,
                border_color=C_BORDER,
            )
            frame.pack(fill=tk.BOTH, expand=True)
            ctk.CTkLabel(
                frame,
                text=message,
                font=FONT_SMALL,
                text_color=C_TEXT_MAIN,
            ).pack(padx=14, pady=10)

            self.root.update_idletasks()
            w, h = 360, 52
            x = self.root.winfo_rootx() + self.root.winfo_width() - w - 20
            y = self.root.winfo_rooty() + 70
            toast.geometry(f"{w}x{h}+{max(20, x)}+{max(20, y)}")
            toast.after(duration_ms, toast.destroy)
        except Exception as e:
            log_exception("_show_toast", e)

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
        if not get_service_key("FINNHUB_API_KEY") and "finnhub_tip" not in self._shown_tips:
            self._shown_tips.add("finnhub_tip")
            self._show_toast("Tip: Add Finnhub key for Earnings Data.")
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
        
        self.current_ticker = t
        self.current_symbol = s
        self.df_fin_q = analysis_clean_datetime_index_df(fin_q)
        self.df_bal_q = analysis_clean_datetime_index_df(bs_q)
        self.df_cf_q = analysis_clean_datetime_index_df(cf_q)
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
        try:
            return analysis_calculate_technical_indicators(df)
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

        radar_wrap = ctk.CTkFrame(self.f_verdict, fg_color=C_BG_INPUT, corner_radius=12, border_width=1, border_color=C_BORDER)
        radar_wrap.pack(fill=tk.BOTH, expand=False, padx=15, pady=(0, 12))
        ctk.CTkLabel(radar_wrap, text="QUALITY RADAR", font=FONT_SMALL, text_color=C_ACCENT).pack(anchor="w", padx=10, pady=(8, 0))
        self.fig_quality, self.ax_quality = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(3.2, 2.6), dpi=100)
        self.fig_quality.patch.set_edgecolor(C_PLOT_FRAME)
        self.fig_quality.patch.set_linewidth(1.0)
        self.canvas_quality = FigureCanvasTkAgg(self.fig_quality, master=radar_wrap)
        self.canvas_quality.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

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
        radar_groups = {
            "Moat": [],
            "Growth": [],
            "Efficiency": [],
            "Safety": [],
            "Capital Allocation": [],
            "Valuation": [],
        }

        def add_m(name, target_display, actual_val, passed, short_ctx, logic, calc, source, include_in_score=True, category: str = "Moat"):
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
            if category not in radar_groups:
                radar_groups[category] = []
            radar_groups[category].append(bool(passed))
            if not include_in_score:
                return 0
            return 1 if passed else 0

        def finite_num(v):
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except Exception:
                pass
            return None

        clean_series = analysis_clean_numeric_series

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
            gm_source,
            category="Moat",
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
            f"{rev_yoy_source} + {gm_source}",
            category="Moat",
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
            "Quarterly Financials",
            category="Growth",
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
            rev_yoy_source,
            category="Growth",
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
            cap_source,
            category="Capital Allocation",
        )

        # Core 6: Smart Capital Allocation
        if fcf_a is None and ocf_a is not None and capex_a is not None:
            fcf_a = ocf_a - capex_a.abs()
        if fcf_q is None and ocf_q is not None and capex_q is not None:
            fcf_q = ocf_q - capex_q.abs()
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
            f"{conv_source} + {bb_source} + Yahoo Info",
            category="Capital Allocation",
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
            roe_source,
            category="Efficiency",
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
            cr_source,
            category="Safety",
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
            val_source,
            category="Valuation",
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
            "Analyst Estimates",
            category="Valuation",
        )

        # Supplemental quality check (not included in core score)
        pio = analysis_calculate_piotroski_f_score(self.df_fin_a, self.df_bal_a, self.df_cf_a)
        if pio is not None:
            p_score = int(pio.get("score", 0))
            p_max = int(pio.get("max_score", 9))
            p_target = max(1, int(np.ceil(p_max * 0.75)))
            p_pass = p_score >= p_target
            p_ctx = "Strong financial momentum" if p_pass else "Mixed financial momentum"
            add_m(
                "Piotroski F-Score",
                f">= {p_target}/{p_max}",
                f"{p_score}/{p_max}",
                p_pass,
                p_ctx,
                "Nine-point financial strength model using profitability, leverage/liquidity, and operating efficiency trends.",
                "Year-over-year checks: NI, OCF, ROA, accruals, leverage, current ratio, dilution, gross margin, asset turnover.",
                "Annual Financial Statements",
                include_in_score=False,
                category="Safety",
            )

        self.tr_check.tag_configure("pass", foreground=C_GREEN)
        self.tr_check.tag_configure("fail", foreground=C_RED)
        self.lbl_score.configure(text=f"SCORE: {score}/10", text_color=C_GREEN if score >= 7 else C_YELLOW)
        
        verdict = "VERDICT: HOLD / WATCH\n\n"
        if score >= 8: verdict = "VERDICT: ELITE COMPOUNDER (STRONG BUY)\n\n"
        elif score <= 4: verdict = "VERDICT: LOW QUALITY / SPECULATIVE\n\n"
        base_note = "Core framework follows CompounderOS Quality Framework: Monopoly, Pricing Power, Operating Leverage, Organic Growth, Capital Light, Smart Capital Allocation.\n\n"
        self.txt_verdict.delete(1.0, tk.END); self.txt_verdict.insert(tk.END, verdict + base_note + "Select any metric in the table above to see the Deep Dive explanation here.")
        self._render_quality_radar(radar_groups)
        self._enhance_quality_deep_dive(symbol)

    def _render_quality_radar(self, grouped_scores: Dict[str, List[bool]]):
        if not hasattr(self, "ax_quality"):
            return
        self.ax_quality.clear()
        self.ax_quality.set_facecolor(C_BG_CARD)
        try:
            self.ax_quality.spines["polar"].set_color(C_PLOT_FRAME)
        except Exception:
            pass
        categories = [k for k, v in grouped_scores.items() if v]
        if not categories:
            self.ax_quality.text(0.5, 0.5, "No quality data", ha="center", va="center", color=C_TEXT_SUB, transform=self.ax_quality.transAxes)
            self.canvas_quality.draw_idle()
            return
        values = [float(np.mean(grouped_scores[k])) for k in categories]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        self.ax_quality.plot(angles, values, color=C_ACCENT, linewidth=2)
        self.ax_quality.fill(angles, values, color=C_ACCENT, alpha=0.22)
        self.ax_quality.set_xticks(angles[:-1])
        self.ax_quality.set_xticklabels(categories, color=C_TEXT_MAIN, fontsize=8)
        self.ax_quality.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        self.ax_quality.set_yticklabels([])
        self.ax_quality.grid(True, color=C_PLOT_FRAME, linestyle="--", alpha=0.35)
        try:
            self.fig_quality.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.95)
        except Exception:
            pass
        self.canvas_quality.draw_idle()

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
        cb = self.make_combo(bar, variable=self.chart_var, values=CHART_OPTIONS, width=250, command=self.redraw_chart, **COMBO_KW)
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
        self.fig.patch.set_edgecolor(C_PLOT_FRAME)
        self.fig.patch.set_linewidth(1.2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

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
            df = df.set_index("Date").sort_index()
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

    def _apply_axes_frame(self, axis):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(C_PLOT_FRAME)
            spine.set_linewidth(1.0)

    def _apply_layout(self):
        try:
            # Extra top/bottom room prevents title and x-label clipping on high-DPI displays.
            self.fig.subplots_adjust(bottom=0.25, left=0.08, right=0.97, top=0.82)
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

    # --- GRAPHING ---
    def redraw_chart(self, _=None):
        # Debounce rapid redraws
        if self._redraw_job:
            try: self.root.after_cancel(self._redraw_job)
            except: pass
        self._redraw_job = self.root.after(REDRAW_DEBOUNCE_MS, self._redraw_chart_now)

    def _redraw_chart_now(self):
        redraw_chart_now(
            self,
            palette={
                "C_ACCENT": C_ACCENT,
                "C_BORDER": C_BORDER,
                "C_GREEN": C_GREEN,
                "C_RED": C_RED,
                "C_YELLOW": C_YELLOW,
                "C_PINK": C_PINK,
                "C_BG_CARD": C_BG_CARD,
                "C_TEXT_MAIN": C_TEXT_MAIN,
                "C_TEXT_SUB": C_TEXT_SUB,
                "C_PLOT_FRAME": C_PLOT_FRAME,
            },
            anim_frames=ANIM_FRAMES,
        )


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
        f = self._parse_dcf_float(self.f_fcf, "FCF / EPS per share")
        g_pct = self._parse_dcf_float(self.f_gr, "Growth Rate (%)")
        r_pct = self._parse_dcf_float(self.f_dr, "Discount Rate (%)")
        m = self._parse_dcf_float(self.f_tm, "Terminal Multiple")
        if None in (f, g_pct, r_pct, m):
            return
        g = g_pct / 100.0
        r = r_pct / 100.0
        if r <= -0.99:
            self.lbl_status.configure(text="DCF error: Discount rate must be > -99%.", text_color=C_RED)
            return
        vals = [f * ((1 + g) ** i) for i in range(1, 6)]
        term = vals[-1] * m
        pv = sum([self.safe_div(v, ((1 + r) ** (i + 1))) for i, v in enumerate(vals)]) + self.safe_div(term, ((1 + r) ** 5))
        self.lbl_fwd.configure(text=f"${pv:.2f}")
        if self.current_symbol:
            target = self._parse_dcf_float(self.r_ret, "Target Return (%)", required=False)
            self.store.upsert_dcf_inputs(
                self.current_symbol,
                self.dcf_type.get(),
                f,
                g,
                r,
                m,
                (target / 100.0) if target is not None else 0.0,
            )

    def calc_rev(self):
        f = self._parse_dcf_float(self.r_fcf, "FCF / EPS per share")
        g_pct = self._parse_dcf_float(self.r_gr, "Growth Rate (%)")
        ret_pct = self._parse_dcf_float(self.r_ret, "Target Return (%)")
        m = self._parse_dcf_float(self.r_tm, "Terminal Multiple")
        if None in (f, g_pct, ret_pct, m):
            return
        g = g_pct / 100.0
        ret = ret_pct / 100.0
        if ret <= -0.99:
            self.lbl_status.configure(text="DCF error: Target return must be > -99%.", text_color=C_RED)
            return
        vals = [f * ((1 + g) ** i) for i in range(1, 6)]
        term = vals[-1] * m
        max_price = sum([self.safe_div(v, ((1 + ret) ** (i + 1))) for i, v in enumerate(vals)]) + self.safe_div(term, ((1 + ret) ** 5))
        self.lbl_rev.configure(text=f"${max_price:.2f}")
        if self.current_symbol:
            discount = self._parse_dcf_float(self.f_dr, "Discount Rate (%)", required=False)
            self.store.upsert_dcf_inputs(
                self.current_symbol,
                self.dcf_type.get(),
                f,
                g,
                (discount / 100.0) if discount is not None else 0.0,
                m,
                ret,
            )

    def _parse_dcf_float(self, widget, label: str, required: bool = True):
        txt = widget.get().strip()
        if not txt:
            if required:
                self.lbl_status.configure(text=f"DCF error: {label} is required.", text_color=C_RED)
                return None
            return None
        try:
            return float(txt)
        except Exception:
            self.lbl_status.configure(text=f"DCF error: {label} must be numeric.", text_color=C_RED)
            return None

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
        self.ai_engine.set_api_key(self.gemini_api_key)
        return self.ai_engine.model_candidates()

    def _gemini_generate_with_model(self, prompt: str):
        if not self.gemini_api_key:
            return None, None
        self.ai_engine.set_api_key(self.gemini_api_key)
        return self.ai_engine.generate(prompt)

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
        self.fig_wl, self.ax_wl = plt.subplots(figsize=(6,3.4), dpi=100)
        self.fig_wl.patch.set_edgecolor(C_PLOT_FRAME)
        self.fig_wl.patch.set_linewidth(1.2)
        self.canvas_wl = FigureCanvasTkAgg(self.fig_wl, master=chart_frame)
        self.canvas_wl.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
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
        finnhub_key = get_service_key("FINNHUB_API_KEY")
        alpha_key = get_service_key("ALPHAVANTAGE_API_KEY")
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
        if finnhub_key:
            try:
                today = datetime.now().date()
                start = today.strftime("%Y-%m-%d")
                end = (today + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
                url = f"https://finnhub.io/api/v1/calendar/earnings?from={start}&to={end}&token={finnhub_key}"
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
        if alpha_key:
            try:
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={alpha_key}"
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
        finnhub_key = get_service_key("FINNHUB_API_KEY")
        if finnhub_key:
            try:
                url = f"https://finnhub.io/api/v1/stock/earnings?symbol={symbol}&limit={PAST_EARNINGS_COUNT}&token={finnhub_key}"
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
        transcript_text = self._build_transcript_ai_input(content, qa_section, max_chars=16000)
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

    def _truncate_text(self, text: str, max_chars: int):
        if not text:
            return ""
        t = str(text)
        return t if len(t) <= max_chars else (t[:max_chars] + "\n\n[...truncated...]")

    def _build_transcript_ai_input(self, content: str, qa_section: str, max_chars: int = 16000):
        if not content:
            return ""
        body = str(content)
        qa = str(qa_section or "")
        if len(body) <= max_chars:
            return body

        # Keep strongest signal for earnings analysis: opening context + Q&A + ending guidance.
        head_budget = int(max_chars * 0.35)
        qa_budget = int(max_chars * 0.40)
        tail_budget = max_chars - head_budget - qa_budget

        head = self._truncate_text(body[: head_budget * 2], head_budget)
        tail = self._truncate_text(body[-tail_budget * 2 :], tail_budget)
        qa_part = self._truncate_text(qa, qa_budget)

        parts = [
            head,
            "\n\n[...middle omitted...]\n\n",
            "Q&A SECTION (prioritized):\n",
            qa_part or "No explicit Q&A section found.",
            "\n\n[...closing section...]\n\n",
            tail,
        ]
        return "".join(parts)

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

        body = ctk.CTkFrame(container, fg_color="transparent")
        body.pack(fill=tk.BOTH, expand=True)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        table_frame = ctk.CTkFrame(body, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        cols = ("Symbol", "Shares", "Cost Basis", "Price", "Value")
        self.tr_port = ttk.Treeview(table_frame, columns=cols, show='headings')
        for c in cols:
            self.tr_port.heading(c, text=c.upper())
            self.tr_port.column(c, width=110, anchor="center")
        vsb = ctk.CTkScrollbar(table_frame, orientation="vertical", command=self.tr_port.yview)
        self.tr_port.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y, pady=10, padx=(0, 8))
        self.tr_port.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10, padx=(10, 0))

        chart_wrap = ctk.CTkFrame(body, fg_color=C_BG_CARD, corner_radius=16, border_width=1, border_color=C_BORDER)
        chart_wrap.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(chart_wrap, text="Sector Allocation", font=FONT_H2, text_color=C_ACCENT).pack(anchor="w", padx=12, pady=(10, 0))
        self.fig_port, self.ax_port = plt.subplots(figsize=(4.2, 3.2), dpi=100)
        self.fig_port.patch.set_edgecolor(C_PLOT_FRAME)
        self.fig_port.patch.set_linewidth(1.0)
        self.canvas_port = FigureCanvasTkAgg(self.fig_port, master=chart_wrap)
        self.canvas_port.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

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
        with self._view_cache_lock:
            self.portfolio_rows_cache.clear()
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
        with self._view_cache_lock:
            self.portfolio_rows_cache.clear()
        self.portfolio_menu.configure(values=self.store.list_portfolios())
        self.portfolio_name_var.set(name)
        self.on_portfolio_change()

    def refresh_portfolio(self):
        cache_key = self.current_portfolio
        with self._view_cache_lock:
            cached = self.portfolio_rows_cache.get(cache_key)
        if cached is not None:
            rows, total, sector_totals = cached
            self._render_portfolio(rows, total, sector_totals)
            return
        self.submit_task(self._proc_portfolio, cache_key, context="_proc_portfolio")

    def _proc_portfolio(self, cache_key=None):
        rows = []
        total = 0.0
        sector_totals = {}
        for sym, shares, cost in self.store.get_portfolio(self.current_portfolio):
            price = None
            sector = "Unknown"
            info = None
            try:
                with self.cache_lock:
                    info = (self.cache.get(sym, {}) or {}).get("info", {})
                    price = info.get("currentPrice")
                    sector = info.get("sector") or "Unknown"
            except Exception as e:
                log_exception(f"_proc_portfolio.cache:{sym}", e)
                price = None
            if price is None:
                try:
                    info = yf.Ticker(sym).info
                    price = info.get("currentPrice")
                    sector = info.get("sector") or "Unknown"
                    with self.cache_lock:
                        cached = self.cache.get(sym, {})
                        self.cache[sym] = {**cached, "info": info}
                except Exception as e:
                    log_exception(f"_proc_portfolio.fetch:{sym}", e)
                    price = 0
            value = (shares or 0) * (price or 0)
            total += value
            sector_totals[sector] = sector_totals.get(sector, 0.0) + value
            rows.append((sym, shares, cost, price or 0, value))
        if cache_key is not None:
            with self._view_cache_lock:
                self.portfolio_rows_cache[cache_key] = (rows, total, sector_totals)
        self.root.after(0, lambda rows=rows, total=total, sector_totals=sector_totals: self._render_portfolio(rows, total, sector_totals))

    def _render_portfolio(self, rows, total, sector_totals=None):
        for i in self.tr_port.get_children(): self.tr_port.delete(i)
        for sym, shares, cost, price, value in rows:
            self.tr_port.insert("", tk.END, values=(sym, shares, f"{cost:.2f}", f"{price:.2f}", f"{value:.2f}"))
        self.lbl_port_total.configure(text=f"Total: ${total:,.2f}")
        self._render_portfolio_sector_chart(sector_totals or {})

    def _render_portfolio_sector_chart(self, sector_totals):
        if not hasattr(self, "ax_port"):
            return
        self.ax_port.clear()
        self.ax_port.set_facecolor(C_BG_CARD)
        self._apply_axes_frame(self.ax_port)

        non_zero = {k: float(v) for k, v in (sector_totals or {}).items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v) and v > 0}
        if not non_zero:
            self.ax_port.text(0.5, 0.5, "No sector allocation data", ha="center", va="center", color=C_TEXT_SUB, transform=self.ax_port.transAxes)
            self.canvas_port.draw_idle()
            return

        labels = list(non_zero.keys())
        sizes = np.array(list(non_zero.values()), dtype=float)
        colors = [
            C_ACCENT,
            C_GREEN,
            C_YELLOW,
            C_PINK,
            C_RED,
            "#64D2FF",
            "#7EE081",
            "#F4A259",
            "#A78BFA",
            "#B7C0CD",
        ]
        color_list = [colors[i % len(colors)] for i in range(len(labels))]
        self.ax_port.pie(
            sizes,
            labels=labels,
            colors=color_list,
            autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
            startangle=120,
            pctdistance=0.75,
            textprops={"color": C_TEXT_MAIN, "fontsize": 9},
        )
        self.ax_port.axis("equal")
        try:
            self.fig_port.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.04)
        except Exception:
            pass
        self.canvas_port.draw_idle()

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
        with self._view_cache_lock:
            self.portfolio_rows_cache.clear()
        self.refresh_portfolio()
        self.refresh_earnings()

    def remove_portfolio(self):
        sel = self.tr_port.selection()
        if not sel:
            return
        sym = self.tr_port.item(sel[0])['values'][0]
        self.store.remove_portfolio_symbol(self.current_portfolio, sym)
        with self._view_cache_lock:
            self.portfolio_rows_cache.clear()
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
        with self._view_cache_lock:
            self.watchlist_rows_cache.clear()
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
            with self._view_cache_lock:
                self.watchlist_rows_cache.clear()
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
        with self._view_cache_lock:
            self.watchlist_rows_cache.clear()
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
        with self._view_cache_lock:
            self.watchlist_rows_cache.clear()
        self.watchlist_menu.configure(values=self.store.list_watchlists())
        self.watchlist_name_var.set(name)
        self.on_watchlist_change()
    
    def refresh_watchlist(self):
        if self.watchlist_loading: return # Prevent double refresh
        symbols = [str(s).upper().strip() for s in self.watchlist_data if isinstance(s, str) and s.strip()]
        cache_key = (self.current_watchlist, tuple(symbols))
        with self._view_cache_lock:
            cached_rows = self.watchlist_rows_cache.get(cache_key)
        if cached_rows is not None:
            self._render_watchlist(cached_rows)
            return
        self.watchlist_loading = True
        self.submit_task(self._proc_watch, cache_key, context="_proc_watch")

    def _proc_watch(self, cache_key=None):
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
            if cache_key is not None:
                with self._view_cache_lock:
                    self.watchlist_rows_cache[cache_key] = rows
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
        self.ax_wl.set_facecolor(C_BG_CARD)
        self.ax_wl.grid(True, linestyle="--", alpha=0.16)
        self._apply_axes_frame(self.ax_wl)
        self.ax_wl.tick_params(axis='both', colors=C_TEXT_SUB, labelsize=8)
        self.ax_wl.plot(close.index, close.values, color=color, linewidth=1.5, label="Price")
        self.ax_wl.plot(sma.index, sma.values, color=C_YELLOW, linewidth=1.2, label="SMA200")
        self.ax_wl.set_title(f"Dip Finder: {sym}", fontweight="bold", color=C_TEXT_MAIN, pad=6, y=0.98)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        self.ax_wl.xaxis.set_major_locator(locator)
        self.ax_wl.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax_wl.legend(loc="upper left")
        try:
            self.fig_wl.subplots_adjust(bottom=0.27, left=0.08, right=0.98, top=0.80)
        except Exception:
            pass
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
            if hasattr(self, "fig_port"):
                plt.close(self.fig_port)
            if hasattr(self, "fig_quality"):
                plt.close(self.fig_quality)
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
            if period == "1d":
                # Use intraday granularity and fall back to last trading session if the market is closed.
                h = t.history(period="5d", interval="5m", auto_adjust=False, prepost=False)
                if h is not None and not h.empty:
                    h = analysis_clean_datetime_index_df(h)
                    last_day = h.index.max().date()
                    h = h[h.index.date == last_day]
                if h is None or h.empty:
                    h = t.history(period="5d", interval="1d", auto_adjust=False, prepost=False)
            else:
                h = t.history(period=period)

            if h is not None and not h.empty:
                h = analysis_clean_datetime_index_df(h)
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
        if h is None:
            h = pd.DataFrame()
        if h is not None and not h.empty:
            h = analysis_clean_datetime_index_df(h)
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
            text = self.http_get_text(url, headers={"User-Agent": self.sec_user_agent}, timeout=15)
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
            text = self.http_get_text(url, headers={"User-Agent": self.sec_user_agent}, timeout=20)
            if not text:
                return None
            return json.loads(text)
        except Exception as e:
            log_exception("fetch_sec_companyfacts", e)
            return None

    def fetch_sec_submissions(self, cik: str):
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            text = self.http_get_text(url, headers={"User-Agent": self.sec_user_agent}, timeout=20)
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
            index_text = self.http_get_text(index_url, headers={"User-Agent": self.sec_user_agent}, timeout=20)
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
            doc_text = self.http_get_text(doc_url, headers={"User-Agent": self.sec_user_agent}, timeout=20)
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
        alpha_key = get_service_key("ALPHAVANTAGE_API_KEY")
        if self._should_skip_transcript_fetch(symbol, "AlphaVantage", force):
            self.root.after(0, lambda: self._mark_transcript_fetch_done("AlphaVantage"))
            return
        if not alpha_key:
            self.transcript_fetch_status = "Alpha Vantage: API key missing."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("AlphaVantage"))
            return
        try:
            quarters = self.recent_quarters(AV_TRANSCRIPTS_QUARTERS)
            for q in quarters:
                title = f"Earnings Call Transcript {q} (Alpha Vantage)"
                if self.store.transcript_exists(symbol, q, title):
                    continue
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={symbol}&quarter={q}&apikey={alpha_key}"
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
        ninjas_key = get_service_key("API_NINJAS_KEY")
        if self._should_skip_transcript_fetch(symbol, "APINinjas", force):
            self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))
            return
        if not ninjas_key:
            self.transcript_fetch_status = "API Ninjas: API key missing."
            self.root.after(0, lambda: self._mark_transcript_fetch_done("APINinjas"))
            return
        try:
            url = f"https://api.api-ninjas.com/v1/earningstranscript?ticker={symbol}"
            headers = {"X-Api-Key": ninjas_key}
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
        return shared_http_get_text(url, headers=headers, timeout=timeout)

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
        try:
            return analysis_align_to_quarter_end(series)
        except Exception as e:
            log_exception("align_to_quarter_end", e)
            return series

    def align_to_year_end(self, series: Optional[pd.Series]):
        try:
            return analysis_align_to_year_end(series)
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
            fcf = ocf - capex.abs()
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
            fcf_a = ocf_a - capex_a.abs()
        cf_a = pd.DataFrame({
            "OperatingCashFlow": ocf_a,
            "CapitalExpenditure": capex_a,
            "FreeCashFlow": fcf_a,
            "StockBasedCompensation": sbc_a,
        }).dropna(how="all").sort_index()

        return fin_q, fin_a, bal_q, bal_a, cf_q, cf_a
