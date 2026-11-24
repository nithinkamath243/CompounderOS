# filepath: compounderos_app/database.py
from datetime import datetime
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class DataStore:
    def __init__(self, path: str, role: str = "sor", check_same_thread: bool = False):
        self.path = path
        self.role = role
        self.conn = sqlite3.connect(path, check_same_thread=check_same_thread)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._init_tables()
        self._repair_legacy_tables()
        self._migrate_v6()
        self._migrate_v7()
        self._migrate_v8()
        self._migrate_v9()
        self._ensure_provenance_columns()

    def _repair_legacy_tables(self):
        """Idempotently upgrade pre-v2 desktop tables (e.g. ticker_map.ticker -> symbol, price_history.truncated_window)."""
        with self.lock:
            cur = self.conn.cursor()
            # 1. ticker_map: pre-v2 schema used (ticker, cik, name, updated_at) without 'symbol'
            tm_cols = {row[1] for row in cur.execute("PRAGMA table_info(ticker_map)").fetchall()}
            if tm_cols and "symbol" not in tm_cols and "ticker" in tm_cols:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ticker_map_v2 (
                        symbol TEXT PRIMARY KEY,
                        cik TEXT,
                        name TEXT,
                        sector TEXT,
                        industry TEXT,
                        classification_as_of TEXT
                    )
                    """
                )
                as_of_expr = "updated_at" if "updated_at" in tm_cols else "NULL"
                sector_expr = "sector" if "sector" in tm_cols else "''"
                industry_expr = "industry" if "industry" in tm_cols else "''"
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO ticker_map_v2 (symbol, cik, name, sector, industry, classification_as_of)
                    SELECT UPPER(TRIM(ticker)), cik, name, {sector_expr}, {industry_expr}, {as_of_expr}
                    FROM ticker_map
                    WHERE ticker IS NOT NULL AND TRIM(ticker) != ''
                    """
                )
                cur.execute("DROP TABLE ticker_map")
                cur.execute("ALTER TABLE ticker_map_v2 RENAME TO ticker_map")
            elif tm_cols:
                for col_def in ("sector TEXT", "industry TEXT", "classification_as_of TEXT"):
                    col_name = col_def.split()[0]
                    if col_name not in tm_cols:
                        try:
                            cur.execute(f"ALTER TABLE ticker_map ADD COLUMN {col_def}")
                        except sqlite3.OperationalError:
                            pass

            # 2. price_history: pre-v2 schema lacked 'truncated_window'
            ph_cols = {row[1] for row in cur.execute("PRAGMA table_info(price_history)").fetchall()}
            if ph_cols and "truncated_window" not in ph_cols:
                try:
                    cur.execute("ALTER TABLE price_history ADD COLUMN truncated_window INTEGER DEFAULT 0")
                except sqlite3.OperationalError:
                    pass

            # 3. kpis: pre-v2 schema lacked PRIMARY KEY / unique index on (symbol, name, period)
            kpi_cols = {row[1] for row in cur.execute("PRAGMA table_info(kpis)").fetchall()}
            if kpi_cols and {"symbol", "name", "period"}.issubset(kpi_cols):
                try:
                    cur.execute(
                        "DELETE FROM kpis WHERE rowid NOT IN ("
                        "SELECT MAX(rowid) FROM kpis GROUP BY symbol, name, period)"
                    )
                    cur.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_kpis_symbol_name_period "
                        "ON kpis(symbol, name, period)"
                    )
                except sqlite3.OperationalError:
                    pass

            self.conn.commit()

    def _ensure_provenance_columns(self):
        with self.lock:
            cur = self.conn.cursor()
            for table, col_def in [
                ("sp500_scores_history", "registry_id TEXT"),
                ("sp500_scores_history", "registry_hash TEXT"),
                ("sp500_scores", "registry_id TEXT"),
                ("sp500_scores", "registry_hash TEXT"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass
            self.conn.commit()

    def _require_replica(self):
        if self.role != "replica":
            raise RuntimeError(f"replicate_* operations restricted to role='replica'. Current role: {self.role}")

    def _init_tables(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sp500_scores (
                    symbol TEXT PRIMARY KEY,
                    sector TEXT,
                    score REAL,
                    composite_shrunk REAL,
                    verdict TEXT,
                    model TEXT,
                    evaluated_at TEXT,
                    registry_id TEXT,
                    registry_hash TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sp500_scores_history (
                    symbol            TEXT NOT NULL,
                    scan_date         TEXT NOT NULL,
                    evaluated_at      TEXT NOT NULL,
                    sector            TEXT,
                    score             REAL,
                    score_pre_gate    REAL,
                    composite_shrunk  REAL,
                    b_profitability   REAL,
                    b_efficiency      REAL,
                    b_balance_sheet   REAL,
                    b_capital_alloc   REAL,
                    confidence        REAL NOT NULL,
                    gate_state        TEXT NOT NULL CHECK (gate_state IN ('pass','fired','unverified','fired_unverified')),
                    gates_fired       TEXT,
                    gates_unverified  TEXT,
                    gate_cap_applied  REAL,
                    verdict           TEXT,
                    pe_trailing       REAL,
                    pe_forward        REAL,
                    fcf_yield         REAL,
                    market_cap        REAL,
                    engine_version    TEXT NOT NULL,
                    source            TEXT NOT NULL DEFAULT 'scan' CHECK (source IN ('scan','seeded','vendor')),
                    raw_json          TEXT,
                    subscores_json    TEXT,
                    registry_id       TEXT,
                    registry_hash     TEXT,
                    PRIMARY KEY (symbol, scan_date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_history (
                    symbol TEXT,
                    scan_date TEXT,
                    trajectory REAL,
                    confidence REAL,
                    composition TEXT CHECK (composition IN ('AGB', 'AG', 'AB', 'A', 'WITHHELD')),
                    peer_state TEXT CHECK (peer_state IN ('OK', 'THIN', 'NONE')),
                    granularity_used TEXT,
                    peer_set_json TEXT,
                    raw_inputs_json TEXT,
                    derived_json TEXT,
                    extraction_version TEXT,
                    engine_version TEXT,
                    evaluated_at TEXT,
                    PRIMARY KEY (symbol, scan_date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT,
                    truncated_window INTEGER DEFAULT 0,
                    PRIMARY KEY (symbol, date, source)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticker_map (
                    symbol TEXT PRIMARY KEY,
                    cik TEXT,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    classification_as_of TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    name TEXT,
                    symbol TEXT,
                    PRIMARY KEY (name, symbol)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    name TEXT,
                    symbol TEXT,
                    shares REAL,
                    cost_basis REAL,
                    PRIMARY KEY (name, symbol)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    symbol TEXT,
                    date TEXT,
                    title TEXT,
                    content TEXT,
                    PRIMARY KEY (symbol, date, title)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transcript_analysis (
                    symbol TEXT,
                    date TEXT,
                    title TEXT,
                    summary TEXT,
                    qa TEXT,
                    PRIMARY KEY (symbol, date, title)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kpis (
                    symbol TEXT,
                    name TEXT,
                    period TEXT,
                    value TEXT,
                    unit TEXT,
                    PRIMARY KEY (symbol, name, period)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lenses (
                    lens_id TEXT PRIMARY KEY,
                    name TEXT,
                    lens_hash TEXT,
                    definition_json TEXT,
                    updated_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dcf_inputs (
                    symbol TEXT PRIMARY KEY, data_type TEXT, fcf_or_eps REAL, growth REAL,
                    discount REAL, terminal_mult REAL, target_return REAL, updated_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sec_filings (
                    symbol TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    form TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    report_date TEXT,
                    primary_document TEXT,
                    items TEXT,
                    edgar_url TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, accession_number)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_lots (
                    lot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    shares REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    trade_date TEXT,
                    commission REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    notes TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS macro_quotes (
                    symbol TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    category TEXT NOT NULL,
                    price REAL,
                    change_pct REAL,
                    unit TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_odds (
                    contract_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    probability_pct REAL NOT NULL,
                    volume REAL,
                    url TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_threads (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                    mode TEXT NOT NULL DEFAULT 'chat' CHECK (mode IN ('chat','deep_search')),
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            self.conn.commit()

    def _migrate_v2(self):
        """Backwards-compatibility alias for existing v2 tests."""
        self._migrate_v6()

    def _migrate_v6(self):
        """C: Idempotent unified migration resolving damaged state vs fresh starts."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("PRAGMA user_version")
            v = cur.fetchone()[0]

            if v >= 6:
                return

            cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='sp500_scores'")
            if cur.fetchone()[0] == 0:
                self._init_tables()
                cur.execute("PRAGMA user_version = 6")
                self.conn.commit()
                return

            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK (kind IN ('dip_trigger','score_drop','gate_fired')),
                        registry_tag TEXT NOT NULL DEFAULT 'canonical',
                        dedupe_key TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','acked','resolved')),
                        fired_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        resolved_at TEXT,
                        scan_date TEXT,
                        payload_json TEXT NOT NULL
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status, symbol)")
                cur.execute("CREATE TABLE IF NOT EXISTS scanner_state (key TEXT PRIMARY KEY, value TEXT)")

                if v <= 3:
                    try:
                        cur.execute("ALTER TABLE sp500_scores_history ADD COLUMN registry_hash TEXT")
                    except sqlite3.OperationalError:
                        pass
                    
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS trajectory_history_v6 (
                            symbol TEXT, scan_date TEXT, trajectory REAL, confidence REAL,
                            composition TEXT CHECK (composition IN ('AGB', 'AG', 'AB', 'A', 'WITHHELD')),
                            granularity_used TEXT, peer_set_json TEXT, raw_inputs_json TEXT,
                            engine_version TEXT, evaluated_at TEXT,
                            PRIMARY KEY (symbol, scan_date)
                        )
                    """)
                    cur.execute("""
                        INSERT INTO trajectory_history_v6 
                        (symbol, scan_date, trajectory, confidence, composition, granularity_used, peer_set_json, raw_inputs_json, engine_version, evaluated_at)
                        SELECT symbol, scan_date, trajectory, confidence, composition, granularity_used, peer_set_json, raw_inputs_json, engine_version, evaluated_at 
                        FROM trajectory_history
                    """)
                    cur.execute("DROP TABLE trajectory_history")
                    cur.execute("ALTER TABLE trajectory_history_v6 RENAME TO trajectory_history")

                elif v in (4, 5):
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS trajectory_history_v6 (
                            symbol TEXT, scan_date TEXT, trajectory REAL, confidence REAL,
                            composition TEXT CHECK (composition IN ('AGB', 'AG', 'AB', 'A', 'WITHHELD')),
                            granularity_used TEXT, peer_set_json TEXT, raw_inputs_json TEXT,
                            engine_version TEXT, evaluated_at TEXT,
                            PRIMARY KEY (symbol, scan_date)
                        )
                    """)
                    cur.execute("""
                        INSERT INTO trajectory_history_v6 
                        (symbol, scan_date, trajectory, confidence, composition, granularity_used, peer_set_json, raw_inputs_json, engine_version, evaluated_at)
                        SELECT symbol, scan_date, trajectory, confidence, composition, granularity_used, peer_set_json, raw_inputs_json, engine_version, evaluated_at 
                        FROM trajectory_history
                    """)
                    cur.execute("DROP TABLE trajectory_history")
                    cur.execute("ALTER TABLE trajectory_history_v6 RENAME TO trajectory_history")

                    cur.execute("""
                        CREATE TABLE sp500_scores_history_v6 (
                            symbol TEXT NOT NULL, scan_date TEXT NOT NULL, evaluated_at TEXT NOT NULL,
                            sector TEXT, score REAL, score_pre_gate REAL,
                            b_profitability REAL, b_efficiency REAL, b_balance_sheet REAL, b_capital_alloc REAL,
                            confidence REAL NOT NULL, gate_state TEXT NOT NULL CHECK (gate_state IN ('pass','fired','unverified','fired_unverified')),
                            gates_fired TEXT, gates_unverified TEXT, pe_trailing REAL, pe_forward REAL,
                            fcf_yield REAL, market_cap REAL, engine_version TEXT NOT NULL,
                            source TEXT NOT NULL DEFAULT 'scan' CHECK (source IN ('scan','seeded','vendor')),
                            raw_json TEXT, subscores_json TEXT, registry_hash TEXT,
                            PRIMARY KEY (symbol, scan_date)
                        )
                    """)
                    
                    rows = cur.execute("""
                        SELECT symbol, scan_date, MAX(evaluated_at) as max_eval, score, confidence, gate_state,
                               gates_fired, gates_unverified, engine_version, raw_json, subscores_json, registry_hash,
                               fcf_yield, pe_trailing, source
                        FROM sp500_scores_history
                        GROUP BY symbol, scan_date
                    """).fetchall()
                    
                    from compounderos_app.scoring_engine import score_metrics
                    
                    insert_data = []
                    for r in rows:
                        raw_dict = json.loads(r['raw_json']) if r['raw_json'] else {}
                        try:
                            sm = score_metrics(raw_dict)
                            score_pre_gate = sm.composite_pre_gate
                            b_prof = sm.buckets.get('profitability')
                            b_eff = sm.buckets.get('efficiency')
                            b_bal = sm.buckets.get('balance_sheet')
                            b_cap = sm.buckets.get('capital_alloc')
                        except Exception:
                            score_pre_gate = b_prof = b_eff = b_bal = b_cap = None
                            
                        sec_row = cur.execute("SELECT sector FROM sp500_scores WHERE symbol=?", (r['symbol'],)).fetchone()
                        sector = sec_row[0] if sec_row else None
                        
                        insert_data.append((
                            r['symbol'], r['scan_date'], r['max_eval'], sector, r['score'], score_pre_gate,
                            b_prof, b_eff, b_bal, b_cap, r['confidence'], r['gate_state'], r['gates_fired'],
                            r['gates_unverified'], r['pe_trailing'], None, r['fcf_yield'], None, r['engine_version'],
                            r['source'], r['raw_json'], r['subscores_json'], r['registry_hash']
                        ))
                        
                    cur.executemany("""
                        INSERT INTO sp500_scores_history_v6 
                        (symbol, scan_date, evaluated_at, sector, score, score_pre_gate, 
                         b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc, 
                         confidence, gate_state, gates_fired, gates_unverified, pe_trailing, pe_forward, 
                         fcf_yield, market_cap, engine_version, source, raw_json, subscores_json, registry_hash)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, insert_data)
                    
                    cur.execute("DROP TABLE sp500_scores_history")
                    cur.execute("ALTER TABLE sp500_scores_history_v6 RENAME TO sp500_scores_history")

                cur.execute("PRAGMA user_version = 6")
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(f"Migration v6 failed: {e}")

    def _migrate_v7(self):
        """Migration v7: Additive columns for verdict, gate_cap_applied, composite_shrunk."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("PRAGMA user_version")
            v = cur.fetchone()[0]

            if v < 6:
                raise RuntimeError(f"Precondition failed: database at user_version {v} < 6. Apply v6 remediation first.")

            if v >= 7:
                return

            try:
                for col_def in [
                    ("sp500_scores_history", "verdict TEXT"),
                    ("sp500_scores_history", "gate_cap_applied REAL"),
                    ("sp500_scores_history", "composite_shrunk REAL"),
                    ("sp500_scores", "verdict TEXT"),
                    ("sp500_scores", "composite_shrunk REAL"),
                ]:
                    try:
                        cur.execute(f"ALTER TABLE {col_def[0]} ADD COLUMN {col_def[1]}")
                    except sqlite3.OperationalError:
                        pass

                cur.execute("PRAGMA user_version = 7")
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(f"Migration v7 failed: {e}")

    def _migrate_v8(self):
        """Migration v8: Trajectory v1.1 composition enum, peer_state, derived_json, extraction_version."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("PRAGMA user_version")
            v = cur.fetchone()[0]

            if v < 7:
                raise RuntimeError(
                    f"PRECONDITION FAILED: PRAGMA user_version is {v} (< 7). Cannot migrate to v8."
                )

            if v >= 8:
                return

            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trajectory_history_v8 (
                        symbol TEXT,
                        scan_date TEXT,
                        trajectory REAL,
                        confidence REAL,
                        composition TEXT CHECK (composition IN ('AGB', 'AG', 'AB', 'A', 'WITHHELD')),
                        peer_state TEXT CHECK (peer_state IN ('OK', 'THIN', 'NONE')),
                        granularity_used TEXT,
                        peer_set_json TEXT,
                        raw_inputs_json TEXT,
                        derived_json TEXT,
                        extraction_version TEXT,
                        engine_version TEXT,
                        evaluated_at TEXT,
                        PRIMARY KEY (symbol, scan_date)
                    )
                """)
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='trajectory_history'"
                )
                if cur.fetchone():
                    cur.execute("""
                        INSERT INTO trajectory_history_v8 (
                            symbol, scan_date, trajectory, confidence, composition,
                            peer_state, granularity_used, peer_set_json, raw_inputs_json,
                            derived_json, extraction_version, engine_version, evaluated_at
                        )
                        SELECT
                            symbol,
                            scan_date,
                            trajectory,
                            confidence,
                            CASE composition
                                WHEN 'A_PLUS_B' THEN 'AB'
                                WHEN 'A_ONLY' THEN 'A'
                                WHEN 'WITHHELD' THEN 'WITHHELD'
                                ELSE composition
                            END AS composition,
                            CASE composition
                                WHEN 'A_PLUS_B' THEN 'OK'
                                ELSE 'NONE'
                            END AS peer_state,
                            granularity_used,
                            peer_set_json,
                            raw_inputs_json,
                            NULL AS derived_json,
                            NULL AS extraction_version,
                            engine_version,
                            evaluated_at
                        FROM trajectory_history
                    """)
                    cur.execute("DROP TABLE trajectory_history")
                cur.execute("ALTER TABLE trajectory_history_v8 RENAME TO trajectory_history")
                cur.execute("PRAGMA user_version = 8")
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(f"Migration v8 failed: {e}")

    def _migrate_v9(self):
        """Migration v9: Wave 3 tables (sec_filings, portfolio_lots, macro_quotes, prediction_odds, research_threads)."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("PRAGMA user_version")
            v = cur.fetchone()[0]

            if v < 8:
                raise RuntimeError(
                    f"PRECONDITION FAILED: PRAGMA user_version is {v} (< 8). Cannot migrate to v9."
                )

            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sec_filings (
                        symbol TEXT NOT NULL,
                        accession_number TEXT NOT NULL,
                        form TEXT NOT NULL,
                        filing_date TEXT NOT NULL,
                        report_date TEXT,
                        primary_document TEXT,
                        items TEXT,
                        edgar_url TEXT NOT NULL,
                        fetched_at TEXT NOT NULL,
                        PRIMARY KEY (symbol, accession_number)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_lots (
                        lot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        shares REAL NOT NULL,
                        cost_basis REAL NOT NULL,
                        trade_date TEXT,
                        commission REAL DEFAULT 0.0,
                        realized_pnl REAL DEFAULT 0.0,
                        notes TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS macro_quotes (
                        symbol TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        category TEXT NOT NULL,
                        price REAL,
                        change_pct REAL,
                        unit TEXT DEFAULT '',
                        updated_at TEXT NOT NULL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_odds (
                        contract_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        category TEXT NOT NULL,
                        title TEXT NOT NULL,
                        probability_pct REAL NOT NULL,
                        volume REAL,
                        url TEXT,
                        updated_at TEXT NOT NULL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS research_threads (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                        mode TEXT NOT NULL DEFAULT 'chat' CHECK (mode IN ('chat','deep_search')),
                        content TEXT NOT NULL,
                        citations_json TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                if v < 9:
                    cur.execute("PRAGMA user_version = 9")
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(f"Migration v9 failed: {e}")

    def list_sp500_scores(self) -> List[Tuple[str, str, float, str, str]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT symbol, sector, score, model, evaluated_at FROM sp500_scores ORDER BY COALESCE(composite_shrunk, score) DESC"
            )
            return [tuple(row) for row in cur.fetchall()]

    def record_quality_result(
        self,
        symbol: str,
        sector: str,
        result: Any,
        multiples: dict,
        raw: dict,
        write_snapshot: bool = True,
        verdict: str = "SCORED",
    ):
        from datetime import datetime, timezone
        from compounderos_app.scoring_engine import (
            CANONICAL_REGISTRY_HASH,
            CANONICAL_REGISTRY_ID,
            ENGINE_VERSION,
        )

        if result is not None:
            reg_id = getattr(result, "registry_id", CANONICAL_REGISTRY_ID)
            reg_hash = getattr(result, "registry_hash", "")
            if reg_id != CANONICAL_REGISTRY_ID or (
                reg_hash not in ("", None, CANONICAL_REGISTRY_HASH)
            ):
                raise ValueError(
                    "Write-guard error: Cannot persist custom lens result to canonical tables."
                )

        def _to_float(val):
            if val is None:
                return None
            return float(val)

        with self.lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            scan_date = now_iso[:10]

            try:
                if result is None:
                    existing_null = self.conn.execute(
                        "SELECT confidence FROM sp500_scores_history WHERE symbol = ? AND scan_date = ?",
                        (symbol, scan_date),
                    ).fetchone()
                    if write_snapshot and existing_null is None:
                        self.conn.execute(
                            "REPLACE INTO sp500_scores (symbol, sector, score, composite_shrunk, verdict, model, evaluated_at, registry_id, registry_hash) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?)",
                            (
                                symbol,
                                sector,
                                verdict,
                                ENGINE_VERSION,
                                now_iso,
                                CANONICAL_REGISTRY_ID,
                                CANONICAL_REGISTRY_HASH,
                            ),
                        )
                    self.conn.execute(
                        """INSERT OR IGNORE INTO sp500_scores_history 
                           (symbol, scan_date, evaluated_at, sector, score, score_pre_gate, composite_shrunk,
                            b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                            confidence, gate_state, gates_fired, gates_unverified, gate_cap_applied, verdict,
                            pe_trailing, pe_forward, fcf_yield, market_cap, engine_version, source, raw_json, subscores_json, registry_id, registry_hash)
                           VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.0, 'unverified', '', '', NULL, ?, NULL, NULL, NULL, NULL, ?, 'scan', '{}', '{}', ?, '')""",
                        (symbol, scan_date, now_iso, sector, verdict, ENGINE_VERSION, CANONICAL_REGISTRY_ID),
                    )
                    self.conn.commit()
                    return

                fired_str = ",".join(result.gates_fired)
                unver_str = ",".join(result.gates_unverified)
                gate_state = (
                    "fired_unverified"
                    if (result.gates_fired and result.gates_unverified)
                    else (
                        "fired"
                        if result.gates_fired
                        else ("unverified" if result.gates_unverified else "pass")
                    )
                )
                subscores = {k: v.subscore for k, v in result.metrics.items()}

                b_prof = result.buckets.get("profitability")
                b_eff = result.buckets.get("efficiency")
                b_bal = result.buckets.get("balance_sheet")
                b_cap = result.buckets.get("capital_alloc")

                pe_trailing_val = _to_float(multiples.get("trailing_pe"))
                pe_forward_val = _to_float(multiples.get("forward_pe"))
                fcf_yield_val = _to_float(multiples.get("fcf_yield"))
                market_cap_val = _to_float(multiples.get("market_cap"))

                raw_json_str = json.dumps(raw)
                subscores_json_str = json.dumps(subscores)
                reg_id_val = getattr(result, "registry_id", None) or CANONICAL_REGISTRY_ID
                reg_hash_val = getattr(result, "registry_hash", None) or CANONICAL_REGISTRY_HASH

                existing_hist = self.conn.execute(
                    "SELECT confidence FROM sp500_scores_history WHERE symbol = ? AND scan_date = ?",
                    (symbol, scan_date),
                ).fetchone()
                allow_snapshot_update = existing_hist is None or (
                    result.confidence >= existing_hist[0]
                    and result.composite is not None
                    and raw_json_str != "{}"
                )

                if write_snapshot and allow_snapshot_update:
                    self.conn.execute(
                        "REPLACE INTO sp500_scores (symbol, sector, score, composite_shrunk, verdict, model, evaluated_at, registry_id, registry_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            symbol,
                            sector,
                            result.composite,
                            result.composite_shrunk,
                            verdict,
                            result.engine_version,
                            now_iso,
                            reg_id_val,
                            reg_hash_val,
                        ),
                    )

                self.conn.execute(
                    """INSERT INTO sp500_scores_history 
                       (symbol, scan_date, evaluated_at, sector, score, score_pre_gate, composite_shrunk,
                        b_profitability, b_efficiency, b_balance_sheet, b_capital_alloc,
                        confidence, gate_state, gates_fired, gates_unverified, gate_cap_applied, verdict,
                        pe_trailing, pe_forward, fcf_yield, market_cap, engine_version, source, raw_json, subscores_json, registry_id, registry_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scan', ?, ?, ?, ?)
                       ON CONFLICT (symbol, scan_date) DO UPDATE SET
                            evaluated_at = excluded.evaluated_at,
                            sector = excluded.sector,
                            score = excluded.score,
                            score_pre_gate = excluded.score_pre_gate,
                            composite_shrunk = excluded.composite_shrunk,
                            b_profitability = excluded.b_profitability,
                            b_efficiency = excluded.b_efficiency,
                            b_balance_sheet = excluded.b_balance_sheet,
                            b_capital_alloc = excluded.b_capital_alloc,
                            confidence = excluded.confidence,
                            gate_state = excluded.gate_state,
                            gates_fired = excluded.gates_fired,
                            gates_unverified = excluded.gates_unverified,
                            gate_cap_applied = excluded.gate_cap_applied,
                            verdict = excluded.verdict,
                            pe_trailing = excluded.pe_trailing,
                            pe_forward = excluded.pe_forward,
                            fcf_yield = excluded.fcf_yield,
                            market_cap = excluded.market_cap,
                            engine_version = excluded.engine_version,
                            source = excluded.source,
                            raw_json = excluded.raw_json,
                            subscores_json = excluded.subscores_json,
                            registry_id = excluded.registry_id,
                            registry_hash = excluded.registry_hash
                       WHERE excluded.confidence >= sp500_scores_history.confidence
                         AND excluded.score IS NOT NULL
                         AND excluded.raw_json != '{}'
                       """,
                    (
                        symbol,
                        scan_date,
                        now_iso,
                        sector,
                        result.composite,
                        result.composite_pre_gate,
                        result.composite_shrunk,
                        b_prof,
                        b_eff,
                        b_bal,
                        b_cap,
                        result.confidence,
                        gate_state,
                        fired_str,
                        unver_str,
                        result.gate_cap_applied,
                        verdict,
                        pe_trailing_val,
                        pe_forward_val,
                        fcf_yield_val,
                        market_cap_val,
                        result.engine_version,
                        raw_json_str,
                        subscores_json_str,
                        reg_id_val,
                        reg_hash_val,
                    ),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def record_or_extend_alert(self, symbol: str, kind: str, instance: str, registry_tag: str, payload: dict) -> bool:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        scan_date = now_iso[:10]
        dedupe_key = f"{symbol}|{kind}|{instance}|{registry_tag}"
        payload_str = json.dumps(payload)

        with self.lock:
            cur = self.conn.execute("SELECT alert_id, status FROM alerts WHERE dedupe_key = ?", (dedupe_key,))
            row = cur.fetchone()

            if row:
                if row["status"] in ("open", "acked"):
                    self.conn.execute(
                        "UPDATE alerts SET last_seen_at = ?, payload_json = ? WHERE alert_id = ?",
                        (now_iso, payload_str, row["alert_id"]),
                    )
                    self.conn.commit()
                    return False
                else:
                    self.conn.execute(
                        "UPDATE alerts SET status = 'open', fired_at = ?, last_seen_at = ?, resolved_at = NULL, payload_json = ? WHERE alert_id = ?",
                        (now_iso, now_iso, payload_str, row["alert_id"]),
                    )
                    self.conn.commit()
                    return True
            else:
                self.conn.execute(
                    "INSERT INTO alerts (symbol, kind, registry_tag, dedupe_key, status, fired_at, last_seen_at, scan_date, payload_json) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?)",
                    (symbol, kind, registry_tag, dedupe_key, now_iso, now_iso, scan_date, payload_str),
                )
                self.conn.commit()
                return True

    def resolve_open_alerts(self, symbol: str, kind: str, instance: Optional[str] = None) -> int:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.lock:
            query = "UPDATE alerts SET status = 'resolved', resolved_at = ? WHERE symbol = ? AND kind = ? AND status IN ('open', 'acked')"
            params = [now_iso, symbol, kind]
            if instance:
                query += " AND dedupe_key LIKE ?"
                params.append(f"%|{instance}|%")

            cur = self.conn.execute(query, params)
            self.conn.commit()
            return cur.rowcount

    def ack_alert(self, alert_id: int) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE alerts SET status = 'acked' WHERE alert_id = ? AND status = 'open'",
                (alert_id,),
            )
            self.conn.commit()
            return cur.rowcount > 0

    acknowledge_alert = ack_alert

    def list_alerts(
        self,
        status: Optional[str] = None,
        since_id: int = 0,
        limit: int = 200,
        unacknowledged_only: Optional[bool] = None,
    ) -> List[dict]:
        with self.lock:
            query = "SELECT * FROM alerts WHERE alert_id > ?"
            params: List[Any] = [since_id]
            eff_status = status or ("open" if unacknowledged_only else None)
            if eff_status:
                query += " AND status = ?"
                params.append(eff_status)
            query += " ORDER BY alert_id DESC LIMIT ?"
            params.append(limit)
            cur = self.conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def open_alert_summary(self) -> dict:
        with self.lock:
            count = self.conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'open'").fetchone()[0]
            top = self.conn.execute(
                "SELECT symbol, kind, fired_at FROM alerts WHERE status = 'open' ORDER BY fired_at DESC LIMIT 5"
            ).fetchall()
            return {
                "open_alerts": count,
                "top": [f"{r['symbol']} {r['kind']} ({r['fired_at'][:10]})" for r in top],
            }

    def get_history_since(self, since_iso: str, limit: int) -> List[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM sp500_scores_history WHERE evaluated_at > ? ORDER BY evaluated_at ASC LIMIT ?",
                (since_iso, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_trajectory_since(self, since_iso: str, limit: int) -> List[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM trajectory_history WHERE evaluated_at > ? ORDER BY evaluated_at ASC LIMIT ?",
                (since_iso, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_prices_since(self, since_date: str, limit: int, source: str = "yfinance") -> List[dict]:
        with self.lock:
            if since_date and ":" in since_date:
                since_d, since_sym = since_date.split(":", 1)
                cur = self.conn.execute(
                    "SELECT * FROM price_history WHERE source = ? AND (date > ? OR (date = ? AND symbol > ?)) ORDER BY date ASC, symbol ASC LIMIT ?",
                    (source, since_d, since_d, since_sym, limit),
                )
            else:
                cur = self.conn.execute(
                    "SELECT * FROM price_history WHERE source = ? AND date > ? ORDER BY date ASC, symbol ASC LIMIT ?",
                    (source, since_date, limit),
                )
            return [dict(row) for row in cur.fetchall()]

    get_price_history_since = get_prices_since

    def get_latest_trajectory(self, symbol: str) -> Optional[dict]:
        with self.lock:
            cur = self.conn.execute(
                """SELECT trajectory, confidence, composition, peer_state, granularity_used, peer_set_json, raw_inputs_json, derived_json, extraction_version, evaluated_at, engine_version 
                   FROM trajectory_history WHERE symbol = ? ORDER BY evaluated_at DESC LIMIT 1""",
                (symbol,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def replicate_history_rows(self, rows: List[dict]) -> int:
        self._require_replica()
        if not rows:
            return 0
        cols = list(rows[0].keys())
        query = f"INSERT OR REPLACE INTO sp500_scores_history ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        with self.lock:
            self.conn.executemany(query, [[r[c] for c in cols] for r in rows])
            self.conn.commit()
            return len(rows)

    def replicate_trajectory_rows(self, rows: List[dict]) -> int:
        self._require_replica()
        if not rows:
            return 0
        cols = list(rows[0].keys())
        query = f"INSERT OR REPLACE INTO trajectory_history ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        with self.lock:
            self.conn.executemany(query, [[r[c] for c in cols] for r in rows])
            self.conn.commit()
            return len(rows)

    def replicate_price_rows(self, rows: List[dict]) -> int:
        self._require_replica()
        if not rows:
            return 0
        cols = list(rows[0].keys())
        query = f"INSERT OR REPLACE INTO price_history ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        with self.lock:
            self.conn.executemany(query, [[r[c] for c in cols] for r in rows])
            self.conn.commit()
            return len(rows)

    def replicate_scores_rows(self, rows: List[dict]) -> int:
        self._require_replica()
        with self.lock:
            self.conn.execute("DELETE FROM sp500_scores")
            if not rows:
                return 0
            cols = list(rows[0].keys())
            query = f"INSERT INTO sp500_scores ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
            self.conn.executemany(query, [[r[c] for c in cols] for r in rows])
            self.conn.commit()
            return len(rows)

    def replicate_ticker_map_rows(self, rows: List[dict]) -> int:
        self._require_replica()
        with self.lock:
            self.conn.execute("DELETE FROM ticker_map")
            if not rows:
                return 0
            cols = list(rows[0].keys())
            query = f"INSERT INTO ticker_map ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
            self.conn.executemany(query, [[r[c] for c in cols] for r in rows])
            self.conn.commit()
            return len(rows)

    def export_snapshot(self, dest_path: str) -> None:
        with self.lock:
            self.conn.execute(f"VACUUM INTO '{dest_path}'")

    def get_sp500_score(self, symbol: str) -> Optional[float]:
        with self.lock:
            cur = self.conn.execute("SELECT score FROM sp500_scores WHERE symbol = ?", (symbol.upper(),))
            row = cur.fetchone()
            return row[0] if row else None

    def get_history_last_two(self, symbol: str) -> List[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM sp500_scores_history WHERE symbol = ? AND source IN ('scan', 'seeded') ORDER BY scan_date DESC LIMIT 2",
                (symbol,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_history_window(self, symbol: str, days: int) -> "pd.DataFrame":
        import pandas as pd
        with self.lock:
            query = """
                SELECT scan_date, evaluated_at, score, fcf_yield, pe_trailing, raw_json, gate_state, confidence, engine_version, source, gate_cap_applied
                FROM sp500_scores_history
                WHERE symbol = ? AND source IN ('scan', 'seeded') AND evaluated_at >= date('now', ?)
                ORDER BY evaluated_at ASC
            """
            return pd.read_sql_query(query, self.conn, params=(symbol, f"-{days} days"))

    def record_trajectory_result(
        self,
        symbol: str,
        trajectory: Optional[float],
        confidence: Optional[float],
        composition: str,
        granularity: Optional[str],
        peer_set: list,
        raw_inputs: dict,
        engine_version: str,
        peer_state: str = "NONE",
        derived_json: Optional[Union[str, dict]] = None,
        extraction_version: Optional[str] = None,
    ):
        from datetime import datetime, timezone
        if composition not in ("AGB", "AG", "AB", "A", "WITHHELD"):
            raise ValueError(f"Invalid composition '{composition}'")
        if composition == "WITHHELD" and trajectory is not None:
            raise ValueError("Trajectory must be NULL when composition is WITHHELD")
        if trajectory is None and composition != "WITHHELD":
            raise ValueError("composition must be WITHHELD when trajectory is None")
        if "B" in composition and len(peer_set) < 5:
            raise ValueError(f"Composition '{composition}' requires at least 5 peers in peer_set")
        if peer_state not in ("OK", "THIN", "NONE"):
            raise ValueError(f"Invalid peer_state '{peer_state}'")

        sorted_peer_set = sorted(list(peer_set)) if peer_set else []
        peer_set_json = json.dumps(sorted_peer_set, separators=(",", ":"))
        raw_inputs_json = json.dumps(raw_inputs) if isinstance(raw_inputs, dict) else (raw_inputs or "{}")
        derived_json_str = json.dumps(derived_json) if isinstance(derived_json, dict) else (derived_json or "{}")

        with self.lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            scan_date = now_iso[:10]

            self.conn.execute(
                """INSERT OR REPLACE INTO trajectory_history 
                   (symbol, scan_date, trajectory, confidence, composition, peer_state, granularity_used, peer_set_json, raw_inputs_json, derived_json, extraction_version, engine_version, evaluated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol.upper(),
                    scan_date,
                    trajectory,
                    confidence,
                    composition,
                    peer_state,
                    granularity,
                    peer_set_json,
                    raw_inputs_json,
                    derived_json_str,
                    extraction_version,
                    engine_version,
                    now_iso,
                ),
            )
            self.conn.commit()

    def upsert_price_history(self, symbol: str, df: Any, source: str = "yfinance"):
        if df is None or df.empty:
            return
        rows = []
        for idx, row in df.iterrows():
            d_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            rows.append(
                (
                    symbol,
                    d_str,
                    float(row.get("Open", 0.0)),
                    float(row.get("High", 0.0)),
                    float(row.get("Low", 0.0)),
                    float(row.get("Close", 0.0)),
                    float(row.get("Volume", 0.0)),
                    source,
                    int(row.get("truncated_window", 0)),
                )
            )
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO price_history (symbol, date, open, high, low, close, volume, source, truncated_window) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()

    def flag_price_history_truncated(self, symbol: str):
        with self.lock:
            self.conn.execute("UPDATE price_history SET truncated_window = 1 WHERE symbol = ?", (symbol,))
            self.conn.commit()

    def get_historical_symbols(self) -> List[str]:
        with self.lock:
            cur = self.conn.execute("SELECT DISTINCT symbol FROM sp500_scores_history")
            return [r[0] for r in cur.fetchall()]

    def get_ticker_map(self, symbol: str) -> Optional[Tuple[str, str, str, str]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT cik, name, sector, industry FROM ticker_map WHERE symbol = ?",
                (symbol.upper(),),
            )
            row = cur.fetchone()
            return tuple(row) if row else None

    def resolve_symbol_from_db(self, query: str) -> Optional[str]:
        """Resolve an exact ticker or company name query against local SQLite tables."""
        q = (query or "").strip()
        if not q:
            return None
        q_up = q.upper()
        q_norm = q_up.replace(".", "-")
        with self.lock:
            for cand in (q_up, q_norm):
                row = self.conn.execute(
                    "SELECT symbol FROM ticker_map WHERE symbol = ? LIMIT 1", (cand,)
                ).fetchone()
                if row and row[0]:
                    return str(row[0]).upper()
                row_s = self.conn.execute(
                    "SELECT symbol FROM sp500_scores WHERE symbol = ? LIMIT 1", (cand,)
                ).fetchone()
                if row_s and row_s[0]:
                    return str(row_s[0]).upper()

            q_low = q.lower()
            rows = self.conn.execute(
                "SELECT symbol, name FROM ticker_map WHERE name IS NOT NULL AND name != ''"
            ).fetchall()
        best_sym: Optional[str] = None
        best_rank = 99
        for r in rows:
            sym = str(r[0] or "").strip().upper()
            name_low = str(r[1] or "").strip().lower()
            if not sym or not name_low:
                continue
            # Strip common corporate suffixes for exact matching
            cleaned = name_low
            for suf in (", inc.", " inc.", " inc", ", corp.", " corp.", " corp", " corporation", " company", " co.", " plc", " ltd.", " ltd", " n.v.", " s.a."):
                if cleaned.endswith(suf):
                    cleaned = cleaned[: -len(suf)].strip()
            if cleaned == q_low or name_low == q_low:
                rank = 0
            elif name_low.startswith(q_low + " ") or name_low.startswith(q_low + ","):
                rank = 1
            elif len(q_low) >= 3 and f" {q_low} " in f" {cleaned} ":
                rank = 2
            elif len(q_low) >= 4 and q_low in cleaned:
                rank = 3
            else:
                continue
            # Prefer shorter primary ticker (e.g. AAPL over warrant/preferred tickers)
            if rank < best_rank or (rank == best_rank and best_sym and len(sym) < len(best_sym)):
                best_rank = rank
                best_sym = sym
        return best_sym

    def upsert_ticker_map(self, symbol: str, cik: str, name: str, sector: str = "", industry: str = ""):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO ticker_map (symbol, cik, name, sector, industry) VALUES (?, ?, ?, ?, ?)",
                (symbol.upper(), cik, name, sector, industry),
            )
            self.conn.commit()

    def upsert_ticker_classification(self, symbol: str, name: str, sector: str, industry: str, source_label: str):
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO ticker_map (symbol, cik, name, sector, industry, classification_as_of) VALUES (?, COALESCE((SELECT cik FROM ticker_map WHERE symbol=?), ''), ?, ?, ?, ?)",
                (symbol.upper(), symbol.upper(), name, sector, industry, now_iso),
            )
            self.conn.commit()

    def get_dcf_inputs(self, symbol: str) -> Optional[dict]:
        with self.lock:
            cur = self.conn.execute("SELECT * FROM dcf_inputs WHERE symbol = ?", (symbol.upper(),))
            row = cur.fetchone()
            return dict(row) if row else None

    def save_dcf_inputs(
        self,
        symbol: str,
        data_type: str,
        fcf_or_eps: float,
        growth: float,
        discount: float,
        terminal_mult: float,
        target_return: float,
    ):
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO dcf_inputs 
                   (symbol, data_type, fcf_or_eps, growth, discount, terminal_mult, target_return, updated_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol.upper(),
                    data_type,
                    fcf_or_eps,
                    growth,
                    discount,
                    terminal_mult,
                    target_return,
                    now_iso,
                ),
            )
            self.conn.commit()

    def get_watchlist(self, name: str = "Main") -> List[str]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT symbol FROM watchlists WHERE name = ? ORDER BY symbol ASC", (name,)
            )
            return [r[0] for r in cur.fetchall() if r[0] != "__INIT__"]

    def add_watchlist_symbol(self, name: str, symbol: str):
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO watchlists (name, symbol) VALUES (?, ?)",
                (name, symbol.upper()),
            )
            self.conn.commit()

    def remove_watchlist_symbol(self, name: str, symbol: str):
        with self.lock:
            self.conn.execute(
                "DELETE FROM watchlists WHERE name = ? AND symbol = ?",
                (name, symbol.upper()),
            )
            self.conn.commit()

    def list_watchlists(self) -> List[str]:
        with self.lock:
            cur = self.conn.execute("SELECT DISTINCT name FROM watchlists ORDER BY name ASC")
            return [r[0] for r in cur.fetchall()]

    def get_portfolio(self, name: str = "Main") -> List[Tuple[str, float, float]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT symbol, shares, cost_basis FROM portfolios WHERE name = ? ORDER BY symbol ASC",
                (name,),
            )
            return [(r[0], r[1], r[2]) for r in cur.fetchall() if r[0] != "__INIT__"]

    def upsert_portfolio_symbol(self, name: str, symbol: str, shares: float, cost_basis: float):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO portfolios (name, symbol, shares, cost_basis) VALUES (?, ?, ?, ?)",
                (name, symbol.upper(), shares, cost_basis),
            )
            self.conn.commit()

    def remove_portfolio_symbol(self, name: str, symbol: str):
        with self.lock:
            self.conn.execute(
                "DELETE FROM portfolios WHERE name = ? AND symbol = ?",
                (name, symbol.upper()),
            )
            self.conn.commit()

    def list_portfolios(self) -> List[str]:
        with self.lock:
            cur = self.conn.execute("SELECT DISTINCT name FROM portfolios ORDER BY name ASC")
            return [r[0] for r in cur.fetchall()]

    def add_portfolio_lot(
        self,
        portfolio_name: str,
        symbol: str,
        shares: float,
        cost_basis: float,
        trade_date: Optional[str] = None,
        commission: float = 0.0,
        realized_pnl: float = 0.0,
        notes: Optional[str] = None,
    ) -> int:
        p_name = (portfolio_name or "Main").strip() or "Main"
        sym = (symbol or "").strip().upper()
        if not sym:
            raise ValueError("symbol must not be empty")
        sh = float(shares)
        cb = float(cost_basis)
        comm = float(commission or 0.0)
        rpnl = float(realized_pnl or 0.0)

        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO portfolio_lots
                (portfolio_name, symbol, shares, cost_basis, trade_date, commission, realized_pnl, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (p_name, sym, sh, cb, trade_date, comm, rpnl, notes),
            )
            lot_id = int(cur.lastrowid)

            lots = self.conn.execute(
                "SELECT shares, cost_basis FROM portfolio_lots WHERE portfolio_name = ? AND symbol = ?",
                (p_name, sym),
            ).fetchall()
            tot_sh = sum(float(r["shares"]) for r in lots)
            if tot_sh > 0:
                w_cost = sum(float(r["shares"]) * float(r["cost_basis"]) for r in lots) / tot_sh
                self.conn.execute(
                    "INSERT OR REPLACE INTO portfolios (name, symbol, shares, cost_basis) VALUES (?, ?, ?, ?)",
                    (p_name, sym, tot_sh, w_cost),
                )
            else:
                self.conn.execute(
                    "DELETE FROM portfolios WHERE name = ? AND symbol = ?",
                    (p_name, sym),
                )
            self.conn.commit()
            return lot_id

    def get_portfolio_lots(self, portfolio_name: str = "Main") -> List[Dict[str, Any]]:
        p_name = (portfolio_name or "Main").strip() or "Main"
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM portfolio_lots WHERE portfolio_name = ? ORDER BY lot_id ASC",
                (p_name,),
            )
            return [dict(row) for row in cur.fetchall()]

    def clear_portfolio_lots(self, portfolio_name: str = "Main", symbol: Optional[str] = None):
        p_name = (portfolio_name or "Main").strip() or "Main"
        with self.lock:
            if symbol:
                sym = symbol.strip().upper()
                self.conn.execute(
                    "DELETE FROM portfolio_lots WHERE portfolio_name = ? AND symbol = ?",
                    (p_name, sym),
                )
                self.conn.execute(
                    "DELETE FROM portfolios WHERE name = ? AND symbol = ?",
                    (p_name, sym),
                )
            else:
                self.conn.execute(
                    "DELETE FROM portfolio_lots WHERE portfolio_name = ?",
                    (p_name,),
                )
                self.conn.execute(
                    "DELETE FROM portfolios WHERE name = ?",
                    (p_name,),
                )
            self.conn.commit()

    def list_transcripts(self, symbol: str) -> List[Tuple[str, str, str]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT date, title, content FROM transcripts WHERE symbol = ? ORDER BY date DESC",
                (symbol.upper(),),
            )
            return [tuple(r) for r in cur.fetchall()]

    def upsert_transcript(self, symbol: str, date: str, title: str, content: str):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO transcripts (symbol, date, title, content) VALUES (?, ?, ?, ?)",
                (symbol.upper(), date, title, content),
            )
            self.conn.commit()

    def transcript_exists(self, symbol: str, date: str, title: str) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "SELECT 1 FROM transcripts WHERE symbol = ? AND date = ? AND title = ?",
                (symbol.upper(), date, title),
            )
            return cur.fetchone() is not None

    def search_transcripts(self, symbol: str, query: str) -> List[Tuple[str, str, str]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT date, title, content FROM transcripts WHERE symbol = ? AND content LIKE ? ORDER BY date DESC",
                (symbol.upper(), f"%{query}%"),
            )
            return [tuple(r) for r in cur.fetchall()]

    def get_transcript_analysis(self, symbol: str, date: str, title: str) -> Optional[Tuple[str, str]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT summary, qa FROM transcript_analysis WHERE symbol = ? AND date = ? AND title = ?",
                (symbol.upper(), date, title),
            )
            row = cur.fetchone()
            return tuple(row) if row else None

    def upsert_transcript_analysis(self, symbol: str, date: str, title: str, summary: str, qa: str):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO transcript_analysis (symbol, date, title, summary, qa) VALUES (?, ?, ?, ?, ?)",
                (symbol.upper(), date, title, summary, qa),
            )
            self.conn.commit()

    def list_kpis(self, symbol: str) -> List[Tuple[str, str, str, str]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT name, period, value, unit FROM kpis WHERE symbol = ? ORDER BY name ASC, period DESC",
                (symbol.upper(),),
            )
            return [tuple(r) for r in cur.fetchall()]

    def add_kpi(self, symbol: str, name: str, period: str, value: str, unit: str):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO kpis (symbol, name, period, value, unit) VALUES (?, ?, ?, ?, ?)",
                (symbol.upper(), name, period, value, unit),
            )
            self.conn.commit()

    def clear_kpis(self, symbol: str):
        with self.lock:
            self.conn.execute("DELETE FROM kpis WHERE symbol = ?", (symbol.upper(),))
            self.conn.commit()

    def clear_kpis_for_period(self, symbol: str, period: str):
        with self.lock:
            self.conn.execute(
                "DELETE FROM kpis WHERE symbol = ? AND period = ?",
                (symbol.upper(), period),
            )
            self.conn.commit()

    def upsert_lens(self, lens_id: str, name: str, definition: dict) -> str:
        from datetime import datetime, timezone
        from compounderos_app.scoring_engine import (
            lens_from_definition,
            registry_hash,
            validate_lens,
        )

        reg, bw, gt, gc = lens_from_definition(definition)
        validate_lens(reg, bw, gt, gc)
        h = registry_hash(reg, bw, gt, gc)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO lenses (lens_id, name, lens_hash, definition_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                (lens_id, name, h, json.dumps(definition), now_iso),
            )
            self.conn.commit()
        return h

    def get_lens(self, lens_id_or_hash: str) -> Optional[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT lens_id, name, lens_hash, definition_json, updated_at FROM lenses WHERE lens_id = ? OR lens_hash = ?",
                (lens_id_or_hash, lens_id_or_hash),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "lens_id": row["lens_id"],
                "name": row["name"],
                "lens_hash": row["lens_hash"],
                "definition": json.loads(row["definition_json"]),
                "updated_at": row["updated_at"],
            }

    def upsert_sec_filings(self, symbol: str, filings: List[Dict[str, Any]]) -> int:
        from datetime import datetime, timezone

        sym = (symbol or "").strip().upper()
        if not sym or not filings:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = []
        for f in filings:
            acc = str(f.get("accession_number") or "").strip()
            form = str(f.get("form") or "").strip()
            f_date = str(f.get("filing_date") or "").strip()
            e_url = str(f.get("edgar_url") or "").strip()
            if not acc or not form or not f_date or not e_url:
                continue
            r_date = str(f.get("report_date") or "").strip() or None
            p_doc = str(f.get("primary_document") or "").strip() or None
            items_val = f.get("items")
            if isinstance(items_val, (list, tuple)):
                items_str = ", ".join(str(x) for x in items_val if str(x).strip())
            else:
                items_str = str(items_val or "").strip()
            fetched = str(f.get("fetched_at") or now_iso)
            rows.append((sym, acc, form, f_date, r_date, p_doc, items_str, e_url, fetched))

        if not rows:
            return 0
        with self.lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO sec_filings
                (symbol, accession_number, form, filing_date, report_date, primary_document, items, edgar_url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()
        return len(rows)

    def get_sec_filings(
        self, symbol: str, forms: Optional[List[str]] = None, limit: int = 25
    ) -> List[Dict[str, Any]]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return []
        with self.lock:
            query = "SELECT * FROM sec_filings WHERE symbol = ?"
            params: List[Any] = [sym]
            if forms:
                placeholders = ",".join(["?"] * len(forms))
                query += f" AND form IN ({placeholders})"
                params.extend(str(f).strip() for f in forms)
            query += " ORDER BY filing_date DESC, accession_number DESC LIMIT ?"
            params.append(int(limit))
            cur = self.conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def upsert_macro_quotes(self, quotes: List[Dict[str, Any]]) -> int:
        if not quotes:
            return 0
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for q in quotes:
            if not isinstance(q, dict):
                continue
            sym = str(q.get("symbol") or "").strip().upper()
            if not sym:
                continue
            label = str(q.get("label") or sym).strip()
            category = str(q.get("category") or "indices").strip().lower()
            price = q.get("price")
            change_pct = q.get("change_pct")
            unit = str(q.get("unit") or "").strip()
            updated_at = str(q.get("updated_at") or now_iso).strip()
            rows.append(
                (
                    sym,
                    label,
                    category,
                    float(price) if price is not None and np.isfinite(float(price)) else None,
                    float(change_pct) if change_pct is not None and np.isfinite(float(change_pct)) else None,
                    unit,
                    updated_at,
                )
            )
        if not rows:
            return 0
        with self.lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO macro_quotes
                (symbol, label, category, price, change_pct, unit, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()
        return len(rows)

    def get_macro_quotes(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            if category:
                cur = self.conn.execute(
                    "SELECT * FROM macro_quotes WHERE category = ? ORDER BY symbol ASC",
                    (str(category).strip().lower(),),
                )
            else:
                cur = self.conn.execute("SELECT * FROM macro_quotes ORDER BY category ASC, symbol ASC")
            return [dict(r) for r in cur.fetchall()]

    def get_us10y_yield(self, default: float = 0.0425) -> float:
        with self.lock:
            row = self.conn.execute(
                """
                SELECT price FROM macro_quotes
                WHERE symbol IN ('^TNX', 'US10Y') AND price IS NOT NULL
                ORDER BY CASE WHEN symbol = '^TNX' THEN 0 ELSE 1 END
                LIMIT 1
                """
            ).fetchone()
        if not row or row["price"] is None:
            return float(default)
        try:
            val = float(row["price"])
            if not np.isfinite(val) or val <= 0:
                return float(default)
            return val / 100.0 if val > 0.5 else val
        except (TypeError, ValueError):
            return float(default)

    def upsert_prediction_odds(self, odds: List[Dict[str, Any]]) -> int:
        if not odds:
            return 0
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for item in odds:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("contract_id") or "").strip()
            title = str(item.get("title") or "").strip()
            if not cid or not title:
                continue
            source = str(item.get("source") or "Kalshi").strip()
            category = str(item.get("category") or "macro").strip().lower()
            prob = float(item.get("probability_pct") or 0.0)
            vol = item.get("volume")
            url = str(item.get("url") or "").strip() or None
            updated_at = str(item.get("updated_at") or now_iso).strip()
            rows.append(
                (
                    cid,
                    source,
                    category,
                    title,
                    prob,
                    float(vol) if vol is not None and np.isfinite(float(vol)) else None,
                    url,
                    updated_at,
                )
            )
        if not rows:
            return 0
        with self.lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO prediction_odds
                (contract_id, source, category, title, probability_pct, volume, url, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()
        return len(rows)

    def get_prediction_odds(
        self, category: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        with self.lock:
            if category:
                cur = self.conn.execute(
                    "SELECT * FROM prediction_odds WHERE category = ? ORDER BY COALESCE(volume, 0) DESC, contract_id ASC LIMIT ?",
                    (str(category).strip().lower(), int(limit)),
                )
            else:
                cur = self.conn.execute(
                    "SELECT * FROM prediction_odds ORDER BY COALESCE(volume, 0) DESC, contract_id ASC LIMIT ?",
                    (int(limit),),
                )
            return [dict(r) for r in cur.fetchall()]

    def append_research_message(
        self,
        thread_id: str,
        symbol: str,
        role: str,
        content: str,
        mode: str = "chat",
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        tid = str(thread_id or "").strip()
        sym = str(symbol or "").strip().upper()
        r_norm = str(role or "user").strip().lower()
        if r_norm not in ("user", "assistant", "system"):
            r_norm = "user"
        m_norm = str(mode or "chat").strip().lower()
        if m_norm not in ("chat", "deep_search"):
            m_norm = "chat"
        c_json = json.dumps(citations or [])
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO research_threads
                (thread_id, symbol, role, mode, content, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tid, sym, r_norm, m_norm, str(content or ""), c_json, now_iso),
            )
            msg_id = int(cur.lastrowid or 0)
            self.conn.commit()
            return msg_id

    def get_research_thread(self, thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM research_threads
                    WHERE thread_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                ) ORDER BY message_id ASC
                """,
                (tid, int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["citations"] = json.loads(d.get("citations_json") or "[]")
            except Exception:
                d["citations"] = []
            out.append(d)
        return out

    def close(self):
        with self.lock:
            self.conn.close()
