import json
from typing import Any, Callable, Dict, List, Optional

from .database import DataStore
from .macro_tracker import compute_equity_risk_premium
from .quality_diagnostics import (
    assess_moat_and_five_forces,
    compute_quality_diagnostics,
    synthesize_bull_bear_views,
)
from .scoring_engine import ENGINE_VERSION
from .trajectory_engine import TRAJ_ENGINE_VERSION


class ResearchCopilot:
    """
    Multi-turn grounded research copilot and Deep Search workflow for CompounderOS (W3-06).
    Grounds every response in point-in-time SQLite tables (det-1.1.0, traj-1.1.0, sec_filings,
    transcripts, and macro_quotes) with structured source citations.
    """

    @staticmethod
    def build_grounding_packet(store: DataStore, symbol: str) -> Dict[str, Any]:
        sym = (symbol or "").strip().upper()
        score_row: Optional[Dict[str, Any]] = None
        raw_metrics: Dict[str, Any] = {}
        subscores: Dict[str, Any] = {}
        traj_row: Optional[Dict[str, Any]] = None
        traj_derived: Dict[str, Any] = {}
        transcripts: List[Dict[str, Any]] = []
        transcript_analyses: List[Dict[str, Any]] = []

        with store.lock:
            s_cur = store.conn.execute(
                "SELECT * FROM sp500_scores_history WHERE symbol = ? ORDER BY evaluated_at DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if s_cur:
                score_row = dict(s_cur)
                try:
                    raw_metrics = json.loads(score_row.get("raw_json") or "{}")
                except Exception:
                    raw_metrics = {}
                try:
                    subscores = json.loads(score_row.get("subscores_json") or "{}")
                except Exception:
                    subscores = {}

            t_cur = store.conn.execute(
                "SELECT * FROM trajectory_history WHERE symbol = ? ORDER BY evaluated_at DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if t_cur:
                traj_row = dict(t_cur)
                try:
                    traj_derived = json.loads(traj_row.get("derived_json") or "{}")
                except Exception:
                    traj_derived = {}

            try:
                tr_rows = store.conn.execute(
                    "SELECT symbol, quarter, date, substr(content, 1, 600) AS snippet FROM transcripts WHERE symbol = ? ORDER BY date DESC LIMIT 3",
                    (sym,),
                ).fetchall()
                transcripts = [dict(r) for r in tr_rows]
            except Exception:
                transcripts = []

            try:
                ta_rows = store.conn.execute(
                    "SELECT symbol, quarter, analysis_type, substr(result, 1, 600) AS snippet FROM transcript_analysis WHERE symbol = ? ORDER BY created_at DESC LIMIT 3",
                    (sym,),
                ).fetchall()
                transcript_analyses = [dict(r) for r in ta_rows]
            except Exception:
                transcript_analyses = []

        sec_filings = store.get_sec_filings(sym, limit=10) if hasattr(store, "get_sec_filings") else []
        us10y = store.get_us10y_yield() if hasattr(store, "get_us10y_yield") else 0.0425
        fcf_yield = score_row.get("fcf_yield") if score_row else raw_metrics.get("fcf_yield")
        erp = compute_equity_risk_premium(fcf_yield, us10y)

        sector_val = score_row.get("sector") if score_row else None
        diagnostics = compute_quality_diagnostics(raw_metrics=raw_metrics, info={"sector": sector_val})
        moat_assessment = assess_moat_and_five_forces(
            raw_metrics=raw_metrics,
            diagnostics=diagnostics,
            sector=sector_val,
        )

        bull_bear = synthesize_bull_bear_views(
            symbol=sym,
            score_res=score_row,
            raw_metrics=raw_metrics,
            traj_row=traj_row,
            info={"sector": sector_val},
            diagnostics=diagnostics,
        )

        return {
            "symbol": sym,
            "score_row": score_row,
            "raw_metrics": raw_metrics,
            "subscores": subscores,
            "trajectory_row": traj_row,
            "trajectory_derived": traj_derived,
            "sec_filings": sec_filings,
            "transcripts": transcripts,
            "transcript_analyses": transcript_analyses,
            "us10y_yield": us10y,
            "equity_risk_premium": erp,
            "diagnostics": diagnostics,
            "moat_assessment": moat_assessment,
            "bull_bear": bull_bear,
        }

    @staticmethod
    def _build_citations(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        sym = packet["symbol"]
        citations: List[Dict[str, Any]] = []
        s_row = packet.get("score_row")
        if s_row:
            shrunk = s_row.get("composite_shrunk")
            score = s_row.get("score")
            comp_val = shrunk if shrunk is not None else score
            comp_str = f"{float(comp_val):.2f}/10" if comp_val is not None else "N/A"
            citations.append(
                {
                    "source": ENGINE_VERSION,
                    "label": f"{sym} Deterministic Quality ({ENGINE_VERSION})",
                    "detail": f"Composite {comp_str} · Gate State: {str(s_row.get('gate_state') or 'pass').upper()} · Evaluated: {s_row.get('evaluated_at')}",
                    "url": None,
                }
            )
        else:
            citations.append(
                {
                    "source": ENGINE_VERSION,
                    "label": f"{sym} Deterministic Quality ({ENGINE_VERSION})",
                    "detail": "Baseline deterministic engine reference (no scored snapshot in DB)",
                    "url": None,
                }
            )

        t_row = packet.get("trajectory_row")
        if t_row:
            t_val = t_row.get("trajectory")
            t_str = f"{float(t_val):.1f}/100" if t_val is not None else "WITHHELD"
            citations.append(
                {
                    "source": TRAJ_ENGINE_VERSION,
                    "label": f"{sym} Trajectory Momentum ({TRAJ_ENGINE_VERSION})",
                    "detail": f"Trajectory {t_str} ({t_row.get('composition')}) · Peer State: {t_row.get('peer_state')}",
                    "url": None,
                }
            )
        else:
            citations.append(
                {
                    "source": TRAJ_ENGINE_VERSION,
                    "label": f"{sym} Trajectory Momentum ({TRAJ_ENGINE_VERSION})",
                    "detail": "Trajectory engine reference (no trajectory snapshot in DB)",
                    "url": None,
                }
            )

        filings = packet.get("sec_filings") or []
        if filings:
            top_f = filings[0]
            citations.append(
                {
                    "source": "SEC EDGAR",
                    "label": f"{sym} {top_f.get('form')} ({top_f.get('filing_date')})",
                    "detail": f"Accession {top_f.get('accession_number')}",
                    "url": top_f.get("edgar_url"),
                }
            )
        else:
            citations.append(
                {
                    "source": "SEC EDGAR",
                    "label": f"{sym} SEC Filings",
                    "detail": "No cached SEC filings in local SQLite store",
                    "url": None,
                }
            )
        return citations

    @classmethod
    def ask(
        cls,
        store: DataStore,
        symbol: str,
        question: str,
        thread_id: Optional[str] = None,
        mode: str = "chat",
        gemini_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        sym = (symbol or "").strip().upper()
        tid = (thread_id or f"web:{sym}").strip()
        m_norm = "deep_search" if str(mode or "chat").strip().lower() == "deep_search" else "chat"
        q_clean = (question or "").strip() or f"Summarize the investment thesis and risks for {sym}."

        # 1. Persist user message
        store.append_research_message(tid, sym, "user", q_clean, mode=m_norm, citations=[])

        # 2. Build grounding packet and citations
        packet = cls.build_grounding_packet(store, sym)
        citations = cls._build_citations(packet)

        plan_steps = [
            f"1. Audit {sym} deterministic quality subscores, bucket averages, and balance-sheet gates ({ENGINE_VERSION}).",
            f"2. Verify {sym} fundamental revision momentum and peer breadth ({TRAJ_ENGINE_VERSION}).",
            f"3. Evaluate FCF yield, P/E valuation multiples, and Equity Risk Premium vs. US 10Y Treasury ({packet['us10y_yield'] * 100:.2f}%).",
            f"4. Cross-check recent SEC EDGAR filings ({len(packet['sec_filings'])} cached) and earnings transcript notes.",
        ]

        s_row = packet.get("score_row") or {}
        raw_m = packet.get("raw_metrics") or {}
        t_row = packet.get("trajectory_row") or {}
        shrunk = s_row.get("composite_shrunk")
        score = s_row.get("score")
        comp_val = shrunk if shrunk is not None else score
        comp_s = f"{float(comp_val):.2f}/10" if comp_val is not None else "N/A"
        gate_s = str(s_row.get("gate_state") or "pass").upper()
        gates_fired = s_row.get("gates_fired") or "none"
        roic_v = raw_m.get("roic")
        roic_s = f"{float(roic_v) * 100:.1f}%" if roic_v is not None else "N/A"
        fcf_m = raw_m.get("fcf_margin")
        fcf_m_s = f"{float(fcf_m) * 100:.1f}%" if fcf_m is not None else "N/A"
        fy_v = s_row.get("fcf_yield")
        fy_s = f"{float(fy_v) * 100:.2f}%" if fy_v is not None else "N/A"
        erp_v = packet.get("equity_risk_premium")
        erp_s = f"{float(erp_v) * 100:+.2f}% ({float(erp_v) * 10000:+.0f} bps)" if erp_v is not None else "N/A"
        traj_v = t_row.get("trajectory")
        traj_s = f"{float(traj_v):.1f}/100 ({t_row.get('composition', 'N/A')})" if traj_v is not None else "N/A"

        moat_info = packet.get("moat_assessment") or {}
        diag_info = packet.get("diagnostics") or {}
        moat_lbl = moat_info.get("verdict_label") or "Unverified"
        wc_reg = diag_info.get("working_capital_regime") or "STANDARD"

        evidence_lines = [
            f"- Quality Composite ({ENGINE_VERSION}): {comp_s} | Gate State: {gate_s} (Gates Fired: {gates_fired})",
            f"- Core Economics: ROIC {roic_s} | FCF Margin {fcf_m_s} | FCF Yield {fy_s}",
            f"- Moat & Balance-Sheet Regime: {moat_lbl} | Working Capital: {wc_reg}",
            f"- Trajectory ({TRAJ_ENGINE_VERSION}): {traj_s}",
            f"- Macro & Valuation: US 10Y {packet['us10y_yield'] * 100:.2f}% | Equity Risk Premium (ERP) {erp_s}",
        ]
        if diag_info.get("red_flags"):
            evidence_lines.append("- Forensic Red Flags: " + "; ".join(diag_info["red_flags"]))
        if packet["sec_filings"]:
            f0 = packet["sec_filings"][0]
            evidence_lines.append(
                f"- Latest SEC Filing: Form {f0.get('form')} filed {f0.get('filing_date')} ({f0.get('accession_number')})"
            )

        bull_bullets = packet["bull_bear"].get("bullish", [])
        bear_bullets = packet["bull_bear"].get("bearish", [])
        offline_synthesis = (
            f"For {sym} (question: \"{q_clean}\"): "
            f"Deterministic quality stands at {comp_s} (gate state: {gate_s}) with ROIC of {roic_s}, "
            f"FCF yield of {fy_s} (ERP {erp_s} vs 10Y Treasury), and trajectory score {traj_s}.\n"
            f"Key Bullish Catalyst: {bull_bullets[0] if bull_bullets else 'Strong compounder economics.'}\n"
            f"Key Bearish Risk: {bear_bullets[0] if bear_bullets else 'Monitor valuation and cyclicality.'}"
        )

        llm_synthesis = None
        if gemini_fn is not None:
            try:
                prompt = (
                    f"You are the CompounderOS Research Copilot. Answer the user's question about {sym} "
                    f"using ONLY the grounded quantitative evidence below.\n\n"
                    f"QUESTION: {q_clean}\n\n"
                    f"EVIDENCE:\n" + "\n".join(evidence_lines)
                )
                res_txt = gemini_fn(prompt)
                if res_txt and str(res_txt).strip():
                    llm_synthesis = str(res_txt).strip()
            except Exception:
                llm_synthesis = None

        core_answer = llm_synthesis or offline_synthesis

        if m_norm == "deep_search":
            cit_lines = [
                f"- [{c['source']}] {c['label']}: {c['detail']}"
                + (f" ({c['url']})" if c.get("url") else "")
                for c in citations
            ]
            answer_text = (
                "RESEARCH PLAN:\n"
                + "\n".join(plan_steps)
                + "\n\nQUANTITATIVE EVIDENCE:\n"
                + "\n".join(evidence_lines)
                + "\n\nSYNTHESIS & ANSWER:\n"
                + core_answer
                + "\n\nCITATIONS:\n"
                + "\n".join(cit_lines)
            )
        else:
            answer_text = core_answer

        # 3. Persist assistant message
        store.append_research_message(
            tid, sym, "assistant", answer_text, mode=m_norm, citations=citations
        )

        return {
            "thread_id": tid,
            "symbol": sym,
            "mode": m_norm,
            "answer": answer_text,
            "plan": plan_steps,
            "citations": citations,
        }
