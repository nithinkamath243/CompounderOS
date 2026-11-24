# CompounderOS

CompounderOS is a local-first stock research workstation and S&P 500 scanner built around one goal: finding durable, high-ROIC businesses at sensible prices—and avoiding accounting traps along the way.

Most screeners treat every metric as a flat checklist or hide their math behind a black-box rating. CompounderOS pulls raw SEC EDGAR filings and market data into a local SQLite database, scores companies with transparent, deterministic rules, and pairs that data with valuation, portfolio risk, and grounded research tools.

It runs in two parts that can be used together or on their own:
1. **Desktop App (macOS / Python):** Full research workstation with interactive charts, SEC filings, valuation models, earnings transcripts, and portfolio analytics.
2. **Headless S&P 500 Scanner & Web UI (FastAPI / Home Assistant Add-on):** Runs 24/7 on a Raspberry Pi (or any Linux box), continuously scoring the S&P 500 in the background, serving a mobile/iPad-friendly web dashboard, and syncing results back to the desktop app.

---

## What It Does

### 1. Quality Scoring & Forensic Checks
- **12-Metric Quality Engine:** Scores businesses from 0 to 10 across four buckets—**Profitability**, **Capital Efficiency**, **Balance Sheet**, and **Capital Allocation**—with Bayesian shrinkage when data coverage is partial.
- **Hard Risk Gates:** Automatically caps scores when leverage, liquidity, dilution, or cash conversion cross danger thresholds.
- **Forensic Accounting Overlays:** Checks SBC-adjusted owner cash flow, interest coverage (`EBIT / Interest`), receivables-vs-revenue divergence (channel-stuffing check), inventory turnover & GMROI, and dividend FCF payout sustainability.
- **Working-Capital Context:** Distinguishes genuinely cash-strapped companies from float-funded compounders (like Apple or Costco) that run negative working capital by design.
- **Quantitative Moat Check:** Evaluates pricing power, ROIC durability, and Porter's Five Forces strictly against reported numbers—never awarding a moat label if returns on capital don't clear the bar.

### 2. Trajectory & GARP Screening
- **Fundamental Trajectory (0–100):** Tracks whether a company's operating momentum and earnings revisions are improving or slipping relative to its GICS industry peers.
- **Built-in Screener Presets:** Filter the S&P 500 by *Top Compounders*, *GARP: Compounders @ Fair Value*, *Trajectory Improvers*, or *Dip Candidates*.
- **News vs. Noise Filtering:** Ranks verifiable primary catalysts (10-K/10-Q/8-K filings, earnings, buybacks, regulatory actions) ahead of clickbait listicles.

### 3. Valuation & Macro Context
- **Forward & Reverse DCF:** Estimate fair value from normalized free cash flow or solve for the growth rate already priced into today's share price.
- **Live Equity Risk Premium (ERP):** Compares each stock's FCF yield directly against the live 10-Year US Treasury yield (`^TNX`).
- **Macro & Prediction Markets:** Tracks Treasury yields, the 10Y–2Y curve, VIX, and consensus macro probabilities from Kalshi and Polymarket.

### 4. Portfolio & Tax-Lot Analytics
- **Quick Import:** Add positions from a brokerage CSV or plain English (e.g., `25 shares of NVDA at $118.50 and 10 MSFT @ $420`).
- **Look-Through Fundamentals:** Calculates your portfolio's dollar-weighted quality score, ROIC, FCF yield, and trajectory.
- **Risk & Clustering:** Computes annualized volatility, Sharpe ratio, portfolio Beta & $R^2$ vs. `SPY`, single-name/sector concentration alerts, and high-correlation pair warnings (`> 0.80`).
- **Tax-Lot Tracking:** Tracks holding periods for Long-Term (`> 365 days`) vs. Short-Term capital gains, flags tax-loss harvesting candidates, and compares **Specific-ID** vs. **FIFO** lot sales.

### 5. Grounded Research Copilot
- Ask multi-turn questions or generate structured 4-part Deep Search notes on any ticker, grounded in cached SEC filings, transcripts, quality subscores, and macro data—with optional Gemini synthesis if you supply an API key, and a deterministic offline mode if you don't.

---

## Quick Start (Desktop App)

### 1. Set up Python (3.11+)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your SEC User-Agent (and optional API keys)
On first launch, CompounderOS stores its local settings and SQLite databases in `~/CompounderOS/`. Create or edit `~/CompounderOS/.compounderos_config.json`:

```json
{
  "SEC_USER_AGENT": "CompounderOS/1.0 (contact: your_email@example.com)",
  "GEMINI_API_KEY": "",
  "FINNHUB_API_KEY": "",
  "ALPHAVANTAGE_API_KEY": "",
  "API_NINJAS_KEY": "",
  "SCANNER_API_URL": "",
  "SCANNER_API_TOKEN": ""
}
```
> **Note:** The SEC requires a contact email in `SEC_USER_AGENT` for EDGAR XBRL requests. All third-party API keys are optional—core SEC XBRL, Yahoo Finance data, quality scoring, valuation, and portfolio tools work without paid keys.

### 3. Launch the app
```bash
python compounderos.py
```
You can search by ticker (`AAPL`, `MSFT`, `BRK-B`) or company name (`apple`, `berkshire`).

---

## Running the Headless Scanner & Web UI

To run the background S&P 500 scanner and web interface locally:

```bash
pip install -r requirements-scanner.txt
COMPOUNDEROS_LOCAL_DEV=1 SCANNER_API_TOKEN=change-me python -m scanner_service
```
Then open `http://127.0.0.1:8321/ui/ranking` in your browser.

To run it 24/7 on **Home Assistant OS** (Raspberry Pi 5 / `aarch64`), see [`addons/compounderos_scanner/`](addons/compounderos_scanner/) and [`docs/status/DEPLOY.md`](docs/status/DEPLOY.md). Once running, set `SCANNER_API_URL` and `SCANNER_API_TOKEN` in your desktop config to sync rankings, trajectories, and price history automatically over your local network.

---

## Running Tests

The test suite runs completely offline against temporary SQLite databases:

```bash
pytest -q
```

---

## Project Layout

- `compounderos.py` / `run_app.py` — Desktop app entrypoints
- `compounderos_app/` — Core scoring engine (`det-1.1.0`), trajectory engine (`traj-1.1.0`), forensic diagnostics, portfolio/tax analytics, charts, SQLite store, and CustomTkinter GUI tabs
- `scanner_service/` — Headless S&P 500 pipeline, rate-limited Yahoo Finance gateway, FastAPI sync endpoints, and Jinja2 web UI
- `sp500_scanner.py` — Standalone S&P 500 universe fetcher and symbol evaluator
- `addons/compounderos_scanner/` — Home Assistant OS add-on (`Dockerfile` & `config.yaml`)
- `tests/` — Offline unit and regression tests

---

## Disclaimer

This project is for personal financial research and educational use only. Nothing in this repository is investment, tax, or legal advice—always verify numbers against official SEC filings before making financial decisions.
