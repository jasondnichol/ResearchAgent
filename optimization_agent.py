"""
AI Optimization Agent — Dual-Loop System
Loop 1: Donchian bot config tuning (validated via backtest_engine)
Loop 2: Signal methodology config tuning (validated via signal replay)

Runs on EC2 in screen -S optimizer.
Schedule: Weekly Monday 04:00 UTC (Sunday 8 PM PST).
Manual trigger: python optimization_agent.py --now
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

AI_MODEL = "claude-sonnet-4-20250514"
WEEKLY_SCHEDULE_UTC_HOUR = 4   # Monday 04:00 UTC
WEEKLY_SCHEDULE_DAY = 0        # Monday

MIN_RESOLVED_SIGNALS = 20
LOOKBACK_DAYS_SIGNALS = 30
LOOKBACK_DAYS_TRADES = 60

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/optimizer.log"),
    ]
)
logger = logging.getLogger("optimizer")


# ── Supabase DB Client (standalone, same pattern as main.py) ──

class SupabaseDB:
    def __init__(self, url, key):
        self.rest_url = f"{url}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def select(self, table, columns="*", filters=None, gte_filters=None,
               lte_filters=None, order=None, limit=None):
        url = f"{self.rest_url}/{table}?select={columns}"
        if filters:
            for k, v in filters.items():
                url += f"&{k}=eq.{v}"
        if gte_filters:
            for k, v in gte_filters.items():
                url += f"&{k}=gte.{v}"
        if lte_filters:
            for k, v in lte_filters.items():
                url += f"&{k}=lte.{v}"
        if order:
            url += f"&order={order}"
        if limit:
            url += f"&limit={limit}"
        resp = requests.get(url, headers=self.headers, timeout=15)
        return {"data": resp.json() if resp.status_code == 200 else []}

    def insert(self, table, data):
        url = f"{self.rest_url}/{table}"
        resp = requests.post(url, headers=self.headers, json=data, timeout=15)
        return {"data": resp.json() if resp.status_code in (200, 201) else []}

    def update(self, table, data, filters):
        url = f"{self.rest_url}/{table}"
        for k, v in filters.items():
            url += f"?{k}=eq.{v}" if "?" not in url else f"&{k}=eq.{v}"
        resp = requests.patch(url, headers=self.headers, json=data, timeout=15)
        return {"data": resp.json() if resp.status_code == 200 else []}

    def upsert(self, table, data, on_conflict=""):
        url = f"{self.rest_url}/{table}"
        headers = {**self.headers, "Prefer": "return=representation,resolution=merge-duplicates"}
        if on_conflict:
            url += f"?on_conflict={on_conflict}"
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        return {"data": resp.json() if resp.status_code in (200, 201) else []}


db = SupabaseDB(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL else None


# ── Data Collection ──

def collect_signal_data(days=LOOKBACK_DAYS_SIGNALS):
    """Fetch resolved signals, build breakdowns by methodology/TF/symbol."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = db.select(
        "signal_tracker",
        filters={},
        gte_filters={"signal_time": cutoff},
        order="signal_time.desc",
        limit=2000,
    )
    rows = result.get("data") or []

    resolved = [r for r in rows if r.get("outcome") in ("WIN", "LOSS", "NEUTRAL")]
    if len(resolved) < MIN_RESOLVED_SIGNALS:
        return None

    # Build breakdowns
    by_method = {}
    by_tf = {}
    by_symbol = {}
    losses = []

    for r in resolved:
        method = r.get("methodology", "unknown")
        tf = r.get("timeframe", "unknown")
        sym = r.get("symbol", "unknown")
        outcome = r.get("outcome")

        by_method.setdefault(method, {"total": 0, "wins": 0, "losses": 0, "returns": []})
        by_method[method]["total"] += 1
        if outcome == "WIN":
            by_method[method]["wins"] += 1
        elif outcome == "LOSS":
            by_method[method]["losses"] += 1
        by_method[method]["returns"].append(float(r.get("price_change_pct", 0)))

        by_tf.setdefault(tf, {"total": 0, "wins": 0, "losses": 0})
        by_tf[tf]["total"] += 1
        if outcome == "WIN":
            by_tf[tf]["wins"] += 1
        elif outcome == "LOSS":
            by_tf[tf]["losses"] += 1

        by_symbol.setdefault(sym, {"total": 0, "wins": 0})
        by_symbol[sym]["total"] += 1
        if outcome == "WIN":
            by_symbol[sym]["wins"] += 1

        if outcome == "LOSS":
            losses.append(r)

    # Compute win rates and avg returns
    for m in by_method.values():
        m["win_rate"] = round(m["wins"] / m["total"] * 100, 1) if m["total"] > 0 else 0
        m["avg_return"] = round(sum(m["returns"]) / len(m["returns"]), 2) if m["returns"] else 0

    return {
        "total": len(resolved),
        "wins": sum(m["wins"] for m in by_method.values()),
        "losses": sum(m["losses"] for m in by_method.values()),
        "by_methodology": by_method,
        "by_timeframe": by_tf,
        "by_symbol": by_symbol,
        "loss_details": losses[:50],  # cap to avoid huge prompts
    }


def collect_trade_data(days=LOOKBACK_DAYS_TRADES):
    """Fetch bot trade_history for Donchian loop analysis."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = db.select(
        "trade_history",
        gte_filters={"created_at": cutoff},
        order="created_at.desc",
        limit=500,
    )
    rows = result.get("data") or []
    if not rows:
        return None

    wins = [r for r in rows if r.get("action") in ("SELL", "FUTURES_SELL") and float(r.get("pnl_pct", 0)) > 0]
    losses = [r for r in rows if r.get("action") in ("SELL", "FUTURES_SELL") and float(r.get("pnl_pct", 0)) <= 0]

    by_coin = {}
    for r in rows:
        coin = r.get("symbol", "unknown")
        by_coin.setdefault(coin, {"total": 0, "pnl_sum": 0})
        by_coin[coin]["total"] += 1
        by_coin[coin]["pnl_sum"] += float(r.get("pnl_pct", 0))

    return {
        "total": len(rows),
        "closed_trades": len(wins) + len(losses),
        "wins": len(wins),
        "losses": len(losses),
        "avg_return": round(sum(float(r.get("pnl_pct", 0)) for r in rows) / max(1, len(rows)), 2),
        "by_coin": by_coin,
        "recent_trades": rows[:20],
    }


def get_current_config(target):
    """Load current config for a target (bot_config or signal_config_*)."""
    if target == "donchian_config":
        result = db.select("bot_config", limit=1)
        rows = result.get("data") or []
        return rows[0] if rows else {}
    elif target.startswith("signal_config_"):
        methodology = target.replace("signal_config_", "")
        result = db.select("signal_config", filters={"methodology": methodology}, limit=1)
        rows = result.get("data") or []
        return rows[0].get("config", {}) if rows else {}
    return {}


# ── AI Analysis ──

def build_signal_analysis_prompt(data, config, methodology):
    """Build Claude prompt for signal methodology tuning."""
    method_display = {
        "smc": "Smart Money Concepts",
        "elliott": "Elliott Wave",
        "priceaction": "Price Action",
        "technical": "Technical Analysis",
    }.get(methodology, methodology)

    method_data = data["by_methodology"].get(methodology, {})
    losses = [l for l in data.get("loss_details", []) if l.get("methodology") == methodology]

    return f"""You are an expert crypto trading signal optimizer. Analyze the performance data for the {method_display} signal methodology and propose parameter improvements.

## Current Configuration
```json
{json.dumps(config, indent=2)}
```

## Performance Data (Last {LOOKBACK_DAYS_SIGNALS} Days)
- Total signals: {method_data.get('total', 0)}
- Wins: {method_data.get('wins', 0)}, Losses: {method_data.get('losses', 0)}
- Win rate: {method_data.get('win_rate', 0)}%
- Avg return: {method_data.get('avg_return', 0)}%

## Recent Loss Details
{json.dumps(losses[:15], indent=2, default=str)}

## Root Cause Analysis Guidelines
- Are stops too tight? (Low stop_loss distances = getting stopped out prematurely)
- Are entries firing too late? (High confidence thresholds = missing early moves)
- Are TPs too conservative? (Small take_profit distances = leaving money on table)
- Is the signal firing too often on noise? (Low confidence gates = false signals)

## Constraints
- Make conservative changes: 10-30% adjustments per parameter
- Optimize for PROFITABILITY (avg return), not just win rate
- Only change parameters that address the identified root causes
- Do NOT change parameters that are performing well

## Output Format
Return a JSON object with:
```json
{{
  "proposed_changes": {{"param_name": new_value, ...}},
  "rationale": "Brief explanation of each change",
  "expected_impact": "What improvement we expect"
}}
```
If current params look optimal, return: {{"proposed_changes": {{}}, "rationale": "Current parameters are performing well", "expected_impact": "None needed"}}

Return ONLY the JSON, no other text."""


def build_donchian_analysis_prompt(data, config):
    """Build Claude prompt for Donchian bot tuning."""
    return f"""You are an expert crypto trading bot optimizer. Analyze the Donchian breakout bot's trade history and propose parameter improvements.

## Current Bot Configuration (key parameters)
```json
{{
  "atr_mult": {config.get('atr_mult', 4.0)},
  "volume_mult": {config.get('volume_mult', 1.5)},
  "pyramid_gain_pct": {config.get('pyramid_gain_pct', 15.0)},
  "risk_per_trade_pct": {config.get('risk_per_trade_pct', 2.0)},
  "emergency_stop_pct": {config.get('emergency_stop_pct', 15.0)},
  "tp1_gain_pct": {config.get('tp1_gain_pct', 10.0)},
  "tp2_gain_pct": {config.get('tp2_gain_pct', 20.0)},
  "short_atr_mult": {config.get('short_atr_mult', 2.0)},
  "short_volume_mult": {config.get('short_volume_mult', 2.0)}
}}
```

## Trade Performance (Last {LOOKBACK_DAYS_TRADES} Days)
- Total trades: {data.get('total', 0)}
- Closed: {data.get('closed_trades', 0)} (W: {data.get('wins', 0)}, L: {data.get('losses', 0)})
- Avg return: {data.get('avg_return', 0)}%

## Recent Trades
{json.dumps(data.get('recent_trades', [])[:10], indent=2, default=str)}

## Constraints
- Make conservative changes: 10-30% adjustments per parameter
- Optimize for overall portfolio return, not just win rate
- Only change parameters that address actual performance issues
- Do NOT change coins list, leverage, or trading mode

## Output Format
Return ONLY a JSON object:
```json
{{
  "proposed_changes": {{"param_name": new_value, ...}},
  "rationale": "Brief explanation",
  "expected_impact": "What improvement we expect"
}}
```
If current params look optimal: {{"proposed_changes": {{}}, "rationale": "Current parameters are performing well", "expected_impact": "None needed"}}"""


def run_ai_analysis(prompt):
    """Call Claude API, parse JSON response."""
    if not CLAUDE_API_KEY:
        logger.error("CLAUDE_API_KEY not set")
        return None

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": AI_MODEL,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )

        if resp.status_code != 200:
            logger.error(f"Claude API error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        text = data["content"][0]["text"].strip()
        usage = data.get("usage", {})

        # Parse JSON from response (may be wrapped in ```json)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        result["_token_usage"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return None


# ── Validation ──

SIGNAL_CONFIG_RANGES = {
    "smc": {
        "sl_atr_mult": (0.5, 5.0),
        "sl_fallback_pct": (1.0, 10.0),
        "ob_proximity_pct": (1.0, 10.0),
        "ob_impulse_atr_mult": (1.0, 5.0),
        "confidence_gate": (30, 80),
        "ob_score": (10, 50),
        "bos_score": (10, 50),
        "fvg_score": (5, 30),
        "choch_score": (5, 30),
        "conviction_confidence": (40, 90),
        "conviction_rr": (1.0, 5.0),
    },
    "elliott": {
        "confidence_cap": (40, 90),
        "conviction_rr": (1.0, 5.0),
        "conviction_min_stop_pct": (0.5, 5.0),
        "confidence_wave_24_valid": (30, 80),
        "confidence_wave_24_invalid": (15, 50),
        "confidence_wave_5plus_valid": (25, 75),
        "confidence_wave_5plus_invalid": (10, 45),
        "confidence_wave_13_valid": (20, 70),
        "confidence_wave_13_invalid": (10, 40),
    },
    "priceaction": {
        "sl_atr_mult": (0.5, 5.0),
        "distance_gate_pct": (1.0, 10.0),
        "sr_cluster_pct": (0.1, 2.0),
        "volume_spike_mult": (1.0, 5.0),
        "fallback_tp_pct": (2.0, 15.0),
        "rr_gate": (1.0, 5.0),
        "confidence_base": (20, 60),
        "confidence_cap": (50, 100),
        "conviction_confidence": (30, 90),
        "conviction_rr": (1.0, 5.0),
    },
    "technical": {
        "signal_threshold": (20, 70),
        "conviction_confidence": (40, 90),
        "rsi_bull_threshold": (20, 50),
        "rsi_bear_threshold": (50, 80),
        "adx_trend_threshold": (15, 40),
    },
}

DONCHIAN_CONFIG_RANGES = {
    "atr_mult": (1.0, 8.0),
    "volume_mult": (1.0, 3.0),
    "pyramid_gain_pct": (5.0, 30.0),
    "risk_per_trade_pct": (0.5, 5.0),
    "emergency_stop_pct": (5.0, 25.0),
    "tp1_gain_pct": (3.0, 20.0),
    "tp2_gain_pct": (10.0, 40.0),
    "short_atr_mult": (1.0, 5.0),
    "short_volume_mult": (1.0, 4.0),
}


def validate_proposed_changes(changes, target):
    """Validate each proposed change against allowed ranges."""
    if not changes:
        return changes, []

    if target == "donchian_config":
        ranges = DONCHIAN_CONFIG_RANGES
    elif target.startswith("signal_config_"):
        methodology = target.replace("signal_config_", "")
        ranges = SIGNAL_CONFIG_RANGES.get(methodology, {})
    else:
        return changes, [f"Unknown target: {target}"]

    validated = {}
    warnings = []

    for param, value in changes.items():
        if param in ranges:
            lo, hi = ranges[param]
            if isinstance(value, (int, float)):
                if lo <= value <= hi:
                    validated[param] = value
                else:
                    warnings.append(f"{param}={value} out of range [{lo}, {hi}], clamped")
                    validated[param] = max(lo, min(hi, value))
            else:
                validated[param] = value  # non-numeric (e.g., dicts)
        else:
            validated[param] = value  # not range-checked

    return validated, warnings


# ── Backtest Validation (Donchian loop) ──

def run_donchian_backtest(current_config, proposed_changes):
    """Run baseline vs proposed backtest using backtest_engine."""
    try:
        # Try to import from tradesavvy backend
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend'))
        from backtest_engine import run_backtest

        # Baseline
        baseline_params = {
            "mode": "combined",
            "atr_mult": current_config.get("atr_mult", 4.0),
            "volume_mult": current_config.get("volume_mult", 1.5),
            "pyramid_trigger": current_config.get("pyramid_gain_pct", 15.0),
            "risk_per_trade": current_config.get("risk_per_trade_pct", 2.0),
            "period": "full",
        }
        baseline = run_backtest(baseline_params)

        # Proposed
        proposed_params = {**baseline_params}
        for k, v in proposed_changes.items():
            if k in proposed_params:
                proposed_params[k] = v
            elif k == "pyramid_gain_pct":
                proposed_params["pyramid_trigger"] = v
            elif k == "risk_per_trade_pct":
                proposed_params["risk_per_trade"] = v

        proposed = run_backtest(proposed_params)

        return {
            "baseline": {
                "total_return": baseline.get("total_return"),
                "win_rate": baseline.get("win_rate"),
                "profit_factor": baseline.get("profit_factor"),
                "max_drawdown": baseline.get("max_drawdown"),
                "total_trades": baseline.get("total_trades"),
            },
            "proposed": {
                "total_return": proposed.get("total_return"),
                "win_rate": proposed.get("win_rate"),
                "profit_factor": proposed.get("profit_factor"),
                "max_drawdown": proposed.get("max_drawdown"),
                "total_trades": proposed.get("total_trades"),
            },
        }
    except Exception as e:
        logger.error(f"Backtest validation failed: {e}")
        return None


# ── Signal Replay Validation ──

def run_signal_replay(methodology, current_config, proposed_changes):
    """Run signal replay for methodology validation."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend'))
        from signal_replay import replay_signals

        proposed_config = {**current_config, **proposed_changes}
        result = replay_signals(methodology, current_config, proposed_config)
        return result
    except Exception as e:
        logger.error(f"Signal replay failed for {methodology}: {e}")
        return None


# ── HTML Report Generation ──

def generate_html_report(run_id, signal_data, trade_data, proposals):
    """Generate a dark-theme HTML report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    proposals_html = ""
    for p in proposals:
        status_color = {"pending": "#f59e0b", "approved": "#10b981", "rejected": "#ef4444"}.get(p["status"], "#6b7280")
        proposals_html += f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;margin:12px 0">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <h3 style="margin:0;color:#f1f5f9">{p['target']}</h3>
                <span style="background:{status_color};color:#fff;padding:4px 12px;border-radius:4px;font-size:12px">{p['status'].upper()}</span>
            </div>
            <p style="color:#94a3b8;margin:8px 0">{p.get('change_rationale', '')}</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px">
                <div>
                    <h4 style="color:#64748b;margin:0 0 4px">Baseline</h4>
                    <pre style="background:#0f172a;padding:8px;border-radius:4px;color:#e2e8f0;font-size:12px;overflow-x:auto">{json.dumps(p.get('baseline_metrics', {}), indent=2)}</pre>
                </div>
                <div>
                    <h4 style="color:#64748b;margin:0 0 4px">Proposed</h4>
                    <pre style="background:#0f172a;padding:8px;border-radius:4px;color:#e2e8f0;font-size:12px;overflow-x:auto">{json.dumps(p.get('proposed_metrics', {}), indent=2)}</pre>
                </div>
            </div>
            <div style="margin-top:12px">
                <h4 style="color:#64748b;margin:0 0 4px">Proposed Changes</h4>
                <pre style="background:#0f172a;padding:8px;border-radius:4px;color:#a5f3fc;font-size:12px;overflow-x:auto">{json.dumps(p.get('proposed_changes', {}), indent=2)}</pre>
            </div>
        </div>"""

    signal_summary = ""
    if signal_data:
        for method, stats in signal_data.get("by_methodology", {}).items():
            signal_summary += f"<li><b>{method}</b>: {stats['total']} signals, {stats['win_rate']}% WR, {stats['avg_return']}% avg return</li>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Optimization Report — {run_id}</title>
<style>body{{background:#0f172a;color:#e2e8f0;font-family:system-ui;max-width:900px;margin:0 auto;padding:24px}}
h1{{color:#f1f5f9}}h2{{color:#94a3b8;border-bottom:1px solid #334155;padding-bottom:8px}}
a{{color:#38bdf8}}</style></head>
<body>
<h1>Optimization Report</h1>
<p style="color:#64748b">{now} | Run: {run_id}</p>

<h2>Signal Performance Summary</h2>
<ul>{signal_summary}</ul>

<h2>Proposals ({len(proposals)})</h2>
{proposals_html if proposals else '<p style="color:#64748b">No proposals generated — current parameters look optimal.</p>'}
</body></html>"""


# ── Telegram Notification ──

def send_telegram(message):
    """Send notification to admin via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ── Main Orchestration ──

def run_optimization_cycle(trigger="manual"):
    """Execute one full optimization cycle (both loops)."""
    run_id = f"opt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting optimization cycle {run_id} (trigger={trigger})")

    # Get admin user_id (first row in bot_config)
    user_result = db.select("bot_config", columns="user_id", limit=1)
    user_id = (user_result.get("data") or [{}])[0].get("user_id", "admin")

    # Create run record
    db.insert("optimization_runs", {
        "id": run_id,
        "user_id": user_id,
        "status": "running",
        "trigger": trigger,
        "loop_type": "both",
    })

    proposals = []
    total_tokens = 0

    try:
        # ── Loop 1: Donchian Bot Tuning ──
        logger.info("Loop 1: Donchian bot tuning...")
        trade_data = collect_trade_data()

        if trade_data and trade_data.get("closed_trades", 0) >= 5:
            current_config = get_current_config("donchian_config")
            prompt = build_donchian_analysis_prompt(trade_data, current_config)
            ai_result = run_ai_analysis(prompt)

            if ai_result and ai_result.get("proposed_changes"):
                total_tokens += ai_result.get("_token_usage", 0)
                changes = ai_result["proposed_changes"]
                validated, warnings = validate_proposed_changes(changes, "donchian_config")

                if validated:
                    # Validate via backtest
                    backtest_result = run_donchian_backtest(current_config, validated)
                    baseline_metrics = backtest_result["baseline"] if backtest_result else {}
                    proposed_metrics = backtest_result["proposed"] if backtest_result else {}

                    proposal = {
                        "run_id": run_id,
                        "user_id": user_id,
                        "status": "pending",
                        "target": "donchian_config",
                        "proposed_changes": validated,
                        "change_rationale": ai_result.get("rationale", ""),
                        "baseline_metrics": baseline_metrics,
                        "proposed_metrics": proposed_metrics,
                        "improvement_summary": ai_result.get("expected_impact", ""),
                        "current_config": current_config,
                    }
                    db.insert("optimization_proposals", proposal)
                    proposals.append(proposal)
                    logger.info(f"Donchian proposal created: {list(validated.keys())}")
            elif ai_result:
                total_tokens += ai_result.get("_token_usage", 0)
                logger.info("Donchian: No changes proposed (current params optimal)")
        else:
            logger.info("Loop 1: Not enough trade data, skipping")

        # ── Loop 2: Signal Methodology Tuning ──
        logger.info("Loop 2: Signal methodology tuning...")
        signal_data = collect_signal_data()

        if signal_data:
            for methodology in ["smc", "elliott", "priceaction", "technical"]:
                method_stats = signal_data["by_methodology"].get(methodology)
                if not method_stats or method_stats.get("total", 0) < 5:
                    logger.info(f"Signal {methodology}: Not enough data, skipping")
                    continue

                current_config = get_current_config(f"signal_config_{methodology}")
                if not current_config:
                    # Use defaults from signal_config module
                    from signals.signal_config import DEFAULTS
                    current_config = DEFAULTS.get(methodology, {})

                prompt = build_signal_analysis_prompt(signal_data, current_config, methodology)
                ai_result = run_ai_analysis(prompt)

                if ai_result and ai_result.get("proposed_changes"):
                    total_tokens += ai_result.get("_token_usage", 0)
                    changes = ai_result["proposed_changes"]
                    target = f"signal_config_{methodology}"
                    validated, warnings = validate_proposed_changes(changes, target)

                    if validated:
                        # Validate via signal replay (only SMC and PA have replay support)
                        replay_result = run_signal_replay(methodology, current_config, validated)
                        baseline_metrics = replay_result.get("baseline", {}) if replay_result else {}
                        proposed_metrics = replay_result.get("proposed", {}) if replay_result else {}

                        proposal = {
                            "run_id": run_id,
                            "user_id": user_id,
                            "status": "pending",
                            "target": target,
                            "proposed_changes": validated,
                            "change_rationale": ai_result.get("rationale", ""),
                            "baseline_metrics": baseline_metrics,
                            "proposed_metrics": proposed_metrics,
                            "improvement_summary": ai_result.get("expected_impact", ""),
                            "current_config": current_config,
                        }
                        db.insert("optimization_proposals", proposal)
                        proposals.append(proposal)
                        logger.info(f"Signal {methodology} proposal created: {list(validated.keys())}")
                elif ai_result:
                    total_tokens += ai_result.get("_token_usage", 0)
                    logger.info(f"Signal {methodology}: No changes proposed")
        else:
            logger.info("Loop 2: Not enough signal data, skipping")

        # ── Generate Report ──
        html_report = generate_html_report(run_id, signal_data, trade_data, proposals)

        status = "completed" if proposals else "no_action"
        cost = round(total_tokens * 0.000003, 4)  # Sonnet pricing estimate

        db.update("optimization_runs", {
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "signal_counts": signal_data if signal_data else {},
            "trade_counts": trade_data if trade_data else {},
            "ai_model": AI_MODEL,
            "token_usage": total_tokens,
            "cost_usd": cost,
            "html_report": html_report,
            "proposals_generated": len(proposals),
        }, filters={"id": run_id})

        # Telegram notification
        if proposals:
            proposal_summary = "\n".join([f"  • {p['target']}: {len(p['proposed_changes'])} changes" for p in proposals])
            send_telegram(
                f"🔧 <b>Optimizer Report</b>\n"
                f"Run: {run_id}\n"
                f"Proposals: {len(proposals)}\n"
                f"{proposal_summary}\n\n"
                f"Review at tradesavvy.io/optimizer"
            )
        else:
            send_telegram(f"✅ <b>Optimizer Complete</b>\nRun: {run_id}\nNo changes needed — current params optimal.")

        logger.info(f"Optimization cycle {run_id} complete: {len(proposals)} proposals, {total_tokens} tokens, ${cost}")

    except Exception as e:
        logger.error(f"Optimization cycle {run_id} failed: {e}", exc_info=True)
        db.update("optimization_runs", {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": str(e),
        }, filters={"id": run_id})
        send_telegram(f"❌ <b>Optimizer Failed</b>\nRun: {run_id}\nError: {str(e)[:200]}")

    return run_id


# ── Scheduler ──

def should_run_weekly():
    """Check if it's time for the weekly scheduled run (Monday 04:00 UTC)."""
    now = datetime.now(timezone.utc)
    return now.weekday() == WEEKLY_SCHEDULE_DAY and now.hour == WEEKLY_SCHEDULE_UTC_HOUR and now.minute < 10


def check_manual_trigger():
    """Check Supabase for a queued manual trigger."""
    try:
        result = db.select("optimization_runs", filters={"status": "queued"}, limit=1)
        rows = result.get("data") or []
        if rows:
            return rows[0]["id"]
    except Exception:
        pass
    return None


def main():
    """Main loop: weekly schedule + poll for manual triggers."""
    os.makedirs("logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run immediately")
    args = parser.parse_args()

    if not db:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Exiting.")
        sys.exit(1)

    if args.now:
        logger.info("Manual run triggered via --now flag")
        run_optimization_cycle(trigger="manual")
        return

    logger.info("Optimization agent started. Waiting for schedule or manual trigger...")
    last_weekly_run = None

    while True:
        try:
            # Check weekly schedule
            if should_run_weekly():
                today = datetime.now(timezone.utc).date()
                if last_weekly_run != today:
                    run_optimization_cycle(trigger="scheduled")
                    last_weekly_run = today

            # Check manual trigger
            queued_id = check_manual_trigger()
            if queued_id:
                db.update("optimization_runs", {"status": "running"}, filters={"id": queued_id})
                run_optimization_cycle(trigger="manual")

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
