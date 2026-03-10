"""
AI Optimization Agent — Donchian-Focused System
Loop 1: Donchian bot config tuning (validated via backtest_engine, full + OOS)
Loop 2: Signal methodology tuning (DISABLED — Donchian is sole trading bot)

Tunable parameters: atr_mult, volume_mult, pyramid_gain_pct, risk_per_trade_pct,
    emergency_stop_pct, tp1/tp2 gains, short_atr_mult, short_volume_mult,
    rsi_blowoff, volume_blowoff, atr_mult_tight, short_max_hold_days,
    pyramid_max_cash, short_rsi_blowoff, short_atr_mult_tight

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
        url = f"{self.rest_url}/{table}"
        params = {"select": columns}
        if filters:
            for k, v in filters.items():
                params[k] = f"eq.{v}"
        if gte_filters:
            for k, v in gte_filters.items():
                params[k] = f"gte.{v}"
        if lte_filters:
            for k, v in lte_filters.items():
                params[k] = f"lte.{v}"
        if order:
            params["order"] = order
        if limit:
            params["limit"] = str(limit)
        resp = requests.get(url, headers=self.headers, params=params, timeout=15)
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
        cfg_ver = r.get("config_version")

        by_method.setdefault(method, {"total": 0, "wins": 0, "losses": 0, "returns": [], "by_version": {}})
        by_method[method]["total"] += 1
        if outcome == "WIN":
            by_method[method]["wins"] += 1
        elif outcome == "LOSS":
            by_method[method]["losses"] += 1
        pct = r.get("price_change_pct")
        pct = float(pct) if pct is not None else 0.0
        by_method[method]["returns"].append(pct)

        # Track per-version stats
        if cfg_ver is not None:
            ver_key = str(cfg_ver)
            bv = by_method[method]["by_version"]
            bv.setdefault(ver_key, {"total": 0, "wins": 0, "losses": 0, "returns": []})
            bv[ver_key]["total"] += 1
            if outcome == "WIN":
                bv[ver_key]["wins"] += 1
            elif outcome == "LOSS":
                bv[ver_key]["losses"] += 1
            bv[ver_key]["returns"].append(pct)

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
        for v in m.get("by_version", {}).values():
            v["win_rate"] = round(v["wins"] / v["total"] * 100, 1) if v["total"] > 0 else 0
            v["avg_return"] = round(sum(v["returns"]) / len(v["returns"]), 2) if v["returns"] else 0

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

    def _pnl(r):
        v = r.get("pnl_pct")
        return float(v) if v is not None else 0.0

    wins = [r for r in rows if r.get("action") in ("SELL", "FUTURES_SELL") and _pnl(r) > 0]
    losses = [r for r in rows if r.get("action") in ("SELL", "FUTURES_SELL") and _pnl(r) <= 0]

    by_coin = {}
    for r in rows:
        coin = r.get("symbol", "unknown")
        by_coin.setdefault(coin, {"total": 0, "pnl_sum": 0})
        by_coin[coin]["total"] += 1
        by_coin[coin]["pnl_sum"] += _pnl(r)

    return {
        "total": len(rows),
        "closed_trades": len(wins) + len(losses),
        "wins": len(wins),
        "losses": len(losses),
        "avg_return": round(sum(_pnl(r) for r in rows) / max(1, len(rows)), 2),
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


def get_rejected_proposals(target):
    """Fetch recently rejected proposals for a target (to avoid re-proposing)."""
    try:
        result = db.select(
            "optimization_proposals",
            columns="proposed_changes,change_rationale,rejection_reason,approved_at",
            filters={"target": target, "status": "rejected"},
            order="approved_at.desc",
            limit=5,
        )
        rows = result.get("data") or []
        return [{"changes": r.get("proposed_changes", {}),
                 "rationale": r.get("change_rationale", ""),
                 "rejection_reason": r.get("rejection_reason", ""),
                 "rejected_at": r.get("approved_at", "")} for r in rows]
    except Exception:
        return []


def get_config_history(methodology):
    """Fetch config change history for a methodology."""
    try:
        result = db.select(
            "signal_config_history",
            filters={"methodology": methodology},
            order="version.asc",
            limit=20,
        )
        rows = result.get("data") or []
        return [{"version": r["version"], "changes": r.get("changes", {}),
                 "reason": r.get("reason", ""), "source": r.get("source", ""),
                 "created_at": r.get("created_at", "")} for r in rows]
    except Exception:
        return []


# ── AI Analysis ──

def build_signal_analysis_prompt(data, config, methodology):
    """Build Claude prompt for signal methodology tuning."""
    method_display = {
        "smc": "Smart Money Concepts",
        "priceaction": "Price Action",
        "technical": "Technical Analysis",
    }.get(methodology, methodology)

    method_data = data["by_methodology"].get(methodology, {})
    losses = [l for l in data.get("loss_details", []) if l.get("methodology") == methodology]
    history = get_config_history(methodology)
    rejected = get_rejected_proposals(f"signal_config_{methodology}")

    # Build per-version performance breakdown
    by_version = method_data.get("by_version", {})
    version_lines = ""
    if by_version:
        version_lines = "\n## Performance by Config Version\n"
        for ver, stats in sorted(by_version.items(), key=lambda x: int(x[0])):
            version_lines += f"- Version {ver}: {stats['total']} signals, {stats['win_rate']}% WR, {stats['avg_return']}% avg return ({stats['wins']}W/{stats['losses']}L)\n"
        # Find current version
        current_ver = max(int(v) for v in by_version.keys()) if by_version else 1
        version_lines += f"\n**Analyze version {current_ver} data only for tuning decisions.** Prior versions are historical context showing impact of past changes.\n"

    # Build change history section
    history_lines = ""
    if history:
        history_lines = "\n## Config Change History\n"
        for h in history:
            history_lines += f"- v{h['version']} ({h['created_at'][:10]}): {h['reason']} — {json.dumps(h['changes'])}\n"

    # Build rejected proposals section
    rejected_lines = ""
    if rejected:
        rejected_lines = "\n## Previously Rejected Proposals\nThese changes were proposed and REJECTED by the admin. Do NOT re-propose the same changes unless you have significantly stronger evidence.\n"
        for r in rejected:
            rejected_lines += f"- Rejected ({r['rejected_at'][:10] if r['rejected_at'] else 'unknown'}): {json.dumps(r['changes'])}\n"
            if r.get("rejection_reason"):
                rejected_lines += f"  Reason: {r['rejection_reason']}\n"

    return f"""You are an expert crypto trading signal optimizer. Analyze the performance data for the {method_display} signal methodology and propose parameter improvements.

## Current Configuration
```json
{json.dumps(config, indent=2)}
```
{history_lines}{rejected_lines}
## Performance Data (Last {LOOKBACK_DAYS_SIGNALS} Days)
- Total signals: {method_data.get('total', 0)}
- Wins: {method_data.get('wins', 0)}, Losses: {method_data.get('losses', 0)}
- Win rate: {method_data.get('win_rate', 0)}%
- Avg return: {method_data.get('avg_return', 0)}%
{version_lines}
## Recent Loss Details (current version only)
{json.dumps([l for l in losses[:15] if not by_version or str(l.get('config_version', '')) == str(max(int(v) for v in by_version.keys()))] or losses[:15], indent=2, default=str)}

## Root Cause Analysis Guidelines
- Are stops too tight? (Low stop_loss distances = getting stopped out prematurely)
- Are entries firing too late? (High confidence thresholds = missing early moves)
- Are TPs too conservative? (Small take_profit distances = leaving money on table)
- Is the signal firing too often on noise? (Low confidence gates = false signals)
- Review change history carefully before proposing any changes
- If a recent version change improved results, acknowledge that and look for remaining issues

## Constraints
- Make conservative changes: 10-30% adjustments per parameter
- Optimize for PROFITABILITY (avg return), not just win rate
- Only change parameters that address the identified root causes
- Do NOT change parameters that are performing well
- CRITICAL: If your proposed change would move a parameter CLOSER to a value from a PRIOR version (i.e. partially or fully reverting a previous change), you MUST include "reversion_justification" in your JSON explaining why the previous change was wrong and what new evidence supports reverting it. Changes marked as "manual" source in the history were deliberate human decisions based on deep analysis — reverting these requires especially strong justification.

## Output Format
Return a JSON object with:
```json
{{
  "proposed_changes": {{"param_name": new_value, ...}},
  "rationale": "Brief explanation of each change",
  "expected_impact": "What improvement we expect",
  "reversion_justification": "REQUIRED if any change reverts a prior version's parameter. Omit if no reversions."
}}
```
If current params look optimal, return: {{"proposed_changes": {{}}, "rationale": "Current parameters are performing well", "expected_impact": "None needed"}}

Return ONLY the JSON, no other text."""


def build_donchian_analysis_prompt(data, config):
    """Build Claude prompt for Donchian bot tuning."""
    rejected = get_rejected_proposals("donchian_config")
    rejected_lines = ""
    if rejected:
        rejected_lines = "\n## Previously Rejected Proposals\nThese changes were proposed and REJECTED by the admin. Do NOT re-propose the same changes unless you have significantly stronger evidence.\n"
        for r in rejected:
            rejected_lines += f"- Rejected ({r['rejected_at'][:10] if r['rejected_at'] else 'unknown'}): {json.dumps(r['changes'])}\n"
            if r.get("rejection_reason"):
                rejected_lines += f"  Reason: {r['rejection_reason']}\n"

    return f"""You are an expert crypto trading bot optimizer. Analyze the Donchian breakout bot's trade history and propose parameter improvements.

## Current Bot Configuration (key parameters)
```json
{{
  "atr_mult": {config.get('atr_mult', 4.5)},
  "volume_mult": {config.get('volume_mult', 1.5)},
  "pyramid_gain_pct": {config.get('pyramid_gain_pct', 15.0)},
  "risk_per_trade_pct": {config.get('risk_per_trade_pct', 2.0)},
  "emergency_stop_pct": {config.get('emergency_stop_pct', 18.0)},
  "tp1_gain_pct": {config.get('tp1_gain_pct', 10.0)},
  "tp2_gain_pct": {config.get('tp2_gain_pct', 25.0)},
  "short_atr_mult": {config.get('short_atr_mult', 2.0)},
  "short_volume_mult": {config.get('short_volume_mult', 2.0)},
  "rsi_blowoff": {config.get('rsi_blowoff', 80)},
  "volume_blowoff": {config.get('volume_blowoff', 3.0)},
  "atr_mult_tight": {config.get('atr_mult_tight', 1.5)},
  "short_max_hold_days": {config.get('short_max_hold_days', 25)},
  "pyramid_max_cash": {config.get('pyramid_max_cash', 0.5)},
  "short_rsi_blowoff": {config.get('short_rsi_blowoff', 25)},
  "short_atr_mult_tight": {config.get('short_atr_mult_tight', 2.0)}
}}
```

{rejected_lines}
## Trade Performance
{"(BACKTEST-ONLY MODE — no live trades yet, optimize using backtested metrics)" if data.get("_backtest_only") else f"Last {LOOKBACK_DAYS_TRADES} days: {data.get('closed_trades', 0)} closed (W: {data.get('wins', 0)}, L: {data.get('losses', 0)}), Avg return: {data.get('avg_return', 0)}%"}

Note: The backtester will be run automatically on both baseline and proposed configs to validate changes. Both full (4-year) and out-of-sample (2025-2026) periods are tested. Focus on improving BOTH full-period AND OOS performance — avoid overfitting to the training set.

## Constraints
- Make conservative changes: 10-30% adjustments per parameter
- Optimize for overall portfolio return AND profit factor, not just win rate
- Only change parameters that address actual performance issues
- Do NOT change coins list, leverage, or trading mode
- Bull filter is OFF, bear filter is death_cross — do NOT change these
- The backtester uses realistic futures mechanics: intrabar SL (high/low), funding costs, liquidation checks, SL slippage
- New tunable parameters: rsi_blowoff (long blow-off RSI threshold), volume_blowoff (blow-off volume multiplier), atr_mult_tight (tightened stop during blow-off), short_max_hold_days, pyramid_max_cash (max fraction of cash for pyramid add), short_rsi_blowoff, short_atr_mult_tight
- Consider tuning blow-off detection (rsi_blowoff + volume_blowoff + atr_mult_tight) — these have never been optimized

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
    # Newly exposed (previously hardcoded)
    "rsi_blowoff": (65, 90),
    "volume_blowoff": (2.0, 5.0),
    "atr_mult_tight": (0.5, 3.0),
    "short_max_hold_days": (15, 60),
    "pyramid_max_cash": (0.2, 0.8),
    "short_rsi_blowoff": (10, 35),
    "short_atr_mult_tight": (0.5, 2.5),
}

UALGO_CONFIG_RANGES = {
    "supertrend_multiplier": (1.0, 5.0),
    "sl_atr_mult": (1.0, 4.0),
    "sl_min_pct": (0.5, 3.0),
    "flip_window": (2, 10),
    "rsi_period": (5, 21),
    "hma_open_period": (3, 12),
    "hma_close_period": (5, 21),
    "cmo_period": (7, 21),
    "rsi_buy_threshold": (40, 80),
    "cmo_buy_threshold": (-30, 10),
    "rsi_sell_threshold": (20, 60),
    "cmo_sell_threshold": (-10, 30),
    "tp1_fraction": (0.15, 0.40),
    "tp2_fraction": (0.15, 0.40),
    "max_hold_days": (10, 60),
    "risk_per_trade": (1.0, 5.0),
    "short_sl_atr_mult": (1.0, 4.0),
}


def validate_proposed_changes(changes, target):
    """Validate each proposed change against allowed ranges."""
    if not changes:
        return changes, []

    if target == "donchian_config":
        ranges = DONCHIAN_CONFIG_RANGES
    elif target == "ualgo_config":
        ranges = UALGO_CONFIG_RANGES
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
                warnings.append(f"{param}={value} non-scalar, skipped")
        else:
            if isinstance(value, (int, float)):
                validated[param] = value
            else:
                warnings.append(f"{param}={value} non-scalar, skipped")

    return validated, warnings


# ── Backtest Validation (Donchian loop) ──

def _is_regression(baseline_metrics, proposed_metrics, target):
    """Check if proposed metrics are worse than baseline. Returns (is_worse, reason)."""
    if not baseline_metrics or not proposed_metrics:
        return False, ""  # No metrics to compare — allow through for human review

    if target == "donchian_config":
        # Donchian: check total_return and profit_factor
        b_ret = baseline_metrics.get("total_return", 0) or 0
        p_ret = proposed_metrics.get("total_return", 0) or 0
        b_pf = baseline_metrics.get("profit_factor", 0) or 0
        p_pf = proposed_metrics.get("profit_factor", 0) or 0
        if b_ret == p_ret and b_pf == p_pf:
            return True, f"Identical metrics (backtest params likely not applied)"
        if p_ret < b_ret and p_pf < b_pf:
            return True, f"Return {p_ret:.1f}% < {b_ret:.1f}% AND PF {p_pf:.2f} < {b_pf:.2f}"
    else:
        # Signal methods: check win_rate and avg_return
        b_wr = baseline_metrics.get("win_rate", 0) or 0
        p_wr = proposed_metrics.get("win_rate", 0) or 0
        b_ret = baseline_metrics.get("avg_return", 0) or 0
        p_ret = proposed_metrics.get("avg_return", 0) or 0
        if p_wr < b_wr and p_ret < b_ret:
            return True, f"WR {p_wr:.1f}% < {b_wr:.1f}% AND avg return {p_ret:.2f}% < {b_ret:.2f}%"

    return False, ""


def run_donchian_backtest(current_config, proposed_changes):
    """Run baseline vs proposed backtest using backtest_engine."""
    try:
        # Try to import from tradesavvy backend
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend'))
        from backtest_engine import run_backtest

        # Map optimizer param names → backtest engine param names
        PARAM_MAP = {
            "atr_mult": "atr_mult",
            "volume_mult": "volume_mult",
            "pyramid_gain_pct": "pyramid_trigger",
            "risk_per_trade_pct": "risk_per_trade",
            "short_atr_mult": "short_atr_mult",
            "short_volume_mult": "short_volume_mult",
            "tp1_gain_pct": "tp1_gain_pct",
            "tp2_gain_pct": "tp2_gain_pct",
            "emergency_stop_pct": "emergency_stop_pct",
            "rsi_blowoff": "rsi_blowoff",
            "volume_blowoff": "volume_blowoff",
            "atr_mult_tight": "atr_mult_tight",
            "short_max_hold_days": "short_max_hold_days",
            "pyramid_max_cash": "pyramid_max_cash",
            "short_rsi_blowoff": "short_rsi_blowoff",
            "short_atr_mult_tight": "short_atr_mult_tight",
        }

        # Baseline — match production config (bull filter OFF, death cross ON)
        baseline_params = {
            "mode": "futures-combined", "period": "full",
            "bull_filter": "off", "bear_filter": "death_cross",
            "long_leverage": 1.0, "short_leverage": 1.0,
        }
        for opt_key, bt_key in PARAM_MAP.items():
            if opt_key in current_config:
                baseline_params[bt_key] = current_config[opt_key]

        baseline = run_backtest(baseline_params)

        # Also run OOS test period for regression check
        oos_params = {**baseline_params, "period": "test"}
        baseline_oos = run_backtest(oos_params)

        # Proposed
        proposed_params = {**baseline_params}
        for k, v in proposed_changes.items():
            bt_key = PARAM_MAP.get(k, k)
            proposed_params[bt_key] = v

        proposed = run_backtest(proposed_params)

        # OOS for proposed
        proposed_oos_params = {**proposed_params, "period": "test"}
        proposed_oos = run_backtest(proposed_oos_params)

        return {
            "baseline": {
                "total_return": baseline.get("total_return"),
                "win_rate": baseline.get("win_rate"),
                "profit_factor": baseline.get("profit_factor"),
                "max_drawdown": baseline.get("max_drawdown"),
                "total_trades": baseline.get("total_trades"),
                "oos_return": baseline_oos.get("total_return"),
                "oos_pf": baseline_oos.get("profit_factor"),
                "oos_dd": baseline_oos.get("max_drawdown"),
            },
            "proposed": {
                "total_return": proposed.get("total_return"),
                "win_rate": proposed.get("win_rate"),
                "profit_factor": proposed.get("profit_factor"),
                "max_drawdown": proposed.get("max_drawdown"),
                "total_trades": proposed.get("total_trades"),
                "oos_return": proposed_oos.get("total_return"),
                "oos_pf": proposed_oos.get("profit_factor"),
                "oos_dd": proposed_oos.get("max_drawdown"),
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

def run_optimization_cycle(trigger="manual", existing_run_id=None):
    """Execute one full optimization cycle (both loops)."""
    # Ensure tradesavvy backend is on sys.path for signal imports
    tradesavvy_backend = os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend')
    if tradesavvy_backend not in sys.path:
        sys.path.insert(0, tradesavvy_backend)

    # Get admin user_id from env or look up by email
    user_id = os.getenv("OPTIMIZER_USER_ID", "")
    if not user_id:
        profile_result = db.select("profiles", columns="id", filters={"email": "jasonnichol@gmail.com"}, limit=1)
        user_id = (profile_result.get("data") or [{}])[0].get("id", "admin")

    # Reuse existing queued run or create new one
    if existing_run_id:
        run_id = existing_run_id
        db.update("optimization_runs", {"status": "running"}, filters={"id": run_id})
    else:
        run_id = f"opt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        db.insert("optimization_runs", {
            "id": run_id,
            "user_id": user_id,
            "status": "running",
            "trigger": trigger,
            "loop_type": "both",
        })

    logger.info(f"Starting optimization cycle {run_id} (trigger={trigger})")
    proposals = []
    total_tokens = 0

    try:
        # ── Loop 1: Donchian Bot Tuning ──
        logger.info("Loop 1: Donchian bot tuning...")
        trade_data = collect_trade_data()

        # Run even without live trades — use backtest metrics for optimization
        has_live_trades = trade_data and trade_data.get("closed_trades", 0) >= 5
        if not has_live_trades:
            logger.info("No live trade data — using backtest-only optimization mode")
            trade_data = trade_data or {}
            trade_data["_backtest_only"] = True

        if True:  # Always run Donchian loop (backtest-only if no live trades)
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

                    # Regression gate — auto-reject if proposed is worse
                    is_regression, reason = _is_regression(baseline_metrics, proposed_metrics, "donchian_config")
                    if is_regression:
                        logger.info(f"Donchian proposal auto-rejected (regression): {reason}")
                    else:
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

        # ── Loop 2: Signal Methodology Tuning — DISABLED (Mar 9, 2026) ──
        # Signal methods (SMC, PriceAction, Technical) are signal-only, not trading.
        # Donchian is the sole trading bot. All optimizer cycles focus on Donchian.
        # Re-enable if a signal method is promoted to bot status.
        logger.info("Loop 2: Signal methodology tuning DISABLED (Donchian-only focus)")
        signal_data = collect_signal_data()  # still collect for report context

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


def run_intensive_optimization(max_rounds=10):
    """Run multiple optimization rounds using backtests only (no live data needed).

    Each round:
    1. AI proposes param changes based on current config + history
    2. Backtest validates on full (4yr) + OOS (2025-2026)
    3. Auto-apply if BOTH full AND OOS improve (no regression)
    4. Feed results back for next round

    Stops when: no improvement found, AI says optimal, or max rounds hit.
    """
    tradesavvy_backend = os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend')
    if tradesavvy_backend not in sys.path:
        sys.path.insert(0, tradesavvy_backend)
    from backtest_engine import run_backtest

    logger.info(f"=== INTENSIVE OPTIMIZATION: up to {max_rounds} rounds ===")
    send_telegram(f"🔬 <b>Intensive Optimization Started</b>\nMax rounds: {max_rounds}\nMode: Backtest-only, auto-apply improvements")

    # Load current config as starting point
    current_config = get_current_config("donchian_config")
    if not current_config:
        current_config = {}

    # PARAM_MAP for translating optimizer names → backtest names
    PARAM_MAP = {
        "atr_mult": "atr_mult", "volume_mult": "volume_mult",
        "pyramid_gain_pct": "pyramid_trigger", "risk_per_trade_pct": "risk_per_trade",
        "short_atr_mult": "short_atr_mult", "short_volume_mult": "short_volume_mult",
        "tp1_gain_pct": "tp1_gain_pct", "tp2_gain_pct": "tp2_gain_pct",
        "emergency_stop_pct": "emergency_stop_pct",
        "rsi_blowoff": "rsi_blowoff", "volume_blowoff": "volume_blowoff",
        "atr_mult_tight": "atr_mult_tight", "short_max_hold_days": "short_max_hold_days",
        "pyramid_max_cash": "pyramid_max_cash", "short_rsi_blowoff": "short_rsi_blowoff",
        "short_atr_mult_tight": "short_atr_mult_tight",
    }

    def _build_bt_params(config_overrides=None):
        """Build backtest params from current config + overrides."""
        params = {
            "mode": "futures-combined", "bull_filter": "off", "bear_filter": "death_cross",
            "long_leverage": 1.0, "short_leverage": 1.0,
        }
        for opt_key, bt_key in PARAM_MAP.items():
            if opt_key in current_config:
                params[bt_key] = current_config[opt_key]
        if config_overrides:
            for k, v in config_overrides.items():
                bt_key = PARAM_MAP.get(k, k)
                params[bt_key] = v
        return params

    def _run_full_and_oos(config_overrides=None):
        """Run backtest on full + OOS periods. Returns (full_metrics, oos_metrics)."""
        base = _build_bt_params(config_overrides)
        full_res = run_backtest({**base, "period": "full"})
        oos_res = run_backtest({**base, "period": "test"})
        return (
            {k: full_res.get(k) for k in ["total_return", "profit_factor", "win_rate", "max_drawdown", "total_trades", "sharpe_ratio"]},
            {k: oos_res.get(k) for k in ["total_return", "profit_factor", "win_rate", "max_drawdown", "total_trades", "sharpe_ratio"]},
        )

    # Get baseline metrics
    baseline_full, baseline_oos = _run_full_and_oos()
    best_full = baseline_full
    best_oos = baseline_oos
    best_config = dict(current_config)
    round_history = []
    total_tokens = 0
    improvements = 0

    logger.info(f"Baseline: Full {baseline_full['total_return']:+.1f}% PF {baseline_full['profit_factor']:.2f} | "
                f"OOS {baseline_oos['total_return']:+.1f}% PF {baseline_oos['profit_factor']:.2f}")

    for round_num in range(1, max_rounds + 1):
        logger.info(f"\n--- Round {round_num}/{max_rounds} ---")

        # Build prompt with current best config + history of previous rounds
        history_text = ""
        if round_history:
            history_text = "\n## Previous Rounds This Session\n"
            for rh in round_history:
                status = "APPLIED" if rh["applied"] else "REJECTED"
                history_text += (f"- Round {rh['round']}: {json.dumps(rh['changes'])} → "
                                 f"Full {rh['proposed_full']['total_return']:+.1f}% PF {rh['proposed_full']['profit_factor']:.2f}, "
                                 f"OOS {rh['proposed_oos']['total_return']:+.1f}% PF {rh['proposed_oos']['profit_factor']:.2f} "
                                 f"[{status}: {rh.get('reason', '')}]\n")

        prompt = f"""You are an expert crypto trading bot optimizer running in INTENSIVE BACKTEST MODE.
Your goal: find the optimal Donchian breakout parameters by proposing changes tested against 4-year historical data.

## Current Best Configuration
```json
{json.dumps({k: best_config.get(k, DONCHIAN_CONFIG_RANGES.get(k, [None])) for k in DONCHIAN_CONFIG_RANGES}, indent=2)}
```

## Current Best Performance
- Full period (2022-2026): Return {best_full['total_return']:+.1f}%, PF {best_full['profit_factor']:.2f}, WR {best_full['win_rate']:.1f}%, DD {best_full['max_drawdown']:.1f}%, Sharpe {best_full.get('sharpe_ratio', 0):.2f}
- OOS (2025-2026): Return {best_oos['total_return']:+.1f}%, PF {best_oos['profit_factor']:.2f}, WR {best_oos['win_rate']:.1f}%, DD {best_oos['max_drawdown']:.1f}%

## Baseline (starting point this session)
- Full: Return {baseline_full['total_return']:+.1f}%, PF {baseline_full['profit_factor']:.2f}
- OOS: Return {baseline_oos['total_return']:+.1f}%, PF {baseline_oos['profit_factor']:.2f}

{history_text}

## Round {round_num}/{max_rounds} — Strategy
- This is round {round_num}. {"Start with the most impactful parameters (atr_mult, tp levels, volume_mult)." if round_num <= 3 else "Try fine-tuning blow-off detection, short-side params, or pyramid settings." if round_num <= 6 else "Focus on subtle refinements or combinations not yet tested."}
- Changes are AUTO-APPLIED if they improve BOTH full AND OOS metrics. No human review.
- Avoid overfitting: if full improves but OOS gets worse, the change will be rejected.
- Try 1-3 parameter changes per round. Larger jumps early, smaller refinements later.
- Bull filter is OFF, bear filter is death_cross — do NOT change these.

## Allowed Ranges (propose a SINGLE scalar value within each range, NOT a list/array)
{json.dumps(DONCHIAN_CONFIG_RANGES, indent=2)}

## Output Format
Return ONLY a JSON object. Each value in proposed_changes MUST be a single number (e.g., 3.5), NOT a list or range:
```json
{{
  "proposed_changes": {{"param_name": 3.5, "other_param": 25}},
  "rationale": "Brief explanation of what you're testing and why",
  "expected_impact": "What improvement we expect"
}}
```
If you believe current params are optimal: {{"proposed_changes": {{}}, "rationale": "Converged — no further improvements expected"}}"""

        # Get AI proposal
        ai_result = run_ai_analysis(prompt)
        if not ai_result:
            logger.error(f"Round {round_num}: AI analysis failed, stopping")
            break

        total_tokens += ai_result.get("_token_usage", 0)
        changes = ai_result.get("proposed_changes", {})
        rationale = ai_result.get("rationale", "")

        if not changes:
            logger.info(f"Round {round_num}: AI says optimal — stopping. Rationale: {rationale}")
            break

        # Validate ranges
        validated, warnings = validate_proposed_changes(changes, "donchian_config")
        if warnings:
            logger.info(f"Round {round_num} warnings: {warnings}")

        if not validated:
            logger.info(f"Round {round_num}: No valid changes after validation")
            round_history.append({"round": round_num, "changes": changes, "applied": False,
                                  "reason": "validation failed", "proposed_full": best_full, "proposed_oos": best_oos})
            continue

        logger.info(f"Round {round_num}: Testing {json.dumps(validated)} — {rationale}")

        # Run backtest with proposed changes
        proposed_full, proposed_oos = _run_full_and_oos(validated)

        # Acceptance: full must not regress on EITHER return or PF,
        # OOS must stay within tolerance, DD must not blow up
        full_ret_ok = proposed_full["total_return"] >= best_full["total_return"] - 1.0
        full_pf_ok = proposed_full["profit_factor"] >= best_full["profit_factor"] - 0.05
        full_better = full_ret_ok and full_pf_ok
        oos_not_worse = (proposed_oos["total_return"] >= best_oos["total_return"] - 1.0 and
                         proposed_oos["max_drawdown"] <= best_oos["max_drawdown"] + 2.0)
        dd_ok = proposed_full["max_drawdown"] <= best_full["max_drawdown"] + 3.0
        # At least one metric must actually improve (not just stay flat)
        has_improvement = (proposed_full["total_return"] > best_full["total_return"] + 0.5 or
                           proposed_full["profit_factor"] > best_full["profit_factor"] + 0.02 or
                           proposed_oos["total_return"] > best_oos["total_return"] + 0.5)

        applied = full_better and oos_not_worse and dd_ok and has_improvement

        if applied:
            improvements += 1
            # Apply changes to best_config
            for k, v in validated.items():
                best_config[k] = v
            best_full = proposed_full
            best_oos = proposed_oos
            reason = "improvement"
            logger.info(f"Round {round_num}: ✅ APPLIED — Full {proposed_full['total_return']:+.1f}% PF {proposed_full['profit_factor']:.2f} | "
                        f"OOS {proposed_oos['total_return']:+.1f}% PF {proposed_oos['profit_factor']:.2f}")
        else:
            reasons = []
            if not full_ret_ok:
                reasons.append(f"full return regressed ({proposed_full['total_return']:+.1f}% vs {best_full['total_return']:+.1f}%)")
            if not full_pf_ok:
                reasons.append(f"full PF regressed ({proposed_full['profit_factor']:.2f} vs {best_full['profit_factor']:.2f})")
            if not oos_not_worse:
                reasons.append(f"OOS regressed ({proposed_oos['total_return']:+.1f}% vs {best_oos['total_return']:+.1f}%)")
            if not dd_ok:
                reasons.append(f"DD too high ({proposed_full['max_drawdown']:.1f}% vs {best_full['max_drawdown']:.1f}%)")
            if not has_improvement:
                reasons.append("no meaningful improvement")
            reason = "; ".join(reasons) if reasons else "no criteria met"
            logger.info(f"Round {round_num}: ❌ REJECTED — {reason}")

        round_history.append({
            "round": round_num, "changes": validated, "applied": applied,
            "reason": reason, "rationale": rationale,
            "proposed_full": proposed_full, "proposed_oos": proposed_oos,
        })

    # === Final Report ===
    logger.info(f"\n{'='*60}")
    logger.info(f"INTENSIVE OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Rounds: {len(round_history)}, Improvements: {improvements}")
    logger.info(f"Baseline:  Full {baseline_full['total_return']:+.1f}% PF {baseline_full['profit_factor']:.2f} | "
                f"OOS {baseline_oos['total_return']:+.1f}% PF {baseline_oos['profit_factor']:.2f}")
    logger.info(f"Best:      Full {best_full['total_return']:+.1f}% PF {best_full['profit_factor']:.2f} | "
                f"OOS {best_oos['total_return']:+.1f}% PF {best_oos['profit_factor']:.2f}")
    logger.info(f"Best config: {json.dumps({k: best_config.get(k) for k in DONCHIAN_CONFIG_RANGES if k in best_config}, indent=2)}")
    logger.info(f"Tokens used: {total_tokens}, Est cost: ${total_tokens * 0.000003:.4f}")

    # Save results to Supabase
    try:
        user_id = os.getenv("OPTIMIZER_USER_ID", "")
        if not user_id:
            profile_result = db.select("profiles", columns="id", filters={"email": "jasonnichol@gmail.com"}, limit=1)
            user_id = (profile_result.get("data") or [{}])[0].get("id", "admin")

        run_id = f"intensive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        db.insert("optimization_runs", {
            "id": run_id,
            "user_id": user_id,
            "status": "completed",
            "trigger": "intensive",
            "loop_type": "donchian_intensive",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "ai_model": AI_MODEL,
            "token_usage": total_tokens,
            "cost_usd": round(total_tokens * 0.000003, 4),
            "proposals_generated": improvements,
        })

        # Create a proposal with the final best config for review
        if improvements > 0:
            final_changes = {k: best_config[k] for k in DONCHIAN_CONFIG_RANGES if k in best_config}
            db.insert("optimization_proposals", {
                "run_id": run_id,
                "user_id": user_id,
                "status": "pending",
                "target": "donchian_config",
                "proposed_changes": final_changes,
                "change_rationale": f"Intensive optimization: {improvements} improvements over {len(round_history)} rounds",
                "baseline_metrics": {**baseline_full, "oos_return": baseline_oos["total_return"], "oos_pf": baseline_oos["profit_factor"]},
                "proposed_metrics": {**best_full, "oos_return": best_oos["total_return"], "oos_pf": best_oos["profit_factor"]},
                "improvement_summary": f"Full: {baseline_full['total_return']:+.1f}% → {best_full['total_return']:+.1f}%, OOS: {baseline_oos['total_return']:+.1f}% → {best_oos['total_return']:+.1f}%",
                "current_config": current_config,
            })
    except Exception as e:
        logger.error(f"Failed to save intensive results: {e}")

    # Telegram report
    report = (
        f"🔬 <b>Intensive Optimization Complete</b>\n"
        f"Rounds: {len(round_history)} | Improvements: {improvements}\n\n"
        f"📊 <b>Baseline → Best:</b>\n"
        f"Full: {baseline_full['total_return']:+.1f}% → {best_full['total_return']:+.1f}% (PF {baseline_full['profit_factor']:.2f} → {best_full['profit_factor']:.2f})\n"
        f"OOS: {baseline_oos['total_return']:+.1f}% → {best_oos['total_return']:+.1f}% (PF {baseline_oos['profit_factor']:.2f} → {best_oos['profit_factor']:.2f})\n"
        f"DD: {baseline_full['max_drawdown']:.1f}% → {best_full['max_drawdown']:.1f}%\n\n"
    )
    if improvements > 0:
        report += f"<b>Best config saved as pending proposal in Supabase.</b>\nReview at tradesavvy.io/optimizer"
    else:
        report += "No improvements found — current params are already well-optimized."

    # Add round-by-round summary
    report += "\n\n<b>Round Details:</b>\n"
    for rh in round_history:
        icon = "✅" if rh["applied"] else "❌"
        report += f"{icon} R{rh['round']}: {json.dumps(rh['changes'])} → Full {rh['proposed_full']['total_return']:+.1f}%, OOS {rh['proposed_oos']['total_return']:+.1f}%\n"

    send_telegram(report)
    return best_config, round_history


def run_ualgo_filter_sweep():
    """Sweep all bull/bear filter combinations for UAlgo, then optimize the best combo.

    Tests 49 combinations (7 bull x 7 bear filters) at default UAlgo params.
    Reports full + OOS metrics for each, then kicks off intensive optimization
    on the winning combo.
    """
    tradesavvy_backend = os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend')
    if tradesavvy_backend not in sys.path:
        sys.path.insert(0, tradesavvy_backend)
    from backtest_ualgo_engine import run_backtest

    BULL_FILTERS = ['off', 'sma200', 'golden_cross', 'adx_dmi', 'supertrend', 'sar', 'adx_sma200']
    BEAR_FILTERS = ['off', 'death_cross', 'sma200', 'adx_dmi', 'supertrend', 'sar', 'adx_sma200']

    logger.info("=== UALGO FILTER SWEEP: 49 combinations ===")
    send_telegram("🔬 <b>UAlgo Filter Sweep Started</b>\n49 bull/bear filter combinations")

    results = []
    keys = ["total_return", "profit_factor", "win_rate", "max_drawdown", "total_trades", "sharpe_ratio"]

    for bi, bull in enumerate(BULL_FILTERS):
        for bei, bear in enumerate(BEAR_FILTERS):
            combo_num = bi * len(BEAR_FILTERS) + bei + 1
            logger.info(f"[{combo_num}/49] bull={bull}, bear={bear}")
            try:
                params = {"mode": "futures-combined", "period": "full",
                          "bull_filter": bull, "bear_filter": bear,
                          "long_leverage": 1.0, "short_leverage": 1.0}
                full_res = run_backtest({**params, "period": "full"})
                oos_res = run_backtest({**params, "period": "test"})
                full_m = {k: full_res.get(k) for k in keys}
                oos_m = {k: oos_res.get(k) for k in keys}
                results.append({
                    "bull": bull, "bear": bear,
                    "full": full_m, "oos": oos_m,
                })
                logger.info(f"  Full: {full_m['total_return']:+.1f}% PF {full_m['profit_factor']:.2f} WR {full_m['win_rate']:.1f}% DD {full_m['max_drawdown']:.1f}% ({full_m['total_trades']} trades)")
                logger.info(f"  OOS:  {oos_m['total_return']:+.1f}% PF {oos_m['profit_factor']:.2f} WR {oos_m['win_rate']:.1f}% DD {oos_m['max_drawdown']:.1f}% ({oos_m['total_trades']} trades)")
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results.append({"bull": bull, "bear": bear, "full": None, "oos": None, "error": str(e)})

    # Rank by composite score: OOS return (primary), full PF (secondary), low DD (tertiary)
    valid = [r for r in results if r.get("full") and r.get("oos") and r["full"]["total_trades"] >= 10]
    valid.sort(key=lambda r: (
        r["oos"]["total_return"],
        r["full"]["profit_factor"],
        -r["full"]["max_drawdown"],
    ), reverse=True)

    # Report top 10
    logger.info(f"\n{'='*70}")
    logger.info("UALGO FILTER SWEEP RESULTS — Top 10")
    logger.info(f"{'='*70}")
    report = "🔬 <b>UAlgo Filter Sweep Complete</b>\n\n<b>Top 10 Combos:</b>\n"
    for i, r in enumerate(valid[:10]):
        line = (f"#{i+1} bull={r['bull']}, bear={r['bear']}: "
                f"Full {r['full']['total_return']:+.1f}% PF {r['full']['profit_factor']:.2f} "
                f"({r['full']['total_trades']}t, DD {r['full']['max_drawdown']:.1f}%) | "
                f"OOS {r['oos']['total_return']:+.1f}% PF {r['oos']['profit_factor']:.2f} "
                f"({r['oos']['total_trades']}t, DD {r['oos']['max_drawdown']:.1f}%)")
        logger.info(line)
        report += f"{line}\n"

    if valid:
        best = valid[0]
        report += f"\n🏆 <b>Winner: bull={best['bull']}, bear={best['bear']}</b>"
        logger.info(f"\nWinner: bull={best['bull']}, bear={best['bear']}")
    send_telegram(report)

    return valid


def run_intensive_ualgo(max_rounds=20, bull_filter="off", bear_filter="death_cross"):
    """Run intensive optimization on UAlgo backtest engine.

    Same pattern as Donchian intensive: AI proposes → backtest full + OOS → auto-apply if both improve.
    Uses backtest_ualgo_engine.run_backtest() with UAlgo-specific parameters.
    """
    tradesavvy_backend = os.path.join(os.path.dirname(__file__), '..', 'tradesavvy', 'backend')
    if tradesavvy_backend not in sys.path:
        sys.path.insert(0, tradesavvy_backend)
    from backtest_ualgo_engine import run_backtest

    logger.info(f"=== UALGO INTENSIVE OPTIMIZATION: up to {max_rounds} rounds (bull={bull_filter}, bear={bear_filter}) ===")
    send_telegram(f"🔬 <b>UAlgo Intensive Optimization Started</b>\nMax rounds: {max_rounds}\nFilters: bull={bull_filter}, bear={bear_filter}")

    # UAlgo defaults as starting config
    from backtest_ualgo_engine import UALGO_DEFAULTS
    current_config = dict(UALGO_DEFAULTS)
    # Add top-level params not in UALGO_DEFAULTS but accepted by run_backtest
    current_config['risk_per_trade'] = 2.0
    current_config['short_sl_atr_mult'] = UALGO_DEFAULTS['sl_atr_mult']

    def _run_full_and_oos(config_overrides=None):
        params = {"mode": "futures-combined", "period": "full",
                  "bull_filter": bull_filter, "bear_filter": bear_filter,
                  "long_leverage": 1.0, "short_leverage": 1.0}
        # Apply accumulated best config (closure over best_config)
        for k in UALGO_CONFIG_RANGES:
            if k in best_config:
                params[k] = best_config[k]
        # Apply current round's overrides
        if config_overrides:
            params.update(config_overrides)
        full_res = run_backtest({**params, "period": "full"})
        oos_res = run_backtest({**params, "period": "test"})
        keys = ["total_return", "profit_factor", "win_rate", "max_drawdown", "total_trades", "sharpe_ratio"]
        return ({k: full_res.get(k) for k in keys}, {k: oos_res.get(k) for k in keys})

    best_config = dict(current_config)
    baseline_full, baseline_oos = _run_full_and_oos()
    best_full, best_oos = baseline_full, baseline_oos
    round_history = []
    total_tokens = 0
    improvements = 0

    logger.info(f"UAlgo Baseline: Full {baseline_full['total_return']:+.1f}% PF {baseline_full['profit_factor']:.2f} | "
                f"OOS {baseline_oos['total_return']:+.1f}% PF {baseline_oos['profit_factor']:.2f}")

    for round_num in range(1, max_rounds + 1):
        logger.info(f"\n--- UAlgo Round {round_num}/{max_rounds} ---")

        history_text = ""
        if round_history:
            history_text = "\n## Previous Rounds\n"
            for rh in round_history:
                status = "APPLIED" if rh["applied"] else "REJECTED"
                history_text += (f"- R{rh['round']}: {json.dumps(rh['changes'])} → "
                                 f"Full {rh['proposed_full']['total_return']:+.1f}% PF {rh['proposed_full']['profit_factor']:.2f}, "
                                 f"OOS {rh['proposed_oos']['total_return']:+.1f}% PF {rh['proposed_oos']['profit_factor']:.2f} "
                                 f"[{status}: {rh.get('reason', '')}]\n")

        current_best_params = {k: best_config.get(k) for k in UALGO_CONFIG_RANGES}
        prompt = f"""You are an expert crypto trading bot optimizer. Optimize the UAlgo strategy parameters.

UAlgo uses Supertrend + RSI + CMO + HMA indicators to generate long and short signals on daily candles.
Entry: Supertrend flip + RSI/CMO thresholds + pivot detection
Exit: ATR-Fibonacci partial TPs (1.618/2.618/3.618 * ATR) + opposing signal + max hold + stop loss

## Current Best Configuration
```json
{json.dumps(current_best_params, indent=2)}
```

## Current Best Performance
- Full (2022-2026): Return {best_full['total_return']:+.1f}%, PF {best_full['profit_factor']:.2f}, WR {best_full['win_rate']:.1f}%, DD {best_full['max_drawdown']:.1f}%
- OOS (2025-2026): Return {best_oos['total_return']:+.1f}%, PF {best_oos['profit_factor']:.2f}, WR {best_oos['win_rate']:.1f}%, DD {best_oos['max_drawdown']:.1f}%

## Baseline (starting point)
- Full: {baseline_full['total_return']:+.1f}%, PF {baseline_full['profit_factor']:.2f}
- OOS: {baseline_oos['total_return']:+.1f}%, PF {baseline_oos['profit_factor']:.2f}

{history_text}

## Round {round_num}/{max_rounds}
- {"Focus on high-impact params: supertrend_multiplier, sl_atr_mult, RSI/CMO thresholds" if round_num <= 5 else "Try TP fractions, max_hold_days, flip_window, indicator periods" if round_num <= 12 else "Fine-tune combinations not yet tested"}
- Auto-applied if both full AND OOS improve. No human review.
- Bull filter={bull_filter}, bear filter={bear_filter} — do NOT change these.

## Allowed Ranges (propose SINGLE scalar values, NOT lists)
{json.dumps(UALGO_CONFIG_RANGES, indent=2)}

Return ONLY a JSON object:
```json
{{
  "proposed_changes": {{"param_name": 3.5}},
  "rationale": "Brief explanation",
  "expected_impact": "Expected improvement"
}}
```
If optimal: {{"proposed_changes": {{}}, "rationale": "Converged"}}"""

        ai_result = run_ai_analysis(prompt)
        if not ai_result:
            logger.error(f"UAlgo R{round_num}: AI failed, stopping")
            break

        total_tokens += ai_result.get("_token_usage", 0)
        changes = ai_result.get("proposed_changes", {})
        rationale = ai_result.get("rationale", "")

        if not changes:
            logger.info(f"UAlgo R{round_num}: AI says optimal — {rationale}")
            break

        validated, warnings = validate_proposed_changes(changes, "ualgo_config")
        if warnings:
            logger.info(f"UAlgo R{round_num} warnings: {warnings}")
        if not validated:
            logger.info(f"UAlgo R{round_num}: No valid changes")
            round_history.append({"round": round_num, "changes": changes, "applied": False,
                                  "reason": "validation failed", "proposed_full": best_full, "proposed_oos": best_oos})
            continue

        logger.info(f"UAlgo R{round_num}: Testing {json.dumps(validated)} — {rationale}")
        proposed_full, proposed_oos = _run_full_and_oos(validated)

        # Same acceptance criteria as Donchian
        full_ret_ok = proposed_full["total_return"] >= best_full["total_return"] - 1.0
        full_pf_ok = proposed_full["profit_factor"] >= best_full["profit_factor"] - 0.05
        full_better = full_ret_ok and full_pf_ok
        oos_not_worse = (proposed_oos["total_return"] >= best_oos["total_return"] - 1.0 and
                         proposed_oos["max_drawdown"] <= best_oos["max_drawdown"] + 2.0)
        dd_ok = proposed_full["max_drawdown"] <= best_full["max_drawdown"] + 3.0
        has_improvement = (proposed_full["total_return"] > best_full["total_return"] + 0.5 or
                           proposed_full["profit_factor"] > best_full["profit_factor"] + 0.02 or
                           proposed_oos["total_return"] > best_oos["total_return"] + 0.5)

        applied = full_better and oos_not_worse and dd_ok and has_improvement

        if applied:
            improvements += 1
            for k, v in validated.items():
                best_config[k] = v
            best_full = proposed_full
            best_oos = proposed_oos
            reason = "improvement"
            logger.info(f"UAlgo R{round_num}: ✅ APPLIED — Full {proposed_full['total_return']:+.1f}% PF {proposed_full['profit_factor']:.2f} | "
                        f"OOS {proposed_oos['total_return']:+.1f}% PF {proposed_oos['profit_factor']:.2f}")
        else:
            reasons = []
            if not full_ret_ok: reasons.append(f"full return regressed ({proposed_full['total_return']:+.1f}%)")
            if not full_pf_ok: reasons.append(f"full PF regressed ({proposed_full['profit_factor']:.2f})")
            if not oos_not_worse: reasons.append(f"OOS regressed ({proposed_oos['total_return']:+.1f}%)")
            if not dd_ok: reasons.append(f"DD too high ({proposed_full['max_drawdown']:.1f}%)")
            if not has_improvement: reasons.append("no meaningful improvement")
            reason = "; ".join(reasons) if reasons else "no criteria met"
            logger.info(f"UAlgo R{round_num}: ❌ REJECTED — {reason}")

        round_history.append({
            "round": round_num, "changes": validated, "applied": applied,
            "reason": reason, "rationale": rationale,
            "proposed_full": proposed_full, "proposed_oos": proposed_oos,
        })

    # Final report
    logger.info(f"\n{'='*60}")
    logger.info(f"UALGO INTENSIVE OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Rounds: {len(round_history)}, Improvements: {improvements}")
    logger.info(f"Baseline:  Full {baseline_full['total_return']:+.1f}% PF {baseline_full['profit_factor']:.2f} | OOS {baseline_oos['total_return']:+.1f}% PF {baseline_oos['profit_factor']:.2f}")
    logger.info(f"Best:      Full {best_full['total_return']:+.1f}% PF {best_full['profit_factor']:.2f} | OOS {best_oos['total_return']:+.1f}% PF {best_oos['profit_factor']:.2f}")
    logger.info(f"Best config: {json.dumps({k: best_config.get(k) for k in UALGO_CONFIG_RANGES}, indent=2)}")
    logger.info(f"Tokens used: {total_tokens}, Est cost: ${total_tokens * 0.000003:.4f}")

    report = (
        f"🔬 <b>UAlgo Intensive Optimization Complete</b>\n"
        f"Rounds: {len(round_history)} | Improvements: {improvements}\n\n"
        f"📊 <b>Baseline → Best:</b>\n"
        f"Full: {baseline_full['total_return']:+.1f}% → {best_full['total_return']:+.1f}% (PF {baseline_full['profit_factor']:.2f} → {best_full['profit_factor']:.2f})\n"
        f"OOS: {baseline_oos['total_return']:+.1f}% → {best_oos['total_return']:+.1f}% (PF {baseline_oos['profit_factor']:.2f} → {best_oos['profit_factor']:.2f})\n"
        f"DD: {baseline_full['max_drawdown']:.1f}% → {best_full['max_drawdown']:.1f}%\n\n"
        f"<b>Round Details:</b>\n"
    )
    for rh in round_history:
        icon = "✅" if rh["applied"] else "❌"
        report += f"{icon} R{rh['round']}: {json.dumps(rh['changes'])} → Full {rh['proposed_full']['total_return']:+.1f}%, OOS {rh['proposed_oos']['total_return']:+.1f}%\n"
    send_telegram(report)
    return best_config, round_history


def main():
    """Main loop: weekly schedule + poll for manual triggers."""
    os.makedirs("logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--intensive", type=int, nargs="?", const=10, metavar="ROUNDS",
                        help="Run intensive Donchian backtest optimization (default: 10 rounds)")
    parser.add_argument("--intensive-ualgo", type=int, nargs="?", const=20, metavar="ROUNDS",
                        help="Run intensive UAlgo backtest optimization (default: 20 rounds)")
    parser.add_argument("--ualgo-sweep", action="store_true",
                        help="Run UAlgo filter sweep (49 combos) then intensive optimization on best")
    parser.add_argument("--ualgo-sweep-only", action="store_true",
                        help="Run UAlgo filter sweep only (no optimization after)")
    parser.add_argument("--bull-filter", type=str, default="off",
                        help="Bull filter for UAlgo intensive (default: off)")
    parser.add_argument("--bear-filter", type=str, default="death_cross",
                        help="Bear filter for UAlgo intensive (default: death_cross)")
    args = parser.parse_args()

    if not db:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Exiting.")
        sys.exit(1)

    if args.intensive:
        logger.info(f"Intensive optimization mode: {args.intensive} rounds")
        run_intensive_optimization(max_rounds=args.intensive)
        return

    if args.ualgo_sweep or args.ualgo_sweep_only:
        logger.info("UAlgo filter sweep mode")
        ranked = run_ualgo_filter_sweep()
        if ranked and not args.ualgo_sweep_only:
            best = ranked[0]
            logger.info(f"Starting intensive optimization with best filters: bull={best['bull']}, bear={best['bear']}")
            rounds = args.intensive_ualgo if args.intensive_ualgo else 20
            run_intensive_ualgo(max_rounds=rounds, bull_filter=best['bull'], bear_filter=best['bear'])
        return

    if args.intensive_ualgo:
        logger.info(f"UAlgo intensive optimization mode: {args.intensive_ualgo} rounds (bull={args.bull_filter}, bear={args.bear_filter})")
        run_intensive_ualgo(max_rounds=args.intensive_ualgo, bull_filter=args.bull_filter, bear_filter=args.bear_filter)
        return

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
                run_optimization_cycle(trigger="manual", existing_run_id=queued_id)

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
