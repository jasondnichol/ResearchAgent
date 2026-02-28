"""Supabase sync module for Donchian trading bot.

Provides fire-and-forget sync to Supabase PostgREST API.
Every public method is wrapped in try/except so the bot never
crashes due to a Supabase failure. All writes are additive —
local bot_state.json remains the crash recovery source of truth.
"""
import os
import requests
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tradingbot")


class SupabaseSync:
    """Syncs bot state to Supabase via direct REST API calls."""

    def __init__(self, user_id: str = None):
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_KEY", "")
        self.user_id = user_id or os.getenv("SUPABASE_USER_ID", "")
        self.rest_url = f"{self.url}/rest/v1" if self.url else ""
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.enabled = bool(self.url and self.key and self.user_id)
        if self.enabled:
            logger.info("[SUPABASE] Sync enabled")
        else:
            logger.warning("[SUPABASE] Sync disabled — missing URL, key, or user_id")

    # ------------------------------------------------------------------
    # PRIVATE REST HELPERS
    # ------------------------------------------------------------------

    def _post(self, table, data):
        """Insert a row. Returns response data or None on failure."""
        if not self.enabled:
            return None
        try:
            resp = requests.post(
                f"{self.rest_url}/{table}",
                json=data,
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"[SUPABASE] INSERT {table} failed: {e}")
            return None

    def _patch(self, table, data, filters):
        """Update rows matching filters. Returns response data or None."""
        if not self.enabled:
            return None
        try:
            params = {k: f"eq.{v}" for k, v in filters.items()}
            resp = requests.patch(
                f"{self.rest_url}/{table}",
                json=data,
                params=params,
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"[SUPABASE] UPDATE {table} failed: {e}")
            return None

    def _delete(self, table, filters):
        """Delete rows matching filters. Returns True on success."""
        if not self.enabled:
            return False
        try:
            params = {k: f"eq.{v}" for k, v in filters.items()}
            resp = requests.delete(
                f"{self.rest_url}/{table}",
                params=params,
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[SUPABASE] DELETE {table} failed: {e}")
            return False

    def _get(self, table, filters=None, order=None, limit=None):
        """Select rows. Returns list of dicts or None on failure."""
        if not self.enabled:
            return None
        try:
            params = {"select": "*"}
            if filters:
                for k, v in filters.items():
                    params[k] = f"eq.{v}"
            if order:
                params["order"] = order
            if limit:
                params["limit"] = str(limit)
            resp = requests.get(
                f"{self.rest_url}/{table}",
                params=params,
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"[SUPABASE] SELECT {table} failed: {e}")
            return None

    # ------------------------------------------------------------------
    # POSITIONS
    # ------------------------------------------------------------------

    def sync_position_open(self, symbol, position):
        """Insert a new row into positions after a BUY or SHORT."""
        entry_time = position["entry_time"]
        if isinstance(entry_time, datetime):
            entry_time = entry_time.isoformat()
        side = position.get("side", "LONG")
        data = {
            "user_id": self.user_id,
            "symbol": symbol,
            "side": side,
            "entry_price": position["entry_price"],
            "entry_time": entry_time,
            "high_watermark": position.get("high_watermark"),
            "low_watermark": position.get("low_watermark"),
            "stop_price": position.get("stop_price"),
            "size_usd": position["size_usd"],
            "partials_taken": position.get("partials_taken", 0),
            "remaining_fraction": position.get("remaining_fraction", 1.0),
            "last_atr": position.get("last_atr"),
            "pyramided": position.get("pyramided", False),
            "hold_days": position.get("hold_days", 0),
            "spot_symbol": position.get("spot_symbol"),
        }
        return self._post("positions", data)

    def sync_position_update(self, symbol, position):
        """Update an existing position row (stop, partials, pyramid, etc.)."""
        data = {
            "high_watermark": position.get("high_watermark"),
            "low_watermark": position.get("low_watermark"),
            "stop_price": position.get("stop_price"),
            "size_usd": position["size_usd"],
            "partials_taken": position.get("partials_taken", 0),
            "remaining_fraction": position.get("remaining_fraction", 1.0),
            "last_atr": position.get("last_atr"),
            "pyramided": position.get("pyramided", False),
            "hold_days": position.get("hold_days", 0),
        }
        return self._patch("positions", data,
                           filters={"user_id": self.user_id, "symbol": symbol})

    def sync_position_close(self, symbol):
        """Delete the position row after a full SELL."""
        return self._delete("positions",
                            filters={"user_id": self.user_id, "symbol": symbol})

    # ------------------------------------------------------------------
    # TRADE HISTORY
    # ------------------------------------------------------------------

    def sync_trade(self, symbol, action, entry_price=None, exit_price=None,
                   size_usd=None, pnl_pct=None, pnl_usd=None,
                   exit_reason=None, hold_days=None, trading_mode="paper",
                   side="LONG"):
        """Insert a row into trade_history for every trade action."""
        data = {
            "user_id": self.user_id,
            "symbol": symbol,
            "action": action,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": size_usd,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "pnl_usd": round(pnl_usd, 2) if pnl_usd is not None else None,
            "exit_reason": exit_reason,
            "hold_days": round(hold_days, 1) if hold_days is not None else None,
            "trading_mode": trading_mode,
        }
        return self._post("trade_history", data)

    # ------------------------------------------------------------------
    # EQUITY SNAPSHOTS
    # ------------------------------------------------------------------

    def sync_equity_snapshot(self, equity, cash, positions_value,
                             positions_count, bull_filter_status=None,
                             btc_price=None):
        """Insert a daily equity snapshot for charting."""
        data = {
            "user_id": self.user_id,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "positions_value": round(positions_value, 2),
            "positions_count": positions_count,
            "bull_filter_status": bull_filter_status,
            "btc_price": round(btc_price, 2) if btc_price else None,
        }
        return self._post("equity_snapshots", data)

    # ------------------------------------------------------------------
    # BOT EVENTS
    # ------------------------------------------------------------------

    def sync_event(self, event_type, message, metadata=None):
        """Insert a bot event (startup, shutdown, daily_check, trade, error)."""
        data = {
            "user_id": self.user_id,
            "event_type": event_type,
            "message": message,
            "metadata": metadata,
        }
        return self._post("bot_events", data)

    # ------------------------------------------------------------------
    # BOT CONFIG (read only — bot reads config set via dashboard)
    # ------------------------------------------------------------------

    def load_config(self):
        """Load strategy config from Supabase. Returns dict or None."""
        rows = self._get("bot_config",
                         filters={"user_id": self.user_id},
                         limit=1)
        if rows and len(rows) > 0:
            return rows[0]
        return None
