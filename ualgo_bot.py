"""UAlgo Trend Trading Bot — Tri-mode (Spot Long + Futures Long + Futures Short)

Parallel to donchian_multicoin_bot.py but using UAlgo Trend Signals:
  - Supertrend (ATR 14, mult 2.0) for trend direction
  - HMA momentum + CMO + RSI(9) + pivot detection for entry confirmation
  - Fixed % SL + R-multiple TPs (1R@25%, 2R@25%, 3R@50%)
  - Opposing supertrend flip closes runner
  - Max hold 30 days

Runs in its own screen session on EC2: screen -S ualgo
Daily check at 00:20 UTC (offset from Donchian at 00:15).

Config loaded from Supabase bot_config WHERE strategy='ualgo'.
UAlgo-specific params stored in strategy_params JSONB column.
"""

import json
import time
import os
import requests
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta

from ualgo_strategy import (
    fetch_daily_candles, fetch_current_price,
    compute_ualgo_indicators,
    check_ualgo_long_entry, check_ualgo_short_entry,
    check_ualgo_stop_loss, check_ualgo_partial_tp,
    check_ualgo_opposing_signal, check_ualgo_max_hold,
    check_bull_filter, check_bear_filter,
    DEFAULT_PARAMS,
)
from coinbase_futures import CoinbaseFuturesClient, SPOT_TO_PERP, PERP_TO_SPOT
from notify import send_telegram, send_telegram_user, send_telegram_platform, setup_logging
from supabase_sync import SupabaseSync

# ============================================================================
# DEFAULT CONFIGURATION (overridden by Supabase bot_config WHERE strategy='ualgo')
# ============================================================================

DEFAULT_MAX_POSITIONS = 4
DEFAULT_RISK_PER_TRADE_PCT = 2.0
DEFAULT_STARTING_CAPITAL = 10000.0
DEFAULT_EMERGENCY_STOP_PCT = 15.0

DEFAULT_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD',
]
DEFAULT_SHORT_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]
DEFAULT_FUTURES_LONG_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]

DEFAULT_FUTURES_LONG_ENABLED = True
DEFAULT_FUTURES_LONG_LEVERAGE = 1.0
DEFAULT_SHORT_ENABLED = True

# Timing — offset from Donchian (00:15) to avoid API rate limits
DAILY_CHECK_HOUR = 0
DAILY_CHECK_MINUTE = 20
STOP_CHECK_INTERVAL = 1800  # 30 minutes


# ============================================================================
# UALGO USER BOT
# ============================================================================

class UAlgoUserBot:
    """Per-user UAlgo trading bot instance."""

    def __init__(self, user_id, paper_trading=True):
        self.user_id = user_id
        self.paper_trading = paper_trading
        self.coinbase_api = "https://api.exchange.coinbase.com"

        # Strategy params (UAlgo-specific, loaded from strategy_params JSONB)
        self.strategy_params = dict(DEFAULT_PARAMS)

        # Coin universes
        self.coin_universe = list(DEFAULT_COIN_UNIVERSE)
        self.short_coin_universe = list(DEFAULT_SHORT_COIN_UNIVERSE)
        self.futures_long_coin_universe = list(DEFAULT_FUTURES_LONG_COIN_UNIVERSE)

        # Shared scalars
        self.max_positions = DEFAULT_MAX_POSITIONS
        self.risk_per_trade_pct = DEFAULT_RISK_PER_TRADE_PCT
        self.starting_capital = DEFAULT_STARTING_CAPITAL
        self.emergency_stop_pct = DEFAULT_EMERGENCY_STOP_PCT

        # Mode toggles
        self.bull_filter_enabled = True
        self.bear_filter_enabled = True
        self.short_enabled = DEFAULT_SHORT_ENABLED
        self.futures_long_enabled = DEFAULT_FUTURES_LONG_ENABLED
        self.futures_long_leverage = DEFAULT_FUTURES_LONG_LEVERAGE

        # State file (different from Donchian)
        self.state_file = f'ualgo_bot_state_{user_id[:8]}.json'

        # Portfolio state
        self.cash = self.starting_capital
        self.positions = {}  # key: symbol or perp_id, val: position dict
        self.trades_log = []
        self.total_realized_pnl = 0.0

        # Futures client
        self.futures_client = CoinbaseFuturesClient(paper_mode=paper_trading)

        # Logging
        self.logger, self.trade_logger = setup_logging()

        # Supabase sync (strategy='ualgo')
        self.sync = SupabaseSync(user_id=user_id, strategy='ualgo')

        # Per-user Telegram
        self.telegram_bot_token = ""
        self.telegram_chat_id = ""
        self.notify_telegram = True

        # Load config from Supabase, then restore state
        self.load_remote_config()

        # Enforce paper trading safety
        if not self.paper_trading:
            config = self.sync.load_config()
            if config and config.get("trading_mode") != "live":
                self.paper_trading = True

        self.load_state()

    # ------------------------------------------------------------------
    # REMOTE CONFIG
    # ------------------------------------------------------------------

    def load_remote_config(self):
        """Load config from Supabase bot_config WHERE strategy='ualgo'."""
        config = self.sync.load_config()
        if not config:
            self.logger.info("[UALGO] No remote config found, using defaults")
            return

        self.logger.info("[UALGO] Loaded remote config from Supabase")

        # Mode toggles
        self.bull_filter_enabled = config.get("bull_filter_enabled", True)
        self.bear_filter_enabled = config.get("bear_filter_enabled", True)
        self.short_enabled = config.get("short_enabled", DEFAULT_SHORT_ENABLED)
        self.futures_long_enabled = config.get("futures_long_enabled", DEFAULT_FUTURES_LONG_ENABLED)
        self.futures_long_leverage = min(float(config.get("futures_long_leverage", DEFAULT_FUTURES_LONG_LEVERAGE)), 3.0)

        # Shared scalars
        self.risk_per_trade_pct = float(config.get("risk_per_trade_pct", DEFAULT_RISK_PER_TRADE_PCT))
        self.max_positions = int(config.get("max_positions", DEFAULT_MAX_POSITIONS))
        self.emergency_stop_pct = float(config.get("emergency_stop_pct", DEFAULT_EMERGENCY_STOP_PCT))
        self.starting_capital = float(config.get("starting_capital", DEFAULT_STARTING_CAPITAL))

        # Trading mode
        mode = config.get("trading_mode", "paper")
        self.paper_trading = mode != "live"

        # Coin universes
        coins = config.get("coins")
        if coins and isinstance(coins, list) and len(coins) > 0:
            self.coin_universe = [c for c in coins if isinstance(c, str)]
        short_coins = config.get("short_coins")
        if short_coins and isinstance(short_coins, list) and len(short_coins) > 0:
            self.short_coin_universe = [c for c in short_coins if isinstance(c, str)]
        fl_coins = config.get("futures_long_coins")
        if fl_coins and isinstance(fl_coins, list) and len(fl_coins) > 0:
            self.futures_long_coin_universe = [c for c in fl_coins if isinstance(c, str)]

        # UAlgo-specific strategy params from strategy_params JSONB
        sp = config.get("strategy_params")
        if sp and isinstance(sp, dict):
            for key in DEFAULT_PARAMS:
                if key in sp:
                    self.strategy_params[key] = sp[key]

        # Telegram
        self.telegram_bot_token = config.get("telegram_bot_token", "")
        self.telegram_chat_id = config.get("telegram_chat_id", "")

        self.logger.info(
            f"[UALGO] Config: {len(self.coin_universe)} spot, "
            f"{len(self.futures_long_coin_universe)} FL, "
            f"{len(self.short_coin_universe)} short coins, "
            f"risk={self.risk_per_trade_pct}%, max_pos={self.max_positions}, "
            f"bull={'ON' if self.bull_filter_enabled else 'OFF'}, "
            f"short={'ON' if self.short_enabled else 'OFF'} "
            f"(bear={'ON' if self.bear_filter_enabled else 'OFF'}), "
            f"FL={'ON' if self.futures_long_enabled else 'OFF'} "
            f"(lev={self.futures_long_leverage}x)"
        )

    # ------------------------------------------------------------------
    # STATE PERSISTENCE
    # ------------------------------------------------------------------

    def save_state(self):
        """Save portfolio state for crash recovery."""
        state = {
            'cash': self.cash,
            'total_realized_pnl': self.total_realized_pnl,
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'positions': {},
        }
        for symbol, pos in self.positions.items():
            pos_copy = dict(pos)
            if isinstance(pos_copy.get('entry_time'), datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
            state['positions'][symbol] = pos_copy

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"[UALGO] Failed to save state: {e}")

    def load_state(self):
        """Load portfolio state from crash recovery file."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.cash = state.get('cash', self.starting_capital)
            self.total_realized_pnl = state.get('total_realized_pnl', 0.0)

            for symbol, pos_data in state.get('positions', {}).items():
                pos = dict(pos_data)
                if 'entry_time' in pos and isinstance(pos['entry_time'], str):
                    pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
                # Fill missing spot_symbol for perp positions
                if pos.get('side') in ('SHORT', 'FUTURES_LONG') and 'spot_symbol' not in pos:
                    pos['spot_symbol'] = PERP_TO_SPOT.get(symbol, '')
                self.positions[symbol] = pos

            self.logger.info(
                f"[UALGO] Loaded state: ${self.cash:,.2f} cash, "
                f"{len(self.positions)} positions, "
                f"${self.total_realized_pnl:,.2f} realized P&L"
            )
        except Exception as e:
            self.logger.error(f"[UALGO] Failed to load state: {e}")

    # ------------------------------------------------------------------
    # PRICE HELPERS
    # ------------------------------------------------------------------

    def fetch_all_prices(self):
        """Get current prices for all coins across all universes."""
        prices = {}
        all_coins = set(self.coin_universe) | set(self.short_coin_universe) | set(self.futures_long_coin_universe)
        for symbol in sorted(all_coins):
            try:
                prices[symbol] = fetch_current_price(symbol)
                time.sleep(0.15)
            except Exception as e:
                self.logger.warning(f"[UALGO] Price fetch failed for {symbol}: {e}")

        # Map spot prices to perp IDs for open positions
        for symbol, pos in self.positions.items():
            if pos.get('side') in ('SHORT', 'FUTURES_LONG') and symbol not in prices:
                spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                if spot_sym in prices:
                    prices[symbol] = prices[spot_sym]
        return prices

    # ------------------------------------------------------------------
    # PORTFOLIO HELPERS
    # ------------------------------------------------------------------

    def total_equity(self, prices=None):
        """Calculate total equity: cash + mark-to-market positions."""
        equity = self.cash
        for symbol, pos in self.positions.items():
            price = None
            if prices:
                price = prices.get(symbol)
                if price is None:
                    spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                    price = prices.get(spot_sym)
            if price is None:
                try:
                    spot_sym = pos.get('spot_symbol', symbol)
                    price = fetch_current_price(spot_sym)
                except Exception:
                    price = pos['entry_price']

            size_usd = pos['size_usd'] * pos.get('remaining_fraction', 1.0)
            entry = pos['entry_price']
            side = pos.get('side', 'LONG')

            if side in ('LONG', 'FUTURES_LONG'):
                pnl = size_usd * (price - entry) / entry
            else:  # SHORT
                pnl = size_usd * (entry - price) / entry

            equity += size_usd + pnl

        return equity

    def _count_positions(self):
        """Count current open positions."""
        return len(self.positions)

    def _coin_in_positions(self, symbol):
        """Check if this coin already has a position (spot or perp)."""
        if symbol in self.positions:
            return True
        perp_id = SPOT_TO_PERP.get(symbol, '')
        if perp_id and perp_id in self.positions:
            return True
        # Check reverse: if symbol is a perp_id, check spot
        spot_sym = PERP_TO_SPOT.get(symbol, '')
        if spot_sym and spot_sym in self.positions:
            return True
        return False

    def calculate_position_size(self, entry_price, stop_price, prices=None, leverage=1.0):
        """Size position using risk-based formula.
        risk_usd = equity * risk% ; size = risk_usd / (stop_dist / entry) * leverage
        """
        equity = self.total_equity(prices)
        risk_usd = equity * (self.risk_per_trade_pct / 100.0)
        stop_dist = abs(entry_price - stop_price) / entry_price
        if stop_dist < 0.001:
            return 0

        size = (risk_usd / stop_dist) * leverage
        size = min(size, self.cash * 0.95)
        return max(0, size)

    # ------------------------------------------------------------------
    # TRADE EXECUTION — SPOT LONG
    # ------------------------------------------------------------------

    def execute_buy(self, symbol, entry_price, stop_price, tp_levels, tp_fractions, indicators, prices=None):
        """Open a spot LONG position."""
        if self._coin_in_positions(symbol):
            return False
        if self._count_positions() >= self.max_positions:
            return False

        size = self.calculate_position_size(entry_price, stop_price, prices)
        if size < 10:
            self.logger.info(f"[UALGO] {symbol} BUY skipped — size ${size:.0f} too small")
            return False

        # Paper trade — just track
        self.positions[symbol] = {
            'side': 'LONG',
            'entry_price': entry_price,
            'entry_time': datetime.now(timezone.utc),
            'stop_price': stop_price,
            'size_usd': size,
            'partials_taken': 0,
            'remaining_fraction': 1.0,
            'high_watermark': entry_price,
        }
        self.cash -= size

        self.save_state()

        # Log
        sl_pct = abs(entry_price - stop_price) / entry_price * 100
        msg = (f"[UALGO] BUY {symbol} @ ${entry_price:,.2f} | "
               f"Size: ${size:,.0f} | SL: ${stop_price:,.2f} ({sl_pct:.1f}%) | "
               f"TP1: ${tp_levels[0]:,.2f} TP2: ${tp_levels[1]:,.2f} TP3: ${tp_levels[2]:,.2f}")
        self.logger.info(msg)
        print(msg)

        # Telegram
        self._notify(
            f"🟢 <b>UALGO BUY {symbol}</b>\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"Size: ${size:,.0f}\n"
            f"SL: ${stop_price:,.2f} ({sl_pct:.1f}%)\n"
            f"TP1: ${tp_levels[0]:,.2f} | TP2: ${tp_levels[1]:,.2f} | TP3: ${tp_levels[2]:,.2f}\n"
            f"RSI(9): {indicators.get('rsi9', '?')} | CMO: {indicators.get('cmo', '?')}"
        )

        # Supabase
        self.sync.sync_position_open(symbol, self.positions[symbol])
        self.sync.sync_trade(symbol, 'BUY', entry_price=entry_price, size_usd=size,
                             trading_mode='paper' if self.paper_trading else 'live', side='LONG')
        self.sync.sync_event('trade', f'UAlgo BUY {symbol} @ ${entry_price:,.2f}')

        return True

    def execute_sell(self, symbol, exit_price, reason, fraction=1.0):
        """Close (fully or partially) a spot LONG position."""
        pos = self.positions.get(symbol)
        if not pos:
            return False

        actual_fraction = fraction * pos.get('remaining_fraction', 1.0)
        sell_size = pos['size_usd'] * actual_fraction
        entry = pos['entry_price']
        pnl_pct = (exit_price - entry) / entry * 100
        pnl_usd = sell_size * (exit_price - entry) / entry

        self.cash += sell_size + pnl_usd
        self.total_realized_pnl += pnl_usd

        # Update position or remove
        if fraction >= 0.99:
            self.positions.pop(symbol, None)
            self.sync.sync_position_close(symbol)
        else:
            pos['remaining_fraction'] *= (1.0 - fraction)
            pos['partials_taken'] = pos.get('partials_taken', 0) + 1
            self.sync.sync_position_update(symbol, pos)

        self.save_state()

        entry_time = pos.get('entry_time', datetime.now(timezone.utc))
        hold_days = (datetime.now(timezone.utc) - entry_time).total_seconds() / 86400

        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        msg = (f"[UALGO] SELL {symbol} @ ${exit_price:,.2f} | "
               f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.2f}) | "
               f"Reason: {reason} | Fraction: {fraction:.0%}")
        self.logger.info(msg)
        print(msg)

        self._notify(
            f"{emoji} <b>UALGO SELL {symbol}</b>\n"
            f"Exit: ${exit_price:,.2f} | {reason}\n"
            f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.2f})\n"
            f"Hold: {hold_days:.1f} days | Fraction: {fraction:.0%}"
        )

        self.sync.sync_trade(symbol, 'SELL', entry_price=entry, exit_price=exit_price,
                             size_usd=sell_size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                             exit_reason=reason, hold_days=hold_days,
                             trading_mode='paper' if self.paper_trading else 'live', side='LONG')
        return True

    # ------------------------------------------------------------------
    # TRADE EXECUTION — FUTURES LONG
    # ------------------------------------------------------------------

    def execute_futures_long_open(self, symbol, entry_price, stop_price, tp_levels, tp_fractions, indicators, prices=None):
        """Open a FUTURES_LONG position via CFM perps."""
        if self._coin_in_positions(symbol):
            return False
        if self._count_positions() >= self.max_positions:
            return False

        perp_id = SPOT_TO_PERP.get(symbol)
        if not perp_id:
            self.logger.warning(f"[UALGO] No perp mapping for {symbol}")
            return False

        size = self.calculate_position_size(entry_price, stop_price, prices, leverage=self.futures_long_leverage)
        if size < 10:
            return False

        # Execute via futures client
        result = self.futures_client.market_order(perp_id, 'BUY', size_usd=size)
        if not result.get('success'):
            self.logger.warning(f"[UALGO] Futures long open failed for {perp_id}: {result.get('error')}")
            return False

        fill_price = result.get('fill_price', entry_price)

        self.positions[perp_id] = {
            'side': 'FUTURES_LONG',
            'entry_price': fill_price,
            'entry_time': datetime.now(timezone.utc),
            'stop_price': stop_price,
            'size_usd': size,
            'partials_taken': 0,
            'remaining_fraction': 1.0,
            'high_watermark': fill_price,
            'spot_symbol': symbol,
        }
        self.cash -= size

        self.save_state()

        sl_pct = abs(fill_price - stop_price) / fill_price * 100
        msg = (f"[UALGO] FUTURES_BUY {perp_id} @ ${fill_price:,.2f} | "
               f"Size: ${size:,.0f} ({self.futures_long_leverage}x) | "
               f"SL: ${stop_price:,.2f} ({sl_pct:.1f}%)")
        self.logger.info(msg)
        print(msg)

        self._notify(
            f"🟢 <b>UALGO FL BUY {symbol}</b>\n"
            f"Entry: ${fill_price:,.2f} ({self.futures_long_leverage}x)\n"
            f"Size: ${size:,.0f}\n"
            f"SL: ${stop_price:,.2f} ({sl_pct:.1f}%)\n"
            f"RSI(9): {indicators.get('rsi9', '?')} | CMO: {indicators.get('cmo', '?')}"
        )

        self.sync.sync_position_open(perp_id, self.positions[perp_id])
        self.sync.sync_trade(perp_id, 'FUTURES_BUY', entry_price=fill_price, size_usd=size,
                             trading_mode='paper' if self.paper_trading else 'live', side='FUTURES_LONG')
        return True

    def execute_futures_long_close(self, symbol, exit_price, reason, fraction=1.0):
        """Close (fully or partially) a FUTURES_LONG position."""
        pos = self.positions.get(symbol)
        if not pos:
            return False

        actual_fraction = fraction * pos.get('remaining_fraction', 1.0)
        close_size = pos['size_usd'] * actual_fraction
        entry = pos['entry_price']
        pnl_pct = (exit_price - entry) / entry * 100 * self.futures_long_leverage
        pnl_usd = close_size * (exit_price - entry) / entry

        # Execute via futures client
        if fraction >= 0.99:
            self.futures_client.close_position(symbol)
        else:
            self.futures_client.market_order(symbol, 'SELL', size_usd=close_size)

        self.cash += close_size + pnl_usd
        self.total_realized_pnl += pnl_usd

        if fraction >= 0.99:
            self.positions.pop(symbol, None)
            self.sync.sync_position_close(symbol)
        else:
            pos['remaining_fraction'] *= (1.0 - fraction)
            pos['partials_taken'] = pos.get('partials_taken', 0) + 1
            self.sync.sync_position_update(symbol, pos)

        self.save_state()

        entry_time = pos.get('entry_time', datetime.now(timezone.utc))
        hold_days = (datetime.now(timezone.utc) - entry_time).total_seconds() / 86400

        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        msg = (f"[UALGO] FUTURES_SELL {symbol} @ ${exit_price:,.2f} | "
               f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.2f}) | {reason}")
        self.logger.info(msg)
        print(msg)

        self._notify(
            f"{emoji} <b>UALGO FL SELL {PERP_TO_SPOT.get(symbol, symbol)}</b>\n"
            f"Exit: ${exit_price:,.2f} | {reason}\n"
            f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.2f})\n"
            f"Hold: {hold_days:.1f} days | Fraction: {fraction:.0%}"
        )

        action = 'FUTURES_SELL' if fraction >= 0.99 else f'FUTURES_PARTIAL_TP{pos.get("partials_taken", 0)}'
        self.sync.sync_trade(symbol, action, entry_price=entry, exit_price=exit_price,
                             size_usd=close_size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                             exit_reason=reason, hold_days=hold_days,
                             trading_mode='paper' if self.paper_trading else 'live', side='FUTURES_LONG')
        return True

    # ------------------------------------------------------------------
    # TRADE EXECUTION — SHORT
    # ------------------------------------------------------------------

    def execute_short_open(self, symbol, entry_price, stop_price, tp_levels, tp_fractions, indicators, prices=None):
        """Open a SHORT position via CFM perps."""
        if self._coin_in_positions(symbol):
            return False
        if self._count_positions() >= self.max_positions:
            return False

        perp_id = SPOT_TO_PERP.get(symbol)
        if not perp_id:
            self.logger.warning(f"[UALGO] No perp mapping for {symbol}")
            return False

        size = self.calculate_position_size(entry_price, stop_price, prices)
        if size < 10:
            return False

        result = self.futures_client.market_order(perp_id, 'SELL', size_usd=size)
        if not result.get('success'):
            self.logger.warning(f"[UALGO] Short open failed for {perp_id}: {result.get('error')}")
            return False

        fill_price = result.get('fill_price', entry_price)

        self.positions[perp_id] = {
            'side': 'SHORT',
            'entry_price': fill_price,
            'entry_time': datetime.now(timezone.utc),
            'stop_price': stop_price,
            'size_usd': size,
            'partials_taken': 0,
            'remaining_fraction': 1.0,
            'low_watermark': fill_price,
            'hold_days': 0,
            'spot_symbol': symbol,
        }
        self.cash -= size

        self.save_state()

        sl_pct = abs(stop_price - fill_price) / fill_price * 100
        msg = (f"[UALGO] SHORT {perp_id} @ ${fill_price:,.2f} | "
               f"Size: ${size:,.0f} | SL: ${stop_price:,.2f} ({sl_pct:.1f}%)")
        self.logger.info(msg)
        print(msg)

        self._notify(
            f"🔴 <b>UALGO SHORT {symbol}</b>\n"
            f"Entry: ${fill_price:,.2f}\n"
            f"Size: ${size:,.0f}\n"
            f"SL: ${stop_price:,.2f} ({sl_pct:.1f}%)\n"
            f"TP1: ${tp_levels[0]:,.2f} | TP2: ${tp_levels[1]:,.2f} | TP3: ${tp_levels[2]:,.2f}\n"
            f"RSI(9): {indicators.get('rsi9', '?')} | CMO: {indicators.get('cmo', '?')}"
        )

        self.sync.sync_position_open(perp_id, self.positions[perp_id])
        self.sync.sync_trade(perp_id, 'SHORT_SELL', entry_price=fill_price, size_usd=size,
                             trading_mode='paper' if self.paper_trading else 'live', side='SHORT')
        return True

    def execute_short_close(self, symbol, exit_price, reason, fraction=1.0):
        """Close (fully or partially) a SHORT position."""
        pos = self.positions.get(symbol)
        if not pos:
            return False

        actual_fraction = fraction * pos.get('remaining_fraction', 1.0)
        close_size = pos['size_usd'] * actual_fraction
        entry = pos['entry_price']
        pnl_pct = (entry - exit_price) / entry * 100  # Short: profit when price drops
        pnl_usd = close_size * (entry - exit_price) / entry

        if fraction >= 0.99:
            self.futures_client.close_position(symbol)
        else:
            self.futures_client.market_order(symbol, 'BUY', size_usd=close_size)

        self.cash += close_size + pnl_usd
        self.total_realized_pnl += pnl_usd

        if fraction >= 0.99:
            self.positions.pop(symbol, None)
            self.sync.sync_position_close(symbol)
        else:
            pos['remaining_fraction'] *= (1.0 - fraction)
            pos['partials_taken'] = pos.get('partials_taken', 0) + 1
            self.sync.sync_position_update(symbol, pos)

        self.save_state()

        entry_time = pos.get('entry_time', datetime.now(timezone.utc))
        hold_days = (datetime.now(timezone.utc) - entry_time).total_seconds() / 86400

        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        msg = (f"[UALGO] SHORT_COVER {symbol} @ ${exit_price:,.2f} | "
               f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.2f}) | {reason}")
        self.logger.info(msg)
        print(msg)

        self._notify(
            f"{emoji} <b>UALGO COVER {PERP_TO_SPOT.get(symbol, symbol)}</b>\n"
            f"Exit: ${exit_price:,.2f} | {reason}\n"
            f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.2f})\n"
            f"Hold: {hold_days:.1f} days"
        )

        action = 'SHORT_COVER' if fraction >= 0.99 else f'SHORT_PARTIAL_TP{pos.get("partials_taken", 0)}'
        self.sync.sync_trade(symbol, action, entry_price=entry, exit_price=exit_price,
                             size_usd=close_size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                             exit_reason=reason, hold_days=hold_days,
                             trading_mode='paper' if self.paper_trading else 'live', side='SHORT')
        return True

    # ------------------------------------------------------------------
    # TELEGRAM HELPER
    # ------------------------------------------------------------------

    def _notify(self, message):
        """Send Telegram notification (platform bot or per-user)."""
        try:
            if self.telegram_chat_id:
                send_telegram_platform(message, self.telegram_chat_id)
            else:
                send_telegram(message)
        except Exception as e:
            self.logger.warning(f"[UALGO] Telegram notify failed: {e}")

    # ------------------------------------------------------------------
    # DAILY SIGNAL CHECK
    # ------------------------------------------------------------------

    def daily_check(self):
        """Main daily signal check — exits first, then regime filters, then entries."""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}")
        print(f"[UALGO] Daily check at {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")

        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        print(f"Equity: ${equity:,.2f} | Cash: ${self.cash:,.2f} | "
              f"Positions: {self._count_positions()}/{self.max_positions}")

        # ===== EXITS (never gated by regime) =====
        self._check_daily_exits(prices)

        # ===== REGIME FILTERS =====
        is_bull, bull_details = check_bull_filter(self.bull_filter_enabled)
        is_bear, bear_details = check_bear_filter(self.bear_filter_enabled)

        bull_status = bull_details.get('status', 'UNKNOWN')
        bear_status = bear_details.get('status', 'UNKNOWN')

        if bull_details.get('btc_close'):
            print(f"\nBull filter: {bull_status} | BTC: ${bull_details['btc_close']:,.2f} | "
                  f"SMA200: ${bull_details['sma_200']:,.2f}")
        else:
            print(f"\nBull filter: {bull_status}")
        print(f"Bear filter: {bear_status}")

        # ===== FUTURES LONG ENTRIES (gated by bull + FL enabled) =====
        if is_bull and self.futures_long_enabled:
            self._check_futures_long_entries(prices)
        elif self.futures_long_enabled:
            print(f"  FL ENTRIES BLOCKED — bull filter is {bull_status}")

        # ===== SPOT LONG ENTRIES (gated by bull) =====
        if is_bull:
            self._check_spot_long_entries(prices)
        else:
            print(f"  LONG ENTRIES BLOCKED — bull filter is {bull_status}")

        # ===== SHORT ENTRIES (gated by short_enabled + bear filter) =====
        if self.short_enabled and is_bear:
            self._check_short_entries(prices)
        elif self.short_enabled:
            print(f"  SHORT ENTRIES BLOCKED — bear filter is {bear_status}")

        # ===== DAILY SUMMARY =====
        equity = self.total_equity(prices)
        positions_value = equity - self.cash
        btc_price = prices.get('BTC-USD')

        self.sync.sync_equity_snapshot(
            equity=equity, cash=self.cash,
            positions_value=positions_value,
            positions_count=self._count_positions(),
            bull_filter_status=f"{bull_status}/{bear_status}",
            btc_price=btc_price,
        )

        longs = sum(1 for p in self.positions.values() if p.get('side') == 'LONG')
        fl = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
        self.sync.sync_event("daily_check",
                             f"UAlgo daily scan: {self._count_positions()}/{self.max_positions} positions ({longs}L/{fl}FL/{shorts}S), "
                             f"${equity:,.0f} equity, bull={bull_status}, bear={bear_status}",
                             {"equity": round(equity, 2), "positions": self._count_positions(),
                              "longs": longs, "futures_longs": fl, "shorts": shorts,
                              "bull_status": bull_status, "bear_status": bear_status})

        print(f"\n[UALGO] Daily check complete. Equity: ${equity:,.2f}")

    def _check_daily_exits(self, prices):
        """Check all open positions for exit conditions using daily candle data."""
        symbols_to_check = list(self.positions.keys())

        for symbol in symbols_to_check:
            pos = self.positions.get(symbol)
            if not pos:
                continue

            side = pos.get('side', 'LONG')
            spot_sym = pos.get('spot_symbol', symbol) if side != 'LONG' else symbol

            try:
                df = fetch_daily_candles(spot_sym, limit=self.strategy_params.get('lookback_candles', 250))
                df = compute_ualgo_indicators(df, self.strategy_params)
            except Exception as e:
                self.logger.warning(f"[UALGO] Failed to fetch candles for {spot_sym}: {e}")
                continue

            current_price = prices.get(symbol) or prices.get(spot_sym, pos['entry_price'])

            # Check opposing supertrend flip
            opp = check_ualgo_opposing_signal(pos, df)
            if opp['hit']:
                if side == 'LONG':
                    self.execute_sell(symbol, current_price, opp['exit_reason'])
                elif side == 'FUTURES_LONG':
                    self.execute_futures_long_close(symbol, current_price, opp['exit_reason'])
                elif side == 'SHORT':
                    self.execute_short_close(symbol, current_price, opp['exit_reason'])
                continue

            # Check max hold
            max_hold = check_ualgo_max_hold(pos, self.strategy_params.get('max_hold_days', 30))
            if max_hold['hit']:
                if side == 'LONG':
                    self.execute_sell(symbol, current_price, max_hold['exit_reason'])
                elif side == 'FUTURES_LONG':
                    self.execute_futures_long_close(symbol, current_price, max_hold['exit_reason'])
                elif side == 'SHORT':
                    self.execute_short_close(symbol, current_price, max_hold['exit_reason'])
                continue

            # Check emergency stop
            entry = pos['entry_price']
            if side in ('LONG', 'FUTURES_LONG'):
                loss_pct = (entry - current_price) / entry * 100
            else:
                loss_pct = (current_price - entry) / entry * 100

            if loss_pct >= self.emergency_stop_pct:
                reason = f'Emergency stop ({loss_pct:.1f}%)'
                if side == 'LONG':
                    self.execute_sell(symbol, current_price, reason)
                elif side == 'FUTURES_LONG':
                    self.execute_futures_long_close(symbol, current_price, reason)
                elif side == 'SHORT':
                    self.execute_short_close(symbol, current_price, reason)

    def _check_spot_long_entries(self, prices):
        """Check for UAlgo BUY signals across spot long universe."""
        print(f"\n  Checking {len(self.coin_universe)} coins for UALGO LONG entries...")
        for symbol in self.coin_universe:
            if self._coin_in_positions(symbol):
                continue
            if self._count_positions() >= self.max_positions:
                print(f"  Max positions reached ({self.max_positions})")
                break

            try:
                df = fetch_daily_candles(symbol, limit=self.strategy_params.get('lookback_candles', 250))
                df = compute_ualgo_indicators(df, self.strategy_params)
                result = check_ualgo_long_entry(df, self.strategy_params)
                time.sleep(0.2)

                if result['signal']:
                    print(f"  ✅ {symbol} LONG signal! RSI={result['indicators']['rsi9']}, "
                          f"CMO={result['indicators']['cmo']}")
                    self.execute_buy(
                        symbol, result['entry_price'], result['stop_price'],
                        result['tp_levels'], result['tp_fractions'],
                        result['indicators'], prices,
                    )

            except Exception as e:
                self.logger.warning(f"[UALGO] Entry check failed for {symbol}: {e}")

    def _check_futures_long_entries(self, prices):
        """Check for UAlgo FUTURES_LONG entries."""
        print(f"\n  Checking {len(self.futures_long_coin_universe)} coins for UALGO FL entries...")
        for symbol in self.futures_long_coin_universe:
            if self._coin_in_positions(symbol):
                continue
            if self._count_positions() >= self.max_positions:
                break

            try:
                df = fetch_daily_candles(symbol, limit=self.strategy_params.get('lookback_candles', 250))
                df = compute_ualgo_indicators(df, self.strategy_params)
                result = check_ualgo_long_entry(df, self.strategy_params)
                time.sleep(0.2)

                if result['signal']:
                    print(f"  ✅ {symbol} FL signal! RSI={result['indicators']['rsi9']}, "
                          f"CMO={result['indicators']['cmo']}")
                    self.execute_futures_long_open(
                        symbol, result['entry_price'], result['stop_price'],
                        result['tp_levels'], result['tp_fractions'],
                        result['indicators'], prices,
                    )

            except Exception as e:
                self.logger.warning(f"[UALGO] FL entry check failed for {symbol}: {e}")

    def _check_short_entries(self, prices):
        """Check for UAlgo SHORT entries."""
        print(f"\n  Checking {len(self.short_coin_universe)} coins for UALGO SHORT entries...")
        for symbol in self.short_coin_universe:
            if self._coin_in_positions(symbol):
                continue
            if self._count_positions() >= self.max_positions:
                break

            try:
                df = fetch_daily_candles(symbol, limit=self.strategy_params.get('lookback_candles', 250))
                df = compute_ualgo_indicators(df, self.strategy_params)
                result = check_ualgo_short_entry(df, self.strategy_params)
                time.sleep(0.2)

                if result['signal']:
                    print(f"  ✅ {symbol} SHORT signal! RSI={result['indicators']['rsi9']}, "
                          f"CMO={result['indicators']['cmo']}")
                    self.execute_short_open(
                        symbol, result['entry_price'], result['stop_price'],
                        result['tp_levels'], result['tp_fractions'],
                        result['indicators'], prices,
                    )

            except Exception as e:
                self.logger.warning(f"[UALGO] Short entry check failed for {symbol}: {e}")

    # ------------------------------------------------------------------
    # TRAILING STOP MONITORING (every 30 min)
    # ------------------------------------------------------------------

    def check_trailing_stops(self):
        """Lightweight price-only check for stops and TPs.

        UAlgo uses fixed SL (not trailing) + R-multiple TPs.
        This is simpler than Donchian — just compare price to SL and TP levels.
        """
        if not self.positions:
            return

        prices = self.fetch_all_prices()
        symbols_to_check = list(self.positions.keys())

        for symbol in symbols_to_check:
            pos = self.positions.get(symbol)
            if not pos:
                continue

            side = pos.get('side', 'LONG')
            price = prices.get(symbol)
            if price is None:
                spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                price = prices.get(spot_sym)
            if price is None:
                continue

            # Update watermarks (for equity tracking, not for trailing stop)
            if side in ('LONG', 'FUTURES_LONG'):
                if price > pos.get('high_watermark', 0):
                    pos['high_watermark'] = price
            elif side == 'SHORT':
                if price < pos.get('low_watermark', float('inf')):
                    pos['low_watermark'] = price

            # Check stop loss (fixed, not trailing)
            sl_check = check_ualgo_stop_loss(pos, price)
            if sl_check['hit']:
                if side == 'LONG':
                    self.execute_sell(symbol, pos['stop_price'], sl_check['exit_reason'])
                elif side == 'FUTURES_LONG':
                    self.execute_futures_long_close(symbol, pos['stop_price'], sl_check['exit_reason'])
                elif side == 'SHORT':
                    self.execute_short_close(symbol, pos['stop_price'], sl_check['exit_reason'])
                continue

            # Check emergency stop
            entry = pos['entry_price']
            if side in ('LONG', 'FUTURES_LONG'):
                loss_pct = (entry - price) / entry * 100
            else:
                loss_pct = (price - entry) / entry * 100

            if loss_pct >= self.emergency_stop_pct:
                reason = f'Emergency stop ({loss_pct:.1f}%)'
                if side == 'LONG':
                    self.execute_sell(symbol, price, reason)
                elif side == 'FUTURES_LONG':
                    self.execute_futures_long_close(symbol, price, reason)
                elif side == 'SHORT':
                    self.execute_short_close(symbol, price, reason)
                continue

            # Check partial TPs
            # For longs, use current high proxy (price); for shorts, use price as low proxy
            tp_check = check_ualgo_partial_tp(pos, price)
            if tp_check['hit']:
                frac = tp_check['fraction']
                reason = tp_check['exit_reason']
                if side == 'LONG':
                    self.execute_sell(symbol, tp_check['tp_price'], reason, fraction=frac)
                elif side == 'FUTURES_LONG':
                    self.execute_futures_long_close(symbol, tp_check['tp_price'], reason, fraction=frac)
                elif side == 'SHORT':
                    self.execute_short_close(symbol, tp_check['tp_price'], reason, fraction=frac)

        self.save_state()

    # ------------------------------------------------------------------
    # DAILY SUMMARY
    # ------------------------------------------------------------------

    def send_daily_summary(self):
        """Send daily portfolio summary via Telegram."""
        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)

        lines = [f"📊 <b>UAlgo Daily Summary</b>"]
        lines.append(f"Equity: ${equity:,.2f}")
        lines.append(f"Cash: ${self.cash:,.2f}")
        lines.append(f"Realized P&L: ${self.total_realized_pnl:+,.2f}")
        lines.append(f"Positions: {self._count_positions()}/{self.max_positions}")

        if self.positions:
            lines.append("\n<b>Open Positions:</b>")
            for symbol, pos in self.positions.items():
                side = pos.get('side', 'LONG')
                entry = pos['entry_price']
                price = prices.get(symbol) or prices.get(pos.get('spot_symbol', ''), entry)
                if side in ('LONG', 'FUTURES_LONG'):
                    pnl_pct = (price - entry) / entry * 100
                else:
                    pnl_pct = (entry - price) / entry * 100
                emoji = "🟢" if pnl_pct >= 0 else "🔴"
                display_sym = PERP_TO_SPOT.get(symbol, symbol)
                lines.append(f"  {emoji} {display_sym} ({side}): {pnl_pct:+.1f}%")

        self._notify("\n".join(lines))

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        """Main bot loop. Daily check at 00:20 UTC, trailing stops every 30 min."""
        print(f"\n{'='*60}")
        print(f"[UALGO] UAlgo Trend Bot starting")
        print(f"  User: {self.user_id[:8]}...")
        print(f"  Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        print(f"  Equity: ${self.total_equity():,.2f}")
        print(f"  Positions: {self._count_positions()}")
        print(f"  Daily check: {DAILY_CHECK_HOUR:02d}:{DAILY_CHECK_MINUTE:02d} UTC")
        print(f"{'='*60}\n")

        # Startup event
        self.sync.sync_event('startup', f'UAlgo bot started ({"paper" if self.paper_trading else "live"})')
        self._notify(
            f"🤖 <b>UAlgo Bot Started</b>\n"
            f"Mode: {'Paper' if self.paper_trading else 'Live'}\n"
            f"Equity: ${self.total_equity():,.2f}\n"
            f"Positions: {self._count_positions()}/{self.max_positions}"
        )

        # Check filters on startup
        is_bull, bull_details = check_bull_filter(self.bull_filter_enabled)
        is_bear, bear_details = check_bear_filter(self.bear_filter_enabled)
        print(f"Bull filter: {bull_details.get('status', 'UNKNOWN')}")
        print(f"Bear filter: {bear_details.get('status', 'UNKNOWN')}")

        last_daily_check = None
        last_summary = None

        while True:
            try:
                now = datetime.now(timezone.utc)

                # Daily signal check
                if now.hour == DAILY_CHECK_HOUR and now.minute >= DAILY_CHECK_MINUTE:
                    if last_daily_check != now.date():
                        self.load_remote_config()
                        self.daily_check()
                        last_daily_check = now.date()

                # Daily summary at 20:00 UTC
                if now.hour == 20 and last_summary != now.date():
                    self.send_daily_summary()
                    last_summary = now.date()

                # Trailing stop monitoring
                if self.positions:
                    self.check_trailing_stops()

                time.sleep(STOP_CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n[UALGO] Bot stopped by user")
                self.save_state()
                self.sync.sync_event('shutdown', 'UAlgo bot stopped by user')
                break
            except Exception as e:
                self.logger.error(f"[UALGO] Main loop error: {e}")
                self.sync.sync_event('error', f'Main loop error: {e}')
                time.sleep(60)


# ============================================================================
# BOT MANAGER (multi-tenant)
# ============================================================================

class UAlgoBotManager:
    """Manages multiple UAlgoUserBot instances, one per active user."""

    def __init__(self):
        self.bots = {}
        self.logger, _ = setup_logging()

    def load_active_users(self):
        """Query Supabase for users with UAlgo bot enabled."""
        sync = SupabaseSync(strategy='ualgo')
        rows = sync._get('bot_config', filters={'strategy': 'ualgo', 'bot_enabled': 'true'})
        if not rows:
            return []
        return [{'user_id': r['user_id'], 'trading_mode': r.get('trading_mode', 'paper')}
                for r in rows]

    def refresh_bots(self):
        """Add bots for new active users, remove bots for disabled users."""
        users = self.load_active_users()
        active_ids = {u['user_id'] for u in users}

        # Add new users
        for user in users:
            uid = user['user_id']
            if uid not in self.bots:
                paper = user['trading_mode'] != 'live'
                self.logger.info(f"[UALGO] Adding bot for user {uid[:8]}... (paper={paper})")
                try:
                    bot = UAlgoUserBot(uid, paper_trading=paper)
                    self.bots[uid] = bot
                    bot.sync.sync_event('startup', f'UAlgo bot started ({"paper" if paper else "live"})')
                    bot._notify(
                        f"🤖 <b>UAlgo Bot Started</b>\n"
                        f"Mode: {'Paper' if paper else 'Live'}\n"
                        f"Equity: ${bot.total_equity():,.2f}\n"
                        f"Positions: {bot._count_positions()}/{bot.max_positions}"
                    )
                except Exception as e:
                    self.logger.error(f"[UALGO] Failed to create bot for {uid[:8]}: {e}")

        # Remove disabled users
        for uid in list(self.bots.keys()):
            if uid not in active_ids:
                self.logger.info(f"[UALGO] Removing bot for user {uid[:8]}...")
                try:
                    self.bots[uid].save_state()
                except Exception:
                    pass
                self.bots.pop(uid, None)

    def run(self):
        """Main manager loop."""
        print(f"\n{'='*60}")
        print(f"[UALGO] UAlgo Bot Manager starting")
        print(f"{'='*60}\n")

        self.refresh_bots()
        print(f"[UALGO] Active bots: {len(self.bots)}")

        last_daily_check = None
        last_summary = None
        last_refresh = datetime.now(timezone.utc)

        while True:
            try:
                now = datetime.now(timezone.utc)

                # Refresh bots every 5 minutes
                if (now - last_refresh).total_seconds() >= 300:
                    self.refresh_bots()
                    last_refresh = now

                # Daily signal check
                if now.hour == DAILY_CHECK_HOUR and now.minute >= DAILY_CHECK_MINUTE:
                    if last_daily_check != now.date():
                        for uid, bot in self.bots.items():
                            try:
                                bot.load_remote_config()
                                bot.daily_check()
                            except Exception as e:
                                self.logger.error(f"[UALGO] Daily check failed for {uid[:8]}: {e}")
                        last_daily_check = now.date()

                # Daily summary at 20:00 UTC
                if now.hour == 20 and last_summary != now.date():
                    for uid, bot in self.bots.items():
                        try:
                            bot.send_daily_summary()
                        except Exception as e:
                            self.logger.warning(f"[UALGO] Summary failed for {uid[:8]}: {e}")
                    last_summary = now.date()

                # Trailing stop monitoring
                for uid, bot in self.bots.items():
                    if bot.positions:
                        try:
                            bot.check_trailing_stops()
                        except Exception as e:
                            self.logger.warning(f"[UALGO] Stop check failed for {uid[:8]}: {e}")

                time.sleep(STOP_CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n[UALGO] Manager stopped by user")
                for uid, bot in self.bots.items():
                    bot.save_state()
                break
            except Exception as e:
                self.logger.error(f"[UALGO] Manager loop error: {e}")
                time.sleep(60)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    manager = UAlgoBotManager()
    manager.run()


if __name__ == "__main__":
    main()
