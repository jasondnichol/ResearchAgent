"""Intraday MTF Momentum Trading Bot — Hourly Candles on CFM Futures
==================================================================

Standalone intraday bot (runs alongside the daily Donchian bot):
  - Strategy: Multi-TF Momentum with ADX regime filter
  - Timeframe: 1-hour candles, checks every hour
  - Execution: Coinbase CFM perpetual futures (paper mode default)
  - ADX filter: only trades when BTC daily ADX >= 25 (trending)

Architecture:
  - Hourly signal check at :05 past (after candle close)
  - Trailing stop monitoring every 15 minutes
  - Max 3 concurrent positions, 1.5% risk per trade, 2x leverage
  - 3% daily loss circuit breaker
  - Daily ADX regime computed once per day (using prior day's data)
  - Telegram notifications for all trades + daily summary
  - Crash recovery via state file

Trade coins: ETH, SOL, XRP, SUI, LINK, ADA, DOGE, NEAR (8 coins)
(excludes BTC — used as regime indicator, not traded intraday)

Usage:
    # Paper trading (default):
    venv/Scripts/python intraday_momentum_bot.py

    # Can run alongside donchian_multicoin_bot.py in a separate screen:
    screen -S intraday
"""

import json
import time
import os
import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from coinbase_futures import CoinbaseFuturesClient, SPOT_TO_PERP, PERP_TO_SPOT
from notify import send_telegram, setup_logging
from supabase_sync import SupabaseSync

# ============================================================================
# CONFIGURATION
# ============================================================================

# Trade coins (everything except BTC — BTC is the regime indicator)
TRADE_COINS = [
    'ETH-USD', 'SOL-USD', 'XRP-USD', 'SUI-USD',
    'LINK-USD', 'ADA-USD', 'DOGE-USD', 'NEAR-USD',
]

# Strategy parameters (from walk-forward validated MTF + ADX>=25)
STRATEGY_PARAMS = {
    'adx_threshold': 25,   # Only trade when BTC daily ADX >= 25
    'rsi_entry': 40,       # Long: RSI < 40 dip; Short: RSI > 60 spike
    'atr_stop': 2.0,       # Initial stop: 2x ATR from entry
    'atr_trailing': 1.5,   # Trailing stop: 1.5x ATR from high/low watermark
    'atr_target': 3.0,     # Target: 3x ATR from entry
    'max_hold': 24,        # Max hold: 24 candles (24 hours)
    'vol_threshold': 1.2,  # Volume > 1.2x 20-bar average
}

# Portfolio settings
MAX_POSITIONS = 3
RISK_PER_TRADE_PCT = 1.5  # 1.5% equity risk per trade
STARTING_CAPITAL = 10000.0
LEVERAGE = 2.0
FEE_RATE = 0.0006         # Coinbase CFM: 0.06% per side
MAX_DAILY_LOSS_PCT = 3.0  # Circuit breaker: stop trading if down 3% in a day

# Timing
SIGNAL_CHECK_MINUTE = 5       # Check at :05 past each hour
STOP_CHECK_INTERVAL = 900     # 15 minutes between trailing stop checks
DAILY_SUMMARY_HOUR = 20       # Daily summary at 20:00 UTC (noon PST)

# Indicator periods (for hourly bars)
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
VOL_MA_PERIOD = 20
DAILY_ADX_PERIOD = 14
DAILY_LOOKBACK = 60  # Days of daily candles for ADX calculation

STATE_FILE = 'intraday_state.json'
COINBASE_API = "https://api.exchange.coinbase.com"


# ============================================================================
# INTRADAY MOMENTUM BOT
# ============================================================================

class IntradayMomentumBot:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading

        # Portfolio state
        self.cash = STARTING_CAPITAL
        self.positions = {}  # symbol -> position dict
        self.trades_log = []
        self.total_realized_pnl = 0.0

        # Daily state
        self.daily_adx = None          # Current BTC daily ADX value
        self.daily_trend = None        # 1 = up, -1 = down, 0 = flat
        self.daily_regime_date = None  # Date when regime was last computed
        self.day_pnl = 0.0             # Running P&L for today
        self.day_locked = False        # True if daily loss limit hit
        self.current_day = None

        # Hourly indicator cache per coin
        self.coin_indicators = {}  # symbol -> {rsi, atr, ema_50, vol_ratio, ...}

        # Futures client
        self.futures_client = CoinbaseFuturesClient(paper_mode=paper_trading)

        # Logging
        self.logger, self.trade_logger = setup_logging()

        # Supabase sync
        self.sync = SupabaseSync()

        # Load saved state
        self.load_state()

    # ------------------------------------------------------------------
    # STATE PERSISTENCE
    # ------------------------------------------------------------------

    def save_state(self):
        """Save positions and portfolio state for crash recovery."""
        state = {
            'cash': self.cash,
            'total_realized_pnl': self.total_realized_pnl,
            'daily_adx': self.daily_adx,
            'daily_trend': self.daily_trend,
            'daily_regime_date': self.daily_regime_date.isoformat() if self.daily_regime_date else None,
            'day_pnl': self.day_pnl,
            'current_day': self.current_day.isoformat() if isinstance(self.current_day, datetime) else str(self.current_day) if self.current_day else None,
            'positions': {},
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        for symbol, pos in self.positions.items():
            state['positions'][symbol] = {
                'side': pos['side'],
                'entry_price': pos['entry_price'],
                'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time']),
                'stop_price': pos['stop_price'],
                'target_price': pos['target_price'],
                'size_usd': pos['size_usd'],
                'atr': pos['atr'],
                'hold_candles': pos['hold_candles'],
                'high_watermark': pos.get('high_watermark', pos['entry_price']),
                'low_watermark': pos.get('low_watermark', pos['entry_price']),
            }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load saved state on restart."""
        if not os.path.exists(STATE_FILE):
            return

        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            self.cash = state.get('cash', STARTING_CAPITAL)
            self.total_realized_pnl = state.get('total_realized_pnl', 0)
            self.daily_adx = state.get('daily_adx')
            self.daily_trend = state.get('daily_trend')
            if state.get('daily_regime_date'):
                self.daily_regime_date = datetime.fromisoformat(state['daily_regime_date']).date() if isinstance(state['daily_regime_date'], str) else state['daily_regime_date']
            self.day_pnl = state.get('day_pnl', 0)

            for symbol, pos_data in state.get('positions', {}).items():
                self.positions[symbol] = {
                    'side': pos_data['side'],
                    'entry_price': pos_data['entry_price'],
                    'entry_time': datetime.fromisoformat(pos_data['entry_time']),
                    'stop_price': pos_data['stop_price'],
                    'target_price': pos_data['target_price'],
                    'size_usd': pos_data['size_usd'],
                    'atr': pos_data['atr'],
                    'hold_candles': pos_data.get('hold_candles', 0),
                    'high_watermark': pos_data.get('high_watermark', pos_data['entry_price']),
                    'low_watermark': pos_data.get('low_watermark', pos_data['entry_price']),
                }
            if self.positions:
                longs = sum(1 for p in self.positions.values() if p['side'] == 'long')
                shorts = sum(1 for p in self.positions.values() if p['side'] == 'short')
                print(f"Restored {len(self.positions)} intraday positions ({longs}L/{shorts}S)")
                for sym, pos in self.positions.items():
                    side_tag = "S" if pos['side'] == 'short' else "L"
                    print(f"  {sym} [{side_tag}]: entry ${pos['entry_price']:,.2f}, size ${pos['size_usd']:,.0f}")
        except Exception as e:
            print(f"Warning: Could not load intraday state: {e}")

    # ------------------------------------------------------------------
    # PRICE & CANDLE FETCHING
    # ------------------------------------------------------------------

    def fetch_current_price(self, symbol):
        """Get current spot price."""
        url = f"{COINBASE_API}/products/{symbol}/ticker"
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['price'])

    def fetch_all_prices(self):
        """Get current prices for all trade coins."""
        prices = {}
        for symbol in TRADE_COINS:
            try:
                prices[symbol] = self.fetch_current_price(symbol)
                time.sleep(0.15)
            except Exception as e:
                self.logger.warning(f"Price fetch failed for {symbol}: {e}")
        return prices

    def fetch_hourly_candles(self, symbol, count=250):
        """Fetch hourly candles from Coinbase."""
        url = f"{COINBASE_API}/products/{symbol}/candles"
        params = {'granularity': 3600}  # 1 hour
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            if not data or not isinstance(data, list):
                return None

            df = pd.DataFrame(data[:count], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df = df.sort_values('time').reset_index(drop=True)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            self.logger.error(f"Candle fetch failed for {symbol}: {e}")
            return None

    def fetch_daily_candles(self, symbol='BTC-USD', count=DAILY_LOOKBACK):
        """Fetch daily candles for regime computation."""
        url = f"{COINBASE_API}/products/{symbol}/candles"
        params = {'granularity': 86400}
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            if not data or not isinstance(data, list):
                return None

            df = pd.DataFrame(data[:count], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df = df.sort_values('time').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            self.logger.error(f"Daily candle fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    # INDICATOR COMPUTATION
    # ------------------------------------------------------------------

    def compute_hourly_indicators(self, df):
        """Compute all hourly indicators on a candle DataFrame."""
        if df is None or len(df) < EMA_TREND:
            return None

        # EMAs
        ema_fast = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
        ema_mid = df['close'].ewm(span=EMA_MID, adjust=False).mean()
        ema_slow = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
        ema_trend = df['close'].ewm(span=EMA_TREND, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD).mean()

        # Volume ratio
        vol_ma = df['volume'].rolling(VOL_MA_PERIOD).mean()
        vol_ratio = df['volume'] / vol_ma.replace(0, 1)

        # Return the latest values
        i = len(df) - 1
        return {
            'close': float(df['close'].iloc[i]),
            'high': float(df['high'].iloc[i]),
            'low': float(df['low'].iloc[i]),
            'ema_9': float(ema_fast.iloc[i]),
            'ema_21': float(ema_mid.iloc[i]),
            'ema_50': float(ema_slow.iloc[i]),
            'ema_200': float(ema_trend.iloc[i]),
            'rsi': float(rsi.iloc[i]),
            'rsi_prev': float(rsi.iloc[i-1]) if i > 0 else float(rsi.iloc[i]),
            'atr': float(atr.iloc[i]),
            'vol_ratio': float(vol_ratio.iloc[i]),
            'time': df['time'].iloc[i],
        }

    def compute_daily_regime(self):
        """Compute BTC daily ADX and trend direction.
        Called once per day. Uses prior day's close (no look-ahead).
        """
        today = datetime.now(timezone.utc).date()
        if self.daily_regime_date == today and self.daily_adx is not None:
            return  # Already computed today

        df = self.fetch_daily_candles('BTC-USD', count=DAILY_LOOKBACK)
        if df is None or len(df) < 30:
            self.logger.warning("Not enough daily data for regime computation")
            return

        # Use second-to-last row (prior day's close, today's candle is incomplete)
        # ADX computation
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm_vals = plus_dm.values.copy()
        minus_dm_vals = minus_dm.values.copy()

        # Zero out DM where the other is larger
        for j in range(len(plus_dm_vals)):
            if plus_dm_vals[j] <= minus_dm_vals[j]:
                plus_dm_vals[j] = 0
            if minus_dm_vals[j] <= plus_dm_vals[j]:
                minus_dm_vals[j] = 0

        smooth_tr = pd.Series(tr).rolling(DAILY_ADX_PERIOD).mean()
        smooth_plus = pd.Series(plus_dm_vals).rolling(DAILY_ADX_PERIOD).mean()
        smooth_minus = pd.Series(minus_dm_vals).rolling(DAILY_ADX_PERIOD).mean()

        plus_di = 100 * smooth_plus / smooth_tr.replace(0, np.inf)
        minus_di = 100 * smooth_minus / smooth_tr.replace(0, np.inf)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.inf)
        adx = dx.rolling(DAILY_ADX_PERIOD).mean()

        # EMA(21) for daily trend
        ema_21 = close.ewm(span=21, adjust=False).mean()

        # Use second-to-last values (prior completed day)
        idx = len(df) - 2  # Prior day (today's bar is incomplete)
        if idx < 0:
            idx = len(df) - 1

        self.daily_adx = float(adx.iloc[idx]) if not pd.isna(adx.iloc[idx]) else 0
        btc_close = float(close.iloc[idx])
        btc_ema = float(ema_21.iloc[idx])
        self.daily_trend = 1 if btc_close > btc_ema else (-1 if btc_close < btc_ema else 0)
        self.daily_regime_date = today

        regime = "TRENDING" if self.daily_adx >= STRATEGY_PARAMS['adx_threshold'] else "RANGING"
        trend_str = "UP" if self.daily_trend == 1 else ("DOWN" if self.daily_trend == -1 else "FLAT")
        print(f"  Daily regime: ADX={self.daily_adx:.1f} ({regime}) | Trend: {trend_str} | BTC: ${btc_close:,.0f}")
        self.logger.info(f"Daily regime: ADX={self.daily_adx:.1f} ({regime}), trend={trend_str}")

    # ------------------------------------------------------------------
    # SIGNAL GENERATION
    # ------------------------------------------------------------------

    def check_signal(self, symbol, indicators):
        """Check if we have an MTF momentum signal for this coin.

        Long signal: daily trend UP + hourly RSI dip (< rsi_entry) + RSI bouncing +
                     price > EMA(50) + volume > threshold + ADX >= threshold
        Short signal: daily trend DOWN + hourly RSI spike (> 100-rsi_entry) + RSI dropping +
                      price < EMA(50) + volume > threshold + ADX >= threshold
        """
        if indicators is None:
            return None

        params = STRATEGY_PARAMS

        # ADX regime filter
        if self.daily_adx is None or self.daily_adx < params['adx_threshold']:
            return None

        rsi = indicators['rsi']
        rsi_prev = indicators['rsi_prev']
        close = indicators['close']
        ema_50 = indicators['ema_50']
        vol_ratio = indicators['vol_ratio']
        atr = indicators['atr']

        if atr <= 0 or pd.isna(atr):
            return None

        # Long signal
        if (self.daily_trend == 1 and
            rsi < params['rsi_entry'] and
            rsi > rsi_prev and  # RSI bouncing
            close > ema_50 and
            vol_ratio > params['vol_threshold']):

            stop = close - params['atr_stop'] * atr
            target = close + params['atr_target'] * atr
            return {
                'side': 'long', 'entry': close,
                'stop': stop, 'target': target, 'atr': atr,
                'reason': f"MTF Long: RSI={rsi:.0f} bouncing, ADX={self.daily_adx:.0f}, Vol={vol_ratio:.1f}x",
            }

        # Short signal
        rsi_short_entry = 100 - params['rsi_entry']
        if (self.daily_trend == -1 and
            rsi > rsi_short_entry and
            rsi < rsi_prev and  # RSI dropping
            close < ema_50 and
            vol_ratio > params['vol_threshold']):

            stop = close + params['atr_stop'] * atr
            target = close - params['atr_target'] * atr
            return {
                'side': 'short', 'entry': close,
                'stop': stop, 'target': target, 'atr': atr,
                'reason': f"MTF Short: RSI={rsi:.0f} dropping, ADX={self.daily_adx:.0f}, Vol={vol_ratio:.1f}x",
            }

        return None

    # ------------------------------------------------------------------
    # TRADE EXECUTION
    # ------------------------------------------------------------------

    def total_equity(self, prices=None):
        """Calculate total equity (cash + position value)."""
        equity = self.cash
        if not prices:
            prices = self.fetch_all_prices()
        for symbol, pos in self.positions.items():
            current = prices.get(symbol, pos['entry_price'])
            if pos['side'] == 'long':
                pnl_pct = (current - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - current) / pos['entry_price']
            leveraged_pnl = pnl_pct * LEVERAGE
            equity += pos['size_usd'] * leveraged_pnl
        return equity

    def open_position(self, symbol, signal, prices=None):
        """Open a new position based on signal."""
        entry = signal['entry']
        stop = signal['stop']
        side = signal['side']

        # Position sizing: risk-based
        stop_distance = abs(entry - stop) / entry
        if stop_distance <= 0:
            return

        equity = self.total_equity(prices)
        risk_amount = equity * (RISK_PER_TRADE_PCT / 100)
        position_size = risk_amount / stop_distance
        position_size = min(position_size, equity * 0.95 / LEVERAGE)

        if position_size < 10:  # Minimum $10 position
            return

        # Entry fee
        entry_fee = position_size * FEE_RATE
        self.cash -= entry_fee

        # Perp symbol for futures
        perp_symbol = SPOT_TO_PERP.get(symbol, symbol)

        self.positions[symbol] = {
            'side': side,
            'entry_price': entry,
            'entry_time': datetime.now(timezone.utc),
            'stop_price': stop,
            'target_price': signal['target'],
            'size_usd': position_size,
            'atr': signal['atr'],
            'hold_candles': 0,
            'high_watermark': entry,
            'low_watermark': entry,
            'perp_symbol': perp_symbol,
        }

        side_emoji = "🟢" if side == 'long' else "🔴"
        side_tag = "LONG" if side == 'long' else "SHORT"
        msg = (
            f"{side_emoji} <b>INTRADAY {side_tag} OPEN</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🪙 {symbol}\n"
            f"📍 Entry: ${entry:,.4f}\n"
            f"🛑 Stop: ${stop:,.4f} ({stop_distance*100:.1f}%)\n"
            f"🎯 Target: ${signal['target']:,.4f}\n"
            f"💰 Size: ${position_size:,.0f} ({LEVERAGE}x)\n"
            f"📊 {signal['reason']}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        send_telegram(msg)
        self.logger.info(f"INTRADAY {side_tag} {symbol}: entry=${entry:.4f}, stop=${stop:.4f}, size=${position_size:.0f}")

        # Sync to Supabase
        action = 'INTRADAY_BUY' if side == 'long' else 'INTRADAY_SHORT'
        self.sync.sync_trade(
            symbol=symbol, side=f'INTRADAY_{side_tag}',
            action=action, entry_price=entry,
            size_usd=position_size)
        self.sync.sync_position_update(symbol, self.positions[symbol])
        self.save_state()

    def close_position(self, symbol, exit_price, reason):
        """Close an existing position."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        side = pos['side']
        if side == 'long':
            gross_pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            gross_pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        leveraged_pnl = gross_pnl_pct * LEVERAGE
        net_pnl_pct = leveraged_pnl - 2 * FEE_RATE
        net_pnl_dollar = pos['size_usd'] * net_pnl_pct

        self.cash += net_pnl_dollar + pos['size_usd'] * FEE_RATE  # Return capital + PnL (fee already deducted)
        self.total_realized_pnl += net_pnl_dollar
        self.day_pnl += net_pnl_dollar

        hold_hours = pos['hold_candles']
        win_loss = "WIN" if net_pnl_dollar > 0 else "LOSS"
        side_tag = "LONG" if side == 'long' else "SHORT"
        emoji = "✅" if net_pnl_dollar > 0 else "❌"

        msg = (
            f"{emoji} <b>INTRADAY {side_tag} CLOSED — {win_loss}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🪙 {symbol}\n"
            f"📍 Entry: ${pos['entry_price']:,.4f} → Exit: ${exit_price:,.4f}\n"
            f"💰 P&L: ${net_pnl_dollar:+,.0f} ({net_pnl_pct*100:+.2f}%)\n"
            f"⏱ Hold: {hold_hours}h\n"
            f"📋 {reason}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        send_telegram(msg)
        self.logger.info(f"INTRADAY CLOSE {symbol} [{side_tag}]: PnL ${net_pnl_dollar:+.0f} ({net_pnl_pct*100:+.2f}%), {reason}")

        # Log trade
        self.trades_log.append({
            'symbol': symbol, 'side': side,
            'entry_price': pos['entry_price'], 'exit_price': exit_price,
            'entry_time': pos['entry_time'].isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'net_pnl_dollar': net_pnl_dollar,
            'net_pnl_pct': net_pnl_pct * 100,
            'hold_candles': hold_hours,
            'reason': reason,
        })

        # Sync to Supabase
        action = 'INTRADAY_SELL' if side == 'long' else 'INTRADAY_COVER'
        self.sync.sync_trade(
            symbol=symbol, side=f'INTRADAY_{side_tag}',
            action=action, exit_price=exit_price,
            size_usd=pos['size_usd'],
            pnl_usd=net_pnl_dollar, pnl_pct=net_pnl_pct * 100,
            exit_reason=reason,
            hold_days=pos['hold_candles'] / 24)

        del self.positions[symbol]
        self.save_state()

    # ------------------------------------------------------------------
    # HOURLY SIGNAL CHECK
    # ------------------------------------------------------------------

    def hourly_check(self):
        """Main hourly signal check. Called once per hour."""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}")
        print(f"  HOURLY CHECK — {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")

        # Reset daily state at day boundary
        today = now.date()
        if today != self.current_day:
            self.current_day = today
            self.day_pnl = 0.0
            self.day_locked = False
            print(f"  New day — daily P&L reset")

        # Check daily loss limit
        if self.day_locked:
            print(f"  LOCKED — daily loss limit hit (${self.day_pnl:+,.0f})")
            return

        # Compute daily regime (once per day)
        self.compute_daily_regime()

        regime = "TRENDING" if self.daily_adx and self.daily_adx >= STRATEGY_PARAMS['adx_threshold'] else "RANGING"
        trend_str = "UP" if self.daily_trend == 1 else ("DOWN" if self.daily_trend == -1 else "FLAT")
        print(f"  Regime: {regime} (ADX={(self.daily_adx or 0):.1f}) | Trend: {trend_str}")
        print(f"  Positions: {len(self.positions)}/{MAX_POSITIONS} | Day P&L: ${self.day_pnl:+,.0f}")

        # Fetch prices for all coins
        prices = self.fetch_all_prices()

        # Step 1: Check exits on existing positions
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos:
                continue

            current_price = prices.get(symbol)
            if current_price is None:
                continue

            pos['hold_candles'] += 1
            self._check_exit(symbol, pos, current_price)

        # Check if daily loss hit after exits
        equity = self.total_equity(prices)
        if self.day_pnl / STARTING_CAPITAL < -(MAX_DAILY_LOSS_PCT / 100):
            self.day_locked = True
            print(f"  DAILY LOSS LIMIT HIT: ${self.day_pnl:+,.0f} ({self.day_pnl/STARTING_CAPITAL*100:+.1f}%)")
            send_telegram(f"🚨 <b>INTRADAY DAILY LOSS LIMIT</b>\nP&L: ${self.day_pnl:+,.0f}\nTrading paused until tomorrow.")
            return

        # Step 2: Check for new entries (only if regime is trending)
        if regime == "RANGING":
            print(f"  Skipping entries — ranging market")
            return

        if len(self.positions) >= MAX_POSITIONS:
            print(f"  Max positions reached ({MAX_POSITIONS})")
            return

        for symbol in TRADE_COINS:
            if symbol in self.positions:
                continue
            if len(self.positions) >= MAX_POSITIONS:
                break

            # Fetch hourly candles and compute indicators
            df = self.fetch_hourly_candles(symbol)
            indicators = self.compute_hourly_indicators(df)
            if indicators is None:
                continue

            signal = self.check_signal(symbol, indicators)
            if signal:
                print(f"  SIGNAL: {symbol} — {signal['reason']}")
                self.open_position(symbol, signal, prices)
                time.sleep(0.5)  # Rate limiting

        print(f"  Equity: ${equity:,.0f} | Day P&L: ${self.day_pnl:+,.0f}")

    # ------------------------------------------------------------------
    # EXIT CHECKING (used by both hourly and trailing stop checks)
    # ------------------------------------------------------------------

    def _check_exit(self, symbol, pos, current_price):
        """Check if a position should be closed."""
        side = pos['side']
        params = STRATEGY_PARAMS

        exit_reason = None
        exit_price = current_price

        if side == 'long':
            # Stop hit
            if current_price <= pos['stop_price']:
                exit_reason = 'Trailing stop hit'
                exit_price = pos['stop_price']
            # Target hit
            elif current_price >= pos['target_price']:
                exit_reason = 'Target reached'
                exit_price = pos['target_price']
            else:
                # Ratchet trailing stop
                if current_price > pos['high_watermark']:
                    pos['high_watermark'] = current_price
                    new_stop = current_price - params['atr_trailing'] * pos['atr']
                    if new_stop > pos['stop_price']:
                        old_stop = pos['stop_price']
                        pos['stop_price'] = new_stop
                        self.save_state()
            # Max hold
            if not exit_reason and pos['hold_candles'] >= params['max_hold']:
                exit_reason = f"Max hold ({params['max_hold']}h)"

        elif side == 'short':
            if current_price >= pos['stop_price']:
                exit_reason = 'Trailing stop hit'
                exit_price = pos['stop_price']
            elif current_price <= pos['target_price']:
                exit_reason = 'Target reached'
                exit_price = pos['target_price']
            else:
                if current_price < pos['low_watermark']:
                    pos['low_watermark'] = current_price
                    new_stop = current_price + params['atr_trailing'] * pos['atr']
                    if new_stop < pos['stop_price']:
                        pos['stop_price'] = new_stop
                        self.save_state()
            if not exit_reason and pos['hold_candles'] >= params['max_hold']:
                exit_reason = f"Max hold ({params['max_hold']}h)"

        if exit_reason:
            self.close_position(symbol, exit_price, exit_reason)

    # ------------------------------------------------------------------
    # TRAILING STOP CHECK (between hourly signals)
    # ------------------------------------------------------------------

    def check_trailing_stops(self):
        """Lightweight price check for trailing stops between hourly checks."""
        if not self.positions:
            return

        now = datetime.now(timezone.utc).strftime('%H:%M')
        prices = self.fetch_all_prices()

        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos:
                continue

            current_price = prices.get(symbol)
            if current_price is None:
                continue

            self._check_exit(symbol, pos, current_price)

            # If still open, print status
            if symbol in self.positions:
                side = pos['side']
                if side == 'long':
                    pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                else:
                    pnl = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
                side_tag = "L" if side == 'long' else "S"
                print(f"  [{now}] {symbol} [{side_tag}]: ${current_price:,.4f} ({pnl:+.1f}%) "
                      f"Stop: ${pos['stop_price']:,.4f}")

    # ------------------------------------------------------------------
    # DAILY SUMMARY
    # ------------------------------------------------------------------

    def send_daily_summary(self):
        """Send intraday portfolio summary via Telegram."""
        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

        # Today's trades
        today = datetime.now(timezone.utc).date()
        today_trades = [t for t in self.trades_log
                        if datetime.fromisoformat(t['exit_time']).date() == today]
        wins = sum(1 for t in today_trades if t['net_pnl_dollar'] > 0)
        losses = len(today_trades) - wins

        regime = "TRENDING" if self.daily_adx and self.daily_adx >= STRATEGY_PARAMS['adx_threshold'] else "RANGING"

        pos_lines = []
        for sym, pos in self.positions.items():
            current = prices.get(sym, pos['entry_price'])
            if pos['side'] == 'long':
                pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
            else:
                pnl = ((pos['entry_price'] - current) / pos['entry_price']) * 100
            side_tag = "L" if pos['side'] == 'long' else "S"
            emoji = "🟢" if pnl >= 0 else "🔴"
            pos_lines.append(f"{emoji} {sym} [{side_tag}]: ${current:,.4f} ({pnl:+.1f}%, {pos['hold_candles']}h)")

        pos_text = "\n".join(pos_lines) if pos_lines else "No open positions"
        return_emoji = "🟢" if total_return >= 0 else "🔴"

        msg = (
            f"📋 <b>INTRADAY DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: <b>${equity:,.0f}</b>\n"
            f"{return_emoji} Total Return: <b>{total_return:+.1f}%</b>\n"
            f"📊 Today: {wins}W/{losses}L | P&L: ${self.day_pnl:+,.0f}\n"
            f"📈 Total trades: {len(self.trades_log)}\n"
            f"🔮 Regime: {regime} (ADX={(self.daily_adx or 0):.1f})\n"
            f"\n<b>Open Positions:</b>\n{pos_text}\n\n"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'} | Strategy: MTF Momentum\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        """Main trading loop."""
        print("=" * 60)
        print("INTRADAY MTF MOMENTUM BOT")
        print("=" * 60)
        print(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'} TRADING")
        print(f"Strategy: MTF Momentum + ADX>={STRATEGY_PARAMS['adx_threshold']} filter")
        print(f"Coins: {', '.join(TRADE_COINS)}")
        print(f"Max positions: {MAX_POSITIONS}")
        print(f"Risk: {RISK_PER_TRADE_PCT}% per trade | Leverage: {LEVERAGE}x")
        print(f"Fee: {FEE_RATE*100:.2f}%/side")
        print(f"Stops: {STRATEGY_PARAMS['atr_stop']}x ATR initial, {STRATEGY_PARAMS['atr_trailing']}x trailing")
        print(f"Target: {STRATEGY_PARAMS['atr_target']}x ATR | Max hold: {STRATEGY_PARAMS['max_hold']}h")
        print(f"Daily loss limit: {MAX_DAILY_LOSS_PCT}%")
        print(f"Signal check: :{SIGNAL_CHECK_MINUTE:02d} past each hour")
        print(f"Stop check: every {STOP_CHECK_INTERVAL // 60} min")
        print("=" * 60)

        # Initial regime check
        self.compute_daily_regime()
        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)

        regime = "TRENDING" if self.daily_adx and self.daily_adx >= STRATEGY_PARAMS['adx_threshold'] else "RANGING"
        trend_str = "UP" if self.daily_trend == 1 else ("DOWN" if self.daily_trend == -1 else "FLAT")

        msg = (
            f"🚀 <b>INTRADAY BOT STARTED</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Strategy: MTF Momentum (1h)\n"
            f"🔮 Regime: {regime} (ADX={(self.daily_adx or 0):.1f}, {trend_str})\n"
            f"🪙 Coins: {len(TRADE_COINS)}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"📈 Positions: {len(self.positions)}/{MAX_POSITIONS}\n"
            f"⚡ Leverage: {LEVERAGE}x | Risk: {RISK_PER_TRADE_PCT}%\n"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)

        last_hourly_check = None
        last_summary = None

        try:
            while True:
                now = datetime.now(timezone.utc)

                # Hourly signal check (at :05 past each hour)
                current_hour_key = now.strftime('%Y-%m-%d-%H')
                if (now.minute >= SIGNAL_CHECK_MINUTE and
                        last_hourly_check != current_hour_key):
                    self.hourly_check()
                    last_hourly_check = current_hour_key

                # Daily summary at 20:00 UTC
                if now.hour == DAILY_SUMMARY_HOUR and last_summary != now.date():
                    self.send_daily_summary()
                    last_summary = now.date()

                # Trailing stop monitoring
                if self.positions:
                    print(f"\n--- Stop check ({now.strftime('%H:%M UTC')}) ---")
                    self.check_trailing_stops()

                # Sleep until next check
                next_min = STOP_CHECK_INTERVAL // 60
                print(f"\nNext check in {next_min} min...")
                time.sleep(STOP_CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nIntraday bot stopped by user")
            self.save_state()
            equity = self.total_equity()
            total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
            print(f"\nFinal equity: ${equity:,.0f} ({total_return:+.1f}%)")
            print(f"Total trades: {len(self.trades_log)}")
            print(f"Open positions: {len(self.positions)}")

            # Send shutdown notification
            msg = (
                f"🛑 <b>INTRADAY BOT STOPPED</b>\n"
                f"💼 Equity: ${equity:,.0f} ({total_return:+.1f}%)\n"
                f"📊 Trades: {len(self.trades_log)}\n"
                f"📈 Positions: {len(self.positions)}"
            )
            send_telegram(msg)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import sys

    if '--scan' in sys.argv:
        # Quick scan mode: check all coins for signals, print status, exit
        os.environ.setdefault('TELEGRAM_BOT_TOKEN', '')  # Suppress Telegram in scan mode
        bot = IntradayMomentumBot(paper_trading=True)
        bot.compute_daily_regime()

        regime = "TRENDING" if bot.daily_adx and bot.daily_adx >= STRATEGY_PARAMS['adx_threshold'] else "RANGING"
        trend_str = "UP" if bot.daily_trend == 1 else ("DOWN" if bot.daily_trend == -1 else "FLAT")
        looking_for = "SHORTS" if bot.daily_trend == -1 else ("LONGS" if bot.daily_trend == 1 else "NOTHING")

        print(f"\nRegime: {regime} (ADX={bot.daily_adx:.1f}) | Trend: {trend_str} | Looking for: {looking_for}")
        print(f"Equity: ${bot.total_equity():,.0f} | Positions: {len(bot.positions)}/{MAX_POSITIONS}\n")

        for symbol in TRADE_COINS:
            df = bot.fetch_hourly_candles(symbol)
            ind = bot.compute_hourly_indicators(df)
            if ind is None:
                print(f"  {symbol}: insufficient data")
                continue
            signal = bot.check_signal(symbol, ind)
            ema_pos = "above" if ind['close'] > ind['ema_50'] else "below"
            if signal:
                print(f"  {symbol}: ** SIGNAL ** {signal['side'].upper()} — {signal['reason']}")
            else:
                print(f"  {symbol}: no signal (RSI={ind['rsi']:.1f}, Vol={ind['vol_ratio']:.2f}, {ema_pos} EMA50)")

    elif '--once' in sys.argv:
        # Single cycle mode: run one hourly check, print results, exit
        os.environ.setdefault('TELEGRAM_BOT_TOKEN', '')
        bot = IntradayMomentumBot(paper_trading=True)
        bot.hourly_check()
        print(f"\nEquity: ${bot.total_equity():,.0f} | Positions: {len(bot.positions)}")

    else:
        bot = IntradayMomentumBot(paper_trading=True)
        bot.run()
