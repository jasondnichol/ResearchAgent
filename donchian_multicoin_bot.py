"""Multi-Coin Donchian Channel Breakout Trading Bot — Daily Candles

Dual-mode production bot: long (spot) in bull markets, short (perpetual futures) in bear markets.
Runs the Donchian breakout strategy on liquid Coinbase coins with realistic position sizing.

Architecture:
  - Daily signal check at 00:15 UTC (after daily candle close)
  - Trailing stop monitoring every 30 minutes
  - Max 4 concurrent positions (longs + shorts shared pool), 2% risk per trade
  - Longs: 4x ATR trailing stop, pyramiding at +15%, partial TP at +10%/+20%
  - Shorts: 2x ATR inverted trailing stop, partial TP at -10%/-20%, 30-day max hold
  - Bull filter (golden cross): gates long entries
  - Bear filter (death cross): gates short entries
  - Shorts via Coinbase CFM perpetual futures (paper_mode default)
  - Telegram notifications for all trades

Long coins: BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR
Short coins: BTC, ETH, SOL, XRP, SUI, LINK, ADA, DOGE (NEAR replaced by DOGE for perps)
"""
import json
import time
import os
import requests
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from donchian_breakout_strategy import DonchianBreakoutStrategy
from coinbase_futures import CoinbaseFuturesClient, SPOT_TO_PERP, PERP_TO_SPOT
from notify import send_telegram, setup_logging
from supabase_sync import SupabaseSync

# ============================================================================
# CONFIGURATION
# ============================================================================

COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD',
]

# Strategy parameters (matches backtest DEFAULT_PARAMS)
STRATEGY_PARAMS = {
    'donchian_period': 20,
    'exit_period': 10,
    'atr_period': 14,
    'atr_mult': 4.0,
    'volume_mult': 1.5,
    'ema_period': 21,
    'rsi_blowoff': 80,
    'volume_blowoff': 3.0,
    'atr_mult_tight': 1.5,
    'tp1_pct': 10.0,
    'tp2_pct': 20.0,
    'tp1_fraction': 0.25,
    'tp2_fraction': 0.25,
    'lookback': 60,
}

# Portfolio settings
MAX_POSITIONS = 4
RISK_PER_TRADE_PCT = 2.0  # risk 2% of equity per trade
STARTING_CAPITAL = 10000.0  # paper trading starting balance
EMERGENCY_STOP_PCT = 15.0  # emergency stop 15% below entry

# Pyramiding (add to winners)
PYRAMID_ENABLED = True
PYRAMID_GAIN_PCT = 15.0    # add to winner when position is up +15%
PYRAMID_RISK_PCT = 1.0     # 1% equity risk on the add-on tranche

# Bull market filter (BTC macro gate) — default, overridden by Supabase config
BULL_FILTER_ENABLED = True
BULL_SMA_FAST = 50   # SMA(50) must be above SMA(200) (golden cross)
BULL_SMA_SLOW = 200  # BTC close must be above SMA(200)
BULL_LOOKBACK = 220  # candles to fetch for SMA(200) computation

# Short-side strategy parameters (from backtest_shorts.py death cross best: L10_A2.0_E15_V2.0)
SHORT_STRATEGY_PARAMS = {
    'donchian_period': 10,       # N-day low for breakdown entry
    'exit_period': 15,           # N-day high for short cover exit
    'atr_period': 14,
    'atr_mult': 2.0,             # 2x ATR trailing (tighter than long's 4x)
    'volume_mult': 2.0,          # volume > 2x avg
    'ema_period': 21,            # price must be below EMA(21)
    'rsi_blowoff': 20,           # bounce risk: RSI < 20 (inverted from long's 80)
    'volume_blowoff': 3.0,
    'atr_mult_tight': 1.0,       # tightened stop on bounce
    'tp1_pct': 10.0,             # partial TP at -10% price drop
    'tp2_pct': 20.0,             # partial TP at -20% price drop
    'tp1_fraction': 0.25,
    'tp2_fraction': 0.25,
    'lookback': 60,
}

# Short-eligible coin universe (perp-only; NEAR has no liquid perp, use DOGE)
SHORT_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]

SHORT_RISK_PER_TRADE_PCT = 2.0
SHORT_EMERGENCY_STOP_PCT = 15.0   # exit if price rises 15% above entry
SHORT_MAX_HOLD_DAYS = 30          # forced exit after 30 days

# Short pyramiding (disabled initially — enable after validation)
SHORT_PYRAMID_ENABLED = False

# Bear market filter (death cross) — default, overridden by Supabase config
BEAR_FILTER_ENABLED = True

# Timing
DAILY_CHECK_HOUR = 0      # UTC hour for daily signal check
DAILY_CHECK_MINUTE = 15   # UTC minute (15 min after candle close)
STOP_CHECK_INTERVAL = 1800  # 30 minutes between trailing stop checks

STATE_FILE = 'bot_state.json'


# ============================================================================
# BOT CLASS
# ============================================================================

class DonchianMultiCoinBot:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.coinbase_api = "https://api.exchange.coinbase.com"

        # Portfolio state
        self.cash = STARTING_CAPITAL
        self.positions = {}  # symbol -> position dict
        self.trades_log = []
        self.total_realized_pnl = 0.0

        # Strategy instances (one per coin for longs)
        self.strategies = {}
        for symbol in COIN_UNIVERSE:
            self.strategies[symbol] = DonchianBreakoutStrategy(STRATEGY_PARAMS)

        # Short strategy instances (separate params for shorts)
        self.short_strategies = {}
        for symbol in SHORT_COIN_UNIVERSE:
            self.short_strategies[symbol] = DonchianBreakoutStrategy(SHORT_STRATEGY_PARAMS)

        # Futures client (for short positions via Coinbase CFM perps)
        self.futures_client = CoinbaseFuturesClient(paper_mode=paper_trading)

        # Logging
        self.logger, self.trade_logger = setup_logging()

        # Supabase sync (fire-and-forget, never crashes the bot)
        self.sync = SupabaseSync()

        # Bull filter — default from constant, overridden by Supabase config
        self.bull_filter_enabled = BULL_FILTER_ENABLED
        # Bear filter (death cross) — gates short entries
        self.bear_filter_enabled = BEAR_FILTER_ENABLED
        self.load_remote_config()

        # Load saved state if exists
        self.load_state()

    # ------------------------------------------------------------------
    # REMOTE CONFIG (from TradeSavvy dashboard)
    # ------------------------------------------------------------------

    def load_remote_config(self):
        """Load config from Supabase. Updates bull_filter_enabled.
        Falls back to local constants if Supabase is unavailable."""
        try:
            config = self.sync.load_config()
            if config:
                self.bull_filter_enabled = config.get("bull_filter_enabled", BULL_FILTER_ENABLED)
                self.logger.info(f"[CONFIG] Loaded from Supabase — bull_filter: {'ON' if self.bull_filter_enabled else 'OFF'}")
            else:
                self.logger.info("[CONFIG] No remote config found, using local defaults")
        except Exception as e:
            self.logger.warning(f"[CONFIG] Failed to load remote config: {e}")

    # ------------------------------------------------------------------
    # STATE PERSISTENCE
    # ------------------------------------------------------------------

    def save_state(self):
        """Save positions and portfolio state to JSON for crash recovery"""
        state = {
            'cash': self.cash,
            'total_realized_pnl': self.total_realized_pnl,
            'positions': {},
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        for symbol, pos in self.positions.items():
            side = pos.get('side', 'LONG')
            pos_data = {
                'side': side,
                'entry_price': pos['entry_price'],
                'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time']),
                'partials_taken': pos['partials_taken'],
                'remaining_fraction': pos['remaining_fraction'],
                'size_usd': pos['size_usd'],
                'last_atr': pos.get('last_atr', 0),
                'stop_price': pos.get('stop_price', 0),
                'pyramided': pos.get('pyramided', False),
            }
            if side == 'SHORT':
                pos_data['low_watermark'] = pos['low_watermark']
                pos_data['hold_days'] = pos.get('hold_days', 0)
                pos_data['spot_symbol'] = pos.get('spot_symbol', '')
            else:
                pos_data['high_watermark'] = pos['high_watermark']
            state['positions'][symbol] = pos_data
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load saved state on restart"""
        if not os.path.exists(STATE_FILE):
            return

        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            self.cash = state.get('cash', STARTING_CAPITAL)
            self.total_realized_pnl = state.get('total_realized_pnl', 0)
            for symbol, pos_data in state.get('positions', {}).items():
                side = pos_data.get('side', 'LONG')
                pos = {
                    'side': side,
                    'entry_price': pos_data['entry_price'],
                    'entry_time': datetime.fromisoformat(pos_data['entry_time']),
                    'partials_taken': pos_data['partials_taken'],
                    'remaining_fraction': pos_data['remaining_fraction'],
                    'size_usd': pos_data['size_usd'],
                    'last_atr': pos_data.get('last_atr', 0),
                    'stop_price': pos_data.get('stop_price', 0),
                    'pyramided': pos_data.get('pyramided', False),
                }
                if side == 'SHORT':
                    pos['low_watermark'] = pos_data.get('low_watermark', pos_data['entry_price'])
                    pos['hold_days'] = pos_data.get('hold_days', 0)
                    pos['spot_symbol'] = pos_data.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                else:
                    pos['high_watermark'] = pos_data.get('high_watermark', pos_data['entry_price'])
                self.positions[symbol] = pos
            if self.positions:
                longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
                shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
                print(f"Restored {len(self.positions)} open positions from state file ({longs}L/{shorts}S)")
                for sym, pos in self.positions.items():
                    side_tag = " SHORT" if pos.get('side') == 'SHORT' else ""
                    print(f"  {sym}{side_tag}: entry ${pos['entry_price']:,.2f}, size ${pos['size_usd']:,.0f}")
        except Exception as e:
            print(f"Warning: Could not load state: {e}")

    # ------------------------------------------------------------------
    # PRICE FETCHING
    # ------------------------------------------------------------------

    def fetch_current_price(self, symbol):
        """Lightweight current price check"""
        url = f"{self.coinbase_api}/products/{symbol}/ticker"
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['price'])

    def fetch_all_prices(self):
        """Get current prices for all coins (long + short universes)"""
        prices = {}
        # Fetch spot prices for all unique coins
        all_coins = set(COIN_UNIVERSE) | set(SHORT_COIN_UNIVERSE)
        for symbol in sorted(all_coins):
            try:
                prices[symbol] = self.fetch_current_price(symbol)
                time.sleep(0.15)
            except Exception as e:
                self.logger.warning(f"Price fetch failed for {symbol}: {e}")

        # Map spot prices to perp IDs for open short positions
        for symbol, pos in self.positions.items():
            if pos.get('side') == 'SHORT' and symbol not in prices:
                spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                if spot_sym in prices:
                    prices[symbol] = prices[spot_sym]
        return prices

    # ------------------------------------------------------------------
    # BULL MARKET FILTER
    # ------------------------------------------------------------------

    def check_bull_filter(self):
        """Check if BTC is in a bull market.

        Bull = BTC close > SMA(200) AND SMA(50) > SMA(200).
        Returns (is_bull, details_dict).
        """
        if not self.bull_filter_enabled:
            return True, {'status': 'DISABLED'}

        try:
            url = f"{self.coinbase_api}/products/BTC-USD/candles"
            params = {'granularity': 86400}
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            df = pd.DataFrame(data[:BULL_LOOKBACK], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df = df.sort_values('time').reset_index(drop=True)

            if len(df) < BULL_SMA_SLOW:
                self.logger.warning(f"Bull filter: only {len(df)} candles, need {BULL_SMA_SLOW}")
                return True, {'status': 'INSUFFICIENT_DATA'}

            sma_fast = df['close'].rolling(window=BULL_SMA_FAST).mean().iloc[-1]
            sma_slow = df['close'].rolling(window=BULL_SMA_SLOW).mean().iloc[-1]
            btc_close = float(df['close'].iloc[-1])

            above_200 = btc_close > sma_slow
            golden_cross = sma_fast > sma_slow
            is_bull = above_200 and golden_cross

            details = {
                'status': 'BULL' if is_bull else 'BEAR',
                'btc_close': btc_close,
                'sma_50': round(float(sma_fast), 2),
                'sma_200': round(float(sma_slow), 2),
                'above_200': above_200,
                'golden_cross': golden_cross,
            }
            return is_bull, details

        except Exception as e:
            self.logger.error(f"Bull filter check failed: {e}")
            return True, {'status': 'ERROR', 'error': str(e)}

    # ------------------------------------------------------------------
    # BEAR MARKET FILTER (death cross — gates short entries)
    # ------------------------------------------------------------------

    def check_bear_filter(self):
        """Check if BTC is in a bear market (death cross).

        Bear = BTC close < SMA(200) AND SMA(50) < SMA(200).
        Returns (is_bear, details_dict).
        """
        if not self.bear_filter_enabled:
            return False, {'status': 'DISABLED'}

        try:
            url = f"{self.coinbase_api}/products/BTC-USD/candles"
            params = {'granularity': 86400}
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            df = pd.DataFrame(data[:BULL_LOOKBACK], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df = df.sort_values('time').reset_index(drop=True)

            if len(df) < BULL_SMA_SLOW:
                self.logger.warning(f"Bear filter: only {len(df)} candles, need {BULL_SMA_SLOW}")
                return False, {'status': 'INSUFFICIENT_DATA'}

            sma_fast = df['close'].rolling(window=BULL_SMA_FAST).mean().iloc[-1]
            sma_slow = df['close'].rolling(window=BULL_SMA_SLOW).mean().iloc[-1]
            btc_close = float(df['close'].iloc[-1])

            below_200 = btc_close < sma_slow
            death_cross = sma_fast < sma_slow
            is_bear = below_200 and death_cross

            details = {
                'status': 'DEATH_CROSS' if is_bear else ('BELOW_200' if below_200 else 'NEUTRAL'),
                'btc_close': btc_close,
                'sma_50': round(float(sma_fast), 2),
                'sma_200': round(float(sma_slow), 2),
                'below_200': below_200,
                'death_cross': death_cross,
            }
            return is_bear, details

        except Exception as e:
            self.logger.error(f"Bear filter check failed: {e}")
            return False, {'status': 'ERROR', 'error': str(e)}

    # ------------------------------------------------------------------
    # PORTFOLIO HELPERS
    # ------------------------------------------------------------------

    def total_equity(self, prices=None):
        """Calculate total portfolio equity (longs + shorts)"""
        equity = self.cash
        for symbol, pos in self.positions.items():
            side = pos.get('side', 'LONG')
            current_price = None
            if prices:
                if symbol in prices:
                    current_price = prices[symbol]
                elif side == 'SHORT' and pos.get('spot_symbol') in prices:
                    current_price = prices[pos['spot_symbol']]

            if current_price:
                if side == 'SHORT':
                    # Short P&L: profit when price drops
                    price_change = (pos['entry_price'] - current_price) / pos['entry_price']
                    current_val = pos['size_usd'] * (1 + price_change)
                    current_val = max(current_val, 0)  # isolated margin floor
                else:
                    current_val = pos['size_usd'] * (current_price / pos['entry_price'])
                equity += current_val
            else:
                equity += pos['size_usd']
        return equity

    def calculate_position_size(self, symbol, entry_price, atr, prices=None):
        """Calculate position size using 2% risk rule"""
        equity = self.total_equity(prices)
        risk_amount = equity * (RISK_PER_TRADE_PCT / 100)
        stop_distance = STRATEGY_PARAMS['atr_mult'] * atr
        stop_pct = stop_distance / entry_price

        if stop_pct > 0:
            size = risk_amount / stop_pct
        else:
            size = equity / MAX_POSITIONS

        # Cap at available cash (keep 5% reserve)
        size = min(size, self.cash * 0.95)
        return size

    # ------------------------------------------------------------------
    # TRADE EXECUTION
    # ------------------------------------------------------------------

    def execute_buy(self, symbol, price, atr, reason, prices=None):
        """Open a new position"""
        if symbol in self.positions:
            return  # already in position
        if len(self.positions) >= MAX_POSITIONS:
            self.logger.info(f"SKIP BUY {symbol}: max {MAX_POSITIONS} positions reached")
            return

        size = self.calculate_position_size(symbol, price, atr, prices)
        if size < 50:  # minimum position
            self.logger.info(f"SKIP BUY {symbol}: position too small (${size:.0f})")
            return

        self.cash -= size
        stop_price = price - (STRATEGY_PARAMS['atr_mult'] * atr)

        self.positions[symbol] = {
            'entry_price': price,
            'entry_time': datetime.now(timezone.utc),
            'high_watermark': price,
            'partials_taken': 0,
            'remaining_fraction': 1.0,
            'size_usd': size,
            'last_atr': atr,
            'stop_price': stop_price,
            'pyramided': False,
        }

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        pos_count = len(self.positions)
        equity = self.total_equity(prices)

        print(f"\n  BUY {symbol} @ ${price:,.2f} | Size: ${size:,.0f} | Stop: ${stop_price:,.2f}")
        self.logger.info(f"BUY | {symbol} | ${price:,.2f} | Size: ${size:,.0f} | {reason}")
        self.trade_logger.info(f"BUY | {symbol} | ${price:,.2f} | Size: ${size:,.0f} | {reason} | {mode}")

        msg = (
            f"🟢 <b>BUY {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:,.2f}</b>\n"
            f"📦 Size: ${size:,.0f}\n"
            f"🛡 Stop: ${stop_price:,.2f}\n"
            f"📊 {reason}\n"
            f"📈 Positions: {pos_count}/{MAX_POSITIONS}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode}"
        )
        send_telegram(msg)

        # Sync to Supabase
        self.sync.sync_position_open(symbol, self.positions[symbol])
        self.sync.sync_trade(symbol=symbol, action="BUY",
                             entry_price=price, size_usd=size,
                             trading_mode='paper' if self.paper_trading else 'live')
        self.sync.sync_event("trade", f"BUY {symbol} @ ${price:,.2f} (${size:,.0f})",
                             {"action": "BUY", "symbol": symbol, "price": price,
                              "size_usd": size, "stop_price": stop_price})

    def execute_sell(self, symbol, price, reason, fraction=1.0):
        """Close (or partially close) a position"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        sell_size = pos['size_usd'] * fraction
        pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = sell_size * (pnl_pct / 100)

        self.cash += sell_size + pnl_usd
        self.total_realized_pnl += pnl_usd

        is_partial = fraction < 1.0
        if is_partial:
            pos['size_usd'] -= sell_size
        else:
            del self.positions[symbol]

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        equity = self.total_equity()
        pnl_sign = "+" if pnl_pct >= 0 else ""
        hold_days = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 86400

        action = "PARTIAL SELL" if is_partial else "SELL"
        frac_str = f" ({fraction*100:.0f}%)" if is_partial else ""

        print(f"\n  {action} {symbol}{frac_str} @ ${price:,.2f} | P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_usd:+,.0f})")
        self.trade_logger.info(
            f"{action} | {symbol} | Entry: ${pos['entry_price']:,.2f} | Exit: ${price:,.2f} | "
            f"P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_usd:+,.0f}) | {reason} | {mode}"
        )

        emoji = "🎯" if is_partial and pnl_pct > 0 else ("🟢" if pnl_pct >= 0 else "🔴")
        header = f"{'PARTIAL ' if is_partial else ''}SELL {symbol}{frac_str}"
        msg = (
            f"{emoji} <b>{header}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Entry: ${pos['entry_price']:,.2f}\n"
            f"💰 Exit: <b>${price:,.2f}</b>\n"
            f"{'🟢' if pnl_pct >= 0 else '🔴'} P&L: <b>{pnl_sign}{pnl_pct:.2f}%</b> (${pnl_usd:+,.0f})\n"
            f"📊 {reason}\n"
            f"⏱ Held: {hold_days:.1f} days\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode}"
        )
        send_telegram(msg)

        # Sync to Supabase
        if is_partial:
            trade_action = "PARTIAL_TP1" if "TP1" in reason else "PARTIAL_TP2"
        else:
            trade_action = "SELL"
        self.sync.sync_trade(symbol=symbol, action=trade_action,
                             entry_price=pos['entry_price'], exit_price=price,
                             size_usd=sell_size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                             exit_reason=reason, hold_days=hold_days,
                             trading_mode='paper' if self.paper_trading else 'live')
        if is_partial:
            self.sync.sync_position_update(symbol, pos)
        else:
            self.sync.sync_position_close(symbol)
        self.sync.sync_event("trade",
                             f"{trade_action} {symbol} @ ${price:,.2f} ({pnl_sign}{pnl_pct:.1f}%)",
                             {"action": trade_action, "symbol": symbol,
                              "entry_price": pos['entry_price'], "exit_price": price,
                              "pnl_pct": round(pnl_pct, 2), "pnl_usd": round(pnl_usd, 2),
                              "reason": reason})

    # ------------------------------------------------------------------
    # PYRAMIDING (add to winners)
    # ------------------------------------------------------------------

    def execute_pyramid(self, symbol, price, atr, gain_pct, prices=None):
        """Add a tranche to an existing winning position"""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        if pos.get('pyramided'):
            return  # only one add per position

        equity = self.total_equity(prices)
        add_risk = equity * (PYRAMID_RISK_PCT / 100)
        stop_distance = STRATEGY_PARAMS['atr_mult'] * atr
        stop_pct = stop_distance / price

        if stop_pct > 0:
            add_size = add_risk / stop_pct
        else:
            add_size = equity * 0.05

        # Don't use more than half remaining cash
        add_size = min(add_size, self.cash * 0.50)
        if add_size < 50:
            self.logger.info(f"SKIP PYRAMID {symbol}: add too small (${add_size:.0f})")
            return

        self.cash -= add_size
        pos['size_usd'] += add_size
        pos['pyramided'] = True

        # Update stop for new combined position
        pos['stop_price'] = pos['high_watermark'] - (STRATEGY_PARAMS['atr_mult'] * atr)

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        equity = self.total_equity(prices)

        print(f"\n  PYRAMID {symbol} @ ${price:,.2f} | Add: ${add_size:,.0f} | Total: ${pos['size_usd']:,.0f}")
        self.logger.info(f"PYRAMID | {symbol} | ${price:,.2f} | Add: ${add_size:,.0f} | +{gain_pct:.1f}% gain")
        self.trade_logger.info(f"PYRAMID | {symbol} | ${price:,.2f} | Add: ${add_size:,.0f} | +{gain_pct:.1f}% gain | {mode}")

        msg = (
            f"📈 <b>PYRAMID ADD {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:,.2f}</b>\n"
            f"📦 Add size: ${add_size:,.0f}\n"
            f"📦 Total position: ${pos['size_usd']:,.0f}\n"
            f"🎯 Gain at add: +{gain_pct:.1f}%\n"
            f"🛡 Stop: ${pos['stop_price']:,.2f}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode}"
        )
        send_telegram(msg)

        # Sync to Supabase
        self.sync.sync_position_update(symbol, pos)
        self.sync.sync_trade(symbol=symbol, action="PYRAMID",
                             entry_price=price, size_usd=add_size,
                             trading_mode='paper' if self.paper_trading else 'live')
        self.sync.sync_event("trade",
                             f"PYRAMID {symbol} @ ${price:,.2f} (+{gain_pct:.1f}%)",
                             {"action": "PYRAMID", "symbol": symbol, "price": price,
                              "add_size": add_size, "total_size": pos['size_usd'],
                              "gain_pct": round(gain_pct, 1)})

    # ------------------------------------------------------------------
    # SHORT POSITION SIZING
    # ------------------------------------------------------------------

    def calculate_short_position_size(self, symbol, entry_price, atr, prices=None):
        """Calculate position size for shorts using SHORT_RISK_PER_TRADE_PCT"""
        equity = self.total_equity(prices)
        risk_amount = equity * (SHORT_RISK_PER_TRADE_PCT / 100)
        stop_distance = SHORT_STRATEGY_PARAMS['atr_mult'] * atr
        stop_pct = stop_distance / entry_price

        if stop_pct > 0:
            size = risk_amount / stop_pct
        else:
            size = equity / MAX_POSITIONS

        # Cap at available cash (keep 5% reserve)
        size = min(size, self.cash * 0.95)
        return size

    # ------------------------------------------------------------------
    # SHORT TRADE EXECUTION
    # ------------------------------------------------------------------

    def execute_short_open(self, symbol, price, atr, reason, prices=None):
        """Open a new SHORT position via perpetual futures.

        symbol: spot symbol (e.g., 'ETH-USD')
        The position is stored under the perp ID (e.g., 'ETH-PERP-INTX').
        """
        perp_id = SPOT_TO_PERP.get(symbol)
        if not perp_id:
            self.logger.error(f"No perp mapping for {symbol}")
            return

        # Defensive: no same-coin conflict
        if symbol in self.positions or perp_id in self.positions:
            return
        if len(self.positions) >= MAX_POSITIONS:
            self.logger.info(f"SKIP SHORT {symbol}: max {MAX_POSITIONS} positions reached")
            return

        size = self.calculate_short_position_size(symbol, price, atr, prices)
        if size < 50:
            self.logger.info(f"SKIP SHORT {symbol}: position too small (${size:.0f})")
            return

        # Execute via futures client (paper or live)
        result = self.futures_client.market_order(perp_id, 'SELL', size_usd=size)
        if not result.get('success'):
            self.logger.error(f"Short order failed for {perp_id}: {result.get('error', 'Unknown')}")
            return

        self.cash -= size
        stop_price = price + (SHORT_STRATEGY_PARAMS['atr_mult'] * atr)

        self.positions[perp_id] = {
            'side': 'SHORT',
            'entry_price': price,
            'entry_time': datetime.now(timezone.utc),
            'low_watermark': price,
            'partials_taken': 0,
            'remaining_fraction': 1.0,
            'size_usd': size,
            'last_atr': atr,
            'stop_price': stop_price,
            'pyramided': False,
            'hold_days': 0,
            'spot_symbol': symbol,
        }

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        pos_count = len(self.positions)
        equity = self.total_equity(prices)

        print(f"\n  SHORT {perp_id} @ ${price:,.2f} | Size: ${size:,.0f} | Stop: ${stop_price:,.2f}")
        self.logger.info(f"SHORT | {perp_id} | ${price:,.2f} | Size: ${size:,.0f} | {reason}")
        self.trade_logger.info(f"SHORT | {perp_id} | ${price:,.2f} | Size: ${size:,.0f} | {reason} | {mode}")

        msg = (
            f"🔻 <b>SHORT {perp_id}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:,.2f}</b>\n"
            f"📦 Size: ${size:,.0f}\n"
            f"🛡 Stop: ${stop_price:,.2f} (above entry)\n"
            f"📊 {reason}\n"
            f"📈 Positions: {pos_count}/{MAX_POSITIONS}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode} | Futures"
        )
        send_telegram(msg)

        # Sync to Supabase
        self.sync.sync_position_open(perp_id, self.positions[perp_id])
        self.sync.sync_trade(symbol=perp_id, action="SHORT",
                             entry_price=price, size_usd=size,
                             trading_mode='paper' if self.paper_trading else 'live',
                             side='SHORT')
        self.sync.sync_event("trade", f"SHORT {perp_id} @ ${price:,.2f} (${size:,.0f})",
                             {"action": "SHORT", "symbol": perp_id, "price": price,
                              "size_usd": size, "stop_price": stop_price, "side": "SHORT"})

    def execute_short_close(self, symbol, price, reason, fraction=1.0):
        """Close (cover) a SHORT position via perpetual futures.

        symbol: perp ID (e.g., 'ETH-PERP-INTX')
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        sell_size = pos['size_usd'] * fraction
        # Short P&L: profit when price drops
        pnl_pct = ((pos['entry_price'] - price) / pos['entry_price']) * 100
        pnl_usd = sell_size * (pnl_pct / 100)

        # Execute via futures client
        is_partial = fraction < 1.0
        if is_partial:
            self.futures_client.market_order(symbol, 'BUY', size_usd=sell_size)
        else:
            self.futures_client.close_position(symbol)

        self.cash += sell_size + pnl_usd
        self.total_realized_pnl += pnl_usd

        if is_partial:
            pos['size_usd'] -= sell_size
        else:
            del self.positions[symbol]

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        equity = self.total_equity()
        pnl_sign = "+" if pnl_pct >= 0 else ""
        hold_days = pos.get('hold_days', 0)

        action = "PARTIAL COVER" if is_partial else "COVER"
        frac_str = f" ({fraction*100:.0f}%)" if is_partial else ""

        print(f"\n  {action} {symbol}{frac_str} @ ${price:,.2f} | P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_usd:+,.0f})")
        self.trade_logger.info(
            f"{action} | {symbol} | Entry: ${pos['entry_price']:,.2f} | Exit: ${price:,.2f} | "
            f"P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_usd:+,.0f}) | {reason} | {mode}"
        )

        emoji = "🎯" if is_partial and pnl_pct > 0 else ("🟢" if pnl_pct >= 0 else "🔴")
        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        header = f"{'PARTIAL ' if is_partial else ''}COVER {symbol}{frac_str}"
        msg = (
            f"{emoji} <b>{header}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Entry: ${pos['entry_price']:,.2f}\n"
            f"💰 Exit: <b>${price:,.2f}</b>\n"
            f"{pnl_emoji} P&L: <b>{pnl_sign}{pnl_pct:.2f}%</b> (${pnl_usd:+,.0f})\n"
            f"📊 {reason}\n"
            f"⏱ Held: {hold_days:.0f} days\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode} | Futures"
        )
        send_telegram(msg)

        # Sync to Supabase
        if is_partial:
            trade_action = "SHORT_PARTIAL_TP1" if "TP1" in reason else "SHORT_PARTIAL_TP2"
        else:
            trade_action = "COVER"
        self.sync.sync_trade(symbol=symbol, action=trade_action,
                             entry_price=pos['entry_price'], exit_price=price,
                             size_usd=sell_size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                             exit_reason=reason, hold_days=hold_days,
                             trading_mode='paper' if self.paper_trading else 'live',
                             side='SHORT')
        if is_partial:
            self.sync.sync_position_update(symbol, pos)
        else:
            self.sync.sync_position_close(symbol)
        self.sync.sync_event("trade",
                             f"{trade_action} {symbol} @ ${price:,.2f} ({pnl_sign}{pnl_pct:.1f}%)",
                             {"action": trade_action, "symbol": symbol,
                              "entry_price": pos['entry_price'], "exit_price": price,
                              "pnl_pct": round(pnl_pct, 2), "pnl_usd": round(pnl_usd, 2),
                              "reason": reason, "side": "SHORT"})

    # ------------------------------------------------------------------
    # DAILY SIGNAL CHECK
    # ------------------------------------------------------------------

    def daily_check(self):
        """Full daily signal check — long and short entries/exits"""
        # Reload config from dashboard (picks up bull filter toggle, etc.)
        self.load_remote_config()

        now = datetime.now(timezone.utc)
        print(f"\n{'='*80}")
        print(f"DAILY SIGNAL CHECK — {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*80}")

        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
        print(f"Portfolio: ${equity:,.0f} ({len(self.positions)}/{MAX_POSITIONS} positions [{longs}L/{shorts}S], ${self.cash:,.0f} cash)")

        if self.positions:
            print(f"\nOpen positions:")
            for sym, pos in self.positions.items():
                side = pos.get('side', 'LONG')
                spot_sym = pos.get('spot_symbol', sym) if side == 'SHORT' else sym
                current = prices.get(sym) or prices.get(spot_sym, pos['entry_price'])
                if side == 'SHORT':
                    pnl = ((pos['entry_price'] - current) / pos['entry_price']) * 100
                else:
                    pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                days = pos.get('hold_days', (now - pos['entry_time']).total_seconds() / 86400)
                side_tag = " S" if side == 'SHORT' else ""
                print(f"  {sym}{side_tag}: ${current:,.2f} ({pnl:+.1f}%, {days:.0f}d) | Stop: ${pos['stop_price']:,.2f}")

        # ===== LONG EXITS (never gated by regime) =====
        print(f"\nChecking long exits...")
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos.get('side') == 'SHORT':
                continue
            try:
                strategy = self.strategies.get(symbol)
                if not strategy:
                    continue
                df = strategy.fetch_daily_candles(symbol, limit=STRATEGY_PARAMS['lookback'])
                df = strategy.calculate_indicators(df)

                if len(df) < 25:
                    continue

                current = df.iloc[-1]
                prev = df.iloc[-2]

                # Update ATR and high watermark
                pos['last_atr'] = float(current['atr'])
                pos['high_watermark'] = max(pos['high_watermark'], float(current['high']))

                # Blow-off detection
                vol_sma = float(current['volume_sma']) if current['volume_sma'] > 0 else 1
                volume_ratio = float(current['volume']) / vol_sma
                is_blowoff = (volume_ratio > STRATEGY_PARAMS['volume_blowoff']
                              and float(current['rsi']) > STRATEGY_PARAMS['rsi_blowoff'])
                stop_mult = STRATEGY_PARAMS['atr_mult_tight'] if is_blowoff else STRATEGY_PARAMS['atr_mult']

                # Update trailing stop
                pos['stop_price'] = pos['high_watermark'] - (stop_mult * pos['last_atr'])

                # Check exit conditions
                exit_reason = None
                current_close = float(current['close'])

                if current_close <= pos['stop_price']:
                    suffix = " (BLOW-OFF tightened)" if is_blowoff else ""
                    exit_reason = f'Trailing stop ({stop_mult}x ATR){suffix}'

                if not exit_reason and pd.notna(prev['exit_low']) and current_close < float(prev['exit_low']):
                    exit_reason = 'Donchian exit (10-day low)'

                if not exit_reason and current_close <= pos['entry_price'] * (1 - EMERGENCY_STOP_PCT / 100):
                    exit_reason = f'Emergency stop ({EMERGENCY_STOP_PCT}%)'

                # Check partial profit taking
                if not exit_reason:
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= STRATEGY_PARAMS['tp1_pct']:
                        self.execute_sell(symbol, current_close, f'Partial TP1 (+{STRATEGY_PARAMS["tp1_pct"]}%)',
                                          fraction=STRATEGY_PARAMS['tp1_fraction'])
                        pos['partials_taken'] = 1
                        pos['remaining_fraction'] *= (1 - STRATEGY_PARAMS['tp1_fraction'])
                    elif pos['partials_taken'] == 1 and gain_pct >= STRATEGY_PARAMS['tp2_pct']:
                        self.execute_sell(symbol, current_close, f'Partial TP2 (+{STRATEGY_PARAMS["tp2_pct"]}%)',
                                          fraction=STRATEGY_PARAMS['tp2_fraction'])
                        pos['partials_taken'] = 2
                        pos['remaining_fraction'] *= (1 - STRATEGY_PARAMS['tp2_fraction'])

                if exit_reason:
                    self.execute_sell(symbol, current_close, exit_reason)

                self.save_state()
                time.sleep(0.3)

            except Exception as e:
                self.logger.error(f"Long exit check failed for {symbol}: {e}")

        # ===== SHORT EXITS (never gated by regime) =====
        print(f"\nChecking short exits...")
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos.get('side') != 'SHORT':
                continue
            try:
                spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                strategy = self.short_strategies.get(spot_sym)
                if not strategy:
                    continue
                df = strategy.fetch_daily_candles(spot_sym, limit=SHORT_STRATEGY_PARAMS['lookback'])
                df = strategy.calculate_short_indicators(df)

                if len(df) < 25:
                    continue

                current = df.iloc[-1]
                prev = df.iloc[-2]
                current_close = float(current['close'])

                # Increment hold days
                pos['hold_days'] = pos.get('hold_days', 0) + 1

                # Update ATR and low watermark (ratchets down)
                pos['last_atr'] = float(current['atr'])
                pos['low_watermark'] = min(pos['low_watermark'], float(current['low']))

                # Bounce risk detection (inverted blow-off)
                vol_sma = float(current['volume_sma']) if current['volume_sma'] > 0 else 1
                volume_ratio = float(current['volume']) / vol_sma
                is_bounce_risk = (volume_ratio > SHORT_STRATEGY_PARAMS['volume_blowoff']
                                  and float(current['rsi']) < SHORT_STRATEGY_PARAMS['rsi_blowoff'])
                stop_mult = SHORT_STRATEGY_PARAMS['atr_mult_tight'] if is_bounce_risk else SHORT_STRATEGY_PARAMS['atr_mult']

                # Update inverted trailing stop
                pos['stop_price'] = pos['low_watermark'] + (stop_mult * pos['last_atr'])

                exit_reason = None

                # 1. Inverted trailing stop: price above stop
                if current_close >= pos['stop_price']:
                    suffix = " (bounce-risk tightened)" if is_bounce_risk else ""
                    exit_reason = f'Short trailing stop ({stop_mult}x ATR){suffix}'

                # 2. Donchian exit: above N-day high (reversal)
                if not exit_reason and pd.notna(prev.get('exit_high')) and current_close > float(prev['exit_high']):
                    exit_reason = f'Donchian exit ({SHORT_STRATEGY_PARAMS["exit_period"]}-day high)'

                # 3. Emergency stop: price rose too much
                if not exit_reason and current_close >= pos['entry_price'] * (1 + SHORT_EMERGENCY_STOP_PCT / 100):
                    exit_reason = f'Short emergency stop (+{SHORT_EMERGENCY_STOP_PCT}%)'

                # 4. Max hold days
                if not exit_reason and pos['hold_days'] >= SHORT_MAX_HOLD_DAYS:
                    exit_reason = f'Max hold ({SHORT_MAX_HOLD_DAYS} days)'

                # Check partial profit taking (inverted: profit when price drops)
                if not exit_reason:
                    gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= SHORT_STRATEGY_PARAMS['tp1_pct']:
                        self.execute_short_close(symbol, current_close,
                                                 f'Short Partial TP1 (-{SHORT_STRATEGY_PARAMS["tp1_pct"]}% drop)',
                                                 fraction=SHORT_STRATEGY_PARAMS['tp1_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 1
                            self.positions[symbol]['remaining_fraction'] *= (1 - SHORT_STRATEGY_PARAMS['tp1_fraction'])
                    elif pos['partials_taken'] == 1 and gain_pct >= SHORT_STRATEGY_PARAMS['tp2_pct']:
                        self.execute_short_close(symbol, current_close,
                                                 f'Short Partial TP2 (-{SHORT_STRATEGY_PARAMS["tp2_pct"]}% drop)',
                                                 fraction=SHORT_STRATEGY_PARAMS['tp2_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 2
                            self.positions[symbol]['remaining_fraction'] *= (1 - SHORT_STRATEGY_PARAMS['tp2_fraction'])

                if exit_reason:
                    self.execute_short_close(symbol, current_close, exit_reason)

                self.save_state()
                time.sleep(0.3)

            except Exception as e:
                self.logger.error(f"Short exit check failed for {symbol}: {e}")

        # ===== CHECK REGIME FILTERS =====
        is_bull, bull_details = self.check_bull_filter()
        bull_status = bull_details.get('status', 'UNKNOWN')
        is_bear, bear_details = self.check_bear_filter()
        bear_status = bear_details.get('status', 'UNKNOWN')

        if bull_details.get('btc_close'):
            print(f"\nBull filter: {bull_status} | BTC: ${bull_details['btc_close']:,.2f} | "
                  f"SMA50: ${bull_details['sma_50']:,.2f} | SMA200: ${bull_details['sma_200']:,.2f}")
        else:
            print(f"\nBull filter: {bull_status}")
        print(f"Bear filter: {bear_status}")
        self.logger.info(f"Bull filter: {bull_status} | Bear filter: {bear_status}")

        # ===== LONG PYRAMIDING (gated by bull filter) =====
        if PYRAMID_ENABLED and is_bull and self.positions:
            print(f"\nChecking long pyramid opportunities...")
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if pos.get('side') == 'SHORT' or pos.get('pyramided'):
                    continue

                try:
                    strategy = self.strategies.get(symbol)
                    if not strategy:
                        continue
                    df = strategy.fetch_daily_candles(symbol, limit=STRATEGY_PARAMS['lookback'])
                    df = strategy.calculate_indicators(df)
                    if len(df) < 25:
                        continue

                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    current_close = float(current['close'])
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100

                    new_high = (pd.notna(prev['donchian_high'])
                                and current_close > float(prev['donchian_high']))

                    if gain_pct >= PYRAMID_GAIN_PCT and new_high:
                        atr = float(current['atr'])
                        self.execute_pyramid(symbol, current_close, atr, gain_pct, prices)
                    else:
                        if gain_pct >= PYRAMID_GAIN_PCT:
                            print(f"  {symbol}: +{gain_pct:.1f}% but no new 20d high")
                        else:
                            print(f"  {symbol}: +{gain_pct:.1f}% (need +{PYRAMID_GAIN_PCT}%)")

                    time.sleep(0.3)

                except Exception as e:
                    self.logger.error(f"Pyramid check failed for {symbol}: {e}")

        # ===== LONG ENTRIES (gated by bull filter) =====
        print(f"\nChecking long entries...")
        if not is_bull:
            print(f"  LONG ENTRIES BLOCKED — bull filter is {bull_status}")
            self.logger.info(f"Long entries blocked: {bull_status}")
        else:
            for symbol in COIN_UNIVERSE:
                if symbol in self.positions or SPOT_TO_PERP.get(symbol, '') in self.positions:
                    continue
                if len(self.positions) >= MAX_POSITIONS:
                    print(f"  Max positions reached, skipping remaining")
                    break

                try:
                    strategy = self.strategies[symbol]
                    signal = strategy.generate_signal(symbol)

                    if signal['signal'] == 'BUY':
                        atr = signal.get('atr', 0)
                        self.execute_buy(symbol, signal['price'], atr, signal['reason'], prices)
                    else:
                        print(f"  {symbol}: {signal['signal']} — {signal['reason']}")

                    time.sleep(0.3)

                except Exception as e:
                    self.logger.error(f"Long entry check failed for {symbol}: {e}")
                    print(f"  {symbol}: ERROR — {e}")

        # ===== SHORT ENTRIES (gated by bear filter / death cross) =====
        print(f"\nChecking short entries...")
        if not is_bear:
            print(f"  SHORT ENTRIES BLOCKED — bear filter is {bear_status}")
            self.logger.info(f"Short entries blocked: {bear_status}")
        else:
            for symbol in SHORT_COIN_UNIVERSE:
                perp_id = SPOT_TO_PERP.get(symbol, '')
                # No same-coin conflict
                if symbol in self.positions or perp_id in self.positions:
                    continue
                if len(self.positions) >= MAX_POSITIONS:
                    print(f"  Max positions reached, skipping remaining")
                    break

                try:
                    strategy = self.short_strategies[symbol]
                    signal = strategy.generate_short_signal(symbol)

                    if signal['signal'] == 'SHORT_SELL':
                        atr = signal.get('atr', 0)
                        self.execute_short_open(symbol, signal['price'], atr, signal['reason'], prices)
                    else:
                        print(f"  {symbol}: {signal['signal']} — {signal['reason']}")

                    time.sleep(0.3)

                except Exception as e:
                    self.logger.error(f"Short entry check failed for {symbol}: {e}")
                    print(f"  {symbol}: ERROR — {e}")

        # Daily summary
        equity = self.total_equity(prices)
        total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
        longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
        print(f"\nPortfolio: ${equity:,.0f} ({total_return:+.1f}% total return, {longs}L/{shorts}S)")

        # Sync daily check event to Supabase
        self.sync.sync_event("daily_check",
                             f"Daily scan: {len(self.positions)}/{MAX_POSITIONS} positions ({longs}L/{shorts}S), "
                             f"${equity:,.0f} equity, bull={bull_status}, bear={bear_status}",
                             {"equity": round(equity, 2), "positions": len(self.positions),
                              "longs": longs, "shorts": shorts,
                              "bull_status": bull_status, "bear_status": bear_status})

    # ------------------------------------------------------------------
    # TRAILING STOP MONITORING (intra-day)
    # ------------------------------------------------------------------

    def check_trailing_stops(self):
        """Lightweight price check for trailing stops between daily checks"""
        if not self.positions:
            return

        now = datetime.now(timezone.utc).strftime('%H:%M')
        prices = self.fetch_all_prices()

        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos:
                continue
            side = pos.get('side', 'LONG')

            # Get price (spot or perp proxy)
            if side == 'SHORT':
                spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                price = prices.get(symbol) or prices.get(spot_sym)
            else:
                price = prices.get(symbol)
            if price is None:
                continue

            if side == 'LONG':
                # === LONG TRAILING STOP (unchanged) ===
                pnl = ((price - pos['entry_price']) / pos['entry_price']) * 100

                if price > pos['high_watermark']:
                    old_stop = pos['stop_price']
                    pos['high_watermark'] = price
                    if pos['last_atr'] > 0:
                        pos['stop_price'] = pos['high_watermark'] - (STRATEGY_PARAMS['atr_mult'] * pos['last_atr'])
                        if pos['stop_price'] > old_stop:
                            print(f"  {symbol}: Stop ratcheted ${old_stop:,.2f} -> ${pos['stop_price']:,.2f}")
                    self.save_state()
                    self.sync.sync_position_update(symbol, pos)

                sell_reason = None
                if pos['stop_price'] > 0 and price <= pos['stop_price']:
                    sell_reason = f'Trailing stop hit (${pos["stop_price"]:,.2f}) — intra-day monitor'
                elif price <= pos['entry_price'] * (1 - EMERGENCY_STOP_PCT / 100):
                    sell_reason = f'Emergency stop ({EMERGENCY_STOP_PCT}%) — intra-day monitor'

                if not sell_reason:
                    gain_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= STRATEGY_PARAMS['tp1_pct']:
                        self.execute_sell(symbol, price, f'Partial TP1 (+{STRATEGY_PARAMS["tp1_pct"]}%) — intra-day',
                                          fraction=STRATEGY_PARAMS['tp1_fraction'])
                        pos['partials_taken'] = 1
                        pos['remaining_fraction'] *= (1 - STRATEGY_PARAMS['tp1_fraction'])
                        continue
                    elif pos['partials_taken'] == 1 and gain_pct >= STRATEGY_PARAMS['tp2_pct']:
                        self.execute_sell(symbol, price, f'Partial TP2 (+{STRATEGY_PARAMS["tp2_pct"]}%) — intra-day',
                                          fraction=STRATEGY_PARAMS['tp2_fraction'])
                        pos['partials_taken'] = 2
                        pos['remaining_fraction'] *= (1 - STRATEGY_PARAMS['tp2_fraction'])
                        continue

                if sell_reason:
                    self.execute_sell(symbol, price, sell_reason)
                else:
                    stop_str = f" | Stop: ${pos['stop_price']:,.2f}" if pos['stop_price'] else ""
                    print(f"  [{now}] {symbol}: ${price:,.2f} ({pnl:+.1f}%){stop_str}")

            elif side == 'SHORT':
                # === SHORT TRAILING STOP (inverted) ===
                pnl = ((pos['entry_price'] - price) / pos['entry_price']) * 100

                # Update low watermark (ratchets down)
                if price < pos['low_watermark']:
                    old_stop = pos['stop_price']
                    pos['low_watermark'] = price
                    if pos['last_atr'] > 0:
                        pos['stop_price'] = pos['low_watermark'] + (SHORT_STRATEGY_PARAMS['atr_mult'] * pos['last_atr'])
                        if pos['stop_price'] < old_stop:
                            print(f"  {symbol}: Short stop ratcheted ${old_stop:,.2f} -> ${pos['stop_price']:,.2f}")
                    self.save_state()
                    self.sync.sync_position_update(symbol, pos)

                sell_reason = None
                if pos['stop_price'] > 0 and price >= pos['stop_price']:
                    sell_reason = f'Short trailing stop hit (${pos["stop_price"]:,.2f}) — intra-day'
                elif price >= pos['entry_price'] * (1 + SHORT_EMERGENCY_STOP_PCT / 100):
                    sell_reason = f'Short emergency stop (+{SHORT_EMERGENCY_STOP_PCT}%) — intra-day'

                if not sell_reason:
                    gain_pct = pnl  # already inverted
                    if pos['partials_taken'] == 0 and gain_pct >= SHORT_STRATEGY_PARAMS['tp1_pct']:
                        self.execute_short_close(symbol, price,
                                                 f'Short Partial TP1 (-{SHORT_STRATEGY_PARAMS["tp1_pct"]}%) — intra-day',
                                                 fraction=SHORT_STRATEGY_PARAMS['tp1_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 1
                            self.positions[symbol]['remaining_fraction'] *= (1 - SHORT_STRATEGY_PARAMS['tp1_fraction'])
                        continue
                    elif pos['partials_taken'] == 1 and gain_pct >= SHORT_STRATEGY_PARAMS['tp2_pct']:
                        self.execute_short_close(symbol, price,
                                                 f'Short Partial TP2 (-{SHORT_STRATEGY_PARAMS["tp2_pct"]}%) — intra-day',
                                                 fraction=SHORT_STRATEGY_PARAMS['tp2_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 2
                            self.positions[symbol]['remaining_fraction'] *= (1 - SHORT_STRATEGY_PARAMS['tp2_fraction'])
                        continue

                if sell_reason:
                    self.execute_short_close(symbol, price, sell_reason)
                else:
                    stop_str = f" | Stop: ${pos['stop_price']:,.2f}" if pos['stop_price'] else ""
                    print(f"  [{now}] {symbol} S: ${price:,.2f} ({pnl:+.1f}% SHORT){stop_str}")

    # ------------------------------------------------------------------
    # DAILY SUMMARY
    # ------------------------------------------------------------------

    def send_daily_summary(self):
        """Send portfolio summary via Telegram"""
        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
        pnl_sign = "+" if total_return >= 0 else ""

        # Filter statuses
        is_bull, bull_details = self.check_bull_filter()
        bull_status = bull_details.get('status', 'UNKNOWN')
        is_bear, bear_details = self.check_bear_filter()
        bear_status = bear_details.get('status', 'UNKNOWN')

        filter_lines = ""
        if bull_details.get('btc_close'):
            bull_emoji = "🟢" if is_bull else "🔴"
            filter_lines += (f"{bull_emoji} Bull: <b>{bull_status}</b> "
                             f"(BTC ${bull_details['btc_close']:,.0f} vs SMA200 ${bull_details['sma_200']:,.0f})\n")
            bear_emoji = "💀" if is_bear else "⚪"
            filter_lines += f"{bear_emoji} Bear: <b>{bear_status}</b>\n"

        pos_lines = []
        for sym, pos in self.positions.items():
            side = pos.get('side', 'LONG')
            spot_sym = pos.get('spot_symbol', sym) if side == 'SHORT' else sym
            current = prices.get(sym) or prices.get(spot_sym, pos['entry_price'])
            if side == 'SHORT':
                pnl = ((pos['entry_price'] - current) / pos['entry_price']) * 100
                days = pos.get('hold_days', 0)
                side_tag = " 🔻S"
            else:
                pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                days = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 86400
                side_tag = ""
            emoji = "🟢" if pnl >= 0 else "🔴"
            pos_lines.append(f"{emoji}{side_tag} {sym}: ${current:,.2f} ({pnl:+.1f}%, {days:.0f}d)")

        pos_text = "\n".join(pos_lines) if pos_lines else "No open positions"
        longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')

        return_emoji = "🟢" if total_return >= 0 else "🔴"
        msg = (
            f"📋 <b>DAILY PORTFOLIO SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: <b>${equity:,.0f}</b>\n"
            f"{return_emoji} Return: <b>{pnl_sign}{total_return:.1f}%</b>\n"
            f"💵 Cash: ${self.cash:,.0f}\n"
            f"📊 Positions: {len(self.positions)}/{MAX_POSITIONS} ({longs}L/{shorts}S)\n"
            f"{filter_lines}\n"
            f"<b>Open Positions:</b>\n{pos_text}\n\n"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)

        # Sync equity snapshot to Supabase
        positions_value = equity - self.cash
        self.sync.sync_equity_snapshot(
            equity=equity, cash=self.cash,
            positions_value=positions_value,
            positions_count=len(self.positions),
            bull_filter_status=f"{bull_status}/{bear_status}",
            btc_price=bull_details.get('btc_close'))

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        """Main trading loop"""
        print("=" * 80)
        print("DONCHIAN CHANNEL BREAKOUT — DUAL-MODE BOT")
        print("=" * 80)
        print(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'} TRADING")
        print(f"Long coins: {', '.join(COIN_UNIVERSE)}")
        print(f"Short coins: {', '.join(SHORT_COIN_UNIVERSE)}")
        print(f"Max positions: {MAX_POSITIONS} (shared pool)")
        print(f"Risk per trade: {RISK_PER_TRADE_PCT}% (long) / {SHORT_RISK_PER_TRADE_PCT}% (short)")
        print(f"Long trailing: {STRATEGY_PARAMS['atr_mult']}x ATR")
        print(f"Short trailing: {SHORT_STRATEGY_PARAMS['atr_mult']}x ATR")
        print(f"Pyramiding: {'ON (+' + str(PYRAMID_GAIN_PCT) + '% / ' + str(PYRAMID_RISK_PCT) + '% risk)' if PYRAMID_ENABLED else 'OFF'}")
        print(f"Bull filter: {'ON' if self.bull_filter_enabled else 'OFF'}")
        print(f"Bear filter: {'ON' if self.bear_filter_enabled else 'OFF'}")
        print(f"Daily check: {DAILY_CHECK_HOUR:02d}:{DAILY_CHECK_MINUTE:02d} UTC")
        print(f"Stop check: every {STOP_CHECK_INTERVAL // 60} min")
        print("=" * 80)

        # Check filters on startup
        is_bull, bull_details = self.check_bull_filter()
        bull_status = bull_details.get('status', 'UNKNOWN')
        bull_emoji = "🟢" if is_bull else "🔴"
        print(f"Bull filter: {bull_status}")

        is_bear, bear_details = self.check_bear_filter()
        bear_status = bear_details.get('status', 'UNKNOWN')
        bear_emoji = "💀" if is_bear else "🟢"
        print(f"Bear filter: {bear_status}")

        # Count long/short positions
        long_count = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        short_count = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'SHORT')

        # Send startup message
        equity = self.total_equity()
        bull_line = ""
        if self.bull_filter_enabled and bull_details.get('btc_close'):
            bull_line = (f"{bull_emoji} Bull filter: <b>{bull_status}</b> "
                         f"(BTC ${bull_details['btc_close']:,.0f} vs SMA200 ${bull_details['sma_200']:,.0f})\n")
        bear_line = ""
        if self.bear_filter_enabled and bear_details.get('btc_close'):
            bear_line = (f"{bear_emoji} Bear filter: <b>{bear_status}</b> "
                         f"(SMA50 ${bear_details.get('sma_50', 0):,.0f} vs SMA200 ${bear_details.get('sma_200', 0):,.0f})\n")
        pyramid_line = f"📈 Pyramiding: +{PYRAMID_GAIN_PCT}% / {PYRAMID_RISK_PCT}% risk\n" if PYRAMID_ENABLED else ""
        msg = (
            f"🚀 <b>DONCHIAN BOT STARTED (DUAL-MODE)</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Strategy: Donchian Breakout (Daily)\n"
            f"🛡 Long trailing: {STRATEGY_PARAMS['atr_mult']}x ATR\n"
            f"🔻 Short trailing: {SHORT_STRATEGY_PARAMS['atr_mult']}x ATR\n"
            f"🪙 Long coins: {len(COIN_UNIVERSE)} | Short coins: {len(SHORT_COIN_UNIVERSE)}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"📈 Positions: {len(self.positions)}/{MAX_POSITIONS} (L:{long_count} S:{short_count})\n"
            f"{pyramid_line}"
            f"{bull_line}"
            f"{bear_line}"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)

        # Sync startup to Supabase
        self.sync.sync_event("startup",
                             f"Dual-mode bot started ({len(COIN_UNIVERSE)}L/{len(SHORT_COIN_UNIVERSE)}S coins, "
                             f"{'PAPER' if self.paper_trading else 'LIVE'})",
                             {"long_coins": COIN_UNIVERSE, "short_coins": SHORT_COIN_UNIVERSE,
                              "max_positions": MAX_POSITIONS,
                              "bull_filter": bull_status, "bear_filter": bear_status,
                              "equity": equity})
        self.sync.sync_equity_snapshot(
            equity=equity, cash=self.cash,
            positions_value=equity - self.cash,
            positions_count=len(self.positions),
            bull_filter_status=bull_status,
            btc_price=bull_details.get('btc_close'))

        last_daily_check = None
        last_summary = None

        try:
            while True:
                now = datetime.now(timezone.utc)

                # Daily signal check (once per day at configured time)
                if (now.hour == DAILY_CHECK_HOUR and
                        now.minute >= DAILY_CHECK_MINUTE and
                        last_daily_check != now.date()):
                    self.daily_check()
                    last_daily_check = now.date()

                # Daily summary at 20:00 UTC (noon PST)
                if now.hour == 20 and last_summary != now.date():
                    self.send_daily_summary()
                    last_summary = now.date()

                # Trailing stop monitoring
                if self.positions:
                    print(f"\n--- Trailing stop check ({now.strftime('%H:%M UTC')}) ---")
                    self.check_trailing_stops()

                # Sleep until next check
                next_check_min = STOP_CHECK_INTERVAL // 60
                print(f"\nNext check in {next_check_min} min...")
                time.sleep(STOP_CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nBot stopped by user")
            self.save_state()
            equity = self.total_equity()
            total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
            long_count = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
            short_count = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'SHORT')
            self.sync.sync_event("shutdown",
                                 f"Bot stopped. Equity: ${equity:,.0f} ({total_return:+.1f}%)",
                                 {"equity": equity, "total_return": round(total_return, 1),
                                  "open_positions": len(self.positions),
                                  "long_positions": long_count,
                                  "short_positions": short_count})
            print(f"\nFinal equity: ${equity:,.0f} ({total_return:+.1f}%)")
            print(f"Open positions: {len(self.positions)} (L:{long_count} S:{short_count})")
            for sym, pos in self.positions.items():
                side = pos.get('side', 'LONG')
                price = self.fetch_current_price(sym)
                if price and price > 0:
                    if side == 'SHORT':
                        pnl = ((pos['entry_price'] - price) / pos['entry_price']) * 100
                        tag = "S"
                    else:
                        pnl = ((price - pos['entry_price']) / pos['entry_price']) * 100
                        tag = "L"
                    print(f"  [{tag}] {sym}: entry ${pos['entry_price']:,.2f} ({pnl:+.1f}%)")
                else:
                    print(f"  [{'S' if side == 'SHORT' else 'L'}] {sym}: entry ${pos['entry_price']:,.2f} (no price)")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    bot = DonchianMultiCoinBot(paper_trading=True)
    bot.run()


if __name__ == "__main__":
    main()
