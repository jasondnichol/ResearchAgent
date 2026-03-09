"""Multi-Coin Donchian Channel Breakout Trading Bot — Daily Candles

Tri-mode production bot:
  - Spot Long: buy on Coinbase spot exchange (1x, no leverage)
  - Futures Long: buy on Coinbase CFM perpetual futures (1-3x leverage, configurable)
  - Futures Short: sell on Coinbase CFM perpetual futures (1x default)

Architecture:
  - Daily signal check at 00:15 UTC (after daily candle close)
  - Trailing stop monitoring every 30 minutes
  - Max 4 concurrent positions (all modes shared pool), 2% risk per trade
  - Spot Longs: 4x ATR trailing stop, pyramiding at +15%, partial TP at +10%/+20%
  - Futures Longs: same as spot longs but via perps with configurable leverage
  - Shorts: 2x ATR inverted trailing stop, partial TP at -10%/-20%, 30-day max hold
  - Bull filter (price > SMA200): gates spot long + futures long entries
  - Bear filter (death cross): gates short entries
  - Futures via Coinbase CFM perpetual futures (paper_mode default)
  - Telegram notifications for all trades

Spot long coins: BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR
Futures long coins: BTC, ETH, SOL, XRP, SUI, LINK, ADA, DOGE
Short coins: BTC, ETH, SOL, XRP, SUI, LINK, ADA, DOGE
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
from notify import send_telegram, send_telegram_user, send_telegram_platform, setup_logging
from supabase_sync import SupabaseSync

# ============================================================================
# DEFAULT CONFIGURATION (immutable module-level defaults, never mutated at runtime)
# ============================================================================

DEFAULT_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD',
]

DEFAULT_STRATEGY_PARAMS = {
    'donchian_period': 20,
    'exit_period': 10,
    'atr_period': 14,
    'atr_mult': 4.5,
    'volume_mult': 1.5,
    'ema_period': 21,
    'rsi_blowoff': 80,
    'volume_blowoff': 3.0,
    'atr_mult_tight': 1.5,
    'tp1_pct': 10.0,
    'tp2_pct': 25.0,
    'tp1_fraction': 0.25,
    'tp2_fraction': 0.25,
    'lookback': 60,
}

DEFAULT_MAX_POSITIONS = 4
DEFAULT_RISK_PER_TRADE_PCT = 2.0
DEFAULT_STARTING_CAPITAL = 10000.0
DEFAULT_EMERGENCY_STOP_PCT = 18.0
DEFAULT_PYRAMID_ENABLED = True
DEFAULT_PYRAMID_GAIN_PCT = 15.0
DEFAULT_PYRAMID_RISK_PCT = 1.0

BULL_SMA_FAST = 50
BULL_SMA_SLOW = 200
BULL_LOOKBACK = 220

DEFAULT_SHORT_STRATEGY_PARAMS = {
    'donchian_period': 10,
    'exit_period': 15,
    'atr_period': 14,
    'atr_mult': 2.0,
    'volume_mult': 2.0,
    'ema_period': 21,
    'rsi_blowoff': 25,
    'volume_blowoff': 3.0,
    'atr_mult_tight': 2.0,
    'tp1_pct': 10.0,
    'tp2_pct': 25.0,
    'tp1_fraction': 0.25,
    'tp2_fraction': 0.25,
    'lookback': 60,
}

DEFAULT_SHORT_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]

DEFAULT_SHORT_RISK_PER_TRADE_PCT = 2.0
DEFAULT_SHORT_EMERGENCY_STOP_PCT = 18.0
DEFAULT_SHORT_MAX_HOLD_DAYS = 25
DEFAULT_SHORT_PYRAMID_ENABLED = False
DEFAULT_BEAR_FILTER_ENABLED = True

DEFAULT_FUTURES_LONG_ENABLED = False
DEFAULT_FUTURES_LONG_LEVERAGE = 1.0

DEFAULT_FUTURES_LONG_COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]

DEFAULT_FUTURES_LONG_STRATEGY_PARAMS = {
    'donchian_period': 20,
    'exit_period': 10,
    'atr_period': 14,
    'atr_mult': 4.5,
    'volume_mult': 1.5,
    'ema_period': 21,
    'rsi_blowoff': 80,
    'volume_blowoff': 3.0,
    'atr_mult_tight': 1.5,
    'tp1_pct': 10.0,
    'tp2_pct': 25.0,
    'tp1_fraction': 0.25,
    'tp2_fraction': 0.25,
    'lookback': 60,
}

DEFAULT_FUTURES_LONG_RISK_PER_TRADE_PCT = 2.0
DEFAULT_FUTURES_LONG_EMERGENCY_STOP_PCT = 18.0
DEFAULT_FUTURES_LONG_PYRAMID_ENABLED = True
DEFAULT_FUTURES_LONG_PYRAMID_GAIN_PCT = 15.0
DEFAULT_FUTURES_LONG_PYRAMID_RISK_PCT = 1.0

# Timing (shared across all users — not per-user configurable)
DAILY_CHECK_HOUR = 0
DAILY_CHECK_MINUTE = 15
STOP_CHECK_INTERVAL = 1800


# ============================================================================
# BOT CLASS
# ============================================================================

class UserBot:
    """Per-user trading bot instance. All config is instance-scoped (no global mutation)."""

    def __init__(self, user_id, paper_trading=True):
        self.user_id = user_id
        self.paper_trading = paper_trading
        self.coinbase_api = "https://api.exchange.coinbase.com"

        # Copy defaults into instance vars (never mutate module-level constants)
        self.strategy_params = dict(DEFAULT_STRATEGY_PARAMS)
        self.coin_universe = list(DEFAULT_COIN_UNIVERSE)
        self.short_strategy_params = dict(DEFAULT_SHORT_STRATEGY_PARAMS)
        self.short_coin_universe = list(DEFAULT_SHORT_COIN_UNIVERSE)
        self.futures_long_strategy_params = dict(DEFAULT_FUTURES_LONG_STRATEGY_PARAMS)
        self.futures_long_coin_universe = list(DEFAULT_FUTURES_LONG_COIN_UNIVERSE)

        self.max_positions = DEFAULT_MAX_POSITIONS
        self.risk_per_trade_pct = DEFAULT_RISK_PER_TRADE_PCT
        self.starting_capital = DEFAULT_STARTING_CAPITAL
        self.emergency_stop_pct = DEFAULT_EMERGENCY_STOP_PCT
        self.pyramid_enabled = DEFAULT_PYRAMID_ENABLED
        self.pyramid_gain_pct = DEFAULT_PYRAMID_GAIN_PCT
        self.pyramid_risk_pct = DEFAULT_PYRAMID_RISK_PCT

        self.short_risk_per_trade_pct = DEFAULT_SHORT_RISK_PER_TRADE_PCT
        self.short_emergency_stop_pct = DEFAULT_SHORT_EMERGENCY_STOP_PCT
        self.short_max_hold_days = DEFAULT_SHORT_MAX_HOLD_DAYS
        self.short_pyramid_enabled = DEFAULT_SHORT_PYRAMID_ENABLED

        self.futures_long_risk_per_trade_pct = DEFAULT_FUTURES_LONG_RISK_PER_TRADE_PCT
        self.futures_long_emergency_stop_pct = DEFAULT_FUTURES_LONG_EMERGENCY_STOP_PCT
        self.futures_long_pyramid_enabled = DEFAULT_FUTURES_LONG_PYRAMID_ENABLED
        self.futures_long_pyramid_gain_pct = DEFAULT_FUTURES_LONG_PYRAMID_GAIN_PCT
        self.futures_long_pyramid_risk_pct = DEFAULT_FUTURES_LONG_PYRAMID_RISK_PCT

        self.state_file = f'bot_state_{user_id[:8]}.json'

        # Portfolio state
        self.cash = self.starting_capital
        self.positions = {}
        self.trades_log = []
        self.total_realized_pnl = 0.0

        # Strategy instances
        self.strategies = {sym: DonchianBreakoutStrategy(self.strategy_params) for sym in self.coin_universe}
        self.short_strategies = {sym: DonchianBreakoutStrategy(self.short_strategy_params) for sym in self.short_coin_universe}
        self.futures_long_strategies = {sym: DonchianBreakoutStrategy(self.futures_long_strategy_params) for sym in self.futures_long_coin_universe}

        # Futures client
        self.futures_client = CoinbaseFuturesClient(paper_mode=paper_trading)

        # Logging
        self.logger, self.trade_logger = setup_logging()

        # Supabase sync
        self.sync = SupabaseSync(user_id=user_id)

        # Mode toggles
        self.bull_filter_enabled = True
        self.bear_filter_enabled = DEFAULT_BEAR_FILTER_ENABLED
        self.short_enabled = True
        self.futures_long_enabled = DEFAULT_FUTURES_LONG_ENABLED
        self.futures_long_leverage = DEFAULT_FUTURES_LONG_LEVERAGE

        # Per-user Telegram
        self.telegram_bot_token = ""
        self.telegram_chat_id = ""
        self.notify_telegram = True
        self.load_remote_config()

        # Enforce paper trading if not live mode
        if not self.paper_trading:
            # Double-check: only admin-approved users can go live
            config = self.sync.load_config()
            if config and config.get("trading_mode") != "live":
                self.paper_trading = True

        # Load saved state if exists
        self.load_state()

    # ------------------------------------------------------------------
    # REMOTE CONFIG (from TradeSavvy dashboard)
    # ------------------------------------------------------------------

    def load_remote_config(self):
        """Load full strategy config from Supabase into instance vars."""
        try:
            config = self.sync.load_config()
            if not config:
                self.logger.info(f"[CONFIG:{self.user_id[:8]}] No remote config found, using defaults")
                return

            def _v(key):
                v = config.get(key)
                return v if v is not None else None

            # ── Mode toggles ──
            self.bull_filter_enabled = config.get("bull_filter_enabled", True)
            self.bear_filter_enabled = config.get("bear_filter_enabled", self.bear_filter_enabled)
            self.short_enabled = config.get("short_enabled", True)
            self.futures_long_enabled = config.get("futures_long_enabled", self.futures_long_enabled)
            self.futures_long_leverage = max(1.0, min(3.0, float(
                config.get("futures_long_leverage", self.futures_long_leverage))))

            # ── Spot Long strategy params ──
            spot_updates = {}
            if _v('donchian_period') is not None: spot_updates['donchian_period'] = int(_v('donchian_period'))
            if _v('exit_period') is not None: spot_updates['exit_period'] = int(_v('exit_period'))
            if _v('atr_mult') is not None: spot_updates['atr_mult'] = float(_v('atr_mult'))
            if _v('volume_mult') is not None: spot_updates['volume_mult'] = float(_v('volume_mult'))
            if _v('ema_period') is not None: spot_updates['ema_period'] = int(_v('ema_period'))
            if _v('tp1_gain_pct') is not None: spot_updates['tp1_pct'] = float(_v('tp1_gain_pct'))
            if _v('tp2_gain_pct') is not None: spot_updates['tp2_pct'] = float(_v('tp2_gain_pct'))
            if _v('tp1_size_pct') is not None: spot_updates['tp1_fraction'] = float(_v('tp1_size_pct')) / 100.0
            if _v('tp2_size_pct') is not None: spot_updates['tp2_fraction'] = float(_v('tp2_size_pct')) / 100.0
            if spot_updates:
                self.strategy_params.update(spot_updates)

            # ── Spot Long scalars ──
            if _v('risk_per_trade_pct') is not None: self.risk_per_trade_pct = float(_v('risk_per_trade_pct'))
            if _v('max_positions') is not None: self.max_positions = int(_v('max_positions'))
            if _v('emergency_stop_pct') is not None: self.emergency_stop_pct = float(_v('emergency_stop_pct'))
            if _v('pyramid_enabled') is not None: self.pyramid_enabled = bool(_v('pyramid_enabled'))
            if _v('pyramid_gain_pct') is not None: self.pyramid_gain_pct = float(_v('pyramid_gain_pct'))
            if _v('pyramid_risk_pct') is not None: self.pyramid_risk_pct = float(_v('pyramid_risk_pct'))

            # ── Spot Long coin list ──
            coins = _v('coins')
            if coins and isinstance(coins, list) and len(coins) > 0:
                self.coin_universe = list(coins)

            # ── Short strategy params ──
            short_updates = {}
            if _v('short_entry_period') is not None: short_updates['donchian_period'] = int(_v('short_entry_period'))
            if _v('short_exit_period') is not None: short_updates['exit_period'] = int(_v('short_exit_period'))
            if _v('short_atr_mult') is not None: short_updates['atr_mult'] = float(_v('short_atr_mult'))
            if _v('short_volume_mult') is not None: short_updates['volume_mult'] = float(_v('short_volume_mult'))
            if _v('short_tp1_gain_pct') is not None: short_updates['tp1_pct'] = float(_v('short_tp1_gain_pct'))
            if _v('short_tp2_gain_pct') is not None: short_updates['tp2_pct'] = float(_v('short_tp2_gain_pct'))
            if _v('short_tp1_size_pct') is not None: short_updates['tp1_fraction'] = float(_v('short_tp1_size_pct')) / 100.0
            if _v('short_tp2_size_pct') is not None: short_updates['tp2_fraction'] = float(_v('short_tp2_size_pct')) / 100.0
            if short_updates:
                self.short_strategy_params.update(short_updates)

            # ── Short scalars ──
            if _v('short_risk_per_trade_pct') is not None: self.short_risk_per_trade_pct = float(_v('short_risk_per_trade_pct'))
            if _v('short_emergency_stop_pct') is not None: self.short_emergency_stop_pct = float(_v('short_emergency_stop_pct'))
            if _v('short_max_hold_days') is not None: self.short_max_hold_days = int(_v('short_max_hold_days'))

            # ── Short coin list ──
            short_coins = _v('short_coins')
            if short_coins and isinstance(short_coins, list) and len(short_coins) > 0:
                self.short_coin_universe = list(short_coins)

            # ── Futures Long strategy params ──
            fl_updates = {}
            if _v('futures_long_entry_period') is not None: fl_updates['donchian_period'] = int(_v('futures_long_entry_period'))
            if _v('futures_long_exit_period') is not None: fl_updates['exit_period'] = int(_v('futures_long_exit_period'))
            if _v('futures_long_atr_mult') is not None: fl_updates['atr_mult'] = float(_v('futures_long_atr_mult'))
            if _v('futures_long_volume_mult') is not None: fl_updates['volume_mult'] = float(_v('futures_long_volume_mult'))
            if _v('futures_long_tp1_gain_pct') is not None: fl_updates['tp1_pct'] = float(_v('futures_long_tp1_gain_pct'))
            if _v('futures_long_tp2_gain_pct') is not None: fl_updates['tp2_pct'] = float(_v('futures_long_tp2_gain_pct'))
            if _v('futures_long_tp1_size_pct') is not None: fl_updates['tp1_fraction'] = float(_v('futures_long_tp1_size_pct')) / 100.0
            if _v('futures_long_tp2_size_pct') is not None: fl_updates['tp2_fraction'] = float(_v('futures_long_tp2_size_pct')) / 100.0
            if fl_updates:
                self.futures_long_strategy_params.update(fl_updates)

            # ── Futures Long scalars ──
            if _v('futures_long_risk_per_trade_pct') is not None: self.futures_long_risk_per_trade_pct = float(_v('futures_long_risk_per_trade_pct'))
            if _v('futures_long_emergency_stop_pct') is not None: self.futures_long_emergency_stop_pct = float(_v('futures_long_emergency_stop_pct'))
            if _v('futures_long_pyramid_enabled') is not None: self.futures_long_pyramid_enabled = bool(_v('futures_long_pyramid_enabled'))
            if _v('futures_long_pyramid_gain_pct') is not None: self.futures_long_pyramid_gain_pct = float(_v('futures_long_pyramid_gain_pct'))
            if _v('futures_long_pyramid_risk_pct') is not None: self.futures_long_pyramid_risk_pct = float(_v('futures_long_pyramid_risk_pct'))

            # ── Futures Long coin list ──
            fl_coins = _v('futures_long_coins')
            if fl_coins and isinstance(fl_coins, list) and len(fl_coins) > 0:
                self.futures_long_coin_universe = list(fl_coins)

            # ── Telegram per-user credentials ──
            self.telegram_bot_token = config.get("telegram_bot_token", "")
            self.telegram_chat_id = config.get("telegram_chat_id", "")
            self.notify_telegram = config.get("notify_telegram", True)
            self.notify_trade_entries = config.get("notify_trade_entries", True)
            self.notify_daily_summary = config.get("notify_daily_summary", True)

            # ── Recreate strategy instances with updated params ──
            self.strategies = {sym: DonchianBreakoutStrategy(self.strategy_params) for sym in self.coin_universe}
            self.short_strategies = {sym: DonchianBreakoutStrategy(self.short_strategy_params) for sym in self.short_coin_universe}
            self.futures_long_strategies = {sym: DonchianBreakoutStrategy(self.futures_long_strategy_params) for sym in self.futures_long_coin_universe}

            self.logger.info(
                f"[CONFIG:{self.user_id[:8]}] Loaded from Supabase — "
                f"bull: {'ON' if self.bull_filter_enabled else 'OFF'}, "
                f"short: {'ON' if self.short_enabled else 'OFF'} (bear: {'ON' if self.bear_filter_enabled else 'OFF'}), "
                f"FL: {'ON' if self.futures_long_enabled else 'OFF'} ({self.futures_long_leverage}x), "
                f"max_pos: {self.max_positions}, risk: {self.risk_per_trade_pct}%, "
                f"ATR: {self.strategy_params['atr_mult']}x/{self.short_strategy_params['atr_mult']}x/{self.futures_long_strategy_params['atr_mult']}x, "
                f"coins: {len(self.coin_universe)}L/{len(self.futures_long_coin_universe)}FL/{len(self.short_coin_universe)}S"
            )

        except Exception as e:
            self.logger.warning(f"[CONFIG] Failed to load remote config: {e}")

    # ------------------------------------------------------------------
    # NOTIFICATIONS
    # ------------------------------------------------------------------

    def send_notification(self, message, category="trade"):
        """Send Telegram with category-based filtering.
        category: 'trade' (entries/exits), 'summary' (daily/startup)."""
        if not self.notify_telegram:
            return
        if category == "trade" and not getattr(self, 'notify_trade_entries', True):
            return
        if category == "summary" and not getattr(self, 'notify_daily_summary', True):
            return
        # Use platform bot with per-user fallback
        if self.telegram_chat_id:
            send_telegram_platform(message, self.telegram_chat_id,
                                   self.telegram_bot_token or None)
        else:
            send_telegram(message)

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
            elif side == 'FUTURES_LONG':
                pos_data['high_watermark'] = pos['high_watermark']
                pos_data['spot_symbol'] = pos.get('spot_symbol', '')
            else:
                pos_data['high_watermark'] = pos['high_watermark']
            state['positions'][symbol] = pos_data
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load saved state on restart"""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.cash = state.get('cash', self.starting_capital)
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
                elif side == 'FUTURES_LONG':
                    pos['high_watermark'] = pos_data.get('high_watermark', pos_data['entry_price'])
                    pos['spot_symbol'] = pos_data.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                else:
                    pos['high_watermark'] = pos_data.get('high_watermark', pos_data['entry_price'])
                self.positions[symbol] = pos
            if self.positions:
                longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
                fl = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
                shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
                print(f"Restored {len(self.positions)} open positions from state file ({longs}L/{fl}FL/{shorts}S)")
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
        """Get current prices for all coins (long + futures long + short universes)"""
        prices = {}
        # Fetch spot prices for all unique coins
        all_coins = set(self.coin_universe) | set(self.short_coin_universe) | set(self.futures_long_coin_universe)
        for symbol in sorted(all_coins):
            try:
                prices[symbol] = self.fetch_current_price(symbol)
                time.sleep(0.15)
            except Exception as e:
                self.logger.warning(f"Price fetch failed for {symbol}: {e}")

        # Map spot prices to perp IDs for open short and futures long positions
        for symbol, pos in self.positions.items():
            if pos.get('side') in ('SHORT', 'FUTURES_LONG') and symbol not in prices:
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
            is_bull = above_200

            details = {
                'status': 'BULL' if is_bull else 'BEAR',
                'btc_close': btc_close,
                'sma_50': round(float(sma_fast), 2),
                'sma_200': round(float(sma_slow), 2),
                'above_200': above_200,
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
                elif side == 'FUTURES_LONG':
                    # Futures long P&L: same as spot long
                    current_val = pos['size_usd'] * (current_price / pos['entry_price'])
                else:
                    current_val = pos['size_usd'] * (current_price / pos['entry_price'])
                equity += current_val
            else:
                equity += pos['size_usd']
        return equity

    def calculate_position_size(self, symbol, entry_price, atr, prices=None):
        """Calculate position size using 2% risk rule"""
        equity = self.total_equity(prices)
        risk_amount = equity * (self.risk_per_trade_pct / 100)
        stop_distance = self.strategy_params['atr_mult'] * atr
        stop_pct = stop_distance / entry_price

        if stop_pct > 0:
            size = risk_amount / stop_pct
        else:
            size = equity / self.max_positions

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
        if len(self.positions) >= self.max_positions:
            self.logger.info(f"SKIP BUY {symbol}: max {self.max_positions} positions reached")
            return

        size = self.calculate_position_size(symbol, price, atr, prices)
        if size < 50:  # minimum position
            self.logger.info(f"SKIP BUY {symbol}: position too small (${size:.0f})")
            return

        self.cash -= size
        stop_price = price - (self.strategy_params['atr_mult'] * atr)

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
            f"📈 Positions: {pos_count}/{self.max_positions}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode}"
        )
        self.send_notification(msg)

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
        self.send_notification(msg)

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
        add_risk = equity * (self.pyramid_risk_pct / 100)
        stop_distance = self.strategy_params['atr_mult'] * atr
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
        pos['stop_price'] = pos['high_watermark'] - (self.strategy_params['atr_mult'] * atr)

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
        self.send_notification(msg)

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
        """Calculate position size for shorts using self.short_risk_per_trade_pct"""
        equity = self.total_equity(prices)
        risk_amount = equity * (self.short_risk_per_trade_pct / 100)
        stop_distance = self.short_strategy_params['atr_mult'] * atr
        stop_pct = stop_distance / entry_price

        if stop_pct > 0:
            size = risk_amount / stop_pct
        else:
            size = equity / self.max_positions

        # Cap at available cash (keep 5% reserve)
        size = min(size, self.cash * 0.95)
        return size

    # ------------------------------------------------------------------
    # FUTURES LONG POSITION SIZING
    # ------------------------------------------------------------------

    def calculate_futures_long_position_size(self, symbol, entry_price, atr, prices=None):
        """Calculate position size for futures longs with leverage"""
        equity = self.total_equity(prices)
        risk_amount = equity * (self.futures_long_risk_per_trade_pct / 100)
        stop_distance = self.futures_long_strategy_params['atr_mult'] * atr
        stop_pct = stop_distance / entry_price

        if stop_pct > 0:
            size = risk_amount / stop_pct
        else:
            size = equity / self.max_positions

        # Apply leverage (amplifies position size)
        size = size * self.futures_long_leverage

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
        if len(self.positions) >= self.max_positions:
            self.logger.info(f"SKIP SHORT {symbol}: max {self.max_positions} positions reached")
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
        stop_price = price + (self.short_strategy_params['atr_mult'] * atr)

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
            f"📈 Positions: {pos_count}/{self.max_positions}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode} | Futures"
        )
        self.send_notification(msg)

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
        self.send_notification(msg)

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
    # FUTURES LONG TRADE EXECUTION
    # ------------------------------------------------------------------

    def execute_futures_long_open(self, symbol, price, atr, reason, prices=None):
        """Open a new FUTURES_LONG position via perpetual futures.

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
        if len(self.positions) >= self.max_positions:
            self.logger.info(f"SKIP FUTURES_LONG {symbol}: max {self.max_positions} positions reached")
            return

        size = self.calculate_futures_long_position_size(symbol, price, atr, prices)
        if size < 50:
            self.logger.info(f"SKIP FUTURES_LONG {symbol}: position too small (${size:.0f})")
            return

        # Set leverage on futures client before order
        self.futures_client.set_leverage(self.futures_long_leverage)

        # Execute via futures client (paper or live)
        result = self.futures_client.market_order(perp_id, 'BUY', size_usd=size)
        if not result.get('success'):
            self.logger.error(f"Futures long order failed for {perp_id}: {result.get('error', 'Unknown')}")
            return

        self.cash -= size
        stop_price = price - (self.futures_long_strategy_params['atr_mult'] * atr)

        self.positions[perp_id] = {
            'side': 'FUTURES_LONG',
            'entry_price': price,
            'entry_time': datetime.now(timezone.utc),
            'high_watermark': price,
            'partials_taken': 0,
            'remaining_fraction': 1.0,
            'size_usd': size,
            'last_atr': atr,
            'stop_price': stop_price,
            'pyramided': False,
            'spot_symbol': symbol,
        }

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        pos_count = len(self.positions)
        equity = self.total_equity(prices)
        lev_str = f" | {self.futures_long_leverage}x leverage" if self.futures_long_leverage > 1 else ""

        print(f"\n  FUTURES_LONG BUY {perp_id} @ ${price:,.2f} | Size: ${size:,.0f} | Stop: ${stop_price:,.2f}{lev_str}")
        self.logger.info(f"FUTURES_LONG BUY | {perp_id} | ${price:,.2f} | Size: ${size:,.0f} | {reason}")
        self.trade_logger.info(f"FUTURES_LONG BUY | {perp_id} | ${price:,.2f} | Size: ${size:,.0f} | {reason} | {mode}")

        msg = (
            f"🔵 <b>FUTURES LONG BUY {perp_id}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:,.2f}</b>\n"
            f"📦 Size: ${size:,.0f}\n"
            f"🛡 Stop: ${stop_price:,.2f}\n"
            f"📊 {reason}\n"
            f"⚡ Leverage: {self.futures_long_leverage}x\n"
            f"📈 Positions: {pos_count}/{self.max_positions}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode} | Futures Long"
        )
        self.send_notification(msg)

        # Sync to Supabase
        self.sync.sync_position_open(perp_id, self.positions[perp_id])
        self.sync.sync_trade(symbol=perp_id, action="FUTURES_BUY",
                             entry_price=price, size_usd=size,
                             trading_mode='paper' if self.paper_trading else 'live',
                             side='FUTURES_LONG')
        self.sync.sync_event("trade", f"FUTURES_BUY {perp_id} @ ${price:,.2f} (${size:,.0f})",
                             {"action": "FUTURES_BUY", "symbol": perp_id, "price": price,
                              "size_usd": size, "stop_price": stop_price,
                              "leverage": self.futures_long_leverage, "side": "FUTURES_LONG"})

    def execute_futures_long_close(self, symbol, price, reason, fraction=1.0):
        """Close a FUTURES_LONG position via perpetual futures.

        symbol: perp ID (e.g., 'ETH-PERP-INTX')
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        sell_size = pos['size_usd'] * fraction
        # Long P&L: profit when price rises
        pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = sell_size * (pnl_pct / 100)

        # Execute via futures client
        is_partial = fraction < 1.0
        if is_partial:
            self.futures_client.market_order(symbol, 'SELL', size_usd=sell_size)
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
        hold_days = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 86400

        action = "PARTIAL SELL" if is_partial else "SELL"
        frac_str = f" ({fraction*100:.0f}%)" if is_partial else ""

        print(f"\n  FUTURES_LONG {action} {symbol}{frac_str} @ ${price:,.2f} | P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_usd:+,.0f})")
        self.trade_logger.info(
            f"FUTURES_LONG {action} | {symbol} | Entry: ${pos['entry_price']:,.2f} | Exit: ${price:,.2f} | "
            f"P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_usd:+,.0f}) | {reason} | {mode}"
        )

        emoji = "🎯" if is_partial and pnl_pct > 0 else ("🟢" if pnl_pct >= 0 else "🔴")
        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        header = f"FUTURES LONG {'PARTIAL ' if is_partial else ''}SELL {symbol}{frac_str}"
        msg = (
            f"{emoji} <b>{header}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Entry: ${pos['entry_price']:,.2f}\n"
            f"💰 Exit: <b>${price:,.2f}</b>\n"
            f"{pnl_emoji} P&L: <b>{pnl_sign}{pnl_pct:.2f}%</b> (${pnl_usd:+,.0f})\n"
            f"📊 {reason}\n"
            f"⏱ Held: {hold_days:.1f} days\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode} | Futures Long"
        )
        self.send_notification(msg)

        # Sync to Supabase
        if is_partial:
            trade_action = "FUTURES_PARTIAL_TP1" if "TP1" in reason else "FUTURES_PARTIAL_TP2"
        else:
            trade_action = "FUTURES_SELL"
        self.sync.sync_trade(symbol=symbol, action=trade_action,
                             entry_price=pos['entry_price'], exit_price=price,
                             size_usd=sell_size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                             exit_reason=reason, hold_days=hold_days,
                             trading_mode='paper' if self.paper_trading else 'live',
                             side='FUTURES_LONG')
        if is_partial:
            self.sync.sync_position_update(symbol, pos)
        else:
            self.sync.sync_position_close(symbol)
        self.sync.sync_event("trade",
                             f"{trade_action} {symbol} @ ${price:,.2f} ({pnl_sign}{pnl_pct:.1f}%)",
                             {"action": trade_action, "symbol": symbol,
                              "entry_price": pos['entry_price'], "exit_price": price,
                              "pnl_pct": round(pnl_pct, 2), "pnl_usd": round(pnl_usd, 2),
                              "reason": reason, "side": "FUTURES_LONG"})

    def execute_futures_long_pyramid(self, symbol, price, atr, gain_pct, prices=None):
        """Add a tranche to an existing winning FUTURES_LONG position"""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        if pos.get('pyramided'):
            return  # only one add per position

        equity = self.total_equity(prices)
        add_risk = equity * (self.futures_long_pyramid_risk_pct / 100)
        stop_distance = self.futures_long_strategy_params['atr_mult'] * atr
        stop_pct = stop_distance / price

        if stop_pct > 0:
            add_size = add_risk / stop_pct
        else:
            add_size = equity * 0.05

        # Apply leverage
        add_size = add_size * self.futures_long_leverage

        # Don't use more than half remaining cash
        add_size = min(add_size, self.cash * 0.50)
        if add_size < 50:
            self.logger.info(f"SKIP FUTURES_LONG PYRAMID {symbol}: add too small (${add_size:.0f})")
            return

        # Execute via futures client
        self.futures_client.set_leverage(self.futures_long_leverage)
        result = self.futures_client.market_order(symbol, 'BUY', size_usd=add_size)
        if not result.get('success'):
            self.logger.error(f"Futures long pyramid failed for {symbol}: {result.get('error', 'Unknown')}")
            return

        self.cash -= add_size
        pos['size_usd'] += add_size
        pos['pyramided'] = True

        # Update stop for new combined position
        pos['stop_price'] = pos['high_watermark'] - (self.futures_long_strategy_params['atr_mult'] * atr)

        self.save_state()

        mode = 'PAPER' if self.paper_trading else 'LIVE'
        equity = self.total_equity(prices)

        print(f"\n  FUTURES_LONG PYRAMID {symbol} @ ${price:,.2f} | Add: ${add_size:,.0f} | Total: ${pos['size_usd']:,.0f}")
        self.logger.info(f"FUTURES_LONG PYRAMID | {symbol} | ${price:,.2f} | Add: ${add_size:,.0f} | +{gain_pct:.1f}% gain")
        self.trade_logger.info(f"FUTURES_LONG PYRAMID | {symbol} | ${price:,.2f} | Add: ${add_size:,.0f} | +{gain_pct:.1f}% gain | {mode}")

        msg = (
            f"📈 <b>FUTURES LONG PYRAMID {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:,.2f}</b>\n"
            f"📦 Add size: ${add_size:,.0f}\n"
            f"📦 Total position: ${pos['size_usd']:,.0f}\n"
            f"🎯 Gain at add: +{gain_pct:.1f}%\n"
            f"⚡ Leverage: {self.futures_long_leverage}x\n"
            f"🛡 Stop: ${pos['stop_price']:,.2f}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"🏷 Mode: {mode} | Futures Long"
        )
        self.send_notification(msg)

        # Sync to Supabase
        self.sync.sync_position_update(symbol, pos)
        self.sync.sync_trade(symbol=symbol, action="FUTURES_PYRAMID",
                             entry_price=price, size_usd=add_size,
                             trading_mode='paper' if self.paper_trading else 'live',
                             side='FUTURES_LONG')
        self.sync.sync_event("trade",
                             f"FUTURES_PYRAMID {symbol} @ ${price:,.2f} (+{gain_pct:.1f}%)",
                             {"action": "FUTURES_PYRAMID", "symbol": symbol, "price": price,
                              "add_size": add_size, "total_size": pos['size_usd'],
                              "gain_pct": round(gain_pct, 1), "side": "FUTURES_LONG"})

    # ------------------------------------------------------------------
    # DAILY SIGNAL CHECK
    # ------------------------------------------------------------------

    def daily_check(self):
        """Full daily signal check — long and short entries/exits"""
        # Reload full config from Supabase (strategy params, risk, coins, toggles)
        self.load_remote_config()

        now = datetime.now(timezone.utc)
        print(f"\n{'='*80}")
        print(f"DAILY SIGNAL CHECK — {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*80}")

        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        fl = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
        print(f"Portfolio: ${equity:,.0f} ({len(self.positions)}/{self.max_positions} positions [{longs}L/{fl}FL/{shorts}S], ${self.cash:,.0f} cash)")

        if self.positions:
            print(f"\nOpen positions:")
            for sym, pos in self.positions.items():
                side = pos.get('side', 'LONG')
                spot_sym = pos.get('spot_symbol', sym) if side in ('SHORT', 'FUTURES_LONG') else sym
                current = prices.get(sym) or prices.get(spot_sym, pos['entry_price'])
                if side == 'SHORT':
                    pnl = ((pos['entry_price'] - current) / pos['entry_price']) * 100
                else:
                    pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                days = pos.get('hold_days', (now - pos['entry_time']).total_seconds() / 86400)
                side_tag = " S" if side == 'SHORT' else (" FL" if side == 'FUTURES_LONG' else "")
                print(f"  {sym}{side_tag}: ${current:,.2f} ({pnl:+.1f}%, {days:.0f}d) | Stop: ${pos['stop_price']:,.2f}")

        # ===== LONG EXITS (never gated by regime, spot longs only) =====
        print(f"\nChecking long exits...")
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos.get('side') in ('SHORT', 'FUTURES_LONG'):
                continue
            try:
                strategy = self.strategies.get(symbol)
                if not strategy:
                    continue
                df = strategy.fetch_daily_candles(symbol, limit=self.strategy_params['lookback'])
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
                is_blowoff = (volume_ratio > self.strategy_params['volume_blowoff']
                              and float(current['rsi']) > self.strategy_params['rsi_blowoff'])
                stop_mult = self.strategy_params['atr_mult_tight'] if is_blowoff else self.strategy_params['atr_mult']

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

                if not exit_reason and current_close <= pos['entry_price'] * (1 - self.emergency_stop_pct / 100):
                    exit_reason = f'Emergency stop ({self.emergency_stop_pct}%)'

                # Check partial profit taking
                if not exit_reason:
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= self.strategy_params['tp1_pct']:
                        self.execute_sell(symbol, current_close, f'Partial TP1 (+{self.strategy_params["tp1_pct"]}%)',
                                          fraction=self.strategy_params['tp1_fraction'])
                        pos['partials_taken'] = 1
                        pos['remaining_fraction'] *= (1 - self.strategy_params['tp1_fraction'])
                    elif pos['partials_taken'] == 1 and gain_pct >= self.strategy_params['tp2_pct']:
                        self.execute_sell(symbol, current_close, f'Partial TP2 (+{self.strategy_params["tp2_pct"]}%)',
                                          fraction=self.strategy_params['tp2_fraction'])
                        pos['partials_taken'] = 2
                        pos['remaining_fraction'] *= (1 - self.strategy_params['tp2_fraction'])

                if exit_reason:
                    self.execute_sell(symbol, current_close, exit_reason)

                self.save_state()
                time.sleep(0.3)

            except Exception as e:
                self.logger.error(f"Long exit check failed for {symbol}: {e}")

        # ===== FUTURES LONG EXITS (never gated by regime) =====
        print(f"\nChecking futures long exits...")
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos.get('side') != 'FUTURES_LONG':
                continue
            try:
                spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                strategy = self.futures_long_strategies.get(spot_sym)
                if not strategy:
                    continue
                df = strategy.fetch_daily_candles(spot_sym, limit=self.futures_long_strategy_params['lookback'])
                df = strategy.calculate_indicators(df)

                if len(df) < 25:
                    continue

                current = df.iloc[-1]
                prev = df.iloc[-2]

                # Update ATR and high watermark
                pos['last_atr'] = float(current['atr'])
                pos['high_watermark'] = max(pos['high_watermark'], float(current['high']))

                # Blow-off detection (same as spot longs)
                vol_sma = float(current['volume_sma']) if current['volume_sma'] > 0 else 1
                volume_ratio = float(current['volume']) / vol_sma
                is_blowoff = (volume_ratio > self.futures_long_strategy_params['volume_blowoff']
                              and float(current['rsi']) > self.futures_long_strategy_params['rsi_blowoff'])
                stop_mult = self.futures_long_strategy_params['atr_mult_tight'] if is_blowoff else self.futures_long_strategy_params['atr_mult']

                # Update trailing stop
                pos['stop_price'] = pos['high_watermark'] - (stop_mult * pos['last_atr'])

                # Check exit conditions
                exit_reason = None
                current_close = float(current['close'])

                if current_close <= pos['stop_price']:
                    suffix = " (BLOW-OFF tightened)" if is_blowoff else ""
                    exit_reason = f'Trailing stop ({stop_mult}x ATR){suffix}'

                if not exit_reason and pd.notna(prev['exit_low']) and current_close < float(prev['exit_low']):
                    exit_reason = f'Donchian exit ({self.futures_long_strategy_params["exit_period"]}-day low)'

                if not exit_reason and current_close <= pos['entry_price'] * (1 - self.futures_long_emergency_stop_pct / 100):
                    exit_reason = f'Emergency stop ({self.futures_long_emergency_stop_pct}%)'

                # Check partial profit taking
                if not exit_reason:
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= self.futures_long_strategy_params['tp1_pct']:
                        self.execute_futures_long_close(symbol, current_close,
                                                        f'Futures Long Partial TP1 (+{self.futures_long_strategy_params["tp1_pct"]}%)',
                                                        fraction=self.futures_long_strategy_params['tp1_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 1
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.futures_long_strategy_params['tp1_fraction'])
                    elif pos['partials_taken'] == 1 and gain_pct >= self.futures_long_strategy_params['tp2_pct']:
                        self.execute_futures_long_close(symbol, current_close,
                                                        f'Futures Long Partial TP2 (+{self.futures_long_strategy_params["tp2_pct"]}%)',
                                                        fraction=self.futures_long_strategy_params['tp2_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 2
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.futures_long_strategy_params['tp2_fraction'])

                if exit_reason:
                    self.execute_futures_long_close(symbol, current_close, exit_reason)

                self.save_state()
                time.sleep(0.3)

            except Exception as e:
                self.logger.error(f"Futures long exit check failed for {symbol}: {e}")

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
                df = strategy.fetch_daily_candles(spot_sym, limit=self.short_strategy_params['lookback'])
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
                is_bounce_risk = (volume_ratio > self.short_strategy_params['volume_blowoff']
                                  and float(current['rsi']) < self.short_strategy_params['rsi_blowoff'])
                stop_mult = self.short_strategy_params['atr_mult_tight'] if is_bounce_risk else self.short_strategy_params['atr_mult']

                # Update inverted trailing stop
                pos['stop_price'] = pos['low_watermark'] + (stop_mult * pos['last_atr'])

                exit_reason = None

                # 1. Inverted trailing stop: price above stop
                if current_close >= pos['stop_price']:
                    suffix = " (bounce-risk tightened)" if is_bounce_risk else ""
                    exit_reason = f'Short trailing stop ({stop_mult}x ATR){suffix}'

                # 2. Donchian exit: above N-day high (reversal)
                if not exit_reason and pd.notna(prev.get('exit_high')) and current_close > float(prev['exit_high']):
                    exit_reason = f'Donchian exit ({self.short_strategy_params["exit_period"]}-day high)'

                # 3. Emergency stop: price rose too much
                if not exit_reason and current_close >= pos['entry_price'] * (1 + self.short_emergency_stop_pct / 100):
                    exit_reason = f'Short emergency stop (+{self.short_emergency_stop_pct}%)'

                # 4. Max hold days
                if not exit_reason and pos['hold_days'] >= self.short_max_hold_days:
                    exit_reason = f'Max hold ({self.short_max_hold_days} days)'

                # Check partial profit taking (inverted: profit when price drops)
                if not exit_reason:
                    gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= self.short_strategy_params['tp1_pct']:
                        self.execute_short_close(symbol, current_close,
                                                 f'Short Partial TP1 (-{self.short_strategy_params["tp1_pct"]}% drop)',
                                                 fraction=self.short_strategy_params['tp1_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 1
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.short_strategy_params['tp1_fraction'])
                    elif pos['partials_taken'] == 1 and gain_pct >= self.short_strategy_params['tp2_pct']:
                        self.execute_short_close(symbol, current_close,
                                                 f'Short Partial TP2 (-{self.short_strategy_params["tp2_pct"]}% drop)',
                                                 fraction=self.short_strategy_params['tp2_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 2
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.short_strategy_params['tp2_fraction'])

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
                  f"SMA200: ${bull_details['sma_200']:,.2f}")
        else:
            print(f"\nBull filter: {bull_status}")
        print(f"Bear filter: {bear_status}")
        self.logger.info(f"Bull filter: {bull_status} | Bear filter: {bear_status}")

        # ===== FUTURES LONG PYRAMIDING (gated by bull filter) =====
        if self.futures_long_pyramid_enabled and self.futures_long_enabled and is_bull and self.positions:
            print(f"\nChecking futures long pyramid opportunities...")
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if pos.get('side') != 'FUTURES_LONG' or pos.get('pyramided'):
                    continue

                try:
                    spot_sym = pos.get('spot_symbol', PERP_TO_SPOT.get(symbol, ''))
                    strategy = self.futures_long_strategies.get(spot_sym)
                    if not strategy:
                        continue
                    df = strategy.fetch_daily_candles(spot_sym, limit=self.futures_long_strategy_params['lookback'])
                    df = strategy.calculate_indicators(df)
                    if len(df) < 25:
                        continue

                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    current_close = float(current['close'])
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100

                    new_high = (pd.notna(prev['donchian_high'])
                                and current_close > float(prev['donchian_high']))

                    if gain_pct >= self.futures_long_pyramid_gain_pct and new_high:
                        atr = float(current['atr'])
                        self.execute_futures_long_pyramid(symbol, current_close, atr, gain_pct, prices)
                    else:
                        if gain_pct >= self.futures_long_pyramid_gain_pct:
                            print(f"  {symbol} FL: +{gain_pct:.1f}% but no new 20d high")
                        else:
                            print(f"  {symbol} FL: +{gain_pct:.1f}% (need +{self.futures_long_pyramid_gain_pct}%)")

                    time.sleep(0.3)

                except Exception as e:
                    self.logger.error(f"Futures long pyramid check failed for {symbol}: {e}")

        # ===== LONG PYRAMIDING (gated by bull filter, spot longs only) =====
        if self.pyramid_enabled and is_bull and self.positions:
            print(f"\nChecking long pyramid opportunities...")
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if pos.get('side') in ('SHORT', 'FUTURES_LONG') or pos.get('pyramided'):
                    continue

                try:
                    strategy = self.strategies.get(symbol)
                    if not strategy:
                        continue
                    df = strategy.fetch_daily_candles(symbol, limit=self.strategy_params['lookback'])
                    df = strategy.calculate_indicators(df)
                    if len(df) < 25:
                        continue

                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    current_close = float(current['close'])
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100

                    new_high = (pd.notna(prev['donchian_high'])
                                and current_close > float(prev['donchian_high']))

                    if gain_pct >= self.pyramid_gain_pct and new_high:
                        atr = float(current['atr'])
                        self.execute_pyramid(symbol, current_close, atr, gain_pct, prices)
                    else:
                        if gain_pct >= self.pyramid_gain_pct:
                            print(f"  {symbol}: +{gain_pct:.1f}% but no new 20d high")
                        else:
                            print(f"  {symbol}: +{gain_pct:.1f}% (need +{self.pyramid_gain_pct}%)")

                    time.sleep(0.3)

                except Exception as e:
                    self.logger.error(f"Pyramid check failed for {symbol}: {e}")

        # ===== FUTURES LONG ENTRIES (gated by bull filter + enabled, checked FIRST for lower fees) =====
        print(f"\nChecking futures long entries...")
        if not self.futures_long_enabled:
            print(f"  FUTURES LONG DISABLED — skipping")
        elif not is_bull:
            print(f"  FUTURES LONG ENTRIES BLOCKED — bull filter is {bull_status}")
            self.logger.info(f"Futures long entries blocked: {bull_status}")
        else:
            for symbol in self.futures_long_coin_universe:
                perp_id = SPOT_TO_PERP.get(symbol, '')
                # No same-coin conflict (spot or perp)
                if symbol in self.positions or perp_id in self.positions:
                    continue
                if len(self.positions) >= self.max_positions:
                    print(f"  Max positions reached, skipping remaining")
                    break

                try:
                    strategy = self.futures_long_strategies[symbol]
                    signal = strategy.generate_signal(symbol)

                    if signal['signal'] == 'BUY':
                        atr = signal.get('atr', 0)
                        self.execute_futures_long_open(symbol, signal['price'], atr, signal['reason'], prices)
                    else:
                        print(f"  {symbol} FL: {signal['signal']} — {signal['reason']}")

                    time.sleep(0.3)

                except Exception as e:
                    self.logger.error(f"Futures long entry check failed for {symbol}: {e}")
                    print(f"  {symbol} FL: ERROR — {e}")

        # ===== SPOT LONG ENTRIES (gated by bull filter, fallback for coins without perps) =====
        print(f"\nChecking spot long entries...")
        if not is_bull:
            print(f"  LONG ENTRIES BLOCKED — bull filter is {bull_status}")
            self.logger.info(f"Long entries blocked: {bull_status}")
        else:
            for symbol in self.coin_universe:
                if symbol in self.positions or SPOT_TO_PERP.get(symbol, '') in self.positions:
                    continue
                if len(self.positions) >= self.max_positions:
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

        # ===== SHORT ENTRIES (gated by short_enabled + bear filter / death cross) =====
        print(f"\nChecking short entries...")
        if not self.short_enabled:
            print(f"  SHORT ENTRIES BLOCKED — shorts disabled via dashboard")
            self.logger.info(f"Short entries blocked: disabled via dashboard")
        elif not is_bear:
            print(f"  SHORT ENTRIES BLOCKED — bear filter is {bear_status}")
            self.logger.info(f"Short entries blocked: {bear_status}")
        else:
            for symbol in self.short_coin_universe:
                perp_id = SPOT_TO_PERP.get(symbol, '')
                # No same-coin conflict
                if symbol in self.positions or perp_id in self.positions:
                    continue
                if len(self.positions) >= self.max_positions:
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
        total_return = ((equity - self.starting_capital) / self.starting_capital) * 100
        longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        fl = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
        print(f"\nPortfolio: ${equity:,.0f} ({total_return:+.1f}% total return, {longs}L/{fl}FL/{shorts}S)")

        # Sync daily check event to Supabase
        self.sync.sync_event("daily_check",
                             f"Daily scan: {len(self.positions)}/{self.max_positions} positions ({longs}L/{fl}FL/{shorts}S), "
                             f"${equity:,.0f} equity, bull={bull_status}, bear={bear_status}",
                             {"equity": round(equity, 2), "positions": len(self.positions),
                              "longs": longs, "futures_longs": fl, "shorts": shorts,
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
            if side in ('SHORT', 'FUTURES_LONG'):
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
                        pos['stop_price'] = pos['high_watermark'] - (self.strategy_params['atr_mult'] * pos['last_atr'])
                        if pos['stop_price'] > old_stop:
                            print(f"  {symbol}: Stop ratcheted ${old_stop:,.2f} -> ${pos['stop_price']:,.2f}")
                    self.save_state()
                    self.sync.sync_position_update(symbol, pos)

                sell_reason = None
                if pos['stop_price'] > 0 and price <= pos['stop_price']:
                    sell_reason = f'Trailing stop hit (${pos["stop_price"]:,.2f}) — intra-day monitor'
                elif price <= pos['entry_price'] * (1 - self.emergency_stop_pct / 100):
                    sell_reason = f'Emergency stop ({self.emergency_stop_pct}%) — intra-day monitor'

                if not sell_reason:
                    gain_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= self.strategy_params['tp1_pct']:
                        self.execute_sell(symbol, price, f'Partial TP1 (+{self.strategy_params["tp1_pct"]}%) — intra-day',
                                          fraction=self.strategy_params['tp1_fraction'])
                        pos['partials_taken'] = 1
                        pos['remaining_fraction'] *= (1 - self.strategy_params['tp1_fraction'])
                        continue
                    elif pos['partials_taken'] == 1 and gain_pct >= self.strategy_params['tp2_pct']:
                        self.execute_sell(symbol, price, f'Partial TP2 (+{self.strategy_params["tp2_pct"]}%) — intra-day',
                                          fraction=self.strategy_params['tp2_fraction'])
                        pos['partials_taken'] = 2
                        pos['remaining_fraction'] *= (1 - self.strategy_params['tp2_fraction'])
                        continue

                if sell_reason:
                    self.execute_sell(symbol, price, sell_reason)
                else:
                    stop_str = f" | Stop: ${pos['stop_price']:,.2f}" if pos['stop_price'] else ""
                    print(f"  [{now}] {symbol}: ${price:,.2f} ({pnl:+.1f}%){stop_str}")

            elif side == 'FUTURES_LONG':
                # === FUTURES LONG TRAILING STOP (same direction as spot long) ===
                pnl = ((price - pos['entry_price']) / pos['entry_price']) * 100

                if price > pos['high_watermark']:
                    old_stop = pos['stop_price']
                    pos['high_watermark'] = price
                    if pos['last_atr'] > 0:
                        pos['stop_price'] = pos['high_watermark'] - (self.futures_long_strategy_params['atr_mult'] * pos['last_atr'])
                        if pos['stop_price'] > old_stop:
                            print(f"  {symbol} FL: Stop ratcheted ${old_stop:,.2f} -> ${pos['stop_price']:,.2f}")
                    self.save_state()
                    self.sync.sync_position_update(symbol, pos)

                sell_reason = None
                if pos['stop_price'] > 0 and price <= pos['stop_price']:
                    sell_reason = f'Trailing stop hit (${pos["stop_price"]:,.2f}) — intra-day monitor'
                elif price <= pos['entry_price'] * (1 - self.futures_long_emergency_stop_pct / 100):
                    sell_reason = f'Emergency stop ({self.futures_long_emergency_stop_pct}%) — intra-day monitor'

                if not sell_reason:
                    gain_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= self.futures_long_strategy_params['tp1_pct']:
                        self.execute_futures_long_close(symbol, price,
                                                        f'Futures Long Partial TP1 (+{self.futures_long_strategy_params["tp1_pct"]}%) — intra-day',
                                                        fraction=self.futures_long_strategy_params['tp1_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 1
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.futures_long_strategy_params['tp1_fraction'])
                        continue
                    elif pos['partials_taken'] == 1 and gain_pct >= self.futures_long_strategy_params['tp2_pct']:
                        self.execute_futures_long_close(symbol, price,
                                                        f'Futures Long Partial TP2 (+{self.futures_long_strategy_params["tp2_pct"]}%) — intra-day',
                                                        fraction=self.futures_long_strategy_params['tp2_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 2
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.futures_long_strategy_params['tp2_fraction'])
                        continue

                if sell_reason:
                    self.execute_futures_long_close(symbol, price, sell_reason)
                else:
                    stop_str = f" | Stop: ${pos['stop_price']:,.2f}" if pos['stop_price'] else ""
                    print(f"  [{now}] {symbol} FL: ${price:,.2f} ({pnl:+.1f}%){stop_str}")

            elif side == 'SHORT':
                # === SHORT TRAILING STOP (inverted) ===
                pnl = ((pos['entry_price'] - price) / pos['entry_price']) * 100

                # Update low watermark (ratchets down)
                if price < pos['low_watermark']:
                    old_stop = pos['stop_price']
                    pos['low_watermark'] = price
                    if pos['last_atr'] > 0:
                        pos['stop_price'] = pos['low_watermark'] + (self.short_strategy_params['atr_mult'] * pos['last_atr'])
                        if pos['stop_price'] < old_stop:
                            print(f"  {symbol}: Short stop ratcheted ${old_stop:,.2f} -> ${pos['stop_price']:,.2f}")
                    self.save_state()
                    self.sync.sync_position_update(symbol, pos)

                sell_reason = None
                if pos['stop_price'] > 0 and price >= pos['stop_price']:
                    sell_reason = f'Short trailing stop hit (${pos["stop_price"]:,.2f}) — intra-day'
                elif price >= pos['entry_price'] * (1 + self.short_emergency_stop_pct / 100):
                    sell_reason = f'Short emergency stop (+{self.short_emergency_stop_pct}%) — intra-day'

                if not sell_reason:
                    gain_pct = pnl  # already inverted
                    if pos['partials_taken'] == 0 and gain_pct >= self.short_strategy_params['tp1_pct']:
                        self.execute_short_close(symbol, price,
                                                 f'Short Partial TP1 (-{self.short_strategy_params["tp1_pct"]}%) — intra-day',
                                                 fraction=self.short_strategy_params['tp1_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 1
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.short_strategy_params['tp1_fraction'])
                        continue
                    elif pos['partials_taken'] == 1 and gain_pct >= self.short_strategy_params['tp2_pct']:
                        self.execute_short_close(symbol, price,
                                                 f'Short Partial TP2 (-{self.short_strategy_params["tp2_pct"]}%) — intra-day',
                                                 fraction=self.short_strategy_params['tp2_fraction'])
                        if symbol in self.positions:
                            self.positions[symbol]['partials_taken'] = 2
                            self.positions[symbol]['remaining_fraction'] *= (1 - self.short_strategy_params['tp2_fraction'])
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
        total_return = ((equity - self.starting_capital) / self.starting_capital) * 100
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
            spot_sym = pos.get('spot_symbol', sym) if side in ('SHORT', 'FUTURES_LONG') else sym
            current = prices.get(sym) or prices.get(spot_sym, pos['entry_price'])
            if side == 'SHORT':
                pnl = ((pos['entry_price'] - current) / pos['entry_price']) * 100
                days = pos.get('hold_days', 0)
                side_tag = " 🔻S"
            elif side == 'FUTURES_LONG':
                pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                days = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 86400
                side_tag = " 🔵FL"
            else:
                pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                days = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 86400
                side_tag = ""
            emoji = "🟢" if pnl >= 0 else "🔴"
            pos_lines.append(f"{emoji}{side_tag} {sym}: ${current:,.2f} ({pnl:+.1f}%, {days:.0f}d)")

        pos_text = "\n".join(pos_lines) if pos_lines else "No open positions"
        longs = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        fl = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
        shorts = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')

        return_emoji = "🟢" if total_return >= 0 else "🔴"
        msg = (
            f"📋 <b>DAILY PORTFOLIO SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: <b>${equity:,.0f}</b>\n"
            f"{return_emoji} Return: <b>{pnl_sign}{total_return:.1f}%</b>\n"
            f"💵 Cash: ${self.cash:,.0f}\n"
            f"📊 Positions: {len(self.positions)}/{self.max_positions} ({longs}L/{fl}FL/{shorts}S)\n"
            f"{filter_lines}\n"
            f"<b>Open Positions:</b>\n{pos_text}\n\n"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        self.send_notification(msg, category="summary")

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
        print("DONCHIAN CHANNEL BREAKOUT — TRI-MODE BOT")
        print("=" * 80)
        print(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'} TRADING")
        print(f"Spot long coins: {', '.join(self.coin_universe)}")
        print(f"Futures long coins: {', '.join(self.futures_long_coin_universe)} ({'ON' if self.futures_long_enabled else 'OFF'}, {self.futures_long_leverage}x)")
        print(f"Short coins: {', '.join(self.short_coin_universe)}")
        print(f"Max positions: {self.max_positions} (shared pool)")
        print(f"Risk per trade: {self.risk_per_trade_pct}% (spot) / {self.futures_long_risk_per_trade_pct}% (FL) / {self.short_risk_per_trade_pct}% (short)")
        print(f"Spot long trailing: {self.strategy_params['atr_mult']}x ATR")
        print(f"Futures long trailing: {self.futures_long_strategy_params['atr_mult']}x ATR")
        print(f"Short trailing: {self.short_strategy_params['atr_mult']}x ATR")
        print(f"Pyramiding: {'ON (+' + str(self.pyramid_gain_pct) + '% / ' + str(self.pyramid_risk_pct) + '% risk)' if self.pyramid_enabled else 'OFF'}")
        print(f"Bull filter: {'ON' if self.bull_filter_enabled else 'OFF'}")
        print(f"Shorts: {'ON' if self.short_enabled else 'OFF'} (bear filter: {'ON' if self.bear_filter_enabled else 'OFF'})")
        print(f"Config: loaded from Supabase (reloads each daily check)")
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

        # Count positions by side
        long_count = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
        fl_count = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
        short_count = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')

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
        pyramid_line = f"📈 Pyramiding: +{self.pyramid_gain_pct}% / {self.pyramid_risk_pct}% risk\n" if self.pyramid_enabled else ""
        fl_line = f"🔵 Futures Long: <b>{'ON' if self.futures_long_enabled else 'OFF'}</b> ({self.futures_long_leverage}x leverage)\n"
        short_line = f"🟣 Shorts: <b>{'ON' if self.short_enabled else 'OFF'}</b>\n"
        msg = (
            f"🚀 <b>DONCHIAN BOT STARTED (TRI-MODE)</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Strategy: Donchian Breakout (Daily)\n"
            f"🛡 Spot long trailing: {self.strategy_params['atr_mult']}x ATR\n"
            f"🔵 FL trailing: {self.futures_long_strategy_params['atr_mult']}x ATR\n"
            f"🔻 Short trailing: {self.short_strategy_params['atr_mult']}x ATR\n"
            f"🪙 Spot: {len(self.coin_universe)} | FL: {len(self.futures_long_coin_universe)} | Short: {len(self.short_coin_universe)}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"📈 Positions: {len(self.positions)}/{self.max_positions} (L:{long_count} FL:{fl_count} S:{short_count})\n"
            f"{pyramid_line}"
            f"{fl_line}"
            f"{short_line}"
            f"{bull_line}"
            f"{bear_line}"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        self.send_notification(msg, category="summary")

        # Sync startup to Supabase
        self.sync.sync_event("startup",
                             f"Tri-mode bot started ({len(self.coin_universe)}L/{len(self.futures_long_coin_universe)}FL/{len(self.short_coin_universe)}S coins, "
                             f"{'PAPER' if self.paper_trading else 'LIVE'})",
                             {"long_coins": self.coin_universe,
                              "futures_long_coins": self.futures_long_coin_universe,
                              "short_coins": self.short_coin_universe,
                              "max_positions": self.max_positions,
                              "futures_long_enabled": self.futures_long_enabled,
                              "futures_long_leverage": self.futures_long_leverage,
                              "short_enabled": self.short_enabled,
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
            total_return = ((equity - self.starting_capital) / self.starting_capital) * 100
            long_count = sum(1 for p in self.positions.values() if p.get('side', 'LONG') == 'LONG')
            fl_count = sum(1 for p in self.positions.values() if p.get('side') == 'FUTURES_LONG')
            short_count = sum(1 for p in self.positions.values() if p.get('side') == 'SHORT')
            self.sync.sync_event("shutdown",
                                 f"Bot stopped. Equity: ${equity:,.0f} ({total_return:+.1f}%)",
                                 {"equity": equity, "total_return": round(total_return, 1),
                                  "open_positions": len(self.positions),
                                  "long_positions": long_count,
                                  "futures_long_positions": fl_count,
                                  "short_positions": short_count})
            print(f"\nFinal equity: ${equity:,.0f} ({total_return:+.1f}%)")
            print(f"Open positions: {len(self.positions)} (L:{long_count} FL:{fl_count} S:{short_count})")
            for sym, pos in self.positions.items():
                side = pos.get('side', 'LONG')
                spot_sym = pos.get('spot_symbol', sym) if side in ('SHORT', 'FUTURES_LONG') else sym
                try:
                    price = self.fetch_current_price(spot_sym)
                except Exception:
                    price = None
                if price and price > 0:
                    if side == 'SHORT':
                        pnl = ((pos['entry_price'] - price) / pos['entry_price']) * 100
                        tag = "S"
                    elif side == 'FUTURES_LONG':
                        pnl = ((price - pos['entry_price']) / pos['entry_price']) * 100
                        tag = "FL"
                    else:
                        pnl = ((price - pos['entry_price']) / pos['entry_price']) * 100
                        tag = "L"
                    print(f"  [{tag}] {sym}: entry ${pos['entry_price']:,.2f} ({pnl:+.1f}%)")
                else:
                    tag = "S" if side == 'SHORT' else ("FL" if side == 'FUTURES_LONG' else "L")
                    print(f"  [{tag}] {sym}: entry ${pos['entry_price']:,.2f} (no price)")


# ============================================================================
# BOT MANAGER — multi-user orchestrator
# ============================================================================

class BotManager:
    """Manages multiple UserBot instances, one per active user."""

    def __init__(self):
        self.bots: dict[str, UserBot] = {}
        self.logger, _ = setup_logging()

    def load_active_users(self):
        """Query Supabase for bot_config rows where bot_enabled=true."""
        try:
            sync = SupabaseSync()
            if not sync.url or not sync.key:
                return []
            import requests as req
            resp = req.get(
                f"{sync.url}/rest/v1/bot_config",
                params={"select": "user_id,trading_mode,bot_enabled", "bot_enabled": "eq.true"},
                headers={"apikey": sync.key, "Authorization": f"Bearer {sync.key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            self.logger.warning(f"[MANAGER] Failed to load users: {resp.status_code}")
            return []
        except Exception as e:
            self.logger.warning(f"[MANAGER] Failed to load users: {e}")
            return []

    def refresh_bots(self):
        """Add bots for new active users, remove bots for disabled users."""
        users = self.load_active_users()
        active_ids = {u["user_id"] for u in users}

        # Add new users
        for user_row in users:
            uid = user_row["user_id"]
            if uid not in self.bots:
                try:
                    mode = user_row.get("trading_mode", "paper")
                    paper = mode != "live"
                    self.bots[uid] = UserBot(user_id=uid, paper_trading=paper)
                    self.logger.info(f"[MANAGER] Created bot for user {uid[:8]} (mode={mode})")
                except Exception as e:
                    self.logger.error(f"[MANAGER] Failed to create bot for {uid[:8]}: {e}")

        # Remove disabled users
        for uid in list(self.bots.keys()):
            if uid not in active_ids:
                self.logger.info(f"[MANAGER] Removing bot for disabled user {uid[:8]}")
                try:
                    self.bots[uid].save_state()
                except Exception:
                    pass
                del self.bots[uid]

    def run(self):
        """Main loop: refresh users, run daily checks and trailing stops."""
        print("=" * 80)
        print("DONCHIAN CHANNEL BREAKOUT — MULTI-USER BOT MANAGER")
        print("=" * 80)
        print(f"Daily check: {DAILY_CHECK_HOUR:02d}:{DAILY_CHECK_MINUTE:02d} UTC")
        print(f"Stop check: every {STOP_CHECK_INTERVAL // 60} min")
        print("=" * 80)

        self.refresh_bots()
        print(f"Active bots: {len(self.bots)}")

        last_daily_check = None
        last_summary = {}
        last_user_refresh = time.time()

        try:
            while True:
                now = datetime.now(timezone.utc)

                # Refresh user list every 5 min
                if time.time() - last_user_refresh > 300:
                    self.refresh_bots()
                    last_user_refresh = time.time()

                # Daily signal check (once per day at configured time)
                if (now.hour == DAILY_CHECK_HOUR and
                        now.minute >= DAILY_CHECK_MINUTE and
                        last_daily_check != now.date()):
                    for uid, bot in self.bots.items():
                        try:
                            bot.load_remote_config()
                            bot.daily_check()
                        except Exception as e:
                            self.logger.error(f"[MANAGER] Daily check failed for {uid[:8]}: {e}")
                    last_daily_check = now.date()

                # Daily summary at 20:00 UTC
                if now.hour == 20:
                    for uid, bot in self.bots.items():
                        if last_summary.get(uid) != now.date():
                            try:
                                bot.send_daily_summary()
                                last_summary[uid] = now.date()
                            except Exception as e:
                                self.logger.error(f"[MANAGER] Summary failed for {uid[:8]}: {e}")

                # Trailing stop monitoring for bots with positions
                for uid, bot in self.bots.items():
                    if bot.positions:
                        try:
                            bot.check_trailing_stops()
                        except Exception as e:
                            self.logger.error(f"[MANAGER] Stop check failed for {uid[:8]}: {e}")

                # Sleep until next check
                next_check_min = STOP_CHECK_INTERVAL // 60
                print(f"\nNext check in {next_check_min} min... ({len(self.bots)} active bots)")
                time.sleep(STOP_CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nBot manager stopped by user")
            for uid, bot in self.bots.items():
                try:
                    bot.save_state()
                except Exception:
                    pass
            print(f"Saved state for {len(self.bots)} bots")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    manager = BotManager()
    manager.run()


if __name__ == "__main__":
    main()
