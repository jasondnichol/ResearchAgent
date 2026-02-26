"""Multi-Coin Donchian Channel Breakout Trading Bot — Daily Candles

Production bot that runs the Donchian breakout strategy on 8 liquid Coinbase coins.
Designed for bull cycle crypto trading with realistic position sizing.

Architecture:
  - Daily signal check at 00:15 UTC (after daily candle close)
  - Trailing stop monitoring every 30 minutes
  - Max 4 concurrent positions, 2% risk per trade
  - 4x ATR trailing stop (wider to let winners run)
  - Pyramiding: add 1% risk tranche to winners at +15% on new 20-day high
  - Partial profit taking at +10% and +20%
  - BTC bull filter: only enter when BTC > SMA(200) AND SMA(50) > SMA(200)
  - Telegram notifications for all trades

Coins: BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR
Dropped: HBAR (0 wins in 4yr backtest), AVAX (1/9 wins)
"""
import json
import time
import os
import requests
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from donchian_breakout_strategy import DonchianBreakoutStrategy
from notify import send_telegram, setup_logging

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

# Bull market filter (BTC macro gate)
BULL_FILTER_ENABLED = True
BULL_SMA_FAST = 50   # SMA(50) must be above SMA(200) (golden cross)
BULL_SMA_SLOW = 200  # BTC close must be above SMA(200)
BULL_LOOKBACK = 220  # candles to fetch for SMA(200) computation

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

        # Strategy instances (one per coin)
        self.strategies = {}
        for symbol in COIN_UNIVERSE:
            self.strategies[symbol] = DonchianBreakoutStrategy(STRATEGY_PARAMS)

        # Logging
        self.logger, self.trade_logger = setup_logging()

        # Load saved state if exists
        self.load_state()

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
            state['positions'][symbol] = {
                'entry_price': pos['entry_price'],
                'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time']),
                'high_watermark': pos['high_watermark'],
                'partials_taken': pos['partials_taken'],
                'remaining_fraction': pos['remaining_fraction'],
                'size_usd': pos['size_usd'],
                'last_atr': pos.get('last_atr', 0),
                'stop_price': pos.get('stop_price', 0),
                'pyramided': pos.get('pyramided', False),
            }
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
                self.positions[symbol] = {
                    'entry_price': pos_data['entry_price'],
                    'entry_time': datetime.fromisoformat(pos_data['entry_time']),
                    'high_watermark': pos_data['high_watermark'],
                    'partials_taken': pos_data['partials_taken'],
                    'remaining_fraction': pos_data['remaining_fraction'],
                    'size_usd': pos_data['size_usd'],
                    'last_atr': pos_data.get('last_atr', 0),
                    'stop_price': pos_data.get('stop_price', 0),
                    'pyramided': pos_data.get('pyramided', False),
                }
            if self.positions:
                print(f"Restored {len(self.positions)} open positions from state file")
                for sym, pos in self.positions.items():
                    print(f"  {sym}: entry ${pos['entry_price']:,.2f}, size ${pos['size_usd']:,.0f}")
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
        """Get current prices for all coins in universe"""
        prices = {}
        for symbol in COIN_UNIVERSE:
            try:
                prices[symbol] = self.fetch_current_price(symbol)
                time.sleep(0.15)
            except Exception as e:
                self.logger.warning(f"Price fetch failed for {symbol}: {e}")
        return prices

    # ------------------------------------------------------------------
    # BULL MARKET FILTER
    # ------------------------------------------------------------------

    def check_bull_filter(self):
        """Check if BTC is in a bull market.

        Bull = BTC close > SMA(200) AND SMA(50) > SMA(200).
        Returns (is_bull, details_dict).
        """
        if not BULL_FILTER_ENABLED:
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
    # PORTFOLIO HELPERS
    # ------------------------------------------------------------------

    def total_equity(self, prices=None):
        """Calculate total portfolio equity"""
        equity = self.cash
        for symbol, pos in self.positions.items():
            if prices and symbol in prices:
                current_val = pos['size_usd'] * (prices[symbol] / pos['entry_price'])
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

    # ------------------------------------------------------------------
    # DAILY SIGNAL CHECK
    # ------------------------------------------------------------------

    def daily_check(self):
        """Full daily signal check — entries and indicator-based exits"""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*80}")
        print(f"DAILY SIGNAL CHECK — {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*80}")

        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        print(f"Portfolio: ${equity:,.0f} ({len(self.positions)}/{MAX_POSITIONS} positions, ${self.cash:,.0f} cash)")

        if self.positions:
            print(f"\nOpen positions:")
            for sym, pos in self.positions.items():
                current = prices.get(sym, pos['entry_price'])
                pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                days = (now - pos['entry_time']).total_seconds() / 86400
                print(f"  {sym}: ${current:,.2f} ({pnl:+.1f}%, {days:.0f}d) | Stop: ${pos['stop_price']:,.2f}")

        # Check exits first for open positions
        print(f"\nChecking exits...")
        for symbol in list(self.positions.keys()):
            try:
                strategy = self.strategies[symbol]
                df = strategy.fetch_daily_candles(symbol, limit=STRATEGY_PARAMS['lookback'])
                df = strategy.calculate_indicators(df)

                if len(df) < 25:
                    continue

                current = df.iloc[-1]
                prev = df.iloc[-2]
                pos = self.positions[symbol]

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

                # 1. Trailing stop
                if current_close <= pos['stop_price']:
                    suffix = " (BLOW-OFF tightened)" if is_blowoff else ""
                    exit_reason = f'Trailing stop ({stop_mult}x ATR){suffix}'

                # 2. Donchian exit: below 10-day low
                if not exit_reason and pd.notna(prev['exit_low']) and current_close < float(prev['exit_low']):
                    exit_reason = 'Donchian exit (10-day low)'

                # 3. Emergency stop
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
                self.logger.error(f"Exit check failed for {symbol}: {e}")

        # Check bull filter before scanning for entries / pyramiding
        is_bull, bull_details = self.check_bull_filter()
        bull_status = bull_details.get('status', 'UNKNOWN')
        if bull_details.get('btc_close'):
            print(f"\nBull filter: {bull_status} | BTC: ${bull_details['btc_close']:,.2f} | "
                  f"SMA50: ${bull_details['sma_50']:,.2f} | SMA200: ${bull_details['sma_200']:,.2f}")
        else:
            print(f"\nBull filter: {bull_status}")
        self.logger.info(f"Bull filter: {bull_status} | {bull_details}")

        # Check pyramiding on existing positions (gated by bull filter)
        if PYRAMID_ENABLED and is_bull and self.positions:
            print(f"\nChecking pyramid opportunities...")
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if pos.get('pyramided'):
                    continue

                try:
                    strategy = self.strategies[symbol]
                    df = strategy.fetch_daily_candles(symbol, limit=STRATEGY_PARAMS['lookback'])
                    df = strategy.calculate_indicators(df)
                    if len(df) < 25:
                        continue

                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    current_close = float(current['close'])
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100

                    # Pyramid: up +15% AND making new 20-day high
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

        # Check entries for coins we're not in (gated by bull filter)
        print(f"\nChecking entries...")
        if not is_bull:
            print(f"  ENTRIES BLOCKED — bull filter is {bull_status}")
            print(f"  BTC must be above SMA(200) with golden cross to enter new positions")
            self.logger.info(f"Entries blocked by bull filter: {bull_status}")
        else:
            for symbol in COIN_UNIVERSE:
                if symbol in self.positions:
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
                    self.logger.error(f"Entry check failed for {symbol}: {e}")
                    print(f"  {symbol}: ERROR — {e}")

        # Daily summary
        equity = self.total_equity(prices)
        total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
        print(f"\nPortfolio: ${equity:,.0f} ({total_return:+.1f}% total return)")

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
            if symbol not in prices:
                continue

            pos = self.positions[symbol]
            price = prices[symbol]
            pnl = ((price - pos['entry_price']) / pos['entry_price']) * 100

            # Update high watermark (ratchets up)
            if price > pos['high_watermark']:
                old_stop = pos['stop_price']
                pos['high_watermark'] = price
                if pos['last_atr'] > 0:
                    pos['stop_price'] = pos['high_watermark'] - (STRATEGY_PARAMS['atr_mult'] * pos['last_atr'])
                    if pos['stop_price'] > old_stop:
                        print(f"  {symbol}: Stop ratcheted ${old_stop:,.2f} -> ${pos['stop_price']:,.2f}")
                self.save_state()

            # Check trailing stop
            sell_reason = None
            if pos['stop_price'] > 0 and price <= pos['stop_price']:
                sell_reason = f'Trailing stop hit (${pos["stop_price"]:,.2f}) — intra-day monitor'

            # Emergency stop
            elif price <= pos['entry_price'] * (1 - EMERGENCY_STOP_PCT / 100):
                sell_reason = f'Emergency stop ({EMERGENCY_STOP_PCT}%) — intra-day monitor'

            # Check partial TP on intra-day moves
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

    # ------------------------------------------------------------------
    # DAILY SUMMARY
    # ------------------------------------------------------------------

    def send_daily_summary(self):
        """Send portfolio summary via Telegram"""
        prices = self.fetch_all_prices()
        equity = self.total_equity(prices)
        total_return = ((equity - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
        pnl_sign = "+" if total_return >= 0 else ""

        # Bull filter status
        is_bull, bull_details = self.check_bull_filter()
        bull_emoji = "🟢" if is_bull else "🔴"
        bull_status = bull_details.get('status', 'UNKNOWN')
        bull_line = ""
        if BULL_FILTER_ENABLED and bull_details.get('btc_close'):
            bull_line = (f"{bull_emoji} Bull filter: <b>{bull_status}</b> "
                         f"(BTC ${bull_details['btc_close']:,.0f} vs SMA200 ${bull_details['sma_200']:,.0f})\n")

        pos_lines = []
        for sym, pos in self.positions.items():
            current = prices.get(sym, pos['entry_price'])
            pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
            days = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 86400
            emoji = "🟢" if pnl >= 0 else "🔴"
            pos_lines.append(f"{emoji} {sym}: ${current:,.2f} ({pnl:+.1f}%, {days:.0f}d)")

        pos_text = "\n".join(pos_lines) if pos_lines else "No open positions"

        msg = (
            f"📋 <b>DAILY PORTFOLIO SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: <b>${equity:,.0f}</b>\n"
            f"{'🟢' if total_return >= 0 else '🔴'} Return: <b>{pnl_sign}{total_return:.1f}%</b>\n"
            f"💵 Cash: ${self.cash:,.0f}\n"
            f"📊 Positions: {len(self.positions)}/{MAX_POSITIONS}\n"
            f"{bull_line}\n"
            f"<b>Open Positions:</b>\n{pos_text}\n\n"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        """Main trading loop"""
        print("=" * 80)
        print("DONCHIAN CHANNEL BREAKOUT — MULTI-COIN BOT")
        print("=" * 80)
        print(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'} TRADING")
        print(f"Coins: {', '.join(COIN_UNIVERSE)}")
        print(f"Max positions: {MAX_POSITIONS}")
        print(f"Risk per trade: {RISK_PER_TRADE_PCT}%")
        print(f"Trailing stop: {STRATEGY_PARAMS['atr_mult']}x ATR")
        print(f"Pyramiding: {'ON (+' + str(PYRAMID_GAIN_PCT) + '% / ' + str(PYRAMID_RISK_PCT) + '% risk)' if PYRAMID_ENABLED else 'OFF'}")
        print(f"Bull filter: {'ON' if BULL_FILTER_ENABLED else 'OFF'}")
        print(f"Daily check: {DAILY_CHECK_HOUR:02d}:{DAILY_CHECK_MINUTE:02d} UTC")
        print(f"Stop check: every {STOP_CHECK_INTERVAL // 60} min")
        print("=" * 80)

        # Check bull filter on startup
        is_bull, bull_details = self.check_bull_filter()
        bull_status = bull_details.get('status', 'UNKNOWN')
        bull_emoji = "🟢" if is_bull else "🔴"
        print(f"Bull filter: {bull_status}")

        # Send startup message
        equity = self.total_equity()
        bull_line = ""
        if BULL_FILTER_ENABLED and bull_details.get('btc_close'):
            bull_line = (f"{bull_emoji} Bull filter: <b>{bull_status}</b> "
                         f"(BTC ${bull_details['btc_close']:,.0f} vs SMA200 ${bull_details['sma_200']:,.0f})\n")
        pyramid_line = f"📈 Pyramiding: +{PYRAMID_GAIN_PCT}% / {PYRAMID_RISK_PCT}% risk\n" if PYRAMID_ENABLED else ""
        msg = (
            f"🚀 <b>DONCHIAN BOT STARTED</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Strategy: Donchian Breakout (Daily)\n"
            f"🛡 Trailing: {STRATEGY_PARAMS['atr_mult']}x ATR\n"
            f"🪙 Coins: {len(COIN_UNIVERSE)}\n"
            f"💼 Equity: ${equity:,.0f}\n"
            f"📈 Positions: {len(self.positions)}/{MAX_POSITIONS}\n"
            f"{pyramid_line}"
            f"{bull_line}"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)

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
            print(f"\nFinal equity: ${equity:,.0f} ({total_return:+.1f}%)")
            print(f"Open positions: {len(self.positions)}")
            for sym, pos in self.positions.items():
                pnl = ((self.fetch_current_price(sym) - pos['entry_price']) / pos['entry_price']) * 100
                print(f"  {sym}: entry ${pos['entry_price']:,.2f} ({pnl:+.1f}%)")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    bot = DonchianMultiCoinBot(paper_trading=True)
    bot.run()


if __name__ == "__main__":
    main()
