"""Integrated Strategy Switcher with Williams %R, ADX Momentum, and BB Reversion"""
import json
import time
import requests
from datetime import datetime
from market_regime import MarketRegimeDetector
from williams_r_strategy import WilliamsRStrategy
from adx_momentum_strategy import ADXMomentumStrategy
from bb_reversion_strategy import BBReversionStrategy
from notify import send_telegram, send_startup_message

class IntegratedSwitcher:
    def __init__(self, paper_trading=True):
        self.library_file = 'strategy_library.json'
        self.paper_trading = paper_trading
        self.position = None
        self.entry_price = 0
        self.trades_log = []

        # Regime change tracking for force-sell
        self.current_regime = None
        self.non_matching_regime_count = 0

        # Stop/take-profit monitoring between hourly signals
        self.stop_price = None
        self.take_profit_price = None
        self.high_watermark = None
        self.last_atr = None
        self.active_strategy_name = None

        # Initialize strategies
        self.williams_r = WilliamsRStrategy()
        self.adx_momentum = ADXMomentumStrategy()
        self.bb_reversion = BBReversionStrategy()
    
    def load_library(self):
        """Load strategy library"""
        with open(self.library_file, 'r') as f:
            return json.load(f)
    
    def detect_regime(self):
        """Detect current market regime"""
        detector = MarketRegimeDetector()
        return detector.detect_regime('BTC-USD')
    
    def get_strategy_for_regime(self, regime_type):
        """Get strategy for regime"""
        library = self.load_library()
        
        matches = [
            s for s in library['strategies']
            if s['regime'] == regime_type 
            and s.get('timeframe') == '1H'
            and s['status'] == 'APPROVED'
        ]
        
        if not matches:
            return None
        
        return max(matches, key=lambda x: x['profit_factor'])
    
    def execute_trade(self, signal):
        """Execute trade (paper or live)"""
        
        if signal['signal'] == 'BUY' and not self.position:
            # Enter position
            self.position = 'LONG'
            self.entry_price = signal['price']
            
            print(f"\n🟢 BUY EXECUTED")
            print(f"   Price: ${signal['price']:,.2f}")
            print(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
            
            self.trades_log.append({
                'action': 'BUY',
                'price': signal['price'],
                'time': signal['time'],
                'mode': 'PAPER' if self.paper_trading else 'LIVE'
            })
        
        elif signal['signal'] == 'SELL' and self.position:
            # Exit position
            exit_price = signal['price']
            pnl = ((exit_price - self.entry_price) / self.entry_price) * 100
            
            print(f"\n🔴 SELL EXECUTED")
            print(f"   Entry: ${self.entry_price:,.2f}")
            print(f"   Exit: ${exit_price:,.2f}")
            print(f"   P&L: {pnl:+.2f}%")
            print(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
            
            self.trades_log.append({
                'action': 'SELL',
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'time': signal['time'],
                'mode': 'PAPER' if self.paper_trading else 'LIVE'
            })
            
            self.position = None
            self.entry_price = 0
            self.stop_price = None
            self.take_profit_price = None
            self.high_watermark = None
            self.last_atr = None
            self.active_strategy_name = None

        else:
            print(f"\n⚪ HOLD")
            print(f"   Reason: {signal['reason']}")
            if self.position:
                current_pnl = ((signal['price'] - self.entry_price) / self.entry_price) * 100
                print(f"   Open Position P&L: {current_pnl:+.2f}%")
    
    def force_sell(self, current_price, regime_type):
        """Force-sell open position due to regime change with no matching strategy"""
        if not self.position:
            return

        exit_price = current_price
        pnl = ((exit_price - self.entry_price) / self.entry_price) * 100

        print(f"\n⚠️  FORCE-SELL: No strategy for {regime_type} regime (3 consecutive checks)")
        print(f"   Entry: ${self.entry_price:,.2f}")
        print(f"   Exit: ${exit_price:,.2f}")
        print(f"   P&L: {pnl:+.2f}%")

        self.trades_log.append({
            'action': 'SELL',
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'time': datetime.now().isoformat(),
            'mode': 'PAPER' if self.paper_trading else 'LIVE',
            'exit_reason': f'Force-sell: regime changed to {regime_type}'
        })

        # Telegram notification
        pnl_sign = "+" if pnl >= 0 else ""
        msg = (
            f"⚠️ <b>FORCE-SELL (Regime Change)</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Entry: ${self.entry_price:,.2f}\n"
            f"💰 Exit: <b>${exit_price:,.2f}</b>\n"
            f"{'🟢' if pnl >= 0 else '🔴'} P&L: <b>{pnl_sign}{pnl:.2f}%</b>\n"
            f"📊 Regime: {regime_type} (no strategy)\n"
            f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}"
        )
        send_telegram(msg)

        self.position = None
        self.entry_price = 0
        self.stop_price = None
        self.take_profit_price = None
        self.high_watermark = None
        self.last_atr = None
        self.active_strategy_name = None

    def fetch_current_price(self, symbol='BTC-USD'):
        """Lightweight price check — just current price, no candle history"""
        url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['price'])

    def check_between_signals(self):
        """Lightweight 5-min price check for stop loss monitoring"""
        if not self.position:
            return

        try:
            price = self.fetch_current_price()
        except Exception as e:
            print(f"   ⚠️  Price check failed: {e}")
            return

        current_pnl = ((price - self.entry_price) / self.entry_price) * 100
        now = datetime.now().strftime('%H:%M:%S')

        # Update trailing stop for ADX Momentum
        if self.active_strategy_name == 'ADX Momentum Thrust' and self.high_watermark and self.last_atr:
            if price > self.high_watermark:
                self.high_watermark = price
                self.stop_price = self.high_watermark - (1.5 * self.last_atr)

        # Check take-profit
        sell_reason = None
        if self.take_profit_price and price >= self.take_profit_price:
            sell_reason = f'Take-profit hit (${self.take_profit_price:,.2f}) — 5-min monitor'
        # Check stop loss
        elif self.stop_price and price <= self.stop_price:
            sell_reason = f'Stop loss hit (${self.stop_price:,.2f}) — 5-min monitor'
        # Emergency stop: 5% below entry regardless of strategy
        elif price <= self.entry_price * 0.95:
            sell_reason = f'Emergency stop (-5% from entry ${self.entry_price:,.2f})'

        if sell_reason:
            print(f"\n   🚨 [{now}] STOP TRIGGERED at ${price:,.2f}")
            print(f"      Reason: {sell_reason}")

            signal = {
                'signal': 'SELL',
                'reason': sell_reason,
                'price': price,
                'time': datetime.now().isoformat()
            }
            self.execute_trade(signal)

            # Telegram alert
            pnl = ((price - self.entry_price) / self.entry_price) * 100
            pnl_sign = "+" if pnl >= 0 else ""
            is_profit = 'Take-profit' in sell_reason
            header = "TARGET HIT (5-min Monitor)" if is_profit else "STOP LOSS (5-min Monitor)"
            emoji = "🎯" if is_profit else "🚨"
            msg = (
                f"{emoji} <b>{header}</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 Entry: ${self.entry_price:,.2f}\n"
                f"💰 Exit: <b>${price:,.2f}</b>\n"
                f"{'🟢' if pnl >= 0 else '🔴'} P&L: <b>{pnl_sign}{pnl:.2f}%</b>\n"
                f"📊 Reason: {sell_reason}\n"
                f"🏷 Mode: {'PAPER' if self.paper_trading else 'LIVE'}"
            )
            send_telegram(msg)
        else:
            stop_info = f" | Stop: ${self.stop_price:,.2f}" if self.stop_price else ""
            tp_info = f" | TP: ${self.take_profit_price:,.2f}" if self.take_profit_price else ""
            print(f"   📡 [{now}] BTC: ${price:,.2f} | P&L: {current_pnl:+.2f}%{stop_info}{tp_info}")

    def run_once(self):
        """Single iteration"""
        print(f"\n{'='*80}")
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # 1. Detect regime
        regime = self.detect_regime()
        regime_type = regime['regime_type']

        print(f"\n📊 Regime: {regime_type}")
        print(f"   BTC: ${regime['current_price']:,.2f}")
        print(f"   Trend: {regime['trend_direction']} ({regime['trend_strength']})")

        # Track regime changes
        if self.current_regime is not None and regime_type != self.current_regime:
            print(f"   🔄 Regime changed: {self.current_regime} → {regime_type}")
        self.current_regime = regime_type

        # 2. Get strategy
        strategy = self.get_strategy_for_regime(regime_type)

        if not strategy:
            print(f"\n⏸️  No {regime_type} strategy - sitting out")

            # Force-sell logic: no matching strategy + open position
            if self.position:
                self.non_matching_regime_count += 1
                print(f"   ⚠️  Open position with no strategy ({self.non_matching_regime_count}/3 checks)")

                if self.non_matching_regime_count >= 3:
                    self.force_sell(regime['current_price'], regime_type)
                    self.non_matching_regime_count = 0
            return

        # Reset counter when a matching strategy exists
        self.non_matching_regime_count = 0

        print(f"\n✅ Strategy: {strategy['name']}")
        print(f"   Win Rate: {strategy['win_rate']*100:.1f}% | PF: {strategy['profit_factor']:.2f}")

        # 3. Generate signal
        if strategy['name'] == 'Williams %R Mean Reversion':
            signal = self.williams_r.generate_signal('BTC-USD')
        elif strategy['name'] == 'ADX Momentum Thrust':
            signal = self.adx_momentum.generate_signal('BTC-USD')
        elif strategy['name'] == 'Bollinger Band Mean Reversion':
            signal = self.bb_reversion.generate_signal('BTC-USD')
        else:
            signal = {'signal': 'HOLD', 'reason': 'Strategy not implemented'}

        # 4. Execute
        self.execute_trade(signal)

        # 5. Update stop/take-profit tracking for 5-min monitoring
        if self.position:
            self.active_strategy_name = strategy['name']
            atr = signal.get('atr')
            if atr:
                self.last_atr = atr

            # Stop levels
            if strategy['name'] == 'ADX Momentum Thrust' and self.last_atr:
                self.high_watermark = max(self.high_watermark or 0, signal['price'])
                self.stop_price = self.high_watermark - (1.5 * self.last_atr)
            elif strategy['name'] == 'Bollinger Band Mean Reversion' and self.last_atr:
                self.stop_price = self.entry_price - (1.5 * self.last_atr)
            # Williams %R: no price-based stop (emergency 5% stop still applies)

            # Take-profit levels
            if strategy['name'] == 'Bollinger Band Mean Reversion':
                sma_20 = signal.get('sma_20')
                if sma_20:
                    self.take_profit_price = sma_20
            elif strategy['name'] == 'Williams %R Mean Reversion':
                sma_21 = signal.get('sma_21')
                if sma_21:
                    self.take_profit_price = sma_21 * 1.015
            # ADX Momentum: no fixed target (trailing stop locks in gains)

            if self.stop_price:
                print(f"   🛡️  Stop: ${self.stop_price:,.2f}")
            if self.take_profit_price:
                print(f"   🎯  Take-profit: ${self.take_profit_price:,.2f}")

        # 6. Stats
        if len(self.trades_log) > 0:
            print(f"\n📊 Session Stats:")
            print(f"   Total Trades: {len([t for t in self.trades_log if t['action'] == 'SELL'])}")
            print(f"   Open Position: {self.position or 'None'}")

def main():
    print("="*80)
    print("🤖 INTEGRATED TRADING SYSTEM")
    print("="*80)
    print("Mode: PAPER TRADING")
    print("="*80)
    
    switcher = IntegratedSwitcher(paper_trading=True)
    
    print("\n🚀 Starting trading loop...")
    print("   Full signal check: every 1 hour")
    print("   Stop loss monitor: every 5 min")
    print("   Press Ctrl+C to stop\n")

    send_startup_message()
    
    try:
        while True:
            switcher.run_once()
            # 11 lightweight checks at 5-min intervals = 55 min, then next full run
            for i in range(11):
                mins_left = (11 - i) * 5
                print(f"\n💤 Next full check in {mins_left} min...")
                time.sleep(300)  # 5 minutes
                switcher.check_between_signals()
    except KeyboardInterrupt:
        print("\n\n✅ System stopped")
        
        # Show final stats
        if len(switcher.trades_log) > 0:
            print("\n" + "="*80)
            print("📊 FINAL STATS")
            print("="*80)
            
            sells = [t for t in switcher.trades_log if t['action'] == 'SELL']
            if sells:
                total_pnl = sum(t['pnl_pct'] for t in sells)
                avg_pnl = total_pnl / len(sells)
                wins = len([t for t in sells if t['pnl_pct'] > 0])
                
                print(f"Completed Trades: {len(sells)}")
                print(f"Wins: {wins} ({wins/len(sells)*100:.1f}%)")
                print(f"Total P&L: {total_pnl:+.2f}%")
                print(f"Average P&L: {avg_pnl:+.2f}%")

if __name__ == "__main__":
    main()