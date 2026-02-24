"""Simple Strategy Switcher - 1H timeframe only"""
import json
import time
from datetime import datetime
from market_regime import MarketRegimeDetector

class SimpleSwitcher:
    def __init__(self):
        self.library_file = 'strategy_library.json'
        self.current_regime = None
        self.active_strategy = None
        
    def load_library(self):
        """Load strategy library"""
        with open(self.library_file, 'r') as f:
            return json.load(f)
    
    def detect_regime_1h(self):
        """Detect regime on 1H timeframe"""
        detector = MarketRegimeDetector()
        regime = detector.detect_regime('BTC-USD')
        return regime
    
    def get_strategy_for_regime(self, regime_type):
        """Get 1H strategy for this regime"""
        library = self.load_library()
        
        # Find 1H strategies for this regime
        matches = [
            s for s in library['strategies']
            if s['regime'] == regime_type 
            and s.get('timeframe') == '1H'
            and s['status'] == 'APPROVED'
        ]
        
        if not matches:
            return None
        
        # Return best (highest profit factor)
        return max(matches, key=lambda x: x['profit_factor'])
    
    def run_once(self):
        """Single iteration (called every hour)"""
        print(f"\n{'='*80}")
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # 1. Detect regime
        regime = self.detect_regime_1h()
        regime_type = regime['regime_type']
        
        print(f"\n📊 1H Regime: {regime_type}")
        print(f"   BTC Price: ${regime['current_price']:,.2f}")
        print(f"   Trend: {regime['trend_direction']}")
        print(f"   Strength: {regime['trend_strength']}")
        
        # 2. Load strategy
        strategy = self.get_strategy_for_regime(regime_type)
        
        if strategy:
            print(f"\n✅ Active Strategy: {strategy['name']}")
            print(f"   Timeframe: {strategy['timeframe']}")
            print(f"   Win Rate: {strategy['win_rate']*100:.1f}%")
            print(f"   Profit Factor: {strategy['profit_factor']:.2f}")
            
            # 3. Execute (placeholder - add actual logic later)
            print(f"\n📈 Generating signal...")
            print(f"   → HOLD (strategy implementation pending)")
        else:
            print(f"\n⏸️  No approved {strategy_type} strategy for 1H timeframe")
            print(f"   → Sitting out this regime")
        
        print(f"\n💤 Next check in 1 hour...")

def main():
    print("="*80)
    print("🔄 SIMPLE STRATEGY SWITCHER (1H Timeframe)")
    print("="*80)
    
    switcher = SimpleSwitcher()
    
    # Show library
    library = switcher.load_library()
    print(f"\n📚 Loaded {len(library['strategies'])} strategies")
    
    for s in library['strategies']:
        print(f"   • {s['name']} ({s['regime']}, {s.get('timeframe', '?')})")
    
    print("\n🚀 Starting hourly loop...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            switcher.run_once()
            time.sleep(3600)  # 1 hour = 3600 seconds
    except KeyboardInterrupt:
        print("\n\n✅ Switcher stopped")

if __name__ == "__main__":
    main()