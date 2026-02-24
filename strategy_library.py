"""Strategy Library Manager - Stores and retrieves approved strategies"""
import json
import os
from datetime import datetime

class StrategyLibrary:
    def __init__(self, library_path='strategy_library.json'):
        self.library_path = library_path
        self.library = self.load_library()
    
    def load_library(self):
        """Load existing library or create new one"""
        if os.path.exists(self.library_path):
            with open(self.library_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'strategies': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def save_library(self):
        """Save library to disk"""
        self.library['last_updated'] = datetime.now().isoformat()
        with open(self.library_path, 'w') as f:
            json.dump(self.library, f, indent=2)
        print(f"💾 Library saved to {self.library_path}")
    
    def add_strategy(self, strategy_data):
        """Add a new approved strategy to library"""
        
        # Check if strategy already exists
        for existing in self.library['strategies']:
            if existing['name'] == strategy_data['name'] and existing['regime'] == strategy_data['regime']:
                print(f"⚠️  Strategy '{strategy_data['name']}' for {strategy_data['regime']} already exists")
                return False
        
        # Add strategy
        strategy_entry = {
            'name': strategy_data['name'],
            'regime': strategy_data['regime'],
            'indicators': strategy_data.get('indicators', []),
            'win_rate': strategy_data.get('win_rate', 0),
            'profit_factor': strategy_data.get('profit_factor', 0),
            'total_pnl': strategy_data.get('total_pnl', 0),
            'avg_win': strategy_data.get('avg_win', 0),
            'avg_loss': strategy_data.get('avg_loss', 0),
            'risk_level': strategy_data.get('risk_level', 'Medium'),
            'date_added': datetime.now().isoformat(),
            'status': 'APPROVED',
            'code_file': strategy_data.get('code_file', ''),
            'backtest_file': strategy_data.get('backtest_file', ''),
            'notes': strategy_data.get('notes', '')
        }
        
        self.library['strategies'].append(strategy_entry)
        self.save_library()
        
        print(f"✅ Added '{strategy_data['name']}' to library for {strategy_data['regime']} regime")
        return True
    
    def get_strategies_for_regime(self, regime):
        """Get all strategies for a specific regime"""
        return [s for s in self.library['strategies'] if s['regime'] == regime and s['status'] == 'APPROVED']
    
    def get_best_strategy_for_regime(self, regime):
        """Get highest performing strategy for a regime"""
        regime_strategies = self.get_strategies_for_regime(regime)
        
        if not regime_strategies:
            return None
        
        # Sort by profit factor (best risk/reward)
        best = max(regime_strategies, key=lambda x: x['profit_factor'])
        return best
    
    def display_library(self):
        """Display entire library"""
        print("\n" + "="*80)
        print("📚 STRATEGY LIBRARY")
        print("="*80)
        
        if not self.library['strategies']:
            print("No strategies in library yet.")
            return
        
        # Group by regime
        regimes = {}
        for strategy in self.library['strategies']:
            regime = strategy['regime']
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(strategy)
        
        for regime, strategies in regimes.items():
            print(f"\n{regime} REGIME ({len(strategies)} strategies):")
            print("-" * 80)
            
            for i, s in enumerate(strategies, 1):
                print(f"{i}. {s['name']}")
                print(f"   Win Rate: {s['win_rate']*100:.1f}% | Profit Factor: {s['profit_factor']:.2f} | P&L: {s['total_pnl']:.2f}%")
                print(f"   Risk: {s['risk_level']} | Added: {s['date_added'][:10]}")
                if s['notes']:
                    print(f"   Notes: {s['notes']}")
        
        print("="*80)
    
    def export_for_trading_bot(self, output_file='active_strategies.json'):
        """Export strategies in format TradingBot can use"""
        
        export_data = {}
        
        for regime in ['RANGING', 'TRENDING', 'VOLATILE']:
            best = self.get_best_strategy_for_regime(regime)
            if best:
                export_data[regime] = {
                    'name': best['name'],
                    'code_file': best['code_file'],
                    'win_rate': best['win_rate'],
                    'profit_factor': best['profit_factor']
                }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📤 Exported active strategies to {output_file}")

def main():
    """Add Williams %R to library"""
    print("="*80)
    print("📚 STRATEGY LIBRARY MANAGER")
    print("="*80 + "\n")
    
    library = StrategyLibrary()
    
    # Add Williams %R strategy from our backtest
    williams_r_strategy = {
        'name': 'Williams %R Mean Reversion',
        'regime': 'RANGING',
        'indicators': ['Williams %R (14)', 'SMA (21)'],
        'win_rate': 0.549,  # 54.9%
        'profit_factor': 2.09,
        'total_pnl': 7.15,
        'avg_win': 0.49,
        'avg_loss': -0.29,
        'risk_level': 'Medium',
        'code_file': 'strategies/ranging/williams_r.py',
        'backtest_file': 'backtest_ranging_20260221_135653.json',
        'notes': 'Tested on 1134 ranging periods, 51 trades, excellent profit factor'
    }
    
    library.add_strategy(williams_r_strategy)
    
    # Display library
    library.display_library()
    
    # Show stats
    print(f"\n📊 Library Stats:")
    print(f"   Total Strategies: {len(library.library['strategies'])}")
    print(f"   RANGING: {len(library.get_strategies_for_regime('RANGING'))}")
    print(f"   TRENDING: {len(library.get_strategies_for_regime('TRENDING'))}")
    print(f"   VOLATILE: {len(library.get_strategies_for_regime('VOLATILE'))}")
    
    # Export for TradingBot
    library.export_for_trading_bot()

if __name__ == "__main__":
    main()