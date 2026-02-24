"""Update strategy library to include timeframes"""
import json

# Load existing library
with open('strategy_library.json', 'r') as f:
    library = json.load(f)

# Update Williams %R with timeframe
for strategy in library['strategies']:
    if strategy['name'] == 'Williams %R Mean Reversion':
        strategy['timeframe'] = '1H'  # Williams %R works on 1-hour candles
        strategy['candle_lookback'] = 100  # Needs 100 candles for indicators
        strategy['check_interval'] = '5m'  # Check for signals every 5 minutes

# Save updated library
with open('strategy_library.json', 'w') as f:
    json.dump(library, f, indent=2)

print("✅ Updated library with timeframes")
print("\nStrategy Timeframes:")
for s in library['strategies']:
    print(f"  {s['name']}: {s.get('timeframe', 'NOT SET')} timeframe")