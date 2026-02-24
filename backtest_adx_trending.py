"""Backtest ADX Momentum Thrust on 4-year TRENDING data"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from market_regime import RegimeClassifier


def load_4year_data(cache_file='btc_4year_cache.json'):
    """Load cached 4-year BTC data"""
    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        return None

    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    df = pd.DataFrame(cache_data['data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    print(f"Loaded {len(df)} daily candles")
    print(f"Range: {df['time'].min().date()} to {df['time'].max().date()}")
    return df


def classify_regimes(df):
    """Classify each day's market regime using unified RegimeClassifier"""
    df = RegimeClassifier.classify_dataframe(df, min_warmup=50)

    df = df[df['regime'] != 'UNKNOWN']
    regime_counts = df['regime'].value_counts()
    print(f"\nRegime distribution:")
    print(f"  TRENDING: {regime_counts.get('TRENDING', 0)} days ({regime_counts.get('TRENDING', 0)/len(df)*100:.1f}%)")
    print(f"  RANGING:  {regime_counts.get('RANGING', 0)} days ({regime_counts.get('RANGING', 0)/len(df)*100:.1f}%)")
    print(f"  VOLATILE: {regime_counts.get('VOLATILE', 0)} days ({regime_counts.get('VOLATILE', 0)/len(df)*100:.1f}%)")
    return df, regime_counts


def calculate_strategy_indicators(df):
    """Calculate ADX(14), +DI(14), -DI(14), RSI(14), ATR(14), SMA(50) for trading"""
    df = df.copy()
    period = 14

    # True Range
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)

    # +DM / -DM
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index
    )

    # Wilder's smoothing
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    df['plus_di'] = 100 * smooth_plus_dm / atr
    df['minus_di'] = 100 * smooth_minus_dm / atr

    dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR for stop loss
    df['atr'] = atr

    # SMA(50)
    df['sma_50'] = df['close'].rolling(window=50).mean()

    return df


def backtest(df, regime_counts):
    """Run the ADX Momentum Thrust backtest on TRENDING periods"""
    print("\nRunning ADX Momentum Thrust backtest on TRENDING periods...\n")

    df = calculate_strategy_indicators(df)
    df = df.dropna()

    position = None
    trades = []
    entry_price = 0
    entry_time = None
    high_watermark = 0

    for i in range(5, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]

        # Only trade in TRENDING regime
        if current['regime'] != 'TRENDING':
            if position == 'LONG':
                exit_price = float(current['close'])
                exit_time = current['time']
                pnl = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': float(entry_price),
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl_pct': float(pnl),
                    'exit_reason': 'Regime changed to ' + current['regime'],
                    'win': bool(pnl > 0)
                })
                position = None
            continue

        if position is None:
            # --- BUY CONDITIONS ---
            # 1. ADX > 20 and rising over last 5 bars
            adx_rising = df.iloc[i]['adx'] > df.iloc[i - 5]['adx']
            adx_above_20 = current['adx'] > 20

            # 2. +DI > -DI
            plus_di_above = current['plus_di'] > current['minus_di']

            # 3. +DI crossed above -DI within last 15 bars
            di_cross_recent = False
            for offset in range(0, 15):
                idx = i - offset
                if idx >= 1:
                    bar = df.iloc[idx]
                    bar_prev = df.iloc[idx - 1]
                    if bar['plus_di'] > bar['minus_di'] and bar_prev['plus_di'] <= bar_prev['minus_di']:
                        di_cross_recent = True
                        break

            # 4. RSI between 35-78
            rsi_ok = 35 <= current['rsi'] <= 78

            # 5. Price above SMA(50)
            price_above_sma = current['close'] > current['sma_50']

            if adx_above_20 and adx_rising and plus_di_above and di_cross_recent and rsi_ok and price_above_sma:
                position = 'LONG'
                entry_price = float(current['close'])
                entry_time = current['time']
                high_watermark = float(current['close'])

        elif position == 'LONG':
            # Update trailing stop high watermark
            high_watermark = max(high_watermark, float(current['close']))

            exit_reason = None

            # 1. -DI crosses above +DI
            if current['minus_di'] > current['plus_di'] and prev['minus_di'] <= prev['plus_di']:
                exit_reason = '-DI crossed above +DI'

            # 2. ADX drops below 20
            elif current['adx'] < 20:
                exit_reason = 'ADX dropped below 20'

            # 3. RSI > 75
            elif current['rsi'] > 75:
                exit_reason = 'RSI > 75 (overbought)'

            # 4. Trailing stop: price drops 1.5x ATR below high watermark
            elif current['close'] <= high_watermark - (1.5 * current['atr']):
                exit_reason = f'Trailing stop hit (1.5x ATR from high ${high_watermark:,.2f})'

            if exit_reason:
                exit_price = float(current['close'])
                exit_time = current['time']
                pnl = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': float(entry_price),
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl_pct': float(pnl),
                    'exit_reason': exit_reason,
                    'win': bool(pnl > 0)
                })
                position = None

    return trades, regime_counts


def print_results(trades, regime_counts):
    """Print backtest results"""
    print("=" * 80)
    print("ADX MOMENTUM THRUST - 4-YEAR BACKTEST RESULTS (TRENDING ONLY)")
    print("=" * 80)

    if not trades:
        print("No trades generated in TRENDING periods")
        return

    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    win_rate = wins / len(trades)

    avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losses > 0 else 0
    total_pnl = sum(t['pnl_pct'] for t in trades)

    total_win_pnl = sum(t['pnl_pct'] for t in trades if t['win'])
    total_loss_pnl = abs(sum(t['pnl_pct'] for t in trades if not t['win']))
    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

    print(f"Strategy: ADX Momentum Thrust")
    print(f"Regime: TRENDING only")
    print(f"TRENDING days: {regime_counts.get('TRENDING', 0)}")
    print(f"\nResults:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins}")
    print(f"  Losses: {losses}")
    print(f"  Win Rate: {win_rate * 100:.1f}%")
    print(f"  Average Win: {avg_win:+.2f}%")
    print(f"  Average Loss: {avg_loss:+.2f}%")
    print(f"  Total P&L: {total_pnl:+.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print("=" * 80)

    # Trade log
    print("\nTRADE LOG:")
    for idx, trade in enumerate(trades, 1):
        marker = "WIN " if trade['win'] else "LOSS"
        print(f"  {idx:2d}. [{marker}] {str(trade['entry_time'])[:10]} Entry: ${trade['entry_price']:>10,.2f} -> "
              f"Exit: ${trade['exit_price']:>10,.2f} | P&L: {trade['pnl_pct']:+6.2f}% | {trade['exit_reason']}")

    # Exit reason breakdown
    print("\nExit Reason Breakdown:")
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl_pct']
    for reason, stats in sorted(reasons.items(), key=lambda x: -x[1]['count']):
        print(f"  {reason}: {stats['count']} trades, {stats['pnl']:+.2f}% total P&L")

    # Verdict — two-path approval
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    avg_w = avg_win
    avg_l = abs(avg_loss) or 1
    n_trades = len(trades)
    path_a = win_rate >= 0.55 and profit_factor >= 1.5
    path_b = profit_factor >= 1.8 and (avg_w / avg_l) >= 1.5 and n_trades >= 10
    if path_a or path_b:
        path = "A (win-rate)" if path_a else "B (trend-following)"
        print(f"APPROVED for TRENDING markets via path {path}!")
        print(f"  Win Rate: {win_rate * 100:.1f}% | PF: {profit_factor:.2f}")
    else:
        print(f"NOT APPROVED - needs improvement")
        if win_rate < 0.55:
            print(f"  Win Rate: {win_rate * 100:.1f}% (path A needs 55%+)")
        else:
            print(f"  Win Rate: {win_rate * 100:.1f}% (PASS)")
        if profit_factor < 1.8:
            print(f"  Profit Factor: {profit_factor:.2f} (path B needs 1.8+)")
        else:
            print(f"  Profit Factor: {profit_factor:.2f} (PASS)")
        if n_trades < 10:
            print(f"  Trades: {n_trades} (path B needs 10+)")
    print("=" * 80)

    # Save results
    results = {
        'strategy': 'ADX Momentum Thrust',
        'regime_tested': 'TRENDING',
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'trades': [{**t, 'entry_time': str(t['entry_time']), 'exit_time': str(t['exit_time'])} for t in trades]
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_adx_trending_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    print("=" * 80)
    print("ADX MOMENTUM THRUST - 4-YEAR TRENDING BACKTEST")
    print("=" * 80 + "\n")

    df = load_4year_data()
    if df is None:
        return

    df, regime_counts = classify_regimes(df)
    trades, regime_counts = backtest(df, regime_counts)
    print_results(trades, regime_counts)


if __name__ == "__main__":
    main()
