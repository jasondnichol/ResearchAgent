"""Phase 4: Coin Expansion — Screen new coins for the Donchian universe.

For each candidate coin:
  1. Run the Donchian strategy (4x ATR + pyramid + bull filter) individually
  2. Compare per-coin metrics to existing universe
  3. Run the expanded portfolio backtest + walk-forward

Selection criteria: >35% WR or positive contribution over 4 years.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_donchian_daily import (
    fetch_and_cache_daily_data,
    fetch_all_coins,
    calculate_indicators,
    compute_stats,
    compute_per_coin_stats,
    DEFAULT_PARAMS,
    COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)
from backtest_phase3 import backtest_portfolio_phase3

# Phase 3 params (current production)
PHASE3_PARAMS = {
    **DEFAULT_PARAMS,
    'atr_mult': 4.0,
}


# ============================================================================
# CANDIDATE COINS (2+ years of Coinbase data)
# ============================================================================

# Current universe (8 coins in production)
CURRENT_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD',
]

# Candidates to evaluate (2+ years of data)
CANDIDATES = [
    'DOGE-USD', 'DOT-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD',
    'AAVE-USD', 'FIL-USD', 'SHIB-USD', 'FET-USD',
    'OP-USD', 'INJ-USD', 'APT-USD', 'ARB-USD',
    'SEI-USD', 'TIA-USD', 'RENDER-USD',
]

# Previously dropped (for reference)
DROPPED = ['AVAX-USD', 'HBAR-USD']


# ============================================================================
# SINGLE-COIN SCREENING (quick per-coin backtest)
# ============================================================================

def screen_single_coin(symbol, df, params, bull_filter):
    """Run Donchian strategy on a single coin with bull filter.

    Returns trade list and summary stats.
    """
    df_ind = calculate_indicators(df, params)
    df_ind = df_ind.dropna(subset=['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi'])
    df_ind = df_ind.reset_index(drop=True)

    if len(df_ind) < 30:
        return [], {}

    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    trades = []
    position = None

    for i in range(1, len(df_ind)):
        row = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1]
        date = row['time'].date() if hasattr(row['time'], 'date') else row['time']
        is_bull = bull_filter.get(date, False)

        current_close = float(row['close'])

        # Check exit if in position
        if position is not None:
            position['high_watermark'] = max(position['high_watermark'], float(row['high']))
            current_atr = float(row['atr'])

            exit_reason = None

            # Blow-off detection
            vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
            vol_ratio = float(row['volume']) / vol_sma
            is_blowoff = vol_ratio > params['volume_blowoff'] and float(row['rsi']) > params['rsi_blowoff']
            stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

            trailing_stop = position['high_watermark'] - (stop_mult * current_atr)
            if current_close <= trailing_stop:
                exit_reason = 'Trailing stop'

            if not exit_reason and pd.notna(prev['exit_low']) and current_close < float(prev['exit_low']):
                exit_reason = 'Donchian exit'

            if not exit_reason and current_close <= position['entry_price'] * 0.85:
                exit_reason = 'Emergency stop'

            if exit_reason:
                exit_price = current_close * (1 - cost_per_side)
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': row['time'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'size_usd': 1000,
                    'win': pnl_pct > 0,
                })
                position = None

        # Check entry if flat and bull
        if position is None and is_bull:
            if pd.isna(prev['donchian_high']):
                continue

            breakout = current_close > float(prev['donchian_high'])
            if params['volume_mult'] > 0:
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
            else:
                volume_ok = True
            trend_ok = current_close > float(row['ema_21'])

            if breakout and volume_ok and trend_ok:
                entry_price = current_close * (1 + cost_per_side)
                position = {
                    'entry_price': entry_price,
                    'entry_time': row['time'],
                    'high_watermark': float(row['high']),
                }

    # Close remaining position
    if position is not None:
        last = df_ind.iloc[-1]
        exit_price = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        trades.append({
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': last['time'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of backtest',
            'size_usd': 1000,
            'win': pnl_pct > 0,
        })

    # Compute summary
    if trades:
        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        total_pnl = sum(t['pnl_pct'] for t in trades)
        win_pnl = sum(t['pnl_pct'] for t in wins) if wins else 0
        loss_pnl = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0.001

        summary = {
            'trades': len(trades),
            'wins': len(wins),
            'wr': len(wins) / len(trades) * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'pf': win_pnl / loss_pnl if loss_pnl > 0 else float('inf'),
            'total_pnl': total_pnl,
        }
    else:
        summary = {
            'trades': 0, 'wins': 0, 'wr': 0,
            'avg_win': 0, 'avg_loss': 0, 'pf': 0, 'total_pnl': 0,
        }

    return trades, summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("PHASE 4: COIN EXPANSION — SCREENING NEW CANDIDATES")
    print("  Strategy: Donchian Breakout + Bull Filter + 4x ATR (no pyramid for single-coin screen)")
    print("  Selection: >35% WR or positive total P&L over backtest period")
    print("=" * 100)

    # Fetch BTC for bull filter
    btc_df = fetch_and_cache_daily_data('BTC-USD', years=4)
    bull_filter = compute_btc_bull_filter(btc_df)
    bull_days = sum(1 for v in bull_filter.values() if v)
    total_days = len(bull_filter)
    print(f"\n  BTC bull filter: {bull_days}/{total_days} bull days ({bull_days/total_days*100:.0f}%)")

    # ======================================================================
    # SCREEN ALL COINS (current + candidates)
    # ======================================================================
    print_section("SINGLE-COIN SCREENING (all coins)")

    all_symbols = CURRENT_UNIVERSE + CANDIDATES
    screening_results = []

    for symbol in all_symbols:
        df = fetch_and_cache_daily_data(symbol, years=4)
        if df is None or len(df) < 200:
            print(f"  {symbol}: SKIP (insufficient data)")
            screening_results.append((symbol, {'trades': 0, 'total_pnl': 0, 'wr': 0, 'pf': 0}, 0))
            continue

        trades, summary = screen_single_coin(symbol, df, PHASE3_PARAMS, bull_filter)
        is_current = symbol in CURRENT_UNIVERSE
        screening_results.append((symbol, summary, len(df)))

    # Sort by total P&L
    screening_results.sort(key=lambda x: -x[1].get('total_pnl', 0))

    print(f"\n  {'='*100}")
    print(f"  {'Symbol':<12s} {'Days':>6s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} "
          f"{'AvgWin':>8s} {'AvgLoss':>8s} {'PF':>7s} {'TotalP&L':>10s}  Status")
    print(f"  {'-'*90}")

    keepers = []
    for symbol, summary, n_days in screening_results:
        t = summary.get('trades', 0)
        if t == 0:
            print(f"  {symbol:<12s} {n_days:>6d}       0     0    0.0%     0.0%     0.0%    0.00      0.00%  SKIP")
            continue

        is_current = symbol in CURRENT_UNIVERSE
        is_dropped = symbol in DROPPED

        wr = summary['wr']
        pnl = summary['total_pnl']
        pf = summary['pf']

        # Selection: keep if WR > 35% OR positive total P&L
        keep = wr > 35 or pnl > 0
        tag = "CURRENT" if is_current else ("DROPPED" if is_dropped else ("ADD" if keep else "DROP"))

        print(f"  {symbol:<12s} {n_days:>6d} {t:>7d} {summary['wins']:>5d} {wr:>6.1f}% "
              f"{summary['avg_win']:>+7.1f}% {summary['avg_loss']:>+7.1f}% "
              f"{pf:>7.2f} {pnl:>+9.2f}%  {tag}")

        if keep and not is_dropped:
            keepers.append(symbol)

    # ======================================================================
    # PORTFOLIO BACKTEST: CURRENT 8 vs EXPANDED
    # ======================================================================
    print_section("PORTFOLIO COMPARISON")

    # Fetch all data
    all_keepers = sorted(set(keepers))
    print(f"\n  Proposed expanded universe ({len(all_keepers)} coins): {', '.join(all_keepers)}")
    print(f"  Current universe ({len(CURRENT_UNIVERSE)} coins): {', '.join(CURRENT_UNIVERSE)}")

    # New coins being added
    new_coins = [c for c in all_keepers if c not in CURRENT_UNIVERSE]
    print(f"  New additions: {', '.join(new_coins) if new_coins else 'none'}")

    # Fetch all coin data
    print(f"\n  Loading data for all coins...")
    current_data = {}
    expanded_data = {}

    for symbol in set(CURRENT_UNIVERSE + all_keepers):
        df = fetch_and_cache_daily_data(symbol, years=4)
        if df is not None and len(df) > 100:
            if symbol in CURRENT_UNIVERSE:
                current_data[symbol] = df
            if symbol in all_keepers:
                expanded_data[symbol] = df

    print(f"  Current: {len(current_data)} coins | Expanded: {len(expanded_data)} coins")

    # Full period backtest — current vs expanded
    print(f"\n  Running full-period backtests...")

    current_trades, current_eq, current_cap, current_pa = backtest_portfolio_phase3(
        current_data, PHASE3_PARAMS, bull_filter, pyramiding=True)
    current_stats = compute_stats(current_trades, 'Current 8 coins')

    expanded_trades, expanded_eq, expanded_cap, expanded_pa = backtest_portfolio_phase3(
        expanded_data, PHASE3_PARAMS, bull_filter, pyramiding=True)
    expanded_stats = compute_stats(expanded_trades, f'Expanded {len(expanded_data)} coins')

    print(f"\n  {'='*130}")
    print(f"  FULL PERIOD RESULTS")
    print(f"  {'='*130}")
    print_stats_row(f'Current ({len(current_data)} coins)', current_stats, current_eq)
    print(f"      (pyramid adds: {current_pa})")
    print_stats_row(f'Expanded ({len(expanded_data)} coins)', expanded_stats, expanded_eq)
    print(f"      (pyramid adds: {expanded_pa})")

    # Per-coin breakdown for expanded
    expanded_coin_stats = compute_per_coin_stats(expanded_trades)
    if expanded_coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN (EXPANDED, FULL PERIOD):")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}  New?")
        print(f"  {'-'*65}")
        for sym, cs in sorted(expanded_coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            is_new = sym not in CURRENT_UNIVERSE
            tag = " *NEW*" if is_new else ""
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%{tag}")

    # ======================================================================
    # WALK-FORWARD: CURRENT vs EXPANDED
    # ======================================================================
    print_section("WALK-FORWARD (Train 2022-2024, Test 2025-2026)")

    cutoff = pd.Timestamp('2025-01-01')

    # Split data for both universes
    current_train, current_test = split_coin_data(current_data, cutoff)
    expanded_train, expanded_test = split_coin_data(expanded_data, cutoff)

    btc_train = current_train.get('BTC-USD')
    btc_test = current_test.get('BTC-USD')
    train_bull = compute_btc_bull_filter(btc_train) if btc_train is not None else {}
    test_bull = compute_btc_bull_filter(btc_test) if btc_test is not None else {}

    # Current universe
    ct_trades, ct_eq, _, ct_pa = backtest_portfolio_phase3(
        current_train, PHASE3_PARAMS, train_bull, pyramiding=True)
    ct_stats = compute_stats(ct_trades, 'Current [train]')

    co_trades, co_eq, _, co_pa = backtest_portfolio_phase3(
        current_test, PHASE3_PARAMS, test_bull, pyramiding=True)
    co_stats = compute_stats(co_trades, 'Current [OOS]')

    # Expanded universe
    et_trades, et_eq, _, et_pa = backtest_portfolio_phase3(
        expanded_train, PHASE3_PARAMS, train_bull, pyramiding=True)
    et_stats = compute_stats(et_trades, 'Expanded [train]')

    eo_trades, eo_eq, _, eo_pa = backtest_portfolio_phase3(
        expanded_test, PHASE3_PARAMS, test_bull, pyramiding=True)
    eo_stats = compute_stats(eo_trades, 'Expanded [OOS]')

    print(f"\n  {'='*130}")
    print(f"  WALK-FORWARD RESULTS")
    print(f"  {'='*130}")
    print(f"\n  TRAIN (2022-2024):")
    print_stats_row(f'Current ({len(current_train)} coins)', ct_stats, ct_eq)
    print_stats_row(f'Expanded ({len(expanded_train)} coins)', et_stats, et_eq)

    print(f"\n  TEST / OUT-OF-SAMPLE (2025-2026):")
    print_stats_row(f'Current ({len(current_test)} coins)', co_stats, co_eq)
    print_stats_row(f'Expanded ({len(expanded_test)} coins)', eo_stats, eo_eq)

    # Per-coin OOS for expanded
    eo_coin_stats = compute_per_coin_stats(eo_trades)
    if eo_coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN OOS (EXPANDED):")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}  New?")
        print(f"  {'-'*65}")
        for sym, cs in sorted(eo_coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            is_new = sym not in CURRENT_UNIVERSE
            tag = " *NEW*" if is_new else ""
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%{tag}")

    # ======================================================================
    # VERDICT
    # ======================================================================
    print_section("PHASE 4 VERDICT")

    curr_full_ret = current_stats.get('total_return_pct', 0)
    exp_full_ret = expanded_stats.get('total_return_pct', 0)
    curr_oos_ret = co_stats.get('total_return_pct', 0)
    exp_oos_ret = eo_stats.get('total_return_pct', 0)
    curr_oos_dd = compute_max_drawdown(co_eq)[0]
    exp_oos_dd = compute_max_drawdown(eo_eq)[0]

    print(f"\n  FULL PERIOD:")
    print(f"    Current {len(current_data)} coins:  {curr_full_ret:+.1f}% return, "
          f"PF {current_stats.get('profit_factor',0):.2f}")
    print(f"    Expanded {len(expanded_data)} coins: {exp_full_ret:+.1f}% return, "
          f"PF {expanded_stats.get('profit_factor',0):.2f}")
    print(f"    Improvement: {exp_full_ret - curr_full_ret:+.1f}%")

    print(f"\n  OUT-OF-SAMPLE:")
    print(f"    Current:  {curr_oos_ret:+.1f}%, PF {co_stats.get('profit_factor',0):.2f}, "
          f"MaxDD {curr_oos_dd:.1f}%")
    print(f"    Expanded: {exp_oos_ret:+.1f}%, PF {eo_stats.get('profit_factor',0):.2f}, "
          f"MaxDD {exp_oos_dd:.1f}%")
    print(f"    Improvement: {exp_oos_ret - curr_oos_ret:+.1f}%")

    if exp_oos_ret > curr_oos_ret and exp_full_ret >= curr_full_ret * 0.9:
        print(f"\n  VERDICT: EXPAND — More coins improved OOS performance")
    elif exp_full_ret > curr_full_ret:
        print(f"\n  VERDICT: MARGINAL — Better full-period but check OOS carefully")
    else:
        print(f"\n  VERDICT: HOLD — Current universe is sufficient")

    # Final recommended universe
    print(f"\n  RECOMMENDED UNIVERSE:")
    for sym, summary, n_days in screening_results:
        if sym in all_keepers:
            pnl = summary.get('total_pnl', 0)
            wr = summary.get('wr', 0)
            is_new = sym not in CURRENT_UNIVERSE
            tag = " (NEW)" if is_new else ""
            print(f"    {sym}{tag}: {summary.get('trades',0)} trades, "
                  f"{wr:.0f}% WR, {pnl:+.1f}% P&L")


if __name__ == "__main__":
    main()
