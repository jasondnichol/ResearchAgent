"""Phase 2: Bull Market Filter — Backtest & Walk-Forward Revalidation

Adds a BTC macro filter to the Donchian strategy:
  - BTC close > 200-day SMA
  - Only allow NEW entries when this condition is true
  - Exits are unchanged (trailing stops still work normally)

Also tests dynamic risk: 3% per trade in bull, 2% otherwise.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_donchian_daily import (
    fetch_all_coins,
    calculate_indicators,
    compute_stats,
    compute_per_coin_stats,
    DEFAULT_PARAMS,
    COIN_UNIVERSE,
)
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)


# ============================================================================
# BULL FILTER
# ============================================================================

def compute_btc_bull_filter(btc_df):
    """Compute daily bull/bear signal from BTC price data.

    Bull = BTC close > SMA(200)
    Returns dict: date -> bool
    """
    df = btc_df.copy()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    bull_filter = {}
    for _, row in df.iterrows():
        if pd.isna(row['sma_200']):
            bull_filter[row['time'].date()] = False
            continue

        bull_filter[row['time'].date()] = bool(row['close'] > row['sma_200'])

    # Stats
    total = len(bull_filter)
    bull_days = sum(1 for v in bull_filter.values() if v)
    bear_days = total - bull_days
    print(f"  BTC Bull Filter: {bull_days} bull days ({bull_days/total*100:.1f}%), "
          f"{bear_days} bear days ({bear_days/total*100:.1f}%)")

    return bull_filter


# ============================================================================
# PORTFOLIO BACKTEST WITH BULL FILTER
# ============================================================================

def backtest_portfolio_bull(coin_data, params, bull_filter, dynamic_risk=False):
    """Portfolio backtest with BTC bull filter gate on entries.

    If dynamic_risk=True, uses 3% risk in bull and 2% in bear.
    Entries only allowed when bull_filter[date] is True.
    Exits are unchanged.
    """
    label = params['label']
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    base_risk_pct = params['risk_per_trade_pct'] / 100
    bull_risk_pct = 0.03 if dynamic_risk else base_risk_pct
    bear_risk_pct = base_risk_pct
    capital = params['starting_capital']

    # Pre-calculate indicators for all coins
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_indicators(df, params)
        df_ind = df_ind.dropna(subset=['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi'])
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 30:
            prepared[symbol] = df_ind

    # Build unified daily timeline
    all_dates = set()
    for symbol, df in prepared.items():
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    # Build lookup: symbol -> {date -> row}
    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time'].date()] = row
        lookups[symbol] = lookup

    # Previous day lookups
    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        dates_list = sorted(df['time'].dt.date.tolist())
        for j in range(1, len(dates_list)):
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j-1])
        prev_lookups[symbol] = prev_lookup

    # Portfolio state
    positions = {}
    trades = []
    equity_curve = []
    entries_blocked = 0
    entries_allowed = 0

    for date in all_dates:
        is_bull = bull_filter.get(date, False)
        risk_pct = bull_risk_pct if is_bull else bear_risk_pct

        # === CHECK EXITS (unchanged — always active) ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_high = float(row['high'])
            current_atr = float(row['atr'])
            pos['high_watermark'] = max(pos['high_watermark'], current_high)

            exit_reason = None

            # Blow-off detection
            vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
            volume_ratio = float(row['volume']) / vol_sma
            is_blowoff = (volume_ratio > params['volume_blowoff']
                          and float(row['rsi']) > params['rsi_blowoff'])
            stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

            # 1. Trailing stop
            trailing_stop = pos['high_watermark'] - (stop_mult * current_atr)
            if current_close <= trailing_stop:
                exit_reason = f'Trailing stop ({stop_mult}x ATR)'

            # 2. Donchian exit
            prev_row = prev_lookups[symbol].get(date)
            if not exit_reason and prev_row is not None and pd.notna(prev_row['exit_low']):
                if current_close < float(prev_row['exit_low']):
                    exit_reason = 'Donchian exit (10-day low)'

            # 3. Emergency stop (15%)
            if not exit_reason and current_close <= pos['entry_price'] * 0.85:
                exit_reason = 'Emergency stop (15%)'

            # Partial profit taking
            if not exit_reason:
                gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                    partial_price = current_close * (1 - cost_per_side)
                    partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                    partial_size = pos['size_usd'] * params['tp1_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 1
                    pos['remaining_fraction'] *= (1 - params['tp1_fraction'])
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP1 (+{params["tp1_pct"]}%)',
                        'size_usd': partial_size,
                        'win': True,
                    })
                elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                    partial_price = current_close * (1 - cost_per_side)
                    partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                    partial_size = pos['size_usd'] * params['tp2_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 2
                    pos['remaining_fraction'] *= (1 - params['tp2_fraction'])
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP2 (+{params["tp2_pct"]}%)',
                        'size_usd': partial_size,
                        'win': True,
                    })

            if exit_reason:
                exit_price_adj = current_close * (1 - cost_per_side)
                pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'],
                    'exit_time': row['time'],
                    'exit_price': exit_price_adj,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'size_usd': pos['size_usd'],
                    'win': pnl_pct > 0,
                })
                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del positions[sym]

        # === CHECK ENTRIES (gated by bull filter) ===
        if len(positions) < max_positions and is_bull:
            for symbol in prepared:
                if symbol in positions:
                    continue
                if len(positions) >= max_positions:
                    break

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue
                if pd.isna(prev_row['donchian_high']):
                    continue

                current_close = float(row['close'])

                # Entry conditions (same as original)
                breakout = current_close > float(prev_row['donchian_high'])
                if params['volume_mult'] > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                else:
                    volume_ok = True
                trend_ok = current_close > float(row['ema_21'])

                if breakout and volume_ok and trend_ok:
                    entries_allowed += 1
                    total_equity = capital + sum(p['size_usd'] for p in positions.values())
                    risk_amount = total_equity * risk_pct
                    entry_price = current_close * (1 + cost_per_side)
                    atr_val = float(row['atr'])
                    stop_distance = params['atr_mult'] * atr_val
                    stop_pct = stop_distance / entry_price

                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / max_positions

                    position_size = min(position_size, capital * 0.95)
                    if position_size < 100:
                        continue

                    capital -= position_size
                    positions[symbol] = {
                        'entry_price': entry_price,
                        'entry_time': row['time'],
                        'high_watermark': float(row['high']),
                        'partials_taken': 0,
                        'remaining_fraction': 1.0,
                        'size_usd': position_size,
                    }

        elif len(positions) < max_positions and not is_bull:
            # Count blocked entries (for stats)
            for symbol in prepared:
                if symbol in positions:
                    continue
                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue
                if pd.isna(prev_row['donchian_high']):
                    continue
                current_close = float(row['close'])
                breakout = current_close > float(prev_row['donchian_high'])
                if params['volume_mult'] > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                else:
                    volume_ok = True
                trend_ok = current_close > float(row['ema_21'])
                if breakout and volume_ok and trend_ok:
                    entries_blocked += 1

        # Track equity
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(date)
            if row is not None:
                current_val = pos['size_usd'] * (float(row['close']) / pos['entry_price'])
                total_equity += current_val
            else:
                total_equity += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close remaining positions
    for symbol, pos in list(positions.items()):
        df = prepared[symbol]
        last = df.iloc[-1]
        exit_price_adj = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol,
            'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'],
            'exit_time': last['time'],
            'exit_price': exit_price_adj,
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'],
            'win': pnl_pct > 0,
        })

    return trades, equity_curve, capital, entries_allowed, entries_blocked


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("PHASE 2: BULL MARKET FILTER — BACKTEST & WALK-FORWARD REVALIDATION")
    print("  Filter: BTC close > SMA(200) AND SMA(50) > SMA(200)")
    print("  Only NEW entries gated — exits unchanged")
    print("=" * 100)

    # Fetch data
    coin_data = fetch_all_coins()

    # Compute BTC bull filter from BTC data
    btc_df = coin_data.get('BTC-USD')
    if btc_df is None:
        print("ERROR: BTC-USD data not found")
        return

    print(f"\n  Computing BTC bull filter...")
    bull_filter = compute_btc_bull_filter(btc_df)

    # ======================================================================
    # TEST 1: Full period comparison (with vs without bull filter)
    # ======================================================================
    print_section("FULL PERIOD: WITH vs WITHOUT BULL FILTER (2022-2026)")

    from backtest_donchian_daily import backtest_portfolio
    print(f"\n  --- Original (no filter) ---")
    orig_trades, orig_eq, orig_cap = backtest_portfolio(coin_data, DEFAULT_PARAMS)
    orig_stats = compute_stats(orig_trades, 'Original')

    print(f"\n  --- With bull filter ---")
    bull_params = {**DEFAULT_PARAMS, 'label': 'Bull Filter'}
    bf_trades, bf_eq, bf_cap, bf_allowed, bf_blocked = backtest_portfolio_bull(
        coin_data, bull_params, bull_filter, dynamic_risk=False)
    bf_stats = compute_stats(bf_trades, 'Bull Filter')

    print(f"\n  --- Bull filter + dynamic risk (3% in bull) ---")
    dr_params = {**DEFAULT_PARAMS, 'label': 'Bull + Dynamic Risk'}
    dr_trades, dr_eq, dr_cap, dr_allowed, dr_blocked = backtest_portfolio_bull(
        coin_data, dr_params, bull_filter, dynamic_risk=True)
    dr_stats = compute_stats(dr_trades, 'Bull + Dynamic Risk')

    print(f"\n\n  {'='*120}")
    print(f"  FULL PERIOD COMPARISON")
    print(f"  {'='*120}")
    print_stats_row('Original (no filter)', orig_stats, orig_eq)
    print_stats_row('Bull filter (2% risk)', bf_stats, bf_eq)
    print_stats_row('Bull filter + dynamic (3% risk)', dr_stats, dr_eq)

    print(f"\n  Entries allowed / blocked by bull filter:")
    print(f"    Bull filter (2%):     {bf_allowed} allowed, {bf_blocked} blocked")
    print(f"    Bull + dynamic (3%):  {dr_allowed} allowed, {dr_blocked} blocked")

    # Per-coin for bull filter
    bf_coin_stats = compute_per_coin_stats(bf_trades)
    if bf_coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN (Bull Filter):")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}")
        print(f"  {'-'*55}")
        for sym, cs in sorted(bf_coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%")

    # ======================================================================
    # TEST 2: Walk-forward with bull filter
    # ======================================================================
    print_section("WALK-FORWARD REVALIDATION WITH BULL FILTER")
    print("  Train: 2022-02 to 2024-12 | Test: 2025-01 to 2026-02")

    cutoff = pd.Timestamp('2025-01-01')
    train_data, test_data = split_coin_data(coin_data, cutoff)

    # Bull filter for train/test periods
    btc_train = train_data.get('BTC-USD')
    btc_test = test_data.get('BTC-USD')

    if btc_train is not None:
        print(f"\n  Train period bull filter:")
        train_bull = compute_btc_bull_filter(btc_train)
    else:
        train_bull = {}

    if btc_test is not None:
        print(f"  Test period bull filter:")
        test_bull = compute_btc_bull_filter(btc_test)
    else:
        test_bull = {}

    # Train with bull filter
    print(f"\n  Running train period...")
    train_trades, train_eq, train_cap, _, _ = backtest_portfolio_bull(
        train_data, DEFAULT_PARAMS, train_bull, dynamic_risk=False)
    train_stats = compute_stats(train_trades, 'Train + Bull Filter')

    # Test (OOS) with bull filter
    print(f"  Running test period (out-of-sample)...")
    test_trades, test_eq, test_cap, test_allowed, test_blocked = backtest_portfolio_bull(
        test_data, DEFAULT_PARAMS, test_bull, dynamic_risk=False)
    test_stats = compute_stats(test_trades, 'Test + Bull Filter')

    # Test with dynamic risk
    print(f"  Running test period with dynamic risk...")
    test_dr_trades, test_dr_eq, test_dr_cap, _, _ = backtest_portfolio_bull(
        test_data, DEFAULT_PARAMS, test_bull, dynamic_risk=True)
    test_dr_stats = compute_stats(test_dr_trades, 'Test + Bull + Dynamic')

    # Original walk-forward (no filter) for comparison
    print(f"  Running original test period (no filter)...")
    orig_test_trades, orig_test_eq, _ = backtest_portfolio(test_data, DEFAULT_PARAMS)
    orig_test_stats = compute_stats(orig_test_trades, 'Test (no filter)')

    print(f"\n\n  {'='*120}")
    print(f"  WALK-FORWARD COMPARISON")
    print(f"  {'='*120}")
    print_stats_row('Train + Bull Filter', train_stats, train_eq)
    print_stats_row('Test (no filter) [ORIGINAL]', orig_test_stats, orig_test_eq)
    print_stats_row('Test + Bull Filter [NEW]', test_stats, test_eq)
    print_stats_row('Test + Bull + Dynamic Risk', test_dr_stats, test_dr_eq)

    print(f"\n  Test period entries: {test_allowed} allowed, {test_blocked} blocked by filter")

    # Per-coin OOS with bull filter
    test_coin_stats = compute_per_coin_stats(test_trades)
    if test_coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN (OUT-OF-SAMPLE + BULL FILTER):")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}")
        print(f"  {'-'*55}")
        for sym, cs in sorted(test_coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%")

    # ======================================================================
    # VERDICT
    # ======================================================================
    print_section("PHASE 2 VERDICT")

    orig_test_ret = orig_test_stats.get('total_return_pct', 0)
    orig_test_pf = orig_test_stats.get('profit_factor', 0)
    bf_test_ret = test_stats.get('total_return_pct', 0)
    bf_test_pf = test_stats.get('profit_factor', 0)
    dr_test_ret = test_dr_stats.get('total_return_pct', 0)

    print(f"\n  Out-of-sample (2025-2026) comparison:")
    print(f"    Original:          Return {orig_test_ret:+.1f}%, PF {orig_test_pf:.2f}")
    print(f"    + Bull filter:     Return {bf_test_ret:+.1f}%, PF {bf_test_pf:.2f}")
    print(f"    + Dynamic risk:    Return {dr_test_ret:+.1f}%")

    improvement = bf_test_ret - orig_test_ret
    print(f"\n  Bull filter improvement: {improvement:+.1f}% return in OOS period")

    if bf_test_pf >= 1.0 and bf_test_ret > 0:
        print(f"\n  VERDICT: PASS — Bull filter fixes out-of-sample performance")
        print(f"  Recommendation: Deploy bull filter to production bot")
    elif bf_test_ret > orig_test_ret:
        print(f"\n  VERDICT: IMPROVED — Bull filter helps but OOS still marginal")
        print(f"  Recommendation: Deploy with caution, monitor closely")
    else:
        print(f"\n  VERDICT: NO IMPROVEMENT — Bull filter doesn't help enough")
        print(f"  Recommendation: Investigate other filters or strategy changes")


if __name__ == "__main__":
    main()
