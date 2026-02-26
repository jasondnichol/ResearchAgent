"""Walk-Forward Validation & Slippage Stress Test for Donchian Breakout

Phase 1 validation:
  1. Walk-forward: Train on 2022-2024, test on 2025-2026 (out-of-sample)
  2. Slippage stress test: Run default params with 0.1% to 0.5% slippage
  3. Sharpe ratio and max drawdown analysis

Uses the same backtest engine from backtest_donchian_daily.py.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_donchian_daily import (
    fetch_all_coins,
    backtest_portfolio,
    compute_stats,
    compute_per_coin_stats,
    DEFAULT_PARAMS,
    COIN_UNIVERSE,
)


# ============================================================================
# HELPERS
# ============================================================================

def split_coin_data(coin_data, cutoff_date):
    """Split coin data into train/test sets at a cutoff date"""
    train = {}
    test = {}
    for symbol, df in coin_data.items():
        df_train = df[df['time'] < cutoff_date].copy().reset_index(drop=True)
        df_test = df[df['time'] >= cutoff_date].copy().reset_index(drop=True)

        # Test set needs warmup data for indicators — prepend last 60 days of train
        warmup_start = cutoff_date - pd.Timedelta(days=60)
        df_warmup = df[(df['time'] >= warmup_start) & (df['time'] < cutoff_date)].copy()
        df_test_full = pd.concat([df_warmup, df_test], ignore_index=True)

        if len(df_train) > 100:
            train[symbol] = df_train
        if len(df_test_full) > 60:
            test[symbol] = df_test_full

    return train, test


def compute_sharpe(equity_curve, risk_free_annual=0.05):
    """Compute annualized Sharpe ratio from equity curve"""
    if not equity_curve or len(equity_curve) < 30:
        return 0.0

    equities = [e['equity'] for e in equity_curve]
    returns = pd.Series(equities).pct_change().dropna()

    if returns.std() == 0:
        return 0.0

    # Daily risk-free rate
    rf_daily = (1 + risk_free_annual) ** (1/365) - 1
    excess = returns - rf_daily

    # Annualize (252 trading days)
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
    return sharpe


def compute_max_drawdown(equity_curve):
    """Compute max drawdown % and duration from equity curve"""
    if not equity_curve:
        return 0.0, 0

    peak = equity_curve[0]['equity']
    max_dd = 0.0
    dd_start = equity_curve[0]['date']
    max_dd_duration = 0
    current_dd_start = None

    for point in equity_curve:
        if point['equity'] > peak:
            peak = point['equity']
            current_dd_start = None
        else:
            dd = (peak - point['equity']) / peak * 100
            if dd > max_dd:
                max_dd = dd
            if current_dd_start is None:
                current_dd_start = point['date']

    return max_dd, 0


def print_section(title):
    print(f"\n\n{'#' * 100}")
    print(f"# {title}")
    print(f"{'#' * 100}")


def print_stats_row(label, stats, equity_curve):
    """Print a single row of stats"""
    sharpe = compute_sharpe(equity_curve)
    max_dd, _ = compute_max_drawdown(equity_curve)
    final = equity_curve[-1]['equity'] if equity_curve else 0

    print(f"  {label:<35s} "
          f"{stats.get('total_trades', 0):>5d} trades  "
          f"{stats.get('win_rate', 0)*100:>5.1f}% WR  "
          f"PF {stats.get('profit_factor', 0):>5.2f}  "
          f"Return {stats.get('total_return_pct', 0):>+7.1f}%  "
          f"MaxDD {max_dd:>5.1f}%  "
          f"Sharpe {sharpe:>5.2f}  "
          f"Final ${final:>10,.0f}")


# ============================================================================
# TEST 1: WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward(coin_data):
    """Train on 2022-2024, test on 2025-2026"""
    print_section("WALK-FORWARD VALIDATION")
    print("  Train: 2022-02 to 2024-12 (in-sample)")
    print("  Test:  2025-01 to 2026-02 (out-of-sample)")
    print("  Strategy params are FROZEN from the original backtest — no re-optimization")

    cutoff = pd.Timestamp('2025-01-01')

    train_data, test_data = split_coin_data(coin_data, cutoff)
    print(f"\n  Train coins: {len(train_data)} | Test coins: {len(test_data)}")

    for symbol in train_data:
        df = train_data[symbol]
        print(f"    {symbol}: train {len(df)} bars ({df['time'].min().date()} to {df['time'].max().date()})")
    for symbol in test_data:
        df = test_data[symbol]
        print(f"    {symbol}: test  {len(df)} bars ({df['time'].min().date()} to {df['time'].max().date()})")

    # Run backtest on train period
    print(f"\n  Running train period backtest...")
    train_trades, train_eq, train_cap = backtest_portfolio(train_data, DEFAULT_PARAMS)
    train_stats = compute_stats(train_trades, 'Train (2022-2024)')

    # Run backtest on test period (out-of-sample)
    print(f"\n  Running test period backtest (out-of-sample)...")
    test_trades, test_eq, test_cap = backtest_portfolio(test_data, DEFAULT_PARAMS)
    test_stats = compute_stats(test_trades, 'Test (2025-2026)')

    # Run full period for comparison
    print(f"\n  Running full period backtest...")
    full_trades, full_eq, full_cap = backtest_portfolio(coin_data, DEFAULT_PARAMS)
    full_stats = compute_stats(full_trades, 'Full (2022-2026)')

    # Results
    print(f"\n\n  {'='*110}")
    print(f"  WALK-FORWARD RESULTS")
    print(f"  {'='*110}")

    print_stats_row('Train (2022-2024) [in-sample]', train_stats, train_eq)
    print_stats_row('Test (2025-2026) [OUT-OF-SAMPLE]', test_stats, test_eq)
    print_stats_row('Full (2022-2026) [reference]', full_stats, full_eq)

    # Per-coin breakdown for test period
    test_coin_stats = compute_per_coin_stats(test_trades)
    if test_coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN (OUT-OF-SAMPLE 2025-2026):")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}")
        print(f"  {'-'*55}")
        for sym, cs in sorted(test_coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%")

    # Verdict
    print(f"\n  {'='*80}")
    print(f"  OVERFITTING CHECK:")
    print(f"  {'='*80}")

    train_pf = train_stats.get('profit_factor', 0)
    test_pf = test_stats.get('profit_factor', 0)
    train_wr = train_stats.get('win_rate', 0)
    test_wr = test_stats.get('win_rate', 0)
    train_ret = train_stats.get('total_return_pct', 0)
    test_ret = test_stats.get('total_return_pct', 0)

    pf_decay = ((train_pf - test_pf) / train_pf * 100) if train_pf > 0 else 0
    wr_decay = ((train_wr - test_wr) / train_wr * 100) if train_wr > 0 else 0

    print(f"  Profit Factor decay:  {train_pf:.2f} -> {test_pf:.2f} ({pf_decay:+.1f}%)")
    print(f"  Win Rate decay:       {train_wr*100:.1f}% -> {test_wr*100:.1f}% ({wr_decay:+.1f}%)")
    print(f"  Train return:         {train_ret:+.1f}%")
    print(f"  Test return:          {test_ret:+.1f}%")

    if test_pf >= 1.0 and test_ret > 0:
        if pf_decay < 30:
            print(f"\n  VERDICT: PASS - Strategy holds up out-of-sample with modest decay")
        else:
            print(f"\n  VERDICT: CAUTION - Significant performance decay ({pf_decay:.0f}%), monitor closely")
    else:
        print(f"\n  VERDICT: FAIL - Strategy does not hold up out-of-sample")

    return {
        'train': train_stats,
        'test': test_stats,
        'full': full_stats,
        'pf_decay': pf_decay,
        'wr_decay': wr_decay,
    }


# ============================================================================
# TEST 2: SLIPPAGE STRESS TEST
# ============================================================================

def run_slippage_stress_test(coin_data):
    """Test strategy robustness across different slippage assumptions"""
    print_section("SLIPPAGE STRESS TEST")
    print("  Testing with increasing slippage to find breaking point")
    print("  Base fee: 0.40% taker (Coinbase)")

    slippage_levels = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    results = []
    print(f"\n  {'='*120}")
    print(f"  {'Slippage':>10s} {'Total Cost':>12s} {'Trades':>7s} {'WR':>7s} "
          f"{'PF':>7s} {'Return':>10s} {'MaxDD':>7s} {'Sharpe':>8s} {'Final':>12s}")
    print(f"  {'-'*100}")

    for slip in slippage_levels:
        params = {**DEFAULT_PARAMS, 'slippage_pct': slip}
        total_cost = params['fee_pct'] + slip
        label = f"Slip {slip:.2f}%"

        trades, eq, cap = backtest_portfolio(coin_data, params)
        stats = compute_stats(trades, label)
        sharpe = compute_sharpe(eq)
        max_dd, _ = compute_max_drawdown(eq)
        final = eq[-1]['equity'] if eq else 0

        print(f"  {slip:>9.2f}% {total_cost:>11.2f}% "
              f"{stats.get('total_trades', 0):>7d} "
              f"{stats.get('win_rate', 0)*100:>6.1f}% "
              f"{stats.get('profit_factor', 0):>7.2f} "
              f"{stats.get('total_return_pct', 0):>+9.1f}% "
              f"{max_dd:>6.1f}% "
              f"{sharpe:>7.2f} "
              f"${final:>11,.0f}")

        results.append({
            'slippage_pct': slip,
            'total_cost_pct': total_cost,
            'stats': stats,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'final_equity': final,
        })

    # Find breaking point
    print(f"\n  ANALYSIS:")
    for r in results:
        ret = r['stats'].get('total_return_pct', 0)
        if ret <= 0:
            print(f"  Strategy breaks even at ~{r['total_cost_pct']:.2f}% total cost per side")
            break
    else:
        print(f"  Strategy remains profitable at all tested slippage levels (up to {slippage_levels[-1]}%)")

    base = results[0]['stats'].get('total_return_pct', 0)
    worst = results[-1]['stats'].get('total_return_pct', 0)
    if base > 0:
        degradation = (base - worst) / base * 100
        print(f"  Return degradation from 0% to {slippage_levels[-1]}% slippage: {degradation:.1f}%")
        print(f"  At realistic 0.20% slippage: {results[3]['stats'].get('total_return_pct', 0):+.1f}% return")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("PHASE 1: WALK-FORWARD VALIDATION & SLIPPAGE STRESS TEST")
    print(f"Coins: {', '.join(COIN_UNIVERSE)}")
    print(f"Base params: {DEFAULT_PARAMS['fee_pct']}% fee + {DEFAULT_PARAMS['slippage_pct']}% slippage")
    print("=" * 100)

    # Fetch data
    coin_data = fetch_all_coins()

    # Test 1: Walk-forward
    wf_results = run_walk_forward(coin_data)

    # Test 2: Slippage stress test
    slip_results = run_slippage_stress_test(coin_data)

    # Final summary
    print(f"\n\n{'='*100}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*100}")

    test_stats = wf_results['test']
    test_pf = test_stats.get('profit_factor', 0)
    test_ret = test_stats.get('total_return_pct', 0)
    pf_decay = wf_results['pf_decay']

    print(f"\n  Walk-Forward:")
    print(f"    Out-of-sample PF: {test_pf:.2f} (decay: {pf_decay:+.1f}% from train)")
    print(f"    Out-of-sample return: {test_ret:+.1f}%")
    if test_pf >= 1.0 and test_ret > 0:
        print(f"    Status: PASS")
    else:
        print(f"    Status: FAIL")

    # Find realistic slippage result (0.20%)
    realistic = next((r for r in slip_results if r['slippage_pct'] == 0.20), None)
    if realistic:
        print(f"\n  Slippage Stress Test:")
        print(f"    At 0.20% slippage (realistic): {realistic['stats'].get('total_return_pct', 0):+.1f}% return, "
              f"PF {realistic['stats'].get('profit_factor', 0):.2f}")

    profitable_all = all(r['stats'].get('total_return_pct', 0) > 0 for r in slip_results)
    if profitable_all:
        print(f"    Profitable at all slippage levels up to {slip_results[-1]['slippage_pct']}%")
        print(f"    Status: PASS")
    else:
        break_even = next((r for r in slip_results if r['stats'].get('total_return_pct', 0) <= 0), None)
        if break_even:
            print(f"    Breaks at {break_even['total_cost_pct']:.2f}% total cost")
        print(f"    Status: REVIEW")

    overall = "PASS" if (test_pf >= 1.0 and test_ret > 0 and profitable_all) else "REVIEW NEEDED"
    print(f"\n  Overall Phase 1 Verdict: {overall}")

    if overall == "PASS":
        print(f"\n  Strategy is robust. Safe to proceed to Phase 2 (bull filter + dynamic risk).")
    else:
        print(f"\n  Review results before proceeding. Strategy may need adjustment.")


if __name__ == "__main__":
    main()
