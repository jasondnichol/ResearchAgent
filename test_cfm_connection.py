"""Coinbase CFM Connection Test — Phase F2 Validation

Interactive test script that validates the Coinbase CFM perpetual futures integration.
Run this to confirm API keys, product discovery, pricing, and paper trading all work.

Usage:
    cd C:\\ResearchAgent
    PYTHONIOENCODING=utf-8 venv/Scripts/python test_cfm_connection.py
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_result(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    icon = "[+]" if passed else "[-]"
    msg = f"  {icon} {label}: {status}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def main():
    print_header("COINBASE CFM CONNECTION TEST — PHASE F2")
    print(f"  Testing perpetual futures API integration")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    results = {}

    # ==================================================================
    # TEST 1: API KEY CHECK
    # ==================================================================
    print_header("TEST 1: API Key Configuration")

    api_key = os.getenv('COINBASE_API_KEY') or os.getenv('API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    api_secret_file = os.getenv('API_SECRET_FILE')

    has_key = bool(api_key and len(api_key) > 10)
    # Secret can be inline or in a PEM file
    has_secret_inline = bool(api_secret and len(api_secret) > 10)
    has_secret_file = False
    if api_secret_file:
        secret_path = api_secret_file
        if not os.path.isabs(secret_path):
            secret_path = os.path.join(os.path.dirname(__file__) or '.', secret_path)
        has_secret_file = os.path.isfile(secret_path)
    has_secret = has_secret_inline or has_secret_file

    key_source = 'COINBASE_API_KEY' if os.getenv('COINBASE_API_KEY') else 'API_KEY'
    results['api_key'] = print_result(
        f"{key_source} in .env",
        has_key,
        f"{api_key[:8]}...{api_key[-4:]}" if has_key else "NOT SET"
    )

    if has_secret_inline:
        secret_detail = f"inline ({api_secret[:4]}...{api_secret[-4:]})"
    elif has_secret_file:
        secret_detail = f"PEM file: {api_secret_file}"
    else:
        secret_detail = "NOT SET"
    results['api_secret'] = print_result(
        "API secret",
        has_secret,
        secret_detail
    )

    if not has_key or not has_secret:
        print(f"\n  To set up API keys:")
        print(f"    1. Go to https://www.coinbase.com/settings/api")
        print(f"    2. Create new API key with 'View' and 'Trade' permissions")
        print(f"    3. Add to C:\\ResearchAgent\\.env:")
        print(f"       API_KEY=organizations/.../apiKeys/...")
        print(f"       API_SECRET_FILE=private_key.pem")
        print(f"       (or COINBASE_API_KEY + COINBASE_API_SECRET inline)")
        print(f"\n  Continuing with public-only tests...")

    # ==================================================================
    # TEST 2: CLIENT INITIALIZATION
    # ==================================================================
    print_header("TEST 2: Client Initialization")

    from coinbase_futures import CoinbaseFuturesClient, SPOT_TO_PERP, PERP_PRODUCTS

    client = CoinbaseFuturesClient(paper_mode=True)

    results['client_init'] = print_result(
        "CoinbaseFuturesClient created",
        client._client is not None,
        f"paper_mode={client.paper_mode}, auth={client.is_authenticated()}"
    )

    # ==================================================================
    # TEST 3: AUTHENTICATION (if keys present)
    # ==================================================================
    print_header("TEST 3: Authentication")

    if client.is_authenticated():
        auth_ok, auth_msg = client.test_auth()
        results['auth'] = print_result("API authentication", auth_ok, auth_msg)
    else:
        results['auth'] = print_result(
            "API authentication", False,
            "Skipped — no credentials. Public endpoints still work."
        )

    # ==================================================================
    # TEST 4: PRODUCT DISCOVERY
    # ==================================================================
    print_header("TEST 4: Perpetual Product Discovery")

    products = client.list_perp_products()
    results['products'] = print_result(
        "List perp products",
        len(products) > 0,
        f"Found {len(products)}/{len(PERP_PRODUCTS)} perp contracts"
    )

    if products:
        print(f"\n  {'Product ID':<20s} {'Price':>12s} {'Status':<10s} {'Min Size':<10s} {'Increment':<10s}")
        print(f"  {'-'*62}")
        for p in products:
            print(f"  {p['product_id']:<20s} ${p['price']:>10,.2f} {p['status']:<10s} "
                  f"{p.get('base_min_size', '?'):<10s} {p.get('base_increment', '?'):<10s}")

    # ==================================================================
    # TEST 5: COIN MAPPING
    # ==================================================================
    print_header("TEST 5: Spot-to-Perp Coin Mapping")

    our_coins = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
                 'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD']
    mapped = 0
    print(f"\n  {'Spot Symbol':<12s} {'Perp Product ID':<20s} {'Status'}")
    print(f"  {'-'*44}")
    for coin in our_coins:
        perp_id = client.spot_to_perp(coin)
        if perp_id:
            has_product = any(p['product_id'] == perp_id for p in products)
            status = "AVAILABLE" if has_product else "MAPPED (not in API)"
            print(f"  {coin:<12s} {perp_id:<20s} {status}")
            if has_product:
                mapped += 1
        else:
            print(f"  {coin:<12s} {'NO MAPPING':<20s} MISSING")

    results['mapping'] = print_result(
        "Coin mapping",
        mapped >= 7,
        f"{mapped}/{len(our_coins)} coins mapped to available perps"
    )

    # ==================================================================
    # TEST 6: PRICE FETCHING
    # ==================================================================
    print_header("TEST 6: Current Prices")

    prices = client.get_prices()
    results['prices'] = print_result(
        "Fetch perp prices",
        len(prices) >= 7,
        f"Got prices for {len(prices)} contracts"
    )

    if prices:
        print(f"\n  {'Product':<20s} {'Price':>12s}")
        print(f"  {'-'*32}")
        for pid, price in sorted(prices.items()):
            print(f"  {pid:<20s} ${price:>10,.2f}")

    # ==================================================================
    # TEST 7: CANDLE DATA
    # ==================================================================
    print_header("TEST 7: Candle Data (BTC-PERP-INTX)")

    now = int(time.time())
    start = now - (7 * 86400)  # 7 days ago
    candles = client.get_candles('BTC-PERP-INTX', start=start, end=now,
                                 granularity='ONE_DAY')

    results['candles'] = print_result(
        "Fetch daily candles",
        candles is not None and len(candles) > 0,
        f"Got {len(candles) if candles else 0} candles"
    )

    if candles and len(candles) > 0:
        c = candles[0]
        print(f"  Sample candle keys: {list(c.keys()) if isinstance(c, dict) else dir(c)}")

    # ==================================================================
    # TEST 8: ACCOUNT BALANCE (auth or paper)
    # ==================================================================
    print_header("TEST 8: Account Balance")

    balance = client.get_balance_summary()
    results['balance'] = print_result(
        "Get balance summary",
        balance is not None,
        "Paper mode" if balance and balance.get('paper') else "Live account"
    )

    if balance:
        if balance.get('paper'):
            print(f"\n  Paper Account:")
            print(f"    Total equity:    ${balance.get('total_usd_balance', 0):,.2f}")
            print(f"    Available margin: ${balance.get('available_margin', 0):,.2f}")
            print(f"    Margin used:     ${balance.get('margin_used', 0):,.2f}")
            print(f"    Open positions:  {balance.get('open_positions', 0)}")
        else:
            print(f"\n  Live Account:")
            for k, v in balance.items():
                if k not in ('paper',):
                    print(f"    {k}: {v}")

    # ==================================================================
    # TEST 9: POSITIONS LIST
    # ==================================================================
    print_header("TEST 9: Open Positions")

    positions = client.get_positions()
    results['positions'] = print_result(
        "List positions",
        isinstance(positions, list),
        f"{len(positions)} open positions"
    )

    if positions:
        for pos in positions:
            print(f"  {pos.get('product_id', '?')}: "
                  f"{pos.get('side', '?')} {pos.get('base_size', '?')} "
                  f"@ ${pos.get('avg_entry_price', '?')}")

    # ==================================================================
    # TEST 10: PAPER TRADING DEMO
    # ==================================================================
    print_header("TEST 10: Paper Trading Demo")

    # Reset paper state for clean demo
    client.reset_paper(starting_capital=10000.0)
    print(f"  Reset paper account: $10,000 starting capital")

    btc_price = client.get_current_price('BTC-PERP-INTX')
    if btc_price:
        print(f"  BTC current price: ${btc_price:,.2f}")

        # Open a short position ($100 notional)
        print(f"\n  --- Opening $100 BTC short ---")
        result = client.market_order('BTC-PERP-INTX', 'SELL', size_usd=100)
        short_ok = result.get('success', False)
        results['paper_short'] = print_result(
            "Open paper short",
            short_ok,
            f"${result.get('notional_usd', 0):.2f} @ ${result.get('fill_price', 0):,.2f}"
            if short_ok else result.get('error', 'Unknown error')
        )

        if short_ok:
            # Check position
            pos = client.get_position('BTC-PERP-INTX')
            if pos:
                print(f"  Position: {pos['side']} {pos['base_size']:.6f} BTC "
                      f"@ ${pos['avg_entry_price']:,.2f}")

            # Check balance
            bal = client.get_balance_summary()
            print(f"  Balance: ${bal['total_usd_balance']:,.2f} equity, "
                  f"${bal['available_margin']:,.2f} available")

            # Close the position
            print(f"\n  --- Closing BTC short ---")
            close = client.close_position('BTC-PERP-INTX')
            close_ok = close.get('success', False)
            results['paper_close'] = print_result(
                "Close paper short",
                close_ok,
                f"PnL: ${close.get('pnl', 0):.2f} ({close.get('pnl_pct', 0):+.2f}%)"
                if close_ok else close.get('error', 'Unknown error')
            )

            # Final balance
            bal_final = client.get_balance_summary()
            print(f"  Final balance: ${bal_final['total_usd_balance']:,.2f} "
                  f"({bal_final['total_return_pct']:+.2f}%)")
            print(f"  Trades executed: {bal_final['trade_count']}")
        else:
            results['paper_close'] = print_result("Close paper short", False, "Skipped")
    else:
        results['paper_short'] = print_result("Open paper short", False, "No BTC price")
        results['paper_close'] = print_result("Close paper short", False, "Skipped")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print_header("SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    failed = total - passed

    print(f"\n  Results: {passed}/{total} passed, {failed} failed")
    print()

    if failed > 0:
        print(f"  Failed tests:")
        for name, ok in results.items():
            if not ok:
                print(f"    - {name}")
        print()

    # Auth-specific guidance
    if not results.get('auth', False):
        print(f"  NOTE: Authentication failed or skipped.")
        print(f"  The following features require valid API keys:")
        print(f"    - Live account balance and positions")
        print(f"    - Real order execution (when paper_mode=False)")
        print(f"    - Margin window information")
        print(f"  Paper trading works without authentication.")
        print()

    # Next steps
    print(f"  NEXT STEPS:")
    if passed >= 8:
        print(f"    Phase F2 integration is working.")
        print(f"    1. Add COINBASE_API_KEY/SECRET to .env (if not done)")
        print(f"    2. Re-run this test to validate auth")
        print(f"    3. Proceed to Phase F3: integrate into production bot")
    elif passed >= 5:
        print(f"    Core functionality works. Fix failing tests above.")
    else:
        print(f"    Multiple failures. Check API keys and network connectivity.")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
