"""Coinbase CFM Perpetual Futures — API Client Wrapper

Exchange integration layer for Coinbase Financial Markets perpetual futures.
Supports both paper trading (simulated) and live trading (real orders).

Product IDs: {COIN}-PERP-INTX (e.g., BTC-PERP-INTX, ETH-PERP-INTX)
Venue: INTX (Coinbase International Exchange)
Margin: USDC collateral, isolated margin
Fees: ~0.06% taker (CFM rate, much lower than spot)

Usage:
    from coinbase_futures import CoinbaseFuturesClient

    client = CoinbaseFuturesClient()  # paper mode, reads keys from .env
    products = client.list_perp_products()
    price = client.get_current_price('BTC-PERP-INTX')
    client.market_order('BTC-PERP-INTX', 'SELL', size_usd=100)  # paper short
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_LEVERAGE = 3          # Hard cap — never exceeded
DEFAULT_LEVERAGE = 1      # Start at 1x (no amplification)
TAKER_FEE_PCT = 0.06      # Coinbase CFM taker fee
MIN_ORDER_USDC = 10       # Minimum order size
PAPER_STATE_FILE = 'paper_futures_state.json'

# Map spot symbol (BTC-USD) to perp product ID (BTC-PERP-INTX)
SPOT_TO_PERP = {
    'BTC-USD': 'BTC-PERP-INTX',
    'ETH-USD': 'ETH-PERP-INTX',
    'SOL-USD': 'SOL-PERP-INTX',
    'XRP-USD': 'XRP-PERP-INTX',
    'SUI-USD': 'SUI-PERP-INTX',
    'LINK-USD': 'LINK-PERP-INTX',
    'ADA-USD': 'ADA-PERP-INTX',
    'DOGE-USD': 'DOGE-PERP-INTX',
    'NEAR-USD': 'NEAR-PERP-INTX',
}

# Reverse mapping
PERP_TO_SPOT = {v: k for k, v in SPOT_TO_PERP.items()}

# All supported perp product IDs
PERP_PRODUCTS = list(SPOT_TO_PERP.values())


# ============================================================================
# CLIENT
# ============================================================================

class CoinbaseFuturesClient:
    """API client for Coinbase CFM perpetual futures.

    paper_mode=True (default): read-only API calls + simulated orders in memory.
    paper_mode=False: executes real orders on Coinbase CFM. Requires explicit flag.
    """

    def __init__(self, api_key=None, api_secret=None, paper_mode=True,
                 starting_capital=10000.0):
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        self.paper_mode = paper_mode
        self.leverage = DEFAULT_LEVERAGE
        self._client = None

        # Paper trading state
        self._paper_positions = {}
        self._paper_cash = starting_capital
        self._paper_starting_capital = starting_capital
        self._paper_trades = []
        self._paper_trade_count = 0

        # Initialize SDK client
        if self.api_key and self.api_secret:
            try:
                from coinbase.rest import RESTClient
                self._client = RESTClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )
                logger.info("Coinbase CFM client initialized (authenticated)")
            except Exception as e:
                logger.error(f"Failed to initialize Coinbase client: {e}")
                self._client = None
        else:
            # Public-only client (no auth)
            try:
                from coinbase.rest import RESTClient
                self._client = RESTClient()
                logger.info("Coinbase CFM client initialized (public only, no auth)")
            except Exception as e:
                logger.error(f"Failed to initialize public client: {e}")

        # Load paper state if exists
        if self.paper_mode:
            self._load_paper_state()

    # ========================================================================
    # MARKET DATA (always use real API)
    # ========================================================================

    def list_perp_products(self):
        """List all available perpetual futures contracts.

        Returns list of dicts with product_id, price, status, base_min_size, etc.
        """
        if not self._client:
            return []

        results = []
        for pid in PERP_PRODUCTS:
            try:
                p = self._client.get_public_product(pid)
                results.append({
                    'product_id': p.product_id,
                    'price': float(p.price) if p.price else 0,
                    'status': p.status,
                    'base_min_size': p.base_min_size,
                    'base_increment': p.base_increment,
                    'price_increment': p.price_increment,
                    'volume_24h': p.volume_24h,
                })
            except Exception as e:
                logger.warning(f"Failed to get product {pid}: {e}")

        return results

    def get_product(self, product_id):
        """Get detailed product info for a single perp contract."""
        if not self._client:
            return None
        try:
            p = self._client.get_public_product(product_id)
            return p.to_dict() if hasattr(p, 'to_dict') else p
        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None

    def get_current_price(self, product_id):
        """Get the latest price for a perpetual contract.

        Returns float price or None on error.
        """
        if not self._client:
            return None
        try:
            p = self._client.get_public_product(product_id)
            return float(p.price) if p.price else None
        except Exception as e:
            logger.error(f"Failed to get price for {product_id}: {e}")
            return None

    def get_candles(self, product_id, start, end, granularity='ONE_DAY', limit=300):
        """Get OHLCV candle data for a perp contract.

        Args:
            product_id: e.g. 'BTC-PERP-INTX'
            start: Unix timestamp (str) for start
            end: Unix timestamp (str) for end
            granularity: 'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE',
                        'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR',
                        'SIX_HOUR', 'ONE_DAY'
            limit: max candles (default 300)

        Returns list of candle dicts or None on error.
        """
        if not self._client:
            return None
        try:
            resp = self._client.get_public_candles(
                product_id, start=str(start), end=str(end),
                granularity=granularity, limit=limit
            )
            candles = resp.candles if hasattr(resp, 'candles') else []
            return [c.to_dict() if hasattr(c, 'to_dict') else c for c in candles]
        except Exception as e:
            logger.error(f"Failed to get candles for {product_id}: {e}")
            return None

    def get_prices(self, product_ids=None):
        """Get current prices for multiple perp contracts.

        Returns dict: product_id -> price
        """
        if product_ids is None:
            product_ids = PERP_PRODUCTS

        prices = {}
        for pid in product_ids:
            price = self.get_current_price(pid)
            if price is not None:
                prices[pid] = price
        return prices

    # ========================================================================
    # ACCOUNT (authenticated API)
    # ========================================================================

    def get_balance_summary(self):
        """Get futures account balance summary.

        Returns dict with total_usd_balance, available_margin, etc.
        Or None if not authenticated.
        """
        if not self._client or not self.api_key:
            if self.paper_mode:
                return self._paper_balance_summary()
            return None
        try:
            resp = self._client.get_futures_balance_summary()
            return resp.to_dict() if hasattr(resp, 'to_dict') else resp
        except Exception as e:
            logger.error(f"Failed to get balance summary: {e}")
            if self.paper_mode:
                return self._paper_balance_summary()
            return None

    def get_positions(self):
        """Get all open perpetual futures positions.

        Returns list of position dicts or empty list.
        """
        if self.paper_mode:
            return self._paper_get_positions()

        if not self._client or not self.api_key:
            return []
        try:
            resp = self._client.list_futures_positions()
            positions = resp.positions if hasattr(resp, 'positions') else []
            return [p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            return []

    def get_position(self, product_id):
        """Get details for a single perpetual position.

        Returns position dict or None.
        """
        if self.paper_mode:
            pos = self._paper_positions.get(product_id)
            return pos if pos else None

        if not self._client or not self.api_key:
            return None
        try:
            resp = self._client.get_futures_position(product_id)
            return resp.to_dict() if hasattr(resp, 'to_dict') else resp
        except Exception as e:
            logger.error(f"Failed to get position for {product_id}: {e}")
            return None

    def get_margin_window(self):
        """Get current margin window info (intraday vs overnight rates).

        Returns dict or None.
        """
        if not self._client or not self.api_key:
            return None
        try:
            resp = self._client.get_current_margin_window(
                margin_profile_type='MARGIN_PROFILE_TYPE_REGULAR'
            )
            return resp.to_dict() if hasattr(resp, 'to_dict') else resp
        except Exception as e:
            logger.error(f"Failed to get margin window: {e}")
            return None

    # ========================================================================
    # ORDERS (paper or real)
    # ========================================================================

    def market_order(self, product_id, side, size_usd=None, base_size=None):
        """Place a market order on a perpetual contract.

        Args:
            product_id: e.g. 'BTC-PERP-INTX'
            side: 'BUY' or 'SELL'
            size_usd: notional size in USD (alternative to base_size)
            base_size: size in base currency units

        Returns order result dict.
        """
        side = side.upper()
        if side not in ('BUY', 'SELL'):
            return {'error': f'Invalid side: {side}. Must be BUY or SELL.'}

        # Size validation
        if size_usd is not None and size_usd < MIN_ORDER_USDC:
            return {'error': f'Order size ${size_usd:.2f} below minimum ${MIN_ORDER_USDC}'}

        if self.paper_mode:
            return self._paper_market_order(product_id, side, size_usd, base_size)

        # Live order
        if not self._client or not self.api_key:
            return {'error': 'Not authenticated. Set COINBASE_API_KEY and COINBASE_API_SECRET.'}

        client_order_id = str(uuid.uuid4())
        try:
            kwargs = {}
            if self.leverage > 1:
                kwargs['leverage'] = str(self.leverage)
                kwargs['margin_type'] = 'ISOLATED'

            if side == 'BUY':
                if size_usd is not None:
                    resp = self._client.market_order_buy(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        quote_size=str(size_usd),
                        **kwargs,
                    )
                else:
                    resp = self._client.market_order_buy(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(base_size),
                        **kwargs,
                    )
            else:
                if base_size is None:
                    # For sells, we need base_size. Calculate from USD.
                    price = self.get_current_price(product_id)
                    if price and size_usd:
                        base_size = size_usd / price
                    else:
                        return {'error': 'Cannot determine base_size for sell order'}

                resp = self._client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(base_size),
                    **kwargs,
                )

            result = resp.to_dict() if hasattr(resp, 'to_dict') else resp
            logger.info(f"LIVE market {side} {product_id}: {result}")
            return result

        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return {'error': str(e)}

    def close_position(self, product_id, size=None):
        """Close an open perpetual position (full or partial).

        Args:
            product_id: e.g. 'BTC-PERP-INTX'
            size: base size to close. None = close entire position.

        Returns order result dict.
        """
        if self.paper_mode:
            return self._paper_close_position(product_id, size)

        if not self._client or not self.api_key:
            return {'error': 'Not authenticated'}

        client_order_id = str(uuid.uuid4())
        try:
            resp = self._client.close_position(
                client_order_id=client_order_id,
                product_id=product_id,
                size=str(size) if size else None,
            )
            result = resp.to_dict() if hasattr(resp, 'to_dict') else resp
            logger.info(f"LIVE close position {product_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return {'error': str(e)}

    def limit_order(self, product_id, side, base_size, limit_price):
        """Place a GTC limit order on a perpetual contract.

        Returns order result dict.
        """
        side = side.upper()
        if self.paper_mode:
            # Paper mode doesn't support pending limit orders — execute as market
            logger.info("Paper mode: limit order executed as market")
            return self._paper_market_order(product_id, side,
                                            size_usd=float(base_size) * float(limit_price))

        if not self._client or not self.api_key:
            return {'error': 'Not authenticated'}

        client_order_id = str(uuid.uuid4())
        try:
            kwargs = {}
            if self.leverage > 1:
                kwargs['leverage'] = str(self.leverage)
                kwargs['margin_type'] = 'ISOLATED'

            if side == 'BUY':
                resp = self._client.limit_order_gtc_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(base_size),
                    limit_price=str(limit_price),
                    **kwargs,
                )
            else:
                resp = self._client.limit_order_gtc_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(base_size),
                    limit_price=str(limit_price),
                    **kwargs,
                )

            result = resp.to_dict() if hasattr(resp, 'to_dict') else resp
            logger.info(f"LIVE limit {side} {product_id} @ {limit_price}: {result}")
            return result

        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            return {'error': str(e)}

    # ========================================================================
    # LEVERAGE
    # ========================================================================

    def set_leverage(self, leverage):
        """Set leverage for future orders.

        Hard capped at MAX_LEVERAGE (3x). Returns actual leverage set.
        """
        if leverage < 1:
            leverage = 1
        if leverage > MAX_LEVERAGE:
            logger.warning(f"Leverage {leverage}x exceeds MAX_LEVERAGE ({MAX_LEVERAGE}x). "
                          f"Capping at {MAX_LEVERAGE}x.")
            leverage = MAX_LEVERAGE

        if leverage > 1 and self._paper_trade_count < 30:
            logger.warning(f"Setting leverage to {leverage}x with only "
                          f"{self._paper_trade_count} trades. Recommended: 30+ trades "
                          f"before increasing leverage.")

        self.leverage = leverage
        logger.info(f"Leverage set to {leverage}x")
        return leverage

    def get_leverage(self):
        """Get current leverage setting."""
        return self.leverage

    # ========================================================================
    # PAPER TRADING SIMULATION
    # ========================================================================

    def _paper_market_order(self, product_id, side, size_usd=None, base_size=None):
        """Simulate a market order in paper mode."""
        price = self.get_current_price(product_id)
        if price is None:
            return {'error': f'Cannot get price for {product_id}'}

        # Determine position size
        if size_usd is not None:
            actual_base = size_usd / price
            actual_usd = size_usd
        elif base_size is not None:
            actual_base = float(base_size)
            actual_usd = actual_base * price
        else:
            return {'error': 'Must specify size_usd or base_size'}

        if actual_usd < MIN_ORDER_USDC:
            return {'error': f'Order ${actual_usd:.2f} below minimum ${MIN_ORDER_USDC}'}

        # Check available margin
        margin_required = actual_usd / self.leverage
        if margin_required > self._paper_cash:
            return {'error': f'Insufficient margin. Need ${margin_required:.2f}, '
                    f'have ${self._paper_cash:.2f}'}

        # Apply fee
        fee = actual_usd * TAKER_FEE_PCT / 100
        fill_price = price * (1 + TAKER_FEE_PCT / 100) if side == 'BUY' else \
                     price * (1 - TAKER_FEE_PCT / 100)

        # Update paper state
        existing = self._paper_positions.get(product_id)
        if existing:
            # Add to existing position
            old_size = existing['base_size']
            old_notional = existing['notional_usd']
            if existing['side'] == side:
                # Same direction — increase position
                existing['base_size'] += actual_base
                existing['notional_usd'] += actual_usd
                existing['avg_entry_price'] = existing['notional_usd'] / existing['base_size']
            else:
                # Opposite direction — reduce or flip
                if actual_base >= abs(old_size):
                    # Close and potentially flip
                    pnl = self._calc_paper_pnl(existing, price)
                    self._paper_cash += existing['margin_locked'] + pnl - fee
                    remaining = actual_base - abs(old_size)
                    del self._paper_positions[product_id]
                    if remaining > 0:
                        # Open new position in opposite direction
                        new_usd = remaining * price
                        new_margin = new_usd / self.leverage
                        self._paper_cash -= new_margin
                        self._paper_positions[product_id] = {
                            'product_id': product_id,
                            'side': side,
                            'base_size': remaining,
                            'avg_entry_price': fill_price,
                            'notional_usd': new_usd,
                            'margin_locked': new_margin,
                            'leverage': self.leverage,
                            'entry_time': datetime.now(timezone.utc).isoformat(),
                            'unrealized_pnl': 0,
                        }
                else:
                    # Partial close
                    pnl = self._calc_paper_pnl(existing, price, actual_base)
                    margin_release = existing['margin_locked'] * (actual_base / abs(old_size))
                    existing['base_size'] -= actual_base
                    existing['notional_usd'] = existing['base_size'] * existing['avg_entry_price']
                    existing['margin_locked'] -= margin_release
                    self._paper_cash += margin_release + pnl - fee
        else:
            # New position
            margin_locked = actual_usd / self.leverage
            self._paper_cash -= margin_locked
            self._paper_positions[product_id] = {
                'product_id': product_id,
                'side': side,
                'base_size': actual_base,
                'avg_entry_price': fill_price,
                'notional_usd': actual_usd,
                'margin_locked': margin_locked,
                'leverage': self.leverage,
                'entry_time': datetime.now(timezone.utc).isoformat(),
                'unrealized_pnl': 0,
            }

        self._paper_trade_count += 1
        order_id = f'paper-{uuid.uuid4().hex[:12]}'

        trade = {
            'order_id': order_id,
            'product_id': product_id,
            'side': side,
            'base_size': actual_base,
            'fill_price': fill_price,
            'notional_usd': actual_usd,
            'fee': fee,
            'leverage': self.leverage,
            'time': datetime.now(timezone.utc).isoformat(),
            'paper': True,
        }
        self._paper_trades.append(trade)
        self._save_paper_state()

        logger.info(f"PAPER {side} {product_id}: {actual_base:.6f} @ ${fill_price:.2f} "
                    f"(${actual_usd:.2f} notional, fee ${fee:.2f})")

        return {
            'success': True,
            'order_id': order_id,
            'product_id': product_id,
            'side': side,
            'fill_price': fill_price,
            'base_size': actual_base,
            'notional_usd': actual_usd,
            'fee': fee,
            'paper': True,
        }

    def _paper_close_position(self, product_id, size=None):
        """Close a paper position (full or partial)."""
        pos = self._paper_positions.get(product_id)
        if not pos:
            return {'error': f'No open position for {product_id}'}

        price = self.get_current_price(product_id)
        if price is None:
            return {'error': f'Cannot get price for {product_id}'}

        close_size = float(size) if size else pos['base_size']
        close_size = min(close_size, pos['base_size'])

        # Calculate P&L
        pnl = self._calc_paper_pnl(pos, price, close_size)
        fee = close_size * price * TAKER_FEE_PCT / 100

        # Update state
        fraction = close_size / pos['base_size']
        margin_release = pos['margin_locked'] * fraction

        if close_size >= pos['base_size']:
            # Full close
            del self._paper_positions[product_id]
        else:
            # Partial close
            pos['base_size'] -= close_size
            pos['notional_usd'] = pos['base_size'] * pos['avg_entry_price']
            pos['margin_locked'] -= margin_release

        self._paper_cash += margin_release + pnl - fee
        self._paper_trade_count += 1

        close_side = 'BUY' if pos.get('side', '') == 'SELL' else 'SELL'
        order_id = f'paper-close-{uuid.uuid4().hex[:12]}'

        trade = {
            'order_id': order_id,
            'product_id': product_id,
            'side': close_side,
            'base_size': close_size,
            'fill_price': price,
            'notional_usd': close_size * price,
            'fee': fee,
            'pnl': pnl,
            'time': datetime.now(timezone.utc).isoformat(),
            'paper': True,
        }
        self._paper_trades.append(trade)
        self._save_paper_state()

        pnl_pct = (pnl / (close_size * pos['avg_entry_price'])) * 100

        logger.info(f"PAPER CLOSE {product_id}: {close_size:.6f} @ ${price:.2f}, "
                    f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

        return {
            'success': True,
            'order_id': order_id,
            'product_id': product_id,
            'close_size': close_size,
            'fill_price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'fee': fee,
            'paper': True,
        }

    def _calc_paper_pnl(self, position, current_price, size=None):
        """Calculate unrealized P&L for a paper position."""
        base_size = size if size else position['base_size']
        entry = position['avg_entry_price']

        if position['side'] == 'BUY':
            # Long: profit when price goes up
            pnl = (current_price - entry) * base_size
        else:
            # Short: profit when price goes down
            pnl = (entry - current_price) * base_size

        return pnl

    def _paper_get_positions(self):
        """Get all paper positions with updated unrealized P&L."""
        positions = []
        for pid, pos in self._paper_positions.items():
            price = self.get_current_price(pid)
            if price:
                pos['unrealized_pnl'] = self._calc_paper_pnl(pos, price)
                pos['current_price'] = price
                pnl_pct = (pos['unrealized_pnl'] /
                          (pos['base_size'] * pos['avg_entry_price'])) * 100
                pos['unrealized_pnl_pct'] = pnl_pct
            positions.append(dict(pos))
        return positions

    def _paper_balance_summary(self):
        """Get paper account balance summary."""
        total_margin_locked = sum(p['margin_locked'] for p in self._paper_positions.values())
        total_unrealized = 0
        for pid, pos in self._paper_positions.items():
            price = self.get_current_price(pid)
            if price:
                total_unrealized += self._calc_paper_pnl(pos, price)

        total_equity = self._paper_cash + total_margin_locked + total_unrealized

        return {
            'total_usd_balance': total_equity,
            'available_margin': self._paper_cash,
            'margin_used': total_margin_locked,
            'unrealized_pnl': total_unrealized,
            'open_positions': len(self._paper_positions),
            'trade_count': self._paper_trade_count,
            'starting_capital': self._paper_starting_capital,
            'total_return_pct': ((total_equity / self._paper_starting_capital) - 1) * 100,
            'paper': True,
        }

    def _save_paper_state(self):
        """Save paper trading state to JSON for crash recovery."""
        state = {
            'cash': self._paper_cash,
            'starting_capital': self._paper_starting_capital,
            'positions': self._paper_positions,
            'trade_count': self._paper_trade_count,
            'trades': self._paper_trades[-100:],  # keep last 100
            'leverage': self.leverage,
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(PAPER_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save paper state: {e}")

    def _load_paper_state(self):
        """Load paper trading state from JSON."""
        try:
            with open(PAPER_STATE_FILE, 'r') as f:
                state = json.load(f)
            self._paper_cash = state.get('cash', self._paper_starting_capital)
            self._paper_starting_capital = state.get('starting_capital',
                                                      self._paper_starting_capital)
            self._paper_positions = state.get('positions', {})
            self._paper_trade_count = state.get('trade_count', 0)
            self._paper_trades = state.get('trades', [])
            self.leverage = state.get('leverage', DEFAULT_LEVERAGE)
            logger.info(f"Loaded paper state: ${self._paper_cash:.2f} cash, "
                        f"{len(self._paper_positions)} positions, "
                        f"{self._paper_trade_count} trades")
        except FileNotFoundError:
            pass  # Fresh start
        except Exception as e:
            logger.warning(f"Failed to load paper state: {e}")

    def reset_paper(self, starting_capital=None):
        """Reset paper trading state to fresh start."""
        if starting_capital:
            self._paper_starting_capital = starting_capital
        self._paper_cash = self._paper_starting_capital
        self._paper_positions = {}
        self._paper_trades = []
        self._paper_trade_count = 0
        self.leverage = DEFAULT_LEVERAGE
        self._save_paper_state()
        logger.info(f"Paper state reset: ${self._paper_starting_capital:.2f} capital")

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def spot_to_perp(spot_symbol):
        """Convert spot symbol (BTC-USD) to perp product ID (BTC-PERP-INTX)."""
        return SPOT_TO_PERP.get(spot_symbol)

    @staticmethod
    def perp_to_spot(perp_id):
        """Convert perp product ID (BTC-PERP-INTX) to spot symbol (BTC-USD)."""
        return PERP_TO_SPOT.get(perp_id)

    def is_authenticated(self):
        """Check if client has valid API credentials loaded."""
        return bool(self.api_key and self.api_secret and self._client)

    def test_auth(self):
        """Test API authentication by calling a read-only endpoint.

        Returns (success: bool, message: str).
        """
        if not self.is_authenticated():
            return False, "No API credentials. Set COINBASE_API_KEY and COINBASE_API_SECRET in .env"

        try:
            resp = self._client.get_futures_balance_summary()
            return True, "Authentication successful"
        except Exception as e:
            return False, f"Authentication failed: {e}"
