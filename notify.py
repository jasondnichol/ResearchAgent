"""Trading Bot Logger & Telegram Notifier"""
import requests
import json
import os
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1422959932")

# Setup file logging
def setup_logging(log_dir="logs"):
    """Setup file and console logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("tradingbot")
    logger.setLevel(logging.INFO)
    
    # File handler - all activity
    fh = logging.FileHandler(f"{log_dir}/trading.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    # Trade-only file handler
    th = logging.FileHandler(f"{log_dir}/trades.log")
    th.setLevel(logging.INFO)
    th.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    trade_logger = logging.getLogger("trades")
    trade_logger.setLevel(logging.INFO)
    trade_logger.addHandler(th)
    
    return logger, trade_logger


def send_telegram(message):
    """Send a message via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️  Telegram bot token not set")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"⚠️  Telegram error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"⚠️  Telegram send failed: {e}")
        return False


def notify_buy(price, strategy, regime, mode="PAPER"):
    """Send BUY notification"""
    msg = (
        f"🟢 <b>BUY EXECUTED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💰 Price: <b>${price:,.2f}</b>\n"
        f"📊 Strategy: {strategy}\n"
        f"📈 Regime: {regime}\n"
        f"🏷 Mode: {mode}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram(msg)
    
    trade_logger = logging.getLogger("trades")
    trade_logger.info(f"BUY | ${price:,.2f} | {strategy} | {regime} | {mode}")


def notify_sell(entry_price, exit_price, pnl_pct, strategy, regime, mode="PAPER"):
    """Send SELL notification"""
    emoji = "🟢" if pnl_pct >= 0 else "🔴"
    pnl_sign = "+" if pnl_pct >= 0 else ""
    
    msg = (
        f"🔴 <b>SELL EXECUTED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💰 Entry: ${entry_price:,.2f}\n"
        f"💰 Exit: <b>${exit_price:,.2f}</b>\n"
        f"{emoji} P&L: <b>{pnl_sign}{pnl_pct:.2f}%</b>\n"
        f"📊 Strategy: {strategy}\n"
        f"📈 Regime: {regime}\n"
        f"🏷 Mode: {mode}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram(msg)
    
    trade_logger = logging.getLogger("trades")
    trade_logger.info(f"SELL | Entry: ${entry_price:,.2f} | Exit: ${exit_price:,.2f} | P&L: {pnl_sign}{pnl_pct:.2f}% | {strategy} | {regime} | {mode}")


def notify_regime_change(old_regime, new_regime, price):
    """Notify when market regime changes"""
    msg = (
        f"🔄 <b>REGIME CHANGE</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📊 {old_regime} → <b>{new_regime}</b>\n"
        f"💰 BTC: ${price:,.2f}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram(msg)


def notify_hold(price, regime, reason, strategy=None):
    """Log HOLD signals (file only, no Telegram)"""
    logger = logging.getLogger("tradingbot")
    strat_str = f" | {strategy}" if strategy else ""
    logger.info(f"HOLD | ${price:,.2f} | {regime}{strat_str} | {reason}")


def send_daily_summary(trades_log_path="logs/trades.log"):
    """Send daily summary of last 24 hours"""
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(hours=24)
    
    trades_today = []
    total_pnl = 0.0
    wins = 0
    losses = 0
    
    try:
        with open(trades_log_path, 'r') as f:
            for line in f:
                try:
                    timestamp_str = line.split(' | ')[0].strip()
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    if timestamp >= yesterday:
                        trades_today.append(line.strip())
                        
                        if 'SELL' in line and 'P&L:' in line:
                            pnl_str = line.split('P&L:')[1].split('%')[0].strip()
                            pnl = float(pnl_str)
                            total_pnl += pnl
                            if pnl >= 0:
                                wins += 1
                            else:
                                losses += 1
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        trades_today = []
    
    buy_count = sum(1 for t in trades_today if 'BUY' in t)
    sell_count = sum(1 for t in trades_today if 'SELL' in t)
    total_trades = sell_count
    
    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    pnl_sign = "+" if total_pnl >= 0 else ""
    
    win_rate = f"{(wins/(wins+losses)*100):.0f}%" if (wins + losses) > 0 else "N/A"
    
    msg = (
        f"📋 <b>DAILY SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📅 Last 24 hours\n"
        f"🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"📊 <b>Trades:</b> {total_trades} completed\n"
        f"🟢 Buys: {buy_count} | 🔴 Sells: {sell_count}\n"
        f"✅ Wins: {wins} | ❌ Losses: {losses}\n"
        f"🎯 Win Rate: {win_rate}\n"
        f"{pnl_emoji} <b>Total P&L: {pnl_sign}{total_pnl:.2f}%</b>\n"
    )
    
    if not trades_today:
        msg += "\n💤 No trades in the last 24 hours."
    
    send_telegram(msg)
    return True


def send_startup_message():
    """Send notification that bot has started"""
    msg = (
        f"🚀 <b>TRADING BOT STARTED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🏷 Mode: PAPER TRADING\n"
        f"⏱ Interval: 1 hour\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram(msg)


if __name__ == "__main__":
    print("Testing Telegram connection...")
    success = send_telegram("🧪 <b>Test message</b>\nTrading bot notification system is working!")
    if success:
        print("✅ Telegram message sent!")
    else:
        print("❌ Failed to send. Check your bot token.")
