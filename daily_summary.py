"""Send daily trade summary via Telegram. Run via cron."""

from dotenv import load_dotenv
load_dotenv()

from notify import send_daily_summary

send_daily_summary("logs/trades.log")
