import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# TELEGRAM TOKEN (Railway Environment Variable)
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

DEFAULT_TIMEFRAME = "3y"
DEFAULT_INTERVAL = "1d"

# ค่า default สำหรับ Manual Mode
DEFAULT_TP1_PCT = 5.6
DEFAULT_TP2_PCT = 22.6
DEFAULT_SL_PCT = -3.5

# สีสำหรับกราฟ
COLORS = {
    'bullish': '#26A69A',
    'bearish': '#EF5350',
    'ema20': '#2196F3',
    'ema50': '#FF9800',
    'ema200': '#9C27B0',
    'tp1': '#2E7D32',
    'tp2': '#1B5E20',
    'sl': '#C62828',
    'entry': '#1565C0',
}
