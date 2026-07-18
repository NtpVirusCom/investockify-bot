#!/usr/bin/env python3
"""
🤖 Investockify Bot — Railway Cost-Optimized Edition (v2.2.2)
==========================================================v.25==
ปรับปรุงจาก v2.0.3 สำหรับประหยัดค่าใช้จ่าย Railway:
  • ลด Memory Usage: จำกัด cache size, ใช้ weakref, ลบ df ที่ไม่ใช้
  • ลด CPU Usage: ลดการคำนวณซ้ำ, ใช้ lru_cache, ปิด matplotlib interactive
  • ลด Network Calls: ข่าว cache นานขึ้น, dedup ดีขึ้น, batch requests
  • ลด Disk I/O: ไม่เขียนไฟล์ชั่วคราว, ใช้ BytesIO อย่างเดียว
  • ลด OpenAI API Cost: ใช้ regex มากขึ้น, AI แค่กรณีจำเป็น
  • Graceful Shutdown + Railway Health Check
  • รองรับหุ้นไทย, Crypto, Commodities ครบถ้วน

Environment Variables (บน Railway → Variables):
-----------------------------------------------
TELEGRAM_TOKEN        : required
OPENAI_API_KEY        : optional (สำหรับ AI Summary)
TIMEFRAME             : default "1d"
PORTFOLIO_VALUE       : default "100000"
RISK_PER_TRADE        : default "1.0"
LOG_LEVEL             : default "INFO"
BOT_MODE              : "polling" (default) หรือ "webhook"
PORT                  : default "8080"
WEBHOOK_URL           : บังคับถ้าใช้ webhook
WEBHOOK_SECRET        : optional
"""

import os
import sys
import io
import re
import html
import time
import signal
import logging
import threading
import unicodedata
import gc
import weakref
from functools import lru_cache
from types import ModuleType
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque, Set
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # สำคัญมากสำหรับ headless server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ConversationHandler, filters, ContextTypes
)

# =========================================================
# RAILWAY LOGGING CONFIG (ลด verbosity)
# =========================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO)
)
logger = logging.getLogger(__name__)

# =========================================================
# ENVIRONMENT CONFIG
# =========================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable is required.")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TIMEFRAME = os.environ.get("TIMEFRAME", "1d")
PORTFOLIO_VALUE = float(os.environ.get("PORTFOLIO_VALUE", "100000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "1.0"))
VERSION = "2.2.2"

BOT_MODE = os.environ.get("BOT_MODE", "polling").lower()
PORT = int(os.environ.get("PORT", "8080"))
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

# =========================================================
# 🛡️ DEDUPLICATION & RATE LIMITING (ปรับปรุง)
# =========================================================
class UpdateDeduplicator:
    """กรอง update_id ที่ประมวลผลแล้ว"""
    def __init__(self, max_size: int = 500, ttl_seconds: int = 120):
        self._cache: Set[int] = set()
        self._timestamps: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def is_duplicate(self, update_id: Optional[int]) -> bool:
        if update_id is None:
            return False
        with self._lock:
            now = time.time()
            # Cleanup ถ้า cache ใหญ่เกินไป
            if len(self._cache) > self._max_size:
                self._cache.clear()
                self._timestamps.clear()
            if update_id in self._cache:
                return True
            self._cache.add(update_id)
            self._timestamps[update_id] = now
            return False


class UserRateLimiter:
    """จำกัดความถี่การขอวิเคราะห์ต่อ user"""
    def __init__(self, cooldown_seconds: float = 10.0):
        self._last_request: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._cooldown = cooldown_seconds

    def is_limited(self, user_id: int) -> bool:
        now = time.time()
        with self._lock:
            last = self._last_request.get(user_id, 0)
            if now - last < self._cooldown:
                return True
            self._last_request[user_id] = now
            return False

    def remaining_cooldown(self, user_id: int) -> float:
        now = time.time()
        with self._lock:
            last = self._last_request.get(user_id, 0)
            return max(0.0, self._cooldown - (now - last))


class NewsCache:
    """แคชผลลัพธ์ format_news_section() ต่อ symbol — TTL นานขึ้นเพื่อประหยัด"""
    def __init__(self, ttl_seconds: int = 600):  # 10 นาที (จากเดิม 5 นาที)
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, symbol: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            if symbol in self._cache:
                ts, result = self._cache[symbol]
                if now - ts < self._ttl:
                    return result
                del self._cache[symbol]
            return None

    def set(self, symbol: str, result: str):
        with self._lock:
            # จำกัดจำนวน cache entry
            if len(self._cache) > 100:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest]
            self._cache[symbol] = (time.time(), result)

    def invalidate(self, symbol: str):
        with self._lock:
            self._cache.pop(symbol, None)


class AISummaryManager:
    """จัดการ Cache + Lock สำหรับ AI Summary"""
    def __init__(self, ttl_seconds: int = 600):  # 10 นาที
        self._cache: Dict[str, Tuple[float, str]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._main_lock = threading.Lock()
        self._ttl = ttl_seconds

    def _cleanup(self):
        now = time.time()
        with self._main_lock:
            expired = [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl]
            for k in expired:
                self._cache.pop(k, None)
                self._locks.pop(k, None)

    def get(self, symbol: str) -> Optional[str]:
        self._cleanup()
        with self._main_lock:
            if symbol in self._cache:
                ts, result = self._cache[symbol]
                if time.time() - ts < self._ttl:
                    return result
                del self._cache[symbol]
            return None

    def set(self, symbol: str, result: str):
        with self._main_lock:
            if len(self._cache) > 50:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest]
            self._cache[symbol] = (time.time(), result)

    def get_lock(self, symbol: str) -> threading.Lock:
        with self._main_lock:
            if symbol not in self._locks:
                self._locks[symbol] = threading.Lock()
            return self._locks[symbol]

# Singleton instances
update_dedup = UpdateDeduplicator(max_size=500, ttl_seconds=120)
user_rate_limit = UserRateLimiter(cooldown_seconds=10.0)
news_cache = NewsCache(ttl_seconds=600)
ai_summary_mgr = AISummaryManager(ttl_seconds=600)

# =========================================================
# UNIFIED CONFIG
# =========================================================
COLORS = {
    'bullish': '#26A69A', 'bearish': '#EF5350',
    'ema20': '#2196F3', 'ema50': '#FF9800', 'ema200': '#9C27B0',
    'tp1': '#2E7D32', 'tp2': '#1B5E20', 'sl': '#C62828',
    'entry': '#00695C', 'now': '#1565C0',
    'poc': '#FF9800', 'vah': '#AB47BC', 'val': '#AB47BC',
}
DEFAULT_TP1_PCT = 5.6
DEFAULT_TP2_PCT = 22.6
DEFAULT_SL_PCT = -3.5

# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Level:
    price: float
    bar_index: int
    strength: int
    level_type: str

@dataclass
class VolumeProfile:
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    lookback_days: int = 120

@dataclass
class TradeSetup:
    entry: float
    sl: float
    tp1: float
    tp2: float
    support_levels: List[Level] = field(default_factory=list)
    resistance_levels: List[Level] = field(default_factory=list)
    atr: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    entry_zone_bottom: float = 0.0
    entry_zone_top: float = 0.0
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    weekly_trend: str = "UNKNOWN"
    daily_trend: str = "UNKNOWN"
    volume_avg: float = 0.0

    @property
    def risk(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def reward1(self) -> float:
        return abs(self.tp1 - self.entry)

    @property
    def reward2(self) -> float:
        return abs(self.tp2 - self.entry)

    @property
    def rr1(self) -> float:
        return self.reward1 / self.risk if self.risk != 0 else 0

    @property
    def rr2(self) -> float:
        return self.reward2 / self.risk if self.risk != 0 else 0

    def get_pct_from_entry(self, price: float) -> float:
        return ((price - self.entry) / self.entry) * 100 if self.entry != 0 else 0

    def get_entry_zone(self, atr: float) -> Tuple[float, float]:
        zone_width = atr * 0.2
        return (self.entry - zone_width, self.entry + zone_width)

# =========================================================
# INPUT SANITIZATION & SYMBOL FORMATTING
# =========================================================
THAI_STOCKS: Set[str] = {
    "ADVANC", "AOT", "BBL", "BDMS", "BH", "CPALL", "CPF",
    "CPN", "CRC", "DELTA", "DIF", "GPSC", "GULF", "HMPRO", "IVL",
    "KBANK", "KTB", "KTC", "MINT", "OR", "PTT", "PTTEP",
    "PTTGC", "SCB", "SCC", "SCGP", "THAI", "TLI", "TRUE", "TTB",
    "AP", "AMATA", "EGCO", "LH", "ROJNA", "SCB", "SCC", "TTW"
}

CRYPTO_PAIRS: Set[str] = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "DOGE",
    "AVAX", "LINK", "MATIC", "LTC", "UNI", "AAVE"
}

COMMODITIES = {
    "GOLD": {
        "symbol": "GC=F",
        "name": "ทองคำ (ออนซ์)"
    },
    "SILVER": {
        "symbol": "SI=F",
        "name": "แร่เงิน (ออนซ์)"
    },
    "COPPER": {
        "symbol": "HG=F",
        "name": "ทองแดง (ปอนด์)"
    },
    "OIL": {
        "symbol": "CL=F",
        "name": "น้ำมันดิบ WTI (บาร์เรล)"
    },
    "GAS": {
        "symbol": "NG=F",
        "name": "ก๊าซธรรมชาติ (ล้านบีทียู)"
    }
}


def sanitize_input(text: str) -> Optional[str]:
    """ทำความสะอาด input ก่อนใช้งาน"""
    if not text or not isinstance(text, str):
        return None
    text = unicodedata.normalize('NFKC', text)
    text = text.strip().upper()
    text = re.sub(r'[;|&$<>\`]', '', text)
    if len(text) < 1 or len(text) > 20:
        return None
    return text


def format_symbol(symbol: str) -> str:
    """แปลงชื่อหุ้นให้ yfinance เข้าใจ"""
    symbol = symbol.upper().strip()

    if symbol.endswith(('.BK', '.BKK', '-BK')):
        return symbol.replace('.BKK', '.BK').replace('-BK', '.BK')

    if symbol.endswith(('-USD', '_USD', '.USD', 'USDT', '-USDT')):
        base = symbol.replace('USDT', '').replace('-USD', '').replace('_USD', '').replace('.USD', '')
        return f"{base}-USD"

    if symbol in THAI_STOCKS:
        return f"{symbol}.BK"

    if symbol in CRYPTO_PAIRS:
        return f"{symbol}-USD"
      
    # Commodity Mapping
    if symbol in COMMODITIES:
      return COMMODITIES[symbol]["symbol"]

    return symbol


def get_commodity_name(symbol: str) -> Optional[str]:
    """Return display name for commodity symbol."""
    for item in COMMODITIES.values():
        if item["symbol"] == symbol:
            return item["name"]
    return None
  

def is_valid_stock_input(text: str) -> bool:
    """Validate ชื่อหุ้น"""
    if not text:
        return False
    clean = sanitize_input(text)
    if not clean:
        return False
    pattern = r'^[A-Z\^][A-Z0-9\.\-\_\=]{0,19}$'
    if not re.match(pattern, clean):
        return False
    dangerous = ['../', '..\\', '`', '$(', '${']
    if any(d in clean for d in dangerous):
        return False
    return True


# =========================================================
# UNIFIED DATA FETCHER (Cost-Optimized)
# =========================================================
class DataFetcher:
    """ลด memory usage และ network calls"""
    def __init__(self):
        self._df_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 300  # 5 นาที
        self._max_cache_size = 30  # ลดจาก 50
        self._cache_lock = threading.Lock()

    def _get_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            now = time.time()
            if symbol in self._df_cache:
                ts, df = self._df_cache[symbol]
                if now - ts < self._cache_ttl:
                    return df
                del self._df_cache[symbol]
            return None

    def _set_cache(self, symbol: str, df: pd.DataFrame):
        with self._cache_lock:
            if len(self._df_cache) >= self._max_cache_size:
                oldest = min(self._df_cache.keys(), key=lambda k: self._df_cache[k][0])
                del self._df_cache[oldest]
                gc.collect()
            self._df_cache[symbol] = (time.time(), df)

    def get_stock_data(self, symbol: str, period: str = "3y", interval: str = "1d"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            if df.empty:
                return None, "ไม่พบข้อมูลสำหรับสัญลักษณ์นี้"
            return df, None
        except Exception as e:
            return None, f"เกิดข้อผิดพลาด: {str(e)}"

    def fetch_ohlcv(self, symbol: str, period: str = "2y", interval: str = None):
        if interval is None:
            interval = TIMEFRAME

        cached = self._get_cache(symbol)
        if cached is not None:
            return cached

        try:
            logger.info("Fetching data for %s...", symbol)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            if df.empty:
                logger.warning("No data returned for %s", symbol)
                return None

            # ลด memory โดยเลือกเฉพาะคอลัมน์ที่จำเป็น
            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            keep_cols = ["date", "datetime", "open", "high", "low", "close", "volume"]
            existing = [c for c in keep_cols if c in df.columns]
            df = df[existing]
            if "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            if "datetime" in df.columns:
                df.rename(columns={"datetime": "timestamp"}, inplace=True)
            df.dropna(inplace=True)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            if len(df) < 100:
                logger.warning("Insufficient data for %s: %s rows", symbol, len(df))
                return None
            logger.info("Fetched %s rows for %s", len(df), symbol)

            self._set_cache(symbol, df)
            return df
        except Exception as e:
            logger.error("Error fetching %s: %s", symbol, e)
            return None

    def get_current_price(self, symbol: str):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current = info.get('regularMarketPrice') or info.get('currentPrice')
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
            if current and prev_close:
                change_pct = ((current - prev_close) / prev_close) * 100
                return {
                    'price': current,
                    'change_pct': change_pct,
                    'prev_close': prev_close
                }
            df = ticker.history(period="5d", interval="1d")
            if not df.empty:
                current = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current
                change_pct = ((current - prev_close) / prev_close) * 100 if prev_close else 0
                return {'price': current, 'change_pct': change_pct, 'prev_close': prev_close}
            return None
        except Exception as e:
            logger.warning("get_current_price failed for %s: %s", symbol, e)
            return None


# =========================================================
# UNIFIED TECHNICAL ANALYZER (Cost-Optimized)
# =========================================================
class TechnicalAnalyzer:
    def __init__(self):
        self.PIVOT_LEN = 5
        self.MIN_ATR_STRENGTH = 0.1
        self.MAX_LEVEL_AGE = 300
        self.MAX_ACTIVE_LEVELS_EACH = 5
        self.MERGE_THRESHOLD = 0.5
        self.ZONE_WIDTH = 0.25
        self.BREAK_SENS = 0.1
        self.MAX_BREAKOUT_SIGNALS = 4
        self.ATR_PERIOD = 14
        self.MAX_DISTANCE_FROM_PRICE = 0.5
        self.MIN_BREAKOUT_MOVE_ATR = 0.1
        self.VOLUME_CONFIRMATION_RATIO = 1.5
        self.ENTRY_ZONE_ATR_MULT = 0.2

    def calculate_ema(self, data: pd.Series, period: int):
        return data.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14):
        df = df.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["tr1"] = df["High"] - df["Low"]
        df["tr2"] = abs(df["High"] - df["prev_close"])
        df["tr3"] = abs(df["Low"] - df["prev_close"])
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["tr"].ewm(alpha=1/period, adjust=False).mean()
        return df

    def detect_pivots(self, df):
        highs, lows = [], []
        n = self.PIVOT_LEN
        if len(df) < 2 * n + 1:
            return highs, lows

        left_high = df["High"].rolling(window=n, center=False).max().shift(1)
        right_high = df["High"].rolling(window=n, center=False).max().shift(-n)
        pivot_high_mask = (df["High"] >= left_high) & (df["High"] >= right_high)

        left_low = df["Low"].rolling(window=n, center=False).min().shift(1)
        right_low = df["Low"].rolling(window=n, center=False).min().shift(-n)
        pivot_low_mask = (df["Low"] <= left_low) & (df["Low"] <= right_low)

        atr_series = df["atr"]
        for i in range(n, len(df) - n):
            if pd.isna(atr_series.iloc[i]) or atr_series.iloc[i] == 0:
                continue
            atr = atr_series.iloc[i]
            if pivot_high_mask.iloc[i]:
                current_high = df["High"].iloc[i]
                close = df["Close"].iloc[i]
                strength = (current_high - close) / atr
                if strength >= self.MIN_ATR_STRENGTH:
                    highs.append(Level(price=current_high, bar_index=i, strength=1, level_type='resistance'))
            if pivot_low_mask.iloc[i]:
                current_low = df["Low"].iloc[i]
                close = df["Close"].iloc[i]
                strength = (close - current_low) / atr
                if strength >= self.MIN_ATR_STRENGTH:
                    lows.append(Level(price=current_low, bar_index=i, strength=1, level_type='support'))
        return highs, lows

    def merge_levels(self, levels: List[Level], atr: float):
        if not levels:
            return []
        sorted_levels = sorted(levels, key=lambda x: x.price)
        merged_groups, current_group = [], [sorted_levels[0]]
        for lvl in sorted_levels[1:]:
            avg_price = np.mean([x.price for x in current_group])
            if abs(lvl.price - avg_price) <= atr * self.MERGE_THRESHOLD:
                current_group.append(lvl)
            else:
                merged_groups.append(current_group)
                current_group = [lvl]
        merged_groups.append(current_group)
        result = []
        for group in merged_groups:
            avg_price = np.mean([x.price for x in group])
            latest_bar = max([x.bar_index for x in group])
            result.append(Level(price=round(avg_price, 2), bar_index=latest_bar,
                               strength=len(group), level_type=group[0].level_type))
        return result

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14):
        close = df['Close']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 50, lookback_days: int = 120) -> VolumeProfile:
        vp = VolumeProfile(lookback_days=lookback_days)
        if df.empty or 'Volume' not in df.columns:
            return vp
        recent_df = df.tail(lookback_days).copy()
        if recent_df.empty:
            return vp
        current_price = float(df['Close'].iloc[-1])
        typical_price = (recent_df['High'] + recent_df['Low'] + recent_df['Close']) / 3
        min_price = typical_price.min()
        max_price = typical_price.max()
        if pd.isna(min_price) or pd.isna(max_price):
            return vp
        if max_price < current_price * 0.5 or min_price > current_price * 2.0:
            vwap = (recent_df['Close'] * recent_df['Volume']).sum() / recent_df['Volume'].sum()
            vp.poc = float(vwap) if not pd.isna(vwap) else current_price
            return vp
        bin_width = (max_price - min_price) / bins
        if bin_width == 0:
            vp.poc = float(typical_price.iloc[-1])
            return vp
        volume_profile = {}
        for i in range(bins):
            bin_low = min_price + i * bin_width
            bin_high = min_price + (i + 1) * bin_width
            mask = (typical_price >= bin_low) & (typical_price < bin_high)
            vol_sum = recent_df.loc[mask, 'Volume'].sum()
            bin_center = (bin_low + bin_high) / 2
            volume_profile[bin_center] = vol_sum
        mask_max = typical_price >= max_price
        if mask_max.any():
            volume_profile[max_price] = recent_df.loc[mask_max, 'Volume'].sum()
        if not volume_profile:
            vp.poc = float(typical_price.iloc[-1])
            return vp
        poc = max(volume_profile, key=volume_profile.get)
        vp.poc = float(poc)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.70
        sorted_bins = sorted(volume_profile.items(), key=lambda x: x[0])
        poc_idx = next(i for i, (price, _) in enumerate(sorted_bins) if price == poc)
        accumulated = sorted_bins[poc_idx][1]
        lower_idx = poc_idx
        upper_idx = poc_idx
        while accumulated < target_volume and (lower_idx > 0 or upper_idx < len(sorted_bins) - 1):
            lower_vol = sorted_bins[lower_idx - 1][1] if lower_idx > 0 else 0
            upper_vol = sorted_bins[upper_idx + 1][1] if upper_idx < len(sorted_bins) - 1 else 0
            if lower_vol >= upper_vol and lower_idx > 0:
                lower_idx -= 1
                accumulated += lower_vol
            elif upper_idx < len(sorted_bins) - 1:
                upper_idx += 1
                accumulated += upper_vol
            else:
                break
        vp.val = float(sorted_bins[lower_idx][0])
        vp.vah = float(sorted_bins[upper_idx][0])
        if vp.poc < current_price * 0.6:
            vwap = (recent_df['Close'] * recent_df['Volume']).sum() / recent_df['Volume'].sum()
            vp.poc = float(vwap) if not pd.isna(vwap) else current_price
        return vp

    def _get_sl_multiplier(self, atr: float, current_price: float) -> float:
        atr_pct = atr / current_price if current_price > 0 else 0
        if atr_pct > 0.03:
            return 1.5
        elif atr_pct > 0.015:
            return 2.0
        else:
            return 2.5

    def find_trade_setup(self, df: pd.DataFrame, current_price: float,
                        weekly_trend: str = "UNKNOWN") -> TradeSetup:
        df_atr = self.calculate_atr(df)
        atr = df_atr["atr"].iloc[-1]
        if pd.isna(atr) or atr == 0:
            atr = (df["High"] - df["Low"]).mean()

        highs, lows = self.detect_pivots(df_atr)
        current_bar = len(df)
        highs = [x for x in highs if current_bar - x.bar_index <= self.MAX_LEVEL_AGE]
        lows = [x for x in lows if current_bar - x.bar_index <= self.MAX_LEVEL_AGE]

        merged_highs = self.merge_levels(highs, atr)
        merged_lows = self.merge_levels(lows, atr)

        merged_highs = [x for x in merged_highs
            if x.price > current_price and (x.price - current_price) / current_price <= self.MAX_DISTANCE_FROM_PRICE]
        merged_lows = [x for x in merged_lows
            if x.price < current_price and (current_price - x.price) / current_price <= self.MAX_DISTANCE_FROM_PRICE]

        merged_highs = sorted(merged_highs, key=lambda x: (abs(x.price - current_price), -x.strength))[:self.MAX_ACTIVE_LEVELS_EACH]
        merged_lows = sorted(merged_lows, key=lambda x: (abs(x.price - current_price), -x.strength))[:self.MAX_ACTIVE_LEVELS_EACH]

        if merged_highs:
            tp1_price = merged_highs[0].price
            tp2_price = merged_highs[1].price if len(merged_highs) >= 2 else tp1_price + (atr * 3.0)
        else:
            tp1_price = current_price + (atr * 2.0)
            tp2_price = tp1_price + (atr * 3.0)

        if merged_lows:
            entry_price = merged_lows[0].price
            sl_price = merged_lows[1].price if len(merged_lows) >= 2 else None
        else:
            entry_price = current_price - (atr * 1.5)
            sl_price = None

        sl_multiplier = self._get_sl_multiplier(atr, current_price)
        if sl_price is None:
            sl_buffer = atr * sl_multiplier
            sl_price = entry_price - sl_buffer

        if entry_price >= current_price:
            entry_price = current_price - (atr * 1.5)
            sl_price = entry_price - (atr * sl_multiplier)
        if sl_price >= entry_price:
            sl_price = entry_price * 0.97

        support = merged_lows[0].price if merged_lows else entry_price
        resistance = merged_highs[0].price if merged_highs else tp1_price

        zone_width = atr * self.ENTRY_ZONE_ATR_MULT
        entry_zone_bottom = entry_price - zone_width
        entry_zone_top = entry_price + zone_width

        vp = self.calculate_volume_profile(df)
        volume_avg = df['Volume'].tail(20).mean() if 'Volume' in df.columns else 0

        ema200 = self.calculate_ema(df['Close'], 200).iloc[-1]
        daily_trend = "BULLISH" if current_price > ema200 * 1.02 else "BEARISH"

        return TradeSetup(
            entry=entry_price, sl=sl_price, tp1=tp1_price, tp2=tp2_price,
            support_levels=merged_lows, resistance_levels=merged_highs,
            atr=atr, support=support, resistance=resistance,
            entry_zone_bottom=entry_zone_bottom, entry_zone_top=entry_zone_top,
            poc=vp.poc, vah=vp.vah, val=vp.val,
            weekly_trend=weekly_trend,
            daily_trend=daily_trend,
            volume_avg=volume_avg
        )

# =========================================================
# UNIFIED CHART GENERATOR (Cost-Optimized)
# =========================================================
class ChartGenerator:
    def __init__(self, analyzer: TechnicalAnalyzer):
        self.analyzer = analyzer
        plt.style.use('default')

    def _safe_float(self, val):
        if val is None:
            return None
        if isinstance(val, pd.Series):
            val = val.iloc[-1] if len(val) > 0 else None
        try:
            f = float(val)
            if pd.isna(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    def generate_trading_chart(self, df: pd.DataFrame, symbol: str,
                               setup: TradeSetup = None,
                               use_smart_entry: bool = True,
                               trend: str = "NEUTRAL",
                               entry_price: float = None,
                               tp1_price: float = None,
                               tp2_price: float = None,
                               sl_price: float = None,
                               tp1_pct: float = None,
                               tp2_pct: float = None):
        close = df['Close']
        ema20 = self.analyzer.calculate_ema(close, 20)
        ema50 = self.analyzer.calculate_ema(close, 50)
        ema200 = self.analyzer.calculate_ema(close, 200)

        ema20_last = self._safe_float(ema20.iloc[-1])
        ema50_last = self._safe_float(ema50.iloc[-1])
        ema200_last = self._safe_float(ema200.iloc[-1])
        current_price = self._safe_float(close.iloc[-1])

        if current_price is None or current_price <= 0:
            raise ValueError("Invalid current price data")

        # ลดจำนวนแถวที่แสดงเพื่อประหยัด memory
        df_display = df.tail(60).copy()
        ema20_display = ema20.tail(60)
        ema50_display = ema50.tail(60)
        ema200_display = ema200.tail(60)
        df_plot = df_display

        if entry_price is not None and not use_smart_entry:
            entry_price = float(entry_price)
            if sl_price is None:
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)
            if tp1_price is None and tp1_pct is not None:
                tp1_price = entry_price * (1 + tp1_pct / 100)
            if tp2_price is None and tp2_pct is not None:
                tp2_price = entry_price * (1 + tp2_pct / 100)
            if tp1_price is None:
                tp1_price = entry_price * 1.05
            if tp2_price is None:
                tp2_price = entry_price * 1.10

            atr = (df["High"] - df["Low"]).mean()
            setup = TradeSetup(
                entry=entry_price, sl=sl_price, tp1=tp1_price, tp2=tp2_price,
                support_levels=[], resistance_levels=[],
                atr=atr
            )
        else:
            if setup is None and use_smart_entry:
                weekly_trend = detect_weekly_trend(symbol)
                setup = self.analyzer.find_trade_setup(df, current_price, weekly_trend)

            if setup is None:
                entry_price = current_price
                tp1_price = entry_price * 1.05
                tp2_price = entry_price * 1.10
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)
                setup = TradeSetup(
                    entry=entry_price, sl=sl_price, tp1=tp1_price, tp2=tp2_price,
                    support_levels=[], resistance_levels=[],
                    atr=(df["High"] - df["Low"]).mean()
                )
            else:
                entry_price = setup.entry
                sl_price = setup.sl
                tp1_price = setup.tp1
                tp2_price = setup.tp2
                atr = setup.atr

        sl_pct = setup.get_pct_from_entry(sl_price)
        tp1_pct_display = setup.get_pct_from_entry(tp1_price)
        tp2_pct_display = setup.get_pct_from_entry(tp2_price)

        entry_zone_bottom = setup.entry_zone_bottom
        entry_zone_top = setup.entry_zone_top

        trend_color = '#26A69A' if trend == "BULLISH" else '#EF5350' if trend == "BEARISH" else '#FF9800'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [4, 1]},
                                       sharex=True)

        for i, (idx, row) in enumerate(df_plot.iterrows()):
            x = i
            open_p = row['Open']
            high_p = row['High']
            low_p = row['Low']
            close_p = row['Close']
            color = COLORS['bullish'] if close_p >= open_p else COLORS['bearish']
            height = abs(close_p - open_p)
            bottom = min(open_p, close_p)
            rect = mpatches.FancyBboxPatch(
                (x - 0.4, bottom), 0.8, height,
                boxstyle="square,pad=0",
                facecolor=color, edgecolor=color, linewidth=1
            )
            ax1.add_patch(rect)
            ax1.plot([x, x], [low_p, high_p], color=color, linewidth=1)

        x_range = range(len(df_plot))
        ax1.plot(x_range, ema20_display.values, color=COLORS['ema20'], linewidth=2, label='EMA 20', alpha=0.8)
        ax1.plot(x_range, ema50_display.values, color=COLORS['ema50'], linewidth=2, label='EMA 50', alpha=0.8)
        ax1.plot(x_range, ema200_display.values, color=COLORS['ema200'], linewidth=2, label='EMA 200', alpha=0.8)

        if setup.poc:
            ax1.axhline(y=setup.poc, color=COLORS['poc'], linestyle='-.', linewidth=1.5, alpha=0.7)
        if setup.vah and setup.val:
            ax1.axhline(y=setup.vah, color=COLORS['vah'], linestyle=':', linewidth=1.2, alpha=0.6)
            ax1.axhline(y=setup.val, color=COLORS['val'], linestyle=':', linewidth=1.2, alpha=0.6)
            ax1.axhspan(setup.val, setup.vah, alpha=0.08, color=COLORS['vah'])

        ax1.axhline(y=tp2_price, color=COLORS['tp2'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhline(y=tp1_price, color=COLORS['tp1'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=entry_price, color=COLORS['entry'], linestyle='dotted', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=sl_price, color=COLORS['sl'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=COLORS['entry'])

        all_prices = [df_plot['Low'].min(), sl_price * 0.95]
        if setup.val:
            all_prices.append(setup.val * 0.98)
        price_min = min(all_prices)

        all_prices = [df_plot['High'].max(), tp2_price * 1.05]
        if setup.vah:
            all_prices.append(setup.vah * 1.02)
        price_max = max(all_prices)

        ax1.set_ylim(price_min, price_max)
        ax1.set_xlim(-1, len(df_plot))

        y_range = price_max - price_min
        y_shift = y_range * 0.008
        x_offset = len(df_plot) * 0.02

        ax1.text(x_offset, tp2_price,
                 f"◉ TP2 ${tp2_price:,.2f} ({tp2_pct_display:+.1f}%)",
                 fontsize=11, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp2'],
                          edgecolor='white', alpha=0.9), color='white')

        ax1.text(x_offset, tp1_price,
                 f"◉ TP1 ${tp1_price:,.2f} ({tp1_pct_display:+.1f}%)",
                 fontsize=11, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp1'],
                          edgecolor='white', alpha=0.9), color='white')

        ax1.text(x_offset, current_price,
                 f"▶ NOW ${current_price:,.2f}",
                 fontsize=11, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['now'],
                          edgecolor='white', alpha=0.9), color='white')

        if use_smart_entry and entry_price != current_price:
            entry_text = f"⧫ BUY ZONE ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f}"
        else:
            entry_text = f"⧫ BUY NOW ${entry_price:,.2f}"

        ax1.text(x_offset, entry_price,
                 entry_text,
                 fontsize=10, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['entry'],
                          edgecolor='white', alpha=0.8), color='white')

        ax1.text(x_offset, sl_price,
                 f"⨂ STOP ${sl_price:,.2f} ({sl_pct:+.1f}%)",
                 fontsize=11, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['sl'],
                          edgecolor='white', alpha=0.9), color='white')

        if setup.poc:
            ax1.text(len(df_plot) * 0.7, setup.poc + y_shift,
                     f"POC ${setup.poc:,.2f}",
                     fontsize=9, fontweight='bold', va='bottom',
                     color=COLORS['poc'],
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=COLORS['poc'], alpha=0.9))
        if setup.vah and setup.val:
            ax1.text(len(df_plot) * 0.7, setup.vah + y_shift,
                     f"VAH ${setup.vah:,.2f}",
                     fontsize=8, fontweight='bold', va='bottom',
                     color=COLORS['vah'],
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=COLORS['vah'], alpha=0.8))
            ax1.text(len(df_plot) * 0.7, setup.val - y_shift,
                     f"VAL ${setup.val:,.2f}",
                     fontsize=8, fontweight='bold', va='top',
                     color=COLORS['val'],
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=COLORS['val'], alpha=0.8))

        trend_text = f"{trend}"
        if setup.weekly_trend != "UNKNOWN":
            trend_text = f"{trend} (Week)"
        ax1.text(len(df_plot) * 0.5, price_max - y_shift * 3,
                trend_text,
                fontsize=13, fontweight='bold', va='top', ha='center',
                bbox=dict(boxstyle='round,pad=0.6', facecolor=trend_color,
                         edgecolor='white', alpha=0.95), color='white')

        x_right = len(df_plot) * 0.98
        if ema20_last:
            ax1.text(x_right, ema20_display.iloc[-1] + y_shift,
                    f"EMA20 {ema20_last:,.2f}",
                    fontsize=9, fontweight='bold', va='bottom', ha='right',
                    color=COLORS['ema20'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=COLORS['ema20'], alpha=0.9))
        if ema50_last:
            ax1.text(x_right, ema50_display.iloc[-1] + y_shift,
                    f"EMA50 {ema50_last:,.2f}",
                    fontsize=9, fontweight='bold', va='bottom', ha='right',
                    color=COLORS['ema50'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=COLORS['ema50'], alpha=0.9))
        if ema200_last:
            ax1.text(x_right, ema200_display.iloc[-1] + y_shift,
                    f"EMA200 {ema200_last:,.2f}",
                    fontsize=9, fontweight='bold', va='bottom', ha='right',
                    color=COLORS['ema200'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=COLORS['ema200'], alpha=0.9))

        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_title(f'Investockify Bot v{VERSION} — {symbol} | Pivot S/R + POC/VAH/VAL\n'
                     f'EMA: 20(Blue) 50(Orange) 200(Purple)',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='-')
        ax1.set_axisbelow(True)

        if 'Volume' in df_plot.columns:
            volumes = df_plot['Volume'].values
            max_vol = volumes.max() if len(volumes) > 0 else 1
            for i, (idx, row) in enumerate(df_plot.iterrows()):
                x = i
                color = COLORS['bullish'] if row['Close'] >= row['Open'] else COLORS['bearish']
                ax2.bar(x, row['Volume'], color=color, alpha=0.7, width=0.8)

            def volume_formatter(x, pos):
                if x >= 1e9:
                    return f'{x/1e9:.1f}B'
                elif x >= 1e6:
                    return f'{x/1e6:.0f}M'
                elif x >= 1e3:
                    return f'{x/1e3:.0f}K'
                else:
                    return f'{x:.0f}'

            ax2.yaxis.set_major_formatter(FuncFormatter(volume_formatter))
            ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.set_ylim(0, max_vol * 1.5)
            ax2.grid(True, alpha=0.3, linestyle='-')

        if isinstance(df_plot.index, pd.DatetimeIndex):
            date_labels = [d.strftime('%b %d') for d in df_plot.index[::5]]
        elif 'Timestamp' in df_plot.columns:
            timestamps = pd.to_datetime(df_plot['Timestamp']).iloc[::5]
            date_labels = [d.strftime('%b %d') for d in timestamps]
        else:
            date_labels = [str(i) for i in range(0, len(df_plot), 5)]

        x_ticks = range(0, len(df_plot), 5)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()
        # ล้าง memory หลังสร้างกราฟ
        gc.collect()
        return buf

    def generate_simple_chart(self, df: pd.DataFrame, symbol: str, trend: str = "NEUTRAL"):
        return self.generate_trading_chart(df, symbol, setup=None, use_smart_entry=True, trend=trend)

# =========================================================
# TREND DETECTION
# =========================================================
def detect_trend(df: pd.DataFrame):
    close = df['Close'] if 'Close' in df.columns else df['close']
    ema200 = close.ewm(span=200, adjust=False).mean()
    last_price = close.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    if last_price > last_ema200 * 1.02:
        return "BULLISH", last_ema200
    elif last_price < last_ema200 * 0.98:
        return "BEARISH", last_ema200
    else:
        return "SIDEWAYS", last_ema200


def detect_weekly_trend(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        df_weekly = ticker.history(period="2y", interval="1wk", auto_adjust=False)
        if df_weekly.empty or len(df_weekly) < 50:
            return "UNKNOWN"
        close = df_weekly['Close']
        ema50_weekly = close.ewm(span=50, adjust=False).mean()
        last_price = close.iloc[-1]
        last_ema50 = ema50_weekly.iloc[-1]
        if last_price > last_ema50 * 1.03:
            return "BULLISH"
        elif last_price < last_ema50 * 0.97:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    except Exception as e:
        logger.warning("Weekly trend detection failed for %s: %s", symbol, e)
        return "UNKNOWN"


# =========================================================
# POSITION SIZING
# =========================================================
def calculate_position_size(entry: float, sl: float,
                           portfolio: float = PORTFOLIO_VALUE,
                           risk_pct: float = RISK_PER_TRADE) -> dict:
    if entry <= 0 or sl <= 0:
        return {"shares": 0, "investment": 0, "risk_amount": 0,
                "risk_per_share": 0, "risk_pct": risk_pct, "valid": False, "reason": "Invalid price"}
    risk_amount = portfolio * (risk_pct / 100.0)
    risk_per_share = abs(entry - sl)
    if risk_per_share == 0:
        return {"shares": 0, "investment": 0, "risk_amount": risk_amount,
                "risk_per_share": 0, "risk_pct": risk_pct, "valid": False, "reason": "Entry = SL"}
    if risk_per_share < 0.01:
        return {"shares": 0, "investment": 0, "risk_amount": risk_amount,
                "risk_per_share": risk_per_share, "risk_pct": risk_pct, "valid": False, "reason": "Risk too small"}
    if risk_per_share / entry > 0.20:
        return {"shares": 0, "investment": 0, "risk_amount": risk_amount,
                "risk_per_share": risk_per_share, "risk_pct": risk_pct, "valid": False, "reason": "SL too wide (>20%)"}
    shares = int(risk_amount / risk_per_share)
    investment = shares * entry
    if investment > portfolio * 0.5:
        shares = int((portfolio * 0.5) / entry)
        investment = shares * entry
    return {"shares": shares, "investment": investment, "risk_amount": risk_amount,
            "risk_per_share": risk_per_share, "risk_pct": risk_pct, "valid": True, "reason": "OK"}


# =========================================================
# UX FORMATTERS
# =========================================================
def get_action_card(setup: TradeSetup, current_price: float, trend: str) -> str:
    if setup is None:
        return "⚠️ ไม่พบข้อมูลสำหรับการวิเคราะห์"
    entry = setup.entry
    sl = setup.sl
    tp1 = setup.tp1
    tp2 = setup.tp2
    in_entry_zone = setup.entry_zone_bottom <= current_price <= setup.entry_zone_top
    below_entry = current_price < setup.entry_zone_bottom
    above_entry = current_price > setup.entry_zone_top
    card = []
    card.append("━━━━━━━━━━━━━━━━━━━━\n")
    card.append("📋 คำแนะนำการลงทุน\n")
    card.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
    if in_entry_zone:
        card.append("🟢 ซื้อได้เลย! ราคาอยู่ในโซนซื้อ\n")
        card.append(f"└─ ตั้ง Limit Order ที่ ${entry:,.2f}\n")
    elif below_entry:
        card.append("🔴 ราคาต่ำกว่าโซนซื้อ\n")
        card.append("└─ รอดูว่าจะเด้งกลับหรือไม่\n")
    elif above_entry:
        card.append("🟡 ราคาสูงกว่าโซนซื้อ\n")
        card.append("└─ รอ Pullback เข้าโซนซื้อ\n")
    card.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
    card.append(f"🎯 เป้าหมายกำไร\n")
    card.append(f"├─ TP1 ${tp1:,.2f} (+{setup.get_pct_from_entry(tp1):.1f}%)\n")
    card.append(f"└─ TP2 ${tp2:,.2f} (+{setup.get_pct_from_entry(tp2):.1f}%)\n")
    card.append(f"🛡️ ขาดทุนสูงสุด: ${sl:,.2f} ({setup.get_pct_from_entry(sl):.1f}%)\n")
    card.append(f"└─ ${sl:,.2f} ({setup.get_pct_from_entry(sl):.1f}%)\n")
    return "".join(card)


def get_risk_thermometer(setup: TradeSetup) -> str:
    if setup is None or setup.risk == 0:
        return ""
    rr1 = setup.rr1
    rr2 = setup.rr2
    def make_bar(value, max_val=5, width=20):
        filled = int((value / max_val) * width)
        filled = min(filled, width)
        empty = width - filled
        return "█" * filled + "░" * empty
    lines = []
    lines.append("📊 RISK THERMOMETER\n")
    lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
    lines.append(f"TP1 R:R  1:{rr1:.1f}  {make_bar(rr1)}\n")
    lines.append(f"TP2 R:R  1:{rr2:.1f}  {make_bar(rr2)}\n")
    if rr1 >= 3.0:
        lines.append("🟢 EXCELLENT — ความเสี่ยงต่ำมาก\n")
    elif rr1 >= 2.0:
        lines.append("🟢 GOOD — ความเสี่ยงรับได้\n")
    elif rr1 >= 1.5:
        lines.append("🟡 FAIR — พอใช้ได้\n")
    else:
        lines.append("🔴 POOR — ความเสี่ยงสูง ควรข้าม\n")
    return "".join(lines)


def get_zone_map(setup: TradeSetup, current_price: float) -> str:
    if setup is None:
        return ""
    lines = []
    lines.append("━━━━━━━━━━━━━━━━━━━━\n")
    lines.append("🗺️ PRICE ZONE MAP\n")
    lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
    zones = []
    if setup.tp2:
        zones.append(("🎯 TP2", setup.tp2, "green"))
    if setup.tp1:
        zones.append(("🎯 TP1", setup.tp1, "green"))
    if setup.entry_zone_top:
        zones.append(("🟢 BUY ZONE TOP", setup.entry_zone_top, "light_green"))
    if setup.entry_zone_bottom:
        zones.append(("🟢 BUY ZONE BOTTOM", setup.entry_zone_bottom, "light_green"))
    if setup.poc:
        zones.append(("🟡 FAIR VALUE (POC)", setup.poc, "yellow"))
    if setup.val:
        zones.append(("🔵 VALUE AREA LOW", setup.val, "blue"))
    if setup.sl:
        zones.append(("🔴 STOP LOSS", setup.sl, "red"))
    lines.append(f"📍 YOU ARE HERE: ${current_price:,.2f}\n")
    lines.append("")
    for name, price, color in sorted(zones, key=lambda x: x[1], reverse=True):
        pct = ((price - current_price) / current_price) * 100
        arrow = "⬆️" if price > current_price else "⬇️" if price < current_price else "📍"
        lines.append(f"{arrow} {name}: ${price:,.2f} ({pct:+.1f}%)\n")
    return "".join(lines)


# =========================================================
# UX HELPER
# =========================================================
def _get_entry_context(distance: float, trend: str, weekly_trend: str) -> str:
    if distance > 5:
        if trend == "BULLISH" and weekly_trend == "BULLISH":
            return "ราคาห่างจากโซนซื้อมาก แม้แนวโน้มจะขึ้นแรง แต่ควรรอจังหวะ Pullback"
        elif trend == "BEARISH" or weekly_trend == "BEARISH":
            return "ราคาสูงเกินไปทั้งจุดเข้าและแนวโน้ม ไม่แนะนำให้ซื้อตอนนี้"
        else:
            return "ราคาห่างจากโซนซื้อ ควรรอจังหวะที่ดีกว่า"
    elif distance > -2:
        if trend == "BULLISH" and weekly_trend == "BULLISH":
            return "จังหวะดีมาก ราคาอยู่ในโซนซื้อและแนวโน้มขึ้นแกร่ง"
        elif trend == "BEARISH" and weekly_trend == "BEARISH":
            return "ราคาอยู่ในโซนซื้อ แต่ตลาดเป็นขาลง ควรระมัดระวัง"
        else:
            return "ราคาอยู่ในโซนซื้อ ให้พิจารณาตามแนวโน้มประกอบ"
    else:
        if trend == "BULLISH" and weekly_trend in ("BULLISH", "SIDEWAYS"):
            return "ราคาต่ำกว่าโซนซื้อ อาจเป็นโอกาสซื้อสะสม (Dip Buying)"
        elif trend == "BEARISH" or weekly_trend == "BEARISH":
            return "ราคาต่ำกว่าเป้าในตลาดขาลง อาจเป็นสัญญาณอันตราย"
        else:
            return "ราคาต่ำกว่าโซนซื้อ ให้ตั้งคำสั่งซื้อระวัง"


def get_trend_badge_v2(trend: str, weekly_trend: str) -> str:
    if trend == "BULLISH" and weekly_trend == "BULLISH":
        return "🟢🟢 ตลาดขาขึ้นแกร่ง (Strong Uptrend)"
    elif trend == "BULLISH" and weekly_trend == "BEARISH":
        return "🟢🔴 รายวันเป็นขาขึ้น แต่รายสัปดาห์ยังเป็นขาลง (Caution)"
    elif trend == "BEARISH" and weekly_trend == "BULLISH":
        return "🔴🟢 รายวันเป็นขาลง แต่รายสัปดาห์ยังเป็นขาขึ้น (Dip in Uptrend)"
    elif trend == "BEARISH" and weekly_trend == "BEARISH":
        return "🔴🔴 ตลาดขาลงแกร่ง (Strong Downtrend)"
    elif trend == "SIDEWAYS":
        return "⚪⚪ ตลาดไร้ทิศทางชัดเจน (Sideways / Range-bound)"
    else:
        return f"📊 {trend} (Daily) | {weekly_trend} (Weekly)"


def get_smart_summary_v2(setup, current_price: float, trend: str, weekly_trend: str) -> str:
    if setup is None:
        return "⚠️ ไม่สามารถวิเคราะห์ได้"

    distance = ((current_price - setup.entry) / setup.entry) * 100

    if distance > 5:
        timing_status = "รอซื้อ"
        timing_emoji = "⏳"
    elif distance > -2:
        timing_status = "พร้อมซื้อ"
        timing_emoji = "🟢"
    else:
        timing_status = "ต่ำกว่าเป้า"
        timing_emoji = "🔴"

    trend_icon = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "➡️"
    weekly_trend_icon = "📈" if weekly_trend == "BULLISH" else "📉" if weekly_trend == "BEARISH" else "➡️"
    context = _get_entry_context(distance, trend, weekly_trend)

    lines = [
        f"📍 จังหวะการเข้า: {timing_emoji} {timing_status}",
        f"🌎 แนวโน้มตลาด: {trend} {trend_icon}",
        f"📅 รายสัปดาห์: {weekly_trend} {weekly_trend_icon}",
        f"⚖️ R:R 1:{setup.rr1:.1f} | ห่างจากโซนซื้อ {distance:+.1f}%",
        f"💡 คำแนะนำ: {context}",
        "─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─",
    ]

    return "\n".join(lines)

# =========================================================
# STOCK HEALTH ANALYZER
# =========================================================
def analyze_stock_health(df: pd.DataFrame, setup: TradeSetup) -> str:
    if df is None or df.empty or 'Close' not in df.columns:
        return ""
    close = df['Close']
    current = float(close.iloc[-1])
    lines = []
    lines.append("━━━━━━━━━━━━━━━━━━━━\n")
    lines.append("🕵 วิเคราะห์สุขภาพหุ้น\n")
    lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")

    analyzer = TechnicalAnalyzer()
    ema20_series = analyzer.calculate_ema(close, 20)
    ema50_series = analyzer.calculate_ema(close, 50)
    ema20 = float(ema20_series.iloc[-1])
    ema50 = float(ema50_series.iloc[-1])
    dist_ema20 = ((current - ema20) / ema20) * 100 if ema20 > 0 else 0.0
    if current > ema20 > ema50:
        trend_icon, trend_text = "🟢", "ขาขึ้นแกร่ง"
    elif current > ema20:
        trend_icon, trend_text = "🟢", "ขาขึ้น"
    elif current < ema20 < ema50:
        trend_icon, trend_text = "🔴", "ขาลงแกร่ง"
    else:
        trend_icon, trend_text = "🔴", "ขาลง / ⚠️ ระวัง"
    lines.append(f"• 🌊 Trend: {trend_icon} {trend_text} {dist_ema20:+.2f}% vs EMA20\n")

    rsi_series = analyzer.calculate_rsi(df)
    rsi = float(rsi_series.iloc[-1])
    if pd.isna(rsi):
        rsi_icon, rsi_text = "⚪️", "N/A"
    elif rsi > 70:
        rsi_icon, rsi_text = "🔴", "Overbought"
    elif rsi > 60:
        rsi_icon, rsi_text = "🟡", "Caution (เริ่มร้อน)"
    elif rsi < 30:
        rsi_icon, rsi_text = "🟢", "Oversold"
    elif rsi < 40:
        rsi_icon, rsi_text = "🟡", "สะสม (ใกล้ซื้อ)"
    else:
        rsi_icon, rsi_text = "⚪️", "Neutral"
    lines.append(f"• 🌡 RSI {rsi:.2f}: {rsi_icon} {rsi_text}\n")

    macd_line, signal_line = analyzer.calculate_macd(df)
    macd_val = float(macd_line.iloc[-1])
    signal_val = float(signal_line.iloc[-1])
    prev_macd = float(macd_line.iloc[-2])
    prev_signal = float(signal_line.iloc[-2])
    if macd_val > signal_val and prev_macd <= prev_signal:
        macd_status = "🟢 ทำ Golden Cross"
    elif macd_val < signal_val and prev_macd >= prev_signal:
        macd_status = "🔴 ทำ Dead Cross"
    elif macd_val > signal_val:
        macd_status = "🟢 แรงส่งเพิ่มขึ้น" if macd_val > prev_macd else "🟡 แรงส่งอ่อนลง"
    else:
        macd_status = "🔴 แรงส่งลบแกร่ง" if macd_val < prev_macd else "🟡 แรงส่งลบอ่อนลง"
    lines.append(f"• ⚡️ MACD: {macd_status}  ({macd_val:.2f} / {signal_val:.2f})\n")

    if 'Volume' in df.columns and len(df) >= 2:
        current_vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].tail(20).mean())
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
        prev_close = float(close.iloc[-2])
        price_change = ((current - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
        if vol_ratio > 1.5 and price_change > 1:
            vol_status = "📈 เงินไหลเข้าแรง (Breakout)"
        elif vol_ratio > 1.2 and price_change > 0:
            vol_status = "📈 เงินไหลเข้า (Accumulation)"
        elif vol_ratio > 1.2 and price_change < 0:
            vol_status = "📉 เงินไหลออก (Distribution)"
        elif vol_ratio < 0.7:
            vol_status = "⚪️ สภาพคล่องต่ำ (รอ)"
        else:
            vol_status = "⚪️ ปกติ"
        lines.append(f"• 💧 Volume: {vol_status}\n")
        if current_vol >= 1_000_000:
            vol_str = f"{current_vol/1e6:.2f}M"
            avg_str = f"{avg_vol/1e6:.2f}M"
        elif current_vol >= 1_000:
            vol_str = f"{current_vol/1e3:.2f}K"
            avg_str = f"{avg_vol/1e3:.2f}K"
        else:
            vol_str = f"{current_vol:.2f}"
            avg_str = f"{avg_vol:.2f}"
        lines.append(f"　　 Vol {vol_str} | Avg {avg_str}\n")
    return "".join(lines)

# =========================================================
# NEWS & SENTIMENT ANALYZER (HYBRID: Regex + AI Fallback)
# =========================================================
BULLISH_PATTERNS = [
    r'\b(surge|surges|surged|rally|rallies|rallied|soar|soars|soared|jump|jumps|jumped|climb|climbs|climbed|skyrocket|skyrockets|skyrocketed)\b',
    r'\b(beat|beats|beating)\b.*\b(earnings|revenue|expectation|forecast|consensus)\b',
    r'\b(upgrade|upgrades|upgraded)\b',
    r'\b(strong|stronger|strongest)\b.*\b(growth|demand|sales|performance|quarter|guidance)\b',
    r'\b(buy|overweight|outperform)\b.*\b(rating|target|price target|recommendation)\b',
    r'\b(exceed|exceeds|exceeded|outperform|outperforms|outperformed)\b',
    r'\b(positive|optimistic|promising|recovery|expansion|breakthrough|record high)\b',
    r'\b(raise|raises|raised|lifting|lifts)\b.*\b(guidance|outlook|target|forecast)\b',
    r'\b(approval|approved|clearance|cleared|launch|launches|launched)\b',
]

BEARISH_PATTERNS = [
    r'\b(plunge|plunges|plunged|crash|crashes|crashed|tank|tanks|tanked|dive|dives|dived|slump|slumps|slumped|collapse|collapses|collapsed)\b',
    r'\b(miss|misses|missed)\b.*\b(earnings|revenue|target|forecast|consensus|expectation)\b',
    r'\b(downgrade|downgrades|downgraded)\b',
    r'\b(weak|weaker|weakest)\b.*\b(demand|sales|performance|quarter|guidance|outlook)\b',
    r'\b(sell|underweight|underperform)\b.*\b(rating|target|price target|recommendation)\b',
    r'\b(layoff|layoffs|laying off|lay off|job cuts|workforce reduction)\b',
    r'\b(debt|debts|loss|losses|deficit|bankruptcy|bankrupt|default|defaults)\b',
    r'\b(cut|cuts|cutting|slash|slashes|slashed|lower|lowers|lowered)\b.*\b(guidance|outlook|target|forecast|dividend)\b',
    r'\b(negative|pessimistic|disappointing|warning|downturn|correction|recession|bearish)\b',
    r'\b(investigation|investigating|sec probe|lawsuit|litigation|fine|penalty|scandal)\b',
]


def _parse_timestamp(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = int(raw)
        if val > 1_000_000_000_000:
            return val
        elif val > 1_000_000_000:
            return val * 1000
        else:
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.isdigit():
            val = int(raw)
            return val * 1000 if val < 1_000_000_000_000 else val
        for fmt in ('%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S'):
            try:
                dt = datetime.strptime(raw.replace('Z', '+00:00'), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    return None


def _relative_time(published_ms: Optional[int]) -> str:
    if not published_ms:
        return ""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    diff_m = (now_ms - published_ms) / 60000.0
    if diff_m < 1:
        return "[Now]"
    elif diff_m < 60:
        return f"[{int(diff_m)}m]"
    elif diff_m < 1440:
        return f"[{int(diff_m/60)}h]"
    else:
        return f"[{int(diff_m/1440)}d]"


def _score_news_item(title: str, use_ai_fallback: bool = True) -> tuple:
    """
    Hybrid Scoring:
      Layer 1: Regex Pattern Matching (fast, free, deterministic)
      Layer 2: AI Fallback (expensive, slow) — เฉพาะ ambiguous cases
    """
    t = title.lower()

    # Layer 1: Pattern matching
    bull_count = sum(1 for p in BULLISH_PATTERNS if re.search(p, t))
    bear_count = sum(1 for p in BEARISH_PATTERNS if re.search(p, t))

    # กรณีชัดเจน → ใช้ keyword ได้เลย
    if bull_count > bear_count + 1:
        return min(bull_count * 0.5, 1.0), "🟢"
    if bear_count > bull_count + 1:
        return max(bear_count * -0.5, -1.0), "🔴"

    # กรณี ambiguous → AI Fallback (ถ้าเปิดใช้งานและมี API Key)
    if use_ai_fallback and OPENAI_API_KEY and abs(bull_count - bear_count) <= 1:
        cache_key = f"sent_{hash(title) % 100000}"
        cached = ai_summary_mgr.get(cache_key)
        if cached is not None:
            pass

        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f'Analyze sentiment of this financial news headline. Reply ONLY with: BULLISH, BEARISH, or NEUTRAL.\n\n"{title}"'
                }],
                max_tokens=10,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().upper()
            score_map = {
                "BULLISH": (0.8, "🟢"),
                "BEARISH": (-0.8, "🔴"),
                "NEUTRAL": (0.0, "⚪️")
            }
            sentiment = score_map.get(result, (0.0, "⚪️"))
            ai_summary_mgr.set(cache_key, result)
            return sentiment
        except Exception:
            pass

    return 0.0, "⚪️"


def fetch_stock_news(symbol: str, max_items: int = 5) -> list:
    try:
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news
        if not raw_news:
            return []
        parsed_items = []
        for item in raw_news:
            title = item.get("title", "")
            if not title and "content" in item:
                title = item["content"].get("title", "")
            pub_time = None
            for key in ("published", "providerPublishTime", "pubDate", "publish_time", "time"):
                if key in item and item[key] is not None:
                    pub_time = _parse_timestamp(item[key])
                    if pub_time:
                        break
            if not pub_time and isinstance(item.get("content"), dict):
                for key in ("published", "providerPublishTime", "pubDate", "publish_time", "time"):
                    if key in item["content"] and item["content"][key] is not None:
                        pub_time = _parse_timestamp(item["content"][key])
                        if pub_time:
                            break
            link = item.get("link", "")
            if not link and "content" in item:
                link = item["content"].get("canonicalUrl", {}).get("url", "")
            publisher = item.get("publisher", "")
            if not publisher and "content" in item:
                publisher = item["content"].get("provider", {}).get("displayName", "Yahoo Finance")
            if title:
                parsed_items.append({
                    "title": title,
                    "publisher": publisher or "Yahoo Finance",
                    "link": link,
                    "published_ms": pub_time or 0
                })
        parsed_items.sort(key=lambda x: x["published_ms"], reverse=True)
        results = parsed_items[:max_items]
        results = [r for r in results if r["published_ms"] > 0]
        return results
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", symbol, e)
        return []


def analyze_news_sentiment(news_items: list) -> tuple:
    if not news_items:
        return "⚪️", "ไม่มีข่าวล่าสุด", []
    scored = []
    total_score = 0.0
    valid_count = 0
    for item in news_items:
        score, emoji = _score_news_item(item["title"])
        scored.append({
            **item,
            "score": score,
            "emoji": emoji,
            "rel_time": _relative_time(item["published_ms"])
        })
        if score != 0:
            total_score += score
            valid_count += 1
    if valid_count == 0:
        return "⚪️", "เป็นกลาง", scored
    avg = total_score / valid_count
    if avg >= 0.3:
        return "🟢", "หนุนราคา", scored
    elif avg <= -0.3:
        return "🔴", "กดดันราคา", scored
    else:
        return "⚪️", "เป็นกลาง", scored


def _translate_news_titles(news_items: list) -> list:
    if not news_items:
        return news_items
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='th')
        for item in news_items:
            title = item["title"]
            if len(title) > 120:
                title = title[:117] + "..."
            try:
                item["thai_title"] = translator.translate(title)
            except Exception as e:
                logger.debug("Translation failed for title: %s", e)
                item["thai_title"] = item["title"]
    except ImportError:
        logger.info("deep-translator not installed, using original English titles")
        for item in news_items:
            item["thai_title"] = item["title"]
    except Exception as e:
        logger.warning("News translation setup failed: %s", e)
        for item in news_items:
            item["thai_title"] = item["title"]
    return news_items


def _generate_ai_thai_summary(news_items: list, symbol: str) -> str:
    """
    สร้าง AI Summary ภาษาไทย พร้อม Lock + Cache ต่อ Symbol
    ป้องกัน Race Condition เมื่อหลาย Thread/User ขอหุ้นตัวเดียวกันพร้อมกัน
    """
    if not OPENAI_API_KEY or not news_items:
        return ""

    # 1) Check cache first (fast path)
    cached = ai_summary_mgr.get(symbol)
    if cached is not None:
        return cached

    # 2) Get per-symbol lock (prevent race)
    lock = ai_summary_mgr.get_lock(symbol)

    with lock:
        # 3) Double-check cache after acquiring lock
        cached = ai_summary_mgr.get(symbol)
        if cached is not None:
            return cached

        # 4) Call OpenAI API
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            headlines = "\n".join([f"- {item['title']}" for item in news_items[:5]])

            prompt = f"""คุณเป็นนักวิเคราะห์หุ้นมืออาชีพ จงวิเคราะห์ภาพรวมของข่าวหุ้น {symbol} จากหัวข้อข่าวต่อไปนี้
แล้วสรุปเป็น 1 ย่อหน้าสั้นๆ ไม่เกิน 3 ประโยค ว่าข่าวเหล่านี้มีแนวโน้มส่งผลกระทบต่อราคาหุ้น {symbol} อย่างไรโดยรวม

ห้ามสรุปทีละข่าว ให้สรุปภาพรวมเท่านั้น:

{headlines}

รูปแบบตอบกลับ (ห้ามตอบเกินนี้):
📰 สรุป: [สรุปภาพรวม 1 ย่อหน้า]
🎯 ผลกระทบต่อราคา: [บวก 🟢 / ลบ 🔴 / กลาง ⚪]
💡 มุมมองนักลงทุน: [คำแนะนำสั้นๆ 1 ประโยค]"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            result = response.choices[0].message.content

            # 5) Save to cache
            ai_summary_mgr.set(symbol, result)
            return result

        except Exception as e:
            logger.debug("AI summary generation failed: %s", e)
            return ""


def format_news_section(symbol: str, max_items: int = 5) -> str:
    # ตรวจสอบ cache ก่อน เพื่อป้องกัน AI สรุปซ้ำและลด OpenAI API cost
    cached = news_cache.get(symbol)
    if cached is not None:
        return cached

    news_items = fetch_stock_news(symbol, max_items)
    if not news_items:
        return ""
    news_items = _translate_news_titles(news_items)
    ai_summary = ""
    if OPENAI_API_KEY:
        ai_summary = _generate_ai_thai_summary(news_items, symbol)
        if ai_summary:
            ai_summary = f"🤖 AI สรุปภาพรวมข่าว:\n{ai_summary}\n"
    overall_emoji, overall_text, scored = analyze_news_sentiment(news_items)
    scored.sort(key=lambda x: x.get("published_ms", 0), reverse=True)
    lines = []
    lines.append("━━━━━━━━━━━━━━━━━━━━\n")
    lines.append(f"📰 ข่าวล่าสุด\n")
    lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
    if ai_summary:
        lines.append(ai_summary)
        lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
    bull_count = sum(1 for s in scored if s["emoji"] == "🟢")
    bear_count = sum(1 for s in scored if s["emoji"] == "🔴")
    if bull_count > bear_count:
        lines.append(f"💬 หุ้นได้รับแรงหนุนจากข่าวด้านบวก 🟢\n")
    elif bear_count > bull_count:
        lines.append(f"💬 หุ้นได้รับแรงกดดันจากข่าวด้านลบ 🔴\n")
    else:
        lines.append(f"💬 ข่าวไม่มีนัยสำคัญต่อทิศทางราคา ⚪\n")
    for item in scored:
        thai_title = item.get("thai_title", item["title"])
        display_title = thai_title if len(thai_title) <= 70 else thai_title[:67] + "..."
        safe_title = html.escape(display_title)
        link = item.get("link", "")
        rel = item["rel_time"]
        emo = item["emoji"]
        if link and link.startswith(("http://", "https://")):
            lines.append(f'  {emo} <a href="{link}">{safe_title}</a> {rel}\n')
        else:
            lines.append(f"  {emo} {safe_title} {rel}\n")
        if item.get("thai_title") != item["title"]:
            orig = html.escape(item["title"][:60])
            lines.append(f'     <i>({orig}...)</i>\n')
    lines.append("━━━━━━━━━━━━━━━━━━━━\n")
    result = "".join(lines)

    # บันทึก cache
    news_cache.set(symbol, result)
    return result

# =========================================================
# STATE MANAGEMENT
# =========================================================
class StateManager:
    def __init__(self):
        self._active_levels: Dict[str, List[Dict]] = {}
        self._entry_data: Dict[str, TradeSetup] = {}

    def get_levels(self, symbol: str) -> List[Dict]:
        return self._active_levels.get(symbol, [])

    def set_levels(self, symbol: str, levels: List[Dict]):
        self._active_levels[symbol] = levels

    def get_setup(self, symbol: str) -> Optional[TradeSetup]:
        return self._entry_data.get(symbol)

    def set_setup(self, symbol: str, setup: TradeSetup):
        self._entry_data[symbol] = setup

    def clear_symbol(self, symbol: str):
        self._active_levels.pop(symbol, None)
        self._entry_data.pop(symbol, None)


state_mgr = StateManager()
data_fetcher = DataFetcher()
analyzer = TechnicalAnalyzer()
chart_gen = ChartGenerator(analyzer)


def _df_to_cg(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                  'close': 'Close', 'volume': 'Volume', 'timestamp': 'Timestamp'}
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _df_from_cg(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low',
                  'Close': 'close', 'Volume': 'volume', 'Timestamp': 'timestamp'}
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# =========================================================
# UPDATE LEVELS & BREAKOUTS
# =========================================================
def update_levels(symbol: str, df: pd.DataFrame):
    current_price = float(df["close"].iloc[-1])
    df_cg = _df_to_cg(df)
    weekly_trend = detect_weekly_trend(symbol)
    setup = analyzer.find_trade_setup(df_cg, current_price, weekly_trend)
    atr = setup.atr
    state_mgr.set_setup(symbol, setup)
    levels = []
    for h in setup.resistance_levels:
        levels.append({
            "type": "resistance", "price": round(h.price, 2), "strength": h.strength,
            "zone_top": h.price + atr * analyzer.ZONE_WIDTH,
            "zone_bottom": h.price - atr * analyzer.ZONE_WIDTH
        })
    for l in setup.support_levels:
        levels.append({
            "type": "support", "price": round(l.price, 2), "strength": l.strength,
            "zone_top": l.price + atr * analyzer.ZONE_WIDTH,
            "zone_bottom": l.price - atr * analyzer.ZONE_WIDTH
        })
    levels = sorted(levels, key=lambda x: x["price"], reverse=True)
    state_mgr.set_levels(symbol, levels)
    return levels


def detect_breakout(symbol: str, df: pd.DataFrame):
    levels = state_mgr.get_levels(symbol)
    if not levels:
        return []
    df_atr = analyzer.calculate_atr(_df_to_cg(df))
    close_price = float(df["close"].iloc[-1])
    atr = df_atr["atr"].iloc[-1]
    current_volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0
    volume_avg = df["volume"].tail(20).mean() if "volume" in df.columns else 0
    volume_confirmed = current_volume > volume_avg * analyzer.VOLUME_CONFIRMATION_RATIO
    if pd.isna(atr) or atr == 0:
        return []
    buffer = atr * analyzer.BREAK_SENS
    signals = []
    remaining = []
    breakout_count = 0
    for lvl in levels:
        breakout_move = abs(close_price - lvl["price"]) / atr
        if (lvl["type"] == "resistance" and
            close_price > lvl["zone_top"] + buffer and
            breakout_move >= analyzer.MIN_BREAKOUT_MOVE_ATR):
            if breakout_count < analyzer.MAX_BREAKOUT_SIGNALS:
                vol_status = "✅ Vol Confirmed" if volume_confirmed else "⚠️ Low Volume"
                signals.append(
                    f"🚀 {symbol}\nBreak Resistance\n"
                    f"Level: ${lvl['price']:.2f}\n"
                    f"Current: ${close_price:.2f}\n"
                    f"Strength: {lvl['strength']}\n"
                    f"{vol_status}"
                )
                breakout_count += 1
        elif (lvl["type"] == "support" and
              close_price < lvl["zone_bottom"] - buffer and
              breakout_move >= analyzer.MIN_BREAKOUT_MOVE_ATR):
            if breakout_count < analyzer.MAX_BREAKOUT_SIGNALS:
                vol_status = "✅ Vol Confirmed" if volume_confirmed else "⚠️ Low Volume"
                signals.append(
                    f"🔻 {symbol}\nBreak Support\n"
                    f"Level: ${lvl['price']:.2f}\n"
                    f"Current: ${close_price:.2f}\n"
                    f"Strength: {lvl['strength']}\n"
                    f"{vol_status}"
                )
                breakout_count += 1
        else:
            remaining.append(lvl)
    state_mgr.set_levels(symbol, remaining)
    return signals

# =========================================================
# TELEGRAM HANDLERS
# =========================================================
WELCOME_MESSAGE = """
⚡ ยินดีต้อนรับสู่ Investockify Bot

🤖 ผู้ช่วยลงทุนที่บอกว่า "ควรทำอะไร" ไม่ใช่แค่ตัวเลข

✨ สิ่งที่คุณจะได้รับ:
• 🎯 Action Card — ซื้อ/รอ/ขาย ชัดเจน
• 📊 Risk Thermometer — ความเสี่ยงมองเห็นได้
• 🗺️ Zone Map — แผนที่ราคาแบบ Traffic Light
• 📈 Smart Summary — สรุปทั้งหมดใน 1 บรรทัด
• 📰 ข่าวภาษาไทย — แปลอัตโนมัติ + AI สรุป (ถ้าตั้งค่า)
• 📐 Manual/Quick Chart — กราฟแบบตั้งค่าเองหรือ Smart Entry

👇 พิมพ์ชื่อหุ้น เช่น AAPL หรือกดปุ่มด้านล่าง
"""

MAIN_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🕵 วิเคราะห์หุ้น"), KeyboardButton("📱 เปิดเมนูหลัก")],
        [KeyboardButton("💡 วิธีอ่านผล"), KeyboardButton("📖 คู่มือ")]
    ],
    resize_keyboard=True
)


def get_main_inline_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 AAPL", callback_data="stock_AAPL"),
         InlineKeyboardButton("⚡ NVDA", callback_data="stock_NVDA")],
        [InlineKeyboardButton("🇹🇭 PTT.BK", callback_data="stock_PTT")]
    ])


# =========================================================
# CONVERSATION STATES
# =========================================================
SYMBOL, ENTRY, TP_SL = range(3)
user_data = {}


# =========================================================
# COMMAND HANDLERS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE, reply_markup=MAIN_MENU)
    await update.message.reply_text("📱 เมนูหลัก", reply_markup=get_main_inline_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 *คู่มือการใช้งาน Investockify Bot*\n\n"
        "*คำสั่งพื้นฐาน:*\n"
        "• `/start` - เริ่มต้นใช้งาน\n"
        "• `/analyze <symbol>` - วิเคราะห์หุ้นแบบละเอียด\n"
        "• `/chart <symbol>` - สร้างกราฟพร้อมตั้งค่า TP/SL\n"
        "• `/quick <symbol>` - ดูกราฟเร็วๆ (Smart Entry)\n"
        "• `/help` - ดูคำแนะนำ\n\n"
        "*โหมด Smart Entry:*\n"
        "• ระบบจะหา Swing Low ใกล้ EMA200 อัตโนมัติ\n"
        "• SL คำนวณจาก ATR (Dynamic)\n"
        "• TP คำนวณจาก Pivot Resistance\n\n"
        "*โหมด Manual:*\n"
        "• ใช้ `/chart` เพื่อระบุ Entry และ TP/SL เอง"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """รับคำสั่ง /analyze <ชื่อหุ้น> — วิเคราะห์แบบละเอียด"""
    user_id = update.effective_user.id
    if update.update_id and update_dedup.is_duplicate(update.update_id):
        return
    if user_rate_limit.is_limited(user_id):
        remaining = int(user_rate_limit.remaining_cooldown(user_id))
        await update.message.reply_text(f"⏳ กรุณารออีก {remaining} วินาที ก่อนขอวิเคราะห์ใหม่")
        return

    if not context.args:
        await update.message.reply_text(
            "❌ กรุณาระบุชื่อหุ้น\n\nตัวอย่าง:\n/analyze AAPL\n/analyze NVDA\n/analyze PTT.BK"
        )
        return
    symbol = " ".join(context.args).upper().strip()
    if not is_valid_stock_input(symbol):
        await update.message.reply_text("⚠️ ชื่อหุ้นไม่ถูกต้อง กรุณาตรวจสอบอีกครั้ง")
        return
    await send_real_stock_analysis(update.message, symbol)


async def quick_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """คำสั่ง /quick — ใช้ Smart Entry"""
    user_id = update.effective_user.id
    if update.update_id and update_dedup.is_duplicate(update.update_id):
        return
    if user_rate_limit.is_limited(user_id):
        remaining = int(user_rate_limit.remaining_cooldown(user_id))
        await update.message.reply_text(f"⏳ กรุณารออีก {remaining} วินาที")
        return

    if not context.args:
        await update.message.reply_text(
            "❌ กรุณาระบุสัญลักษณ์\nตัวอย่าง: `/quick GC=F`",
            parse_mode='Markdown'
        )
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"⏳ กำลังวิเคราะห์ {symbol} ด้วย Smart Entry...")

    df, error = data_fetcher.get_stock_data(symbol)
    if error:
        await update.message.reply_text(f"❌ {error}")
        return

    try:
        trend, _ = detect_trend(df)
        chart_buf = chart_gen.generate_simple_chart(df, symbol, trend)
        current_price = df['Close'].iloc[-1]
        await update.message.reply_photo(
            photo=chart_buf,
            caption=(
                f"📊 *{symbol}* | Investockify Smart Chart\n"
                f"ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
                f"💡 *Smart Entry* ระบบหาจุดเข้าอัตโนมัติ\n"
                f"ใช้ `/chart {symbol}` เพื่อตั้งค่าเอง"
            ),
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error("Quick chart error: %s", e)
        await update.message.reply_text(f"❌ เกิดข้อผิดพลาดในการสร้างกราฟ: {str(e)}")


async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """คำสั่ง /chart — ตั้งค่าเอง (ConversationHandler)"""
    user_id = update.effective_user.id
    if update.update_id and update_dedup.is_duplicate(update.update_id):
        return
    if user_rate_limit.is_limited(user_id):
        remaining = int(user_rate_limit.remaining_cooldown(user_id))
        await update.message.reply_text(f"⏳ กรุณารออีก {remaining} วินาที")
        return

    if not context.args:
        await update.message.reply_text(
            "❌ กรุณาระบุสัญลักษณ์\nตัวอย่าง: `/chart GC=F`",
            parse_mode='Markdown'
        )
        return
    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    user_data[user_id] = {'symbol': symbol}

    df, error = data_fetcher.get_stock_data(symbol)
    if error:
        await update.message.reply_text(f"❌ {error}")
        return

    current_price = df['Close'].iloc[-1]
    user_data[user_id]['current_price'] = current_price
    user_data[user_id]['df'] = df

    keyboard = [
        [InlineKeyboardButton(
            f"🤖 Smart Entry (แนะนำ)",
            callback_data="mode_smart"
        )],
        [InlineKeyboardButton(
            f"ใช้ราคาปัจจุบัน (${current_price:,.2f})",
            callback_data="mode_current"
        )],
        [InlineKeyboardButton("ระบุราคาเอง", callback_data="mode_manual")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"📊 *{symbol}*\n"
        f"ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
        f"เลือกโหมด Entry:",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )


# =========================================================
# CALLBACK HANDLERS
# =========================================================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการปุ่ม Inline Keyboard"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    if update.update_id and update_dedup.is_duplicate(update.update_id):
        return
    if user_rate_limit.is_limited(user_id):
        await query.message.reply_text("⏳ กรุณารอสักครู่ ก่อนกดอีกครั้ง")
        return

    data = query.data
    if data == "stock_AAPL":
        await send_real_stock_analysis(query.message, "AAPL")
    elif data == "stock_NVDA":
        await send_real_stock_analysis(query.message, "NVDA")
    elif data == "stock_PTT":
        await send_real_stock_analysis(query.message, "PTT.BK")


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการ callback จากปุ่มเลือกโหมด"""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data

    if data == "mode_smart":
        await query.edit_message_text("⏳ กำลังวิเคราะห์ด้วย Smart Entry...")
        await generate_smart_chart(update, context, user_id)
    elif data == "mode_current":
        user_data[user_id]['entry_price'] = user_data[user_id]['current_price']
        user_data[user_id]['use_smart'] = False
        await ask_tp_sl(update, context)
    elif data == "mode_manual":
        user_data[user_id]['use_smart'] = False
        await query.edit_message_text(
            "พิมพ์ราคา Entry ที่ต้องการ (เช่น 4555.80):"
        )
        return ENTRY


async def generate_smart_chart(update, context, user_id):
    """สร้างกราฟด้วย Smart Entry"""
    data = user_data[user_id]
    symbol = data['symbol']
    df = data['df']

    try:
        trend, _ = detect_trend(df)
        chart_buf = chart_gen.generate_trading_chart(
            df, symbol,
            use_smart_entry=True,
            trend=trend
        )
        current_price = df['Close'].iloc[-1]

        caption = (
            f"📊 *{symbol}* | Investockify Smart Trading Plan\n\n"
            f"🤖 ใช้ Smart Entry (Swing Low + ATR)\n"
            f"📈 ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
            f"ดูกราฟสำหรับระดับ TP/SL ที่คำนวณอัตโนมัติ"
        )

        if hasattr(update, 'callback_query') and update.callback_query:
            await update.callback_query.message.reply_photo(
                photo=chart_buf, caption=caption, parse_mode='Markdown'
            )
        else:
            await update.message.reply_photo(
                photo=chart_buf, caption=caption, parse_mode='Markdown'
            )
    except Exception as e:
        logger.error("Smart chart error: %s", e)
        if hasattr(update, 'callback_query') and update.callback_query:
            await update.callback_query.message.reply_text(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        else:
            await update.message.reply_text(f"❌ เกิดข้อผิดพลาด: {str(e)}")


async def ask_tp_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ถาม TP/SL สำหรับ Manual Entry"""
    user_id = update.effective_user.id

    keyboard = [
        [InlineKeyboardButton(
            (f"✅ ใช้ค่าเริ่มต้น "
             f"(TP1 +{DEFAULT_TP1_PCT}%, TP2 +{DEFAULT_TP2_PCT}%, SL {DEFAULT_SL_PCT}%)"),
            callback_data="tp_default"
        )],
        [InlineKeyboardButton("🔧 ตั้งค่าเอง", callback_data="tp_custom")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_text(
        f"ราคา Entry: `${user_data[user_id]['entry_price']:,.2f}`\n\n"
        f"เลือกการตั้งค่า TP/SL:",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )


async def entry_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการ callback จากปุ่ม Entry (เก่า - compatibility)"""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data

    if data == "entry_current":
        user_data[user_id]['entry_price'] = user_data[user_id]['current_price']
        user_data[user_id]['use_smart'] = False
        await ask_tp_sl(update, context)
    elif data == "entry_custom":
        user_data[user_id]['use_smart'] = False
        await query.edit_message_text(
            "พิมพ์ราคา Entry ที่ต้องการ (เช่น 4555.80):"
        )
        return ENTRY


async def tp_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการ callback TP/SL"""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    if query.data == "tp_default":
        await generate_manual_chart(update, context, user_id)
    elif query.data == "tp_custom":
        await query.edit_message_text(
            "พิมพ์ค่า TP1 TP2 SL คั่นด้วยช่องว่าง\n"
            "ตัวอย่าง: `5.6 22.6 -3.5`"
        )
        return TP_SL


async def custom_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """รับราคา Entry ที่ผู้ใช้พิมพ์เอง"""
    user_id = update.effective_user.id

    try:
        entry_price = float(update.message.text.replace(',', ''))
        user_data[user_id]['entry_price'] = entry_price
        user_data[user_id]['use_smart'] = False

        keyboard = [
            [InlineKeyboardButton("ใช้ค่าเริ่มต้น", callback_data="tp_default")],
            [InlineKeyboardButton("ตั้งค่าเอง", callback_data="tp_custom")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"ราคา Entry: `${entry_price:,.2f}`\n\nเลือก TP/SL:",
            reply_markup=reply_markup
        )
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("❌ ราคาไม่ถูกต้อง กรุณาพิมพ์ตัวเลข")
        return ENTRY


async def custom_tp_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """รับค่า TP/SL ที่ผู้ใช้พิมพ์เอง"""
    user_id = update.effective_user.id

    try:
        values = update.message.text.split()
        if len(values) != 3:
            raise ValueError("ต้องมี 3 ค่า")

        tp1 = float(values[0])
        tp2 = float(values[1])
        sl = float(values[2])

        user_data[user_id]['tp1'] = tp1
        user_data[user_id]['tp2'] = tp2
        user_data[user_id]['sl'] = sl

        await generate_manual_chart(update, context, user_id)
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text(
            "❌ รูปแบบไม่ถูกต้อง\nตัวอย่าง: `5.6 22.6 -3.5`"
        )
        return TP_SL


async def generate_manual_chart(update, context, user_id):
    """สร้างกราฟแบบ Manual Entry"""
    data = user_data[user_id]
    symbol = data['symbol']
    df = data['df']
    entry = data['entry_price']

    tp1 = data.get('tp1', DEFAULT_TP1_PCT)
    tp2 = data.get('tp2', DEFAULT_TP2_PCT)
    sl_pct = data.get('sl', DEFAULT_SL_PCT)

    sl_price = entry * (1 + sl_pct / 100)

    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.edit_message_text("⏳ กำลังสร้างกราฟ...")
    else:
        await update.message.reply_text("⏳ กำลังสร้างกราฟ...")

    try:
        trend, _ = detect_trend(df)
        chart_buf = chart_gen.generate_trading_chart(
            df, symbol,
            use_smart_entry=False,
            entry_price=entry,
            tp1_pct=tp1,
            tp2_pct=tp2,
            sl_price=sl_price,
            trend=trend
        )

        current = df['Close'].iloc[-1]
        change = ((current - entry) / entry) * 100

        caption = (
            f"📊 *{symbol}* | Manual Trading Plan\n\n"
            f"💰 Entry: `${entry:,.2f}`\n"
            f"🎯 TP1: `${entry * (1 + tp1/100):,.2f}` (+{tp1}%)\n"
            f"🎯 TP2: `${entry * (1 + tp2/100):,.2f}` (+{tp2}%)\n"
            f"🛑 SL: `${entry * (1 + sl_pct/100):,.2f}` ({sl_pct}%)\n\n"
            f"📈 ราคาปัจจุบัน: `${current:,.2f}` ({change:+.2f}%)"
        )

        if hasattr(update, 'callback_query') and update.callback_query:
            await update.callback_query.message.reply_photo(
                photo=chart_buf, caption=caption, parse_mode='Markdown'
            )
        else:
            await update.message.reply_photo(
                photo=chart_buf, caption=caption, parse_mode='Markdown'
            )
    except Exception as e:
        logger.error("Manual chart error: %s", e)
        if hasattr(update, 'callback_query') and update.callback_query:
            await update.callback_query.message.reply_text(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        else:
            await update.message.reply_text(f"❌ เกิดข้อผิดพลาด: {str(e)}")

# =========================================================
# ENHANCED ANALYSIS
# =========================================================
async def send_real_stock_analysis(message, stock: str):
    """วิเคราะห์หุ้นแบบละเอียดพร้อมส่งกราฟ"""
    try:
        # Sanitize และ format symbol
        clean_input = sanitize_input(stock)
        if not clean_input:
            await message.reply_text("⚠️ ชื่อหุ้นไม่ถูกต้อง")
            return

        symbol = format_symbol(clean_input)
        logger.info("Analyzing symbol: %s (original: %s)", symbol, clean_input)


        commodity_name = get_commodity_name(symbol)

        if commodity_name:
            company_name = commodity_name
        else:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get("longName") or symbol
            except Exception as e:
                logger.warning("Could not fetch company info for %s: %s", symbol, e)
                company_name = symbol


        df = data_fetcher.fetch_ohlcv(symbol)
        if df is None:
            await message.reply_text(f"❌ ไม่พบข้อมูล {symbol}")
            return

        # Multi-Timeframe Analysis
        weekly_trend = detect_weekly_trend(symbol)
        df_cg = _df_to_cg(df)
        trend, ema200_val = detect_trend(df_cg)

        # Levels & Setup
        levels = update_levels(symbol, df)
        breakouts = detect_breakout(symbol, df)
        current_price = float(df["close"].iloc[-1])
        resistance_levels = [x for x in levels if x["type"] == "resistance"]
        support_levels = [x for x in levels if x["type"] == "support"]
        setup = state_mgr.get_setup(symbol)

        # สร้างกราฟ
        chart_buf = None
        try:
            chart_buf = chart_gen.generate_trading_chart(
                df_cg, symbol, setup=setup, use_smart_entry=True, trend=trend
            )
        except Exception as e:
            logger.error("Chart generation failed: %s", e)

        # UX-FIRST OUTPUT
        summary = get_smart_summary_v2(setup, current_price, trend, weekly_trend)
        trend_badge = get_trend_badge_v2(trend, weekly_trend)
        health_text = analyze_stock_health(df_cg, setup)

        # News ใช้ cache แล้ว ป้องกัน AI สรุปซ้ำ
        news_text = format_news_section(symbol, max_items=5)

        action_card = get_action_card(setup, current_price, trend)
        risk_therm = get_risk_thermometer(setup)
        zone_map = get_zone_map(setup, current_price)

        # Detailed Setup
        detail_lines = []
        if setup:
            entry = setup.entry
            sl = setup.sl
            tp1 = setup.tp1
            tp2 = setup.tp2
            atr = setup.atr
            sl_pct = setup.get_pct_from_entry(sl)
            tp1_pct = setup.get_pct_from_entry(tp1)
            tp2_pct = setup.get_pct_from_entry(tp2)
            risk = entry - sl
            reward1 = tp1 - entry
            reward2 = tp2 - entry
            rr1 = reward1 / risk if risk != 0 else 0
            rr2 = reward2 / risk if risk != 0 else 0

            pos = calculate_position_size(entry, sl, PORTFOLIO_VALUE, RISK_PER_TRADE)

            detail_lines.append("━━━━━━━━━━━━━━━━━━━━\n")
            detail_lines.append("📐 รายละเอียดเทคนิค\n")
            detail_lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
            detail_lines.append(f"📍 Entry: ${entry:,.2f}\n")
            detail_lines.append(f"🟢 Buy Zone: ${setup.entry_zone_bottom:,.2f} — ${setup.entry_zone_top:,.2f}\n")
            detail_lines.append(f"🛑 SL: ${sl:,.2f} ({sl_pct:+.1f}%)\n")
            detail_lines.append(f"🎯 TP1: ${tp1:,.2f} ({tp1_pct:+.1f}%) | R:R 1:{rr1:.1f}\n")
            detail_lines.append(f"🎯 TP2: ${tp2:,.2f} ({tp2_pct:+.1f}%) | R:R 1:{rr2:.1f}\n")
            detail_lines.append(f"📏 ATR: ${atr:,.2f}\n\n")

            if pos["valid"] and pos["shares"] > 0:
                detail_lines.append("")
                detail_lines.append(f"💰 จำนวนหุ้น: {pos['shares']:,} หุ้น\n")
                detail_lines.append(f"เงินลงทุน: ${pos['investment']:,.2f}\n")
                detail_lines.append(f"ขาดทุนสูงสุด: ${pos['risk_amount']:,.2f} ({RISK_PER_TRADE:.1f}%)\n\n")
            elif not pos["valid"]:
                detail_lines.append("")
                detail_lines.append(f"⚠️ ไม่แนะนำ: {pos.get('reason', 'Unknown')}\n\n")

        detail_text = "".join(detail_lines)

        # Key Levels
        key_levels_lines = []
        key_levels_lines.append("🔑 ระดับราคาสำคัญ\n")

        if support_levels:
            key_levels_lines.append(f"🟢 แนวรับใกล้สุด: ${support_levels[0]['price']:.2f}\n")
        if resistance_levels:
            key_levels_lines.append(f"🔴 แนวต้านใกล้สุด: ${resistance_levels[0]['price']:.2f}\n")
        if setup and setup.poc:
            key_levels_lines.append(f"🟡 ราคายุติธรรม (POC): ${setup.poc:.2f}\n")
        if setup and setup.vah and setup.val:
            key_levels_lines.append(f"🔵 โซนความถี่สูง: ${setup.val:.2f} — ${setup.vah:.2f}\n")

        key_levels_text = "".join(key_levels_lines)

        # ASSEMBLE FINAL MESSAGE
        parts = []
        parts.append(f"━━━━━━━━━━━━━━━━━━━━\n")
        parts.append(f"👑 Investockify Bot v{VERSION} Report\n")
        parts.append(f"🤖 วิเคราะห์หุ้น\n")
        parts.append(f"━━━━━━━━━━━━━━━━━━━━\n")
        parts.append(f"📊 {html.escape(symbol)}\n")
        parts.append(f"🏢 {html.escape(company_name)}\n")
        parts.append(f"💵 ราคาปัจจุบัน: ${current_price:,.2f}\n")
        parts.append(f"─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
        parts.append(f"{summary}\n")
        parts.append(f"{trend_badge}\n")

        parts.append(health_text)
        parts.append(action_card)
        parts.append(f"━━━━━━━━━━━━━━━━━━━━\n")
        parts.append(risk_therm)
        parts.append("")
        parts.append(zone_map)
        parts.append("")
        parts.append(detail_text)
        parts.append("")
        parts.append(key_levels_text)

        if news_text:
            parts.append(news_text)

        if breakouts:
            parts.append("")
            parts.append("⚡ Breakouts Detected")
            parts.append("━━━━━━━━━━━━━━━━━━━━")
            parts.append("".join(breakouts))

        result = "".join(parts)

        keyboard = [
            [InlineKeyboardButton("⭐ Watchlist", callback_data="watchlist")],
            [InlineKeyboardButton("🔔 Alert", callback_data="alert"),
             InlineKeyboardButton("📈 Chart", callback_data="chart")]
        ]

        # ส่งรูปและข้อความแยกกัน พร้อม error handling รอบคอบ
        if chart_buf:
            try:
                await message.reply_photo(
                    photo=chart_buf,
                    caption=f"📊 {symbol} — Smart Trading Chart",
                    parse_mode="HTML"
                )
            except Exception as e:
                logger.error("Failed to send photo for %s: %s", symbol, e)

        try:
            await message.reply_text(
                result,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error("Failed to send analysis text for %s: %s", symbol, e)
            try:
                await message.reply_text("❌ ส่งรายงานไม่สำเร็จบางส่วน กรุณาลองใหม่")
            except:
                pass

        logger.info("Analysis sent successfully for %s", symbol)

    except Exception as e:
        logger.error("Error in send_real_stock_analysis: %s", e, exc_info=True)
        try:
            await message.reply_text("❌ เกิดข้อผิดพลาดในการวิเคราะห์ กรุณาลองใหม่")
        except:
            pass


# =========================================================
# MESSAGE HANDLER
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # ข้องข้อความที่เป็นคำสั่ง slash
    if text.startswith('/'):
        return

    # Deduplicate update_id
    if update.update_id and update_dedup.is_duplicate(update.update_id):
        logger.info("Skipped duplicate message update_id=%s", update.update_id)
        return

    # Rate Limit per user
    user_id = update.effective_user.id
    if user_rate_limit.is_limited(user_id):
        remaining = int(user_rate_limit.remaining_cooldown(user_id))
        await update.message.reply_text(f"⏳ กรุณารออีก {remaining} วินาที ก่อนขอวิเคราะห์ใหม่")
        return

    upper_text = text.upper()

    # จัดการเมนูพิเศษ
    if upper_text == "🕵 วิเคราะห์หุ้น":
        await update.message.reply_text(
            "📊 พิมพ์ชื่อหุ้น เช่น:\n\nAAPL\nNVDA\nTSLA\nMSFT\nPTT.BK\nAOT.BK\nBTCUSDT\n\nหรือใช้คำสั่ง /analyze <ชื่อหุ้น>"
        )
        return
    elif upper_text == "📱 เปิดเมนูหลัก":
        await update.message.reply_text("📱 เมนูหลัก", reply_markup=get_main_inline_keyboard())
        return
    elif upper_text == "💡 วิธีอ่านผล":
        await update.message.reply_text(
            "💡 วิธีอ่านผลลัพธ์แบบง่าย\n\n"
            "1️⃣ Smart Summary — สรุปทั้งหมดใน 1 บรรทัด\n"
            "   🟢 = พร้อมซื้อ | ⏳ = รอซื้อ | 🔴 = ต่ำกว่าเป้า\n\n"
            "2️⃣ Action Card — บอกว่าควรทำอะไร\n"
            "   ซื้อได้เลย / รอ Pullback / งดซื้อ\n\n"
            "3️⃣ Risk Thermometer — แท่งความเสี่ยง\n"
            "   ████████ = ดีมาก | ░░░░░░░░ = เสี่ยงสูง\n\n"
            "4️⃣ Zone Map — แผนที่ราคา\n"
            "   🟢 = โซนซื้อ | 🟡 = ราคายุติธรรม | 🔴 = ขาย\n\n"
            "5️⃣ รายละเอียดเทคนิค — ตัวเลขสำหรับนักลงทุนขั้นสูง\n\n"
            "6️⃣ ข่าวภาษาไทย — แปลอัตโนมัติจาก Yahoo Finance"
        )
        return
    elif upper_text == "📖 คู่มือ":
        await update.message.reply_text(
            "📖 คู่มือการใช้งาน\n\n"
            "1️⃣ พิมพ์ชื่อหุ้น (เช่น AAPL)\n"
            "2️⃣ ดู Smart Summary บรรทัดแรก\n"
            "3️⃣ อ่าน Action Card — ทำตามคำแนะนำ\n"
            "4️⃣ ดู Risk Thermometer — ถ้าแดง ควรข้าม\n"
            "5️⃣ ดู Zone Map — รู้ว่าอยู่ตรงไหนของแผนที่\n"
            "6️⃣ อ่านข่าวภาษาไทย — เข้าใจปัจจัยข่าว\n"
            "7️⃣ ตั้งคำสั่งซื้อตาม Buy Zone\n\n"
            "💡 Tip: ถ้าไม่เข้าใจ กด 💡 วิธีอ่านผล อีกครั้ง"
        )
        return

    # ตรวจสอบความถูกต้องของชื่อหุ้น
    if not is_valid_stock_input(text):
        await update.message.reply_text(
            "⚠️ ชื่อหุ้นไม่ถูกต้อง\n\n"
            "รูปแบบที่รองรับ:\n"
            "• หุ้นสหรัฐ: `AAPL`, `NVDA`, `TSLA`, `BRK.B`\n"
            "• หุ้นไทย: `PTT`, `AOT`, `CPALL`, `OR`, `BTS`\n"
            "• Crypto: `BTC`, `ETH`, `SOL`\n"
            "• Commodities: `GOLD`, `SILVER`, `COPPER`, `OIL`, `GAS`\n"
            "• Indices: `^GSPC`, `^DJI`\n\n"
            "หรือใช้ /analyze <ชื่อหุ้น>"
        )
        return

    await send_real_stock_analysis(update.message, text)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ ยกเลิกการดำเนินการ")
    return ConversationHandler.END


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Update %s caused error %s", update, context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง"
        )


# =========================================================
# MAIN (Polling Mode + Graceful Shutdown for Railway)
# =========================================================
def main():
    logger.info("Starting Investockify Bot v%s — Railway Cost-Optimized Edition", VERSION)
    logger.info("Portfolio: $%s | Risk: %s%% | Timeframe: %s", PORTFOLIO_VALUE, RISK_PER_TRADE, TIMEFRAME)
    if OPENAI_API_KEY:
        logger.info("OpenAI AI Summary: ENABLED (Hybrid Mode)")
    else:
        logger.info("OpenAI AI Summary: DISABLED (Regex-only Mode)")
    logger.info("Features: Deduplication | Rate Limiting | News Caching | Data LRU Cache | Input Sanitization")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # ConversationHandler สำหรับ /chart
    chart_conv = ConversationHandler(
        entry_points=[CommandHandler('chart', chart_command)],
        states={
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_entry)],
            TP_SL: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_tp_sl)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    # เพิ่ม Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("quick", quick_chart))
    application.add_handler(chart_conv)
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^stock_'))
    application.add_handler(CallbackQueryHandler(mode_callback, pattern='^mode_'))
    application.add_handler(CallbackQueryHandler(entry_callback, pattern='^entry_'))
    application.add_handler(CallbackQueryHandler(tp_callback, pattern='^tp_'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    # Graceful Shutdown Handler สำหรับ Railway
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal %s, stopping gracefully...", sig)
        try:
            application.stop()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Railway Production: Webhook recommended over Polling
    if BOT_MODE == "webhook" and WEBHOOK_URL:
        webhook_path = "/webhook"
        webhook_full_url = f"{WEBHOOK_URL.rstrip('/')}{webhook_path}"
        logger.info("Running in WEBHOOK mode on port %s", PORT)
        logger.info("Webhook URL: %s", webhook_full_url)
        try:
            application.run_webhook(
                listen="0.0.0.0",
                port=PORT,
                webhook_url=webhook_full_url,
                secret_token=WEBHOOK_SECRET if WEBHOOK_SECRET else None,
                drop_pending_updates=True,
                close_loop=False
            )
        except Exception as e:
            logger.error("Webhook error: %s", e, exc_info=True)
            raise
    else:
        logger.info("Running in POLLING mode")
        logger.info("Bot is running and polling for messages...")
        try:
            application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False
            )
        except Exception as e:
            logger.error("Polling error: %s", e, exc_info=True)
        finally:
            logger.info("Bot stopped")


if __name__ == "__main__":
    main()
