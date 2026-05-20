#!/usr/bin/env python3
"""
🤖 Apexify Bot — Fixed Edition v7.0
แก้ไขปัญหาหลัก:
• รวมระบบคำนวณให้ใช้ TradeSetup เดียวกันทั้งกราฟและรายงาน
• คำนวณ % จาก Entry Price เป็น baseline (ตามมาตรฐานการเทรด)
• แสดง Entry Zone ที่สอดคล้องกับ baseline
• ใช้ logic Support/Resistance เดียวกัน
"""

import os, json, logging, asyncio, re, io, threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, CallbackQueryHandler,
    filters, ContextTypes
)
from fastapi import FastAPI
import uvicorn

# ─── Configuration ───────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7732232153:AAEtA-tJnd1LiRtMKmQWeKE1L98ho9sMt-E")
PORT = int(os.environ.get("PORT", "8080"))
DATA_DIR = os.environ.get("DATA_DIR", "./data")

DEFAULT_TP1_PCT = 5.6
DEFAULT_TP2_PCT = 22.6
DEFAULT_SL_PCT = -3.5

COLORS = {
    'bullish': '#26A69A',
    'bearish': '#EF5350',
    'ema20': '#2196F3',
    'ema50': '#FF9800',
    'ema200': '#9C27B0',
    'tp1': '#2E7D32',
    'tp2': '#1B5E20',
    'sl': '#C62828',
    'entry': '#00695C',
    'now': '#1565C0',
}

ENTRY_SUPPORT_FACTOR = float(os.environ.get("ENTRY_SUPPORT_FACTOR", "0.99"))
ENTRY_SUPPORT_UPPER = float(os.environ.get("ENTRY_SUPPORT_UPPER", "1.01"))
SL_SUPPORT_FACTOR = float(os.environ.get("SL_SUPPORT_FACTOR", "0.93"))
TP2_RESISTANCE_FACTOR = float(os.environ.get("TP2_RESISTANCE_FACTOR", "1.07"))
POSITION_TP_FACTOR = float(os.environ.get("POSITION_TP_FACTOR", "1.50"))
TRAILING_STOP_FACTOR = float(os.environ.get("TRAILING_STOP_FACTOR", "0.95"))
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", "1.5"))

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── JSON Database ───────────────────────────────
class JsonDB:
    def __init__(self, filename):
        self.path = os.path.join(DATA_DIR, filename)
        self.lock = asyncio.Lock()
        self._ensure()

    def _ensure(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    async def load(self):
        async with self.lock:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)

    async def save(self, data):
        async with self.lock:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    async def get_user(self, user_id):
        return (await self.load()).get(str(user_id), {})

    async def set_user(self, user_id, user_data):
        data = await self.load()
        data[str(user_id)] = user_data
        await self.save(data)


db_portfolio = JsonDB("portfolio.json")
db_watchlist = JsonDB("watchlist.json")
db_alerts = JsonDB("alerts.json")
db_settings = JsonDB("settings.json")
db_track = JsonDB("track_record.json")

# ─── Data Classes ────────────────────────────────
@dataclass
class Level:
    price: float
    bar_index: int
    strength: int
    level_type: str


@dataclass
class TradeSetup:
    """Unified Trade Setup - ใช้เป็นตัวกลางสำหรับทั้งกราฟและรายงาน"""
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
        """คำนวณ % จาก Entry Price (มาตรฐานการเทรด)"""
        return ((price - self.entry) / self.entry) * 100

    def get_entry_zone(self, zone_pct: float = 0.009) -> Tuple[float, float]:
        """คำนวณ Entry Zone จาก Entry Price"""
        return (self.entry * (1 - zone_pct), self.entry * (1 + zone_pct))


# ─── StateManager ──────────────────────────────
class StateManager:
    def __init__(self):
        self._active_levels: Dict[str, List[Dict]] = {}
        self._broken_levels: Dict[str, List[Dict]] = {}
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
        self._broken_levels.pop(symbol, None)
        self._entry_data.pop(symbol, None)


state_mgr = StateManager()

# ─── Unified Technical Analysis Engine ─────────
class TechnicalAnalyzer:
    """
    รวม logic การวิเคราะห์ทางเทคนิคให้เป็นที่เดียว
    ใช้ทั้งสำหรับวาดกราฟและสร้างรายงาน
    """
    def __init__(self):
        self.PIVOT_LEN = 5
        self.MIN_ATR_STRENGTH = 0.03
        self.MAX_LEVEL_AGE = 500
        self.MAX_ACTIVE_LEVELS_EACH = 10
        self.MERGE_THRESHOLD = 0.8
        self.ZONE_WIDTH = 0.25
        self.BREAK_SENS = 0.1
        self.MAX_BREAKOUT_SIGNALS = 4
        self.ATR_PERIOD = 14
        self.MAX_DISTANCE_FROM_PRICE = 0.5
        self.MIN_BREAKOUT_MOVE_ATR = 0.1

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

    def detect_pivots_vectorized(self, df: pd.DataFrame):
        highs = []
        lows = []
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
        merged_groups = []
        current_group = [sorted_levels[0]]
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

    def find_trade_setup(self, df: pd.DataFrame, current_price: float) -> TradeSetup:
        """
        หา TradeSetup ที่สอดคล้องกัน - ใช้เป็นตัวกลางสำหรับทั้งกราฟและรายงาน
        """
        df_atr = self.calculate_atr(df)
        atr = df_atr["atr"].iloc[-1]
        if pd.isna(atr) or atr == 0:
            atr = (df["High"] - df["Low"]).mean()

        highs, lows = self.detect_pivots_vectorized(df_atr)
        current_bar = len(df)
        highs = [x for x in highs if current_bar - x.bar_index <= self.MAX_LEVEL_AGE]
        lows = [x for x in lows if current_bar - x.bar_index <= self.MAX_LEVEL_AGE]

        merged_highs = self.merge_levels(highs, atr)
        merged_lows = self.merge_levels(lows, atr)

        merged_highs = [
            x for x in merged_highs
            if x.price > current_price and (x.price - current_price) / current_price <= self.MAX_DISTANCE_FROM_PRICE
        ]
        merged_lows = [
            x for x in merged_lows
            if x.price < current_price and (current_price - x.price) / current_price <= self.MAX_DISTANCE_FROM_PRICE
        ]

        merged_highs = sorted(merged_highs, key=lambda x: (abs(x.price - current_price), -x.strength))[:self.MAX_ACTIVE_LEVELS_EACH]
        merged_lows = sorted(merged_lows, key=lambda x: (abs(x.price - current_price), -x.strength))[:self.MAX_ACTIVE_LEVELS_EACH]

        if merged_highs:
            tp1_price = merged_highs[0].price
            tp2_price = merged_highs[1].price if len(merged_highs) >= 2 else tp1_price * 1.06
        else:
            tp1_price = current_price * 1.05
            tp2_price = tp1_price * 1.06

        if merged_lows:
            entry_price = merged_lows[0].price
            sl_price = merged_lows[1].price if len(merged_lows) >= 2 else None
        else:
            entry_price = current_price * 0.96
            sl_price = None

        if sl_price is None:
            sl_buffer = max(atr * 2, entry_price * 0.03)
            sl_price = entry_price - sl_buffer

        if entry_price >= current_price:
            entry_price = current_price * 0.97
            sl_buffer = max(atr * 1.5, entry_price * 0.025)
            sl_price = entry_price - sl_buffer
        if sl_price >= entry_price:
            sl_price = entry_price * 0.97

        support = merged_lows[0].price if merged_lows else entry_price
        resistance = merged_highs[0].price if merged_highs else tp1_price

        entry_zone_bottom = entry_price * 0.991
        entry_zone_top = entry_price * 1.009

        return TradeSetup(
            entry=entry_price,
            sl=sl_price,
            tp1=tp1_price,
            tp2=tp2_price,
            support_levels=merged_lows,
            resistance_levels=merged_highs,
            atr=atr,
            support=support,
            resistance=resistance,
            entry_zone_bottom=entry_zone_bottom,
            entry_zone_top=entry_zone_top
        )

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14):
        """
        คำนวณ RSI ด้วย Wilder's Smoothing (Exponential Weighted Moving Average)
        ตามมาตรฐานของ telegram_bot143.py — ให้ผลลัพธ์ที่ smoother และตอบสนองต่อ
        การเปลี่ยนแปลงราคาได้ดีกว่า Simple Moving Average
        """
        close = df['Close']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger(self, df: pd.DataFrame, window: int = 20, num_std: int = 2):
        sma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        return sma + (std * num_std), sma - (std * num_std), sma

    def calculate_poc(self, df: pd.DataFrame, bins: int = 50, lookback_days: int = 120):
        if df.empty or 'Volume' not in df.columns:
            return None
        recent_df = df.tail(lookback_days).copy()
        if recent_df.empty:
            return None
        current_price = float(df['Close'].iloc[-1])
        typical_price = (recent_df['High'] + recent_df['Low'] + recent_df['Close']) / 3
        min_price = typical_price.min()
        max_price = typical_price.max()
        if pd.isna(min_price) or pd.isna(max_price):
            return current_price
        if max_price < current_price * 0.5 or min_price > current_price * 2.0:
            vwap = (recent_df['Close'] * recent_df['Volume']).sum() / recent_df['Volume'].sum()
            return float(vwap) if not pd.isna(vwap) else current_price
        bin_width = (max_price - min_price) / bins
        if bin_width == 0:
            return float(typical_price.iloc[-1])
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
            return float(typical_price.iloc[-1])
        poc = max(volume_profile, key=volume_profile.get)
        if poc < current_price * 0.6:
            vwap = (recent_df['Close'] * recent_df['Volume']).sum() / recent_df['Volume'].sum()
            return float(vwap) if not pd.isna(vwap) else current_price
        return float(poc)


# ─── ChartGenerator ────────────────────────────
class ChartGenerator:
    def __init__(self, analyzer: TechnicalAnalyzer):
        self.analyzer = analyzer
        plt.style.use('default')

    def generate_trading_chart(self, df: pd.DataFrame, symbol: str,
                               setup: TradeSetup = None,
                               use_smart_entry: bool = True,
                               tp1_pct: float = None,
                               tp2_pct: float = None,
                               sl_price: float = None):
        """
        วาดกราฟโดยใช้ TradeSetup เป็นตัวกลาง
        คำนวณ % จาก Entry Price (baseline) อย่างสอดคล้องกัน
        """
        close = df['Close']
        ema20 = self.analyzer.calculate_ema(close, 20)
        ema50 = self.analyzer.calculate_ema(close, 50)
        ema200 = self.analyzer.calculate_ema(close, 200)
        ema20_last = ema20.iloc[-1]
        ema50_last = ema50.iloc[-1]
        ema200_last = ema200.iloc[-1]
        current_price = close.iloc[-1]
        df_display = df.tail(60).copy()
        ema20_display = ema20.tail(60)
        ema50_display = ema50.tail(60)
        ema200_display = ema200.tail(60)
        df_plot = df_display

        if setup is None and use_smart_entry:
            setup = self.analyzer.find_trade_setup(df, current_price)

        if setup is None:
            entry_price = current_price
            if tp1_pct is not None:
                tp1_price = entry_price * (1 + tp1_pct / 100)
            else:
                tp1_price = entry_price * 1.05
            if tp2_pct is not None:
                tp2_price = entry_price * (1 + tp2_pct / 100)
            else:
                tp2_price = entry_price * 1.10
            if sl_price is None:
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)
            setup = TradeSetup(
                entry=entry_price, sl=sl_price, tp1=tp1_price, tp2=tp2_price,
                support_levels=[], resistance_levels=[],
                atr=(df["High"] - df["Low"]).mean()
            )

        entry_price = setup.entry
        sl_price = setup.sl
        tp1_price = setup.tp1
        tp2_price = setup.tp2
        support_levels = setup.support_levels
        resistance_levels = setup.resistance_levels
        atr = setup.atr

        # === คำนวณ % จาก Entry Price (baseline) - มาตรฐานการเทรด ===
        baseline = entry_price
        entry_pct = 0.0
        sl_pct = setup.get_pct_from_entry(sl_price)
        tp1_pct_display = setup.get_pct_from_entry(tp1_price)
        tp2_pct_display = setup.get_pct_from_entry(tp2_price)

        entry_zone_bottom = setup.entry_zone_bottom
        entry_zone_top = setup.entry_zone_top
        # ============================================================

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
            edge_color = color
            height = abs(close_p - open_p)
            bottom = min(open_p, close_p)
            rect = mpatches.FancyBboxPatch(
                (x - 0.4, bottom), 0.8, height,
                boxstyle="square,pad=0",
                facecolor=color, edgecolor=edge_color, linewidth=1
            )
            ax1.add_patch(rect)
            ax1.plot([x, x], [low_p, high_p], color=color, linewidth=1)

        x_range = range(len(df_plot))
        ax1.plot(x_range, ema20_display.values, color=COLORS['ema20'], linewidth=2, label='EMA 20', alpha=0.8)
        ax1.plot(x_range, ema50_display.values, color=COLORS['ema50'], linewidth=2, label='EMA 50', alpha=0.8)
        ax1.plot(x_range, ema200_display.values, color=COLORS['ema200'], linewidth=2, label='EMA 200', alpha=0.8)

        for lvl in support_levels:
            ax1.axhline(y=lvl.price, color='#4CAF50', linestyle=':', linewidth=1, alpha=0.5)
        for lvl in resistance_levels:
            ax1.axhline(y=lvl.price, color='#F44336', linestyle=':', linewidth=1, alpha=0.5)

        ax1.axhline(y=tp2_price, color=COLORS['tp2'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhline(y=tp1_price, color=COLORS['tp1'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=entry_price, color=COLORS['entry'], linestyle='dotted', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=sl_price, color=COLORS['sl'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=COLORS['entry'])

        price_min = min(df_plot['Low'].min(), sl_price * 0.95)
        price_max = max(df_plot['High'].max(), tp2_price * 1.05)
        ax1.set_ylim(price_min, price_max)
        ax1.set_xlim(-1, len(df_plot))
        y_range = price_max - price_min
        y_shift = y_range * 0.008
        x_offset = len(df_plot) * 0.02

        # === Label บนกราฟ ใช้สูตรจาก Entry (baseline) ===
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
            #entry_text = f"◫ ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f} ({entry_pct:+.1f}%)"
            entry_text = f"◫ ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f}"
        else:
            #entry_text = f"◫ ENTRY: ${entry_price:,.2f} ({entry_pct:+.1f}%)"
            entry_text = f"◫ ENTRY: ${entry_price:,.2f}"

        ax1.text(x_offset, entry_price,
                 entry_text,
                 fontsize=10, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['entry'],
                          edgecolor='white', alpha=0.8), color='white')
        ax1.text(x_offset, sl_price,
                 f"◍ SL ${sl_price:,.2f} ({sl_pct:+.1f}%)",
                 fontsize=11, fontweight='bold', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['sl'],
                          edgecolor='white', alpha=0.9), color='white')
        # ====================================================

        x_right = len(df_plot) * 0.98
        ax1.text(x_right, ema20_display.iloc[-1] + y_shift,
                 f"EMA20 {ema20_last:,.2f}",
                 fontsize=9, fontweight='bold', va='bottom', ha='right',
                 color=COLORS['ema20'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['ema20'], alpha=0.9))
        ax1.text(x_right, ema50_display.iloc[-1] + y_shift,
                 f"EMA50 {ema50_last:,.2f}",
                 fontsize=9, fontweight='bold', va='bottom', ha='right',
                 color=COLORS['ema50'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['ema50'], alpha=0.9))
        ax1.text(x_right, ema200_display.iloc[-1] + y_shift,
                 f"EMA200 {ema200_last:,.2f}",
                 fontsize=9, fontweight='bold', va='bottom', ha='right',
                 color=COLORS['ema200'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['ema200'], alpha=0.9))

        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_title(f'Investockify — {symbol}  |  Pivot S/R Entry + TP + SL\n'
                     f'EMA: 20(Blue) 50(Orange) 200(Purple)',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='-')
        ax1.set_axisbelow(True)

        volumes = df_plot['Volume'].values
        max_vol = volumes.max() if len(volumes) > 0 else 1
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            x = i
            color = COLORS['bullish'] if row['Close'] >= row['Open'] else COLORS['bearish']
            ax2.bar(x, row['Volume'], color=color, alpha=0.7, width=0.8)

        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylim(0, max_vol * 1.5)
        ax2.grid(True, alpha=0.3, linestyle='-')

        date_labels = [d.strftime('%b %d') for d in df_plot.index[::5]]
        x_ticks = range(0, len(df_plot), 5)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()
        return buf

    def generate_simple_chart(self, df: pd.DataFrame, symbol: str):
        return self.generate_trading_chart(df, symbol, use_smart_entry=True)


# ─── DataFetcher ─────────────────────────────────
class DataFetcher:
    def __init__(self):
        pass

    def get_stock_data(self, symbol: str, period: str = "3y", interval: str = "1d"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return None, "ไม่พบข้อมูลสำหรับสัญลักษณ์นี้"
            return df, None
        except Exception as e:
            return None, f"เกิดข้อผิดพลาด: {str(e)}"

    def get_current_price(self, symbol: str):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current = info.get('regularMarketPrice') or info.get('currentPrice')
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
            if current and prev_close:
                change_pct = ((current - prev_close) / prev_close) * 100
                return {'price': current, 'change_pct': change_pct, 'prev_close': prev_close}
            return None
        except Exception as e:
            logger.warning(f"get_current_price error: {e}")
            return None


# ─── Unified Analyzer ────────────────────────────
analyzer = TechnicalAnalyzer()
chart_gen = ChartGenerator(analyzer)
data_fetcher = DataFetcher()


def analyze_stock(ticker):
    """
    วิเคราะห์หุ้นโดยใช้ TradeSetup จาก TechnicalAnalyzer เดียวกัน
    ทั้งกราฟและรายงานใช้ค่าเดียวกัน
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3y", interval="1d")
        if df.empty or len(df) < 30:
            return None
        info = stock.info
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None

    current_price = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])

    # ใช้ TradeSetup จาก analyzer เดียวกันกับ ChartGenerator
    setup = analyzer.find_trade_setup(df, current_price)

    ema20_series = analyzer.calculate_ema(df['Close'], 20)
    ema50_series = analyzer.calculate_ema(df['Close'], 50)
    ema200_series = analyzer.calculate_ema(df['Close'], 200)
    ema20_val = ema20_series.iloc[-1]
    ema50_val = ema50_series.iloc[-1]
    ema200 = ema200_series.iloc[-1] if len(ema200_series.dropna()) > 0 else None

    rsi = analyzer.calculate_rsi(df, 14).iloc[-1]
    macd, signal = analyzer.calculate_macd(df)
    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]
    bb_upper, bb_lower, bb_mid = analyzer.calculate_bollinger(df)
    bb_upper_val = bb_upper.iloc[-1]
    bb_lower_val = bb_lower.iloc[-1]

    poc = analyzer.calculate_poc(df, lookback_days=120)

    # ใช้ค่าจาก TradeSetup (สอดคล้องกับกราฟ)
    support = setup.support
    resistance = setup.resistance
    entry_low = setup.entry_zone_bottom
    entry_high = setup.entry_zone_top
    sl = setup.sl
    tp1 = setup.tp1
    tp2 = setup.tp2

    # Position Plan
    if poc and poc >= current_price * 0.65:
        position_entry_low = min(entry_low * 0.97, poc * 0.95)
        position_entry_low = max(position_entry_low, current_price * 0.75)
    else:
        position_entry_low = entry_low * 0.97

    position_entry_high = entry_low * 0.99
    position_tp = resistance * POSITION_TP_FACTOR

    trailing_stop_raw = entry_low * TRAILING_STOP_FACTOR
    trailing_stop = max(trailing_stop_raw, support * 0.97, current_price * 0.70)
    trailing_stop = min(trailing_stop, entry_low * 0.92)

    risk = entry_low - sl
    reward1 = tp1 - entry_low
    reward2 = tp2 - entry_low
    rr1 = round(reward1 / risk, 2) if risk > 0 else 0
    rr2 = round(reward2 / risk, 2) if risk > 0 else 0

    # 3-Timeframe Trend
    if current_price > ema20_val * 1.005:
        day_trend = "🟢"; day_text = "ขาขึ้น Bullish"; day_desc = "ราคายืนเหนือ EMA ระยะสั้น"
    elif current_price < ema20_val * 0.995:
        day_trend = "🔴"; day_text = "ขาลง Bearish"; day_desc = "ราคาหลุด EMA ระยะสั้น"
    else:
        day_trend = "⚪️"; day_text = "ทรงตัว Neutral"; day_desc = "ราคาเคลื่อนไหวรอบ EMA ระยะสั้น"

    if ema20_val > ema50_val * 1.01:
        week_trend = "🟢"; week_text = "ขาขึ้น Bullish"; week_desc = "แนวโน้มแข็งแกร่งต่อเนื่อง"
    elif ema20_val < ema50_val * 0.99:
        week_trend = "🔴"; week_text = "ขาลง Bearish"; week_desc = "แนวโน้มอ่อนแอร์ลง"
    else:
        week_trend = "⚪️"; week_text = "ทรงตัว Neutral"; week_desc = "แนวโน้มระยะกลางยังไม่ชัดเจน"

    if ema200 is not None and not pd.isna(ema200):
        if ema50_val > ema200 * 1.01:
            month_trend = "🟢"; month_text = "ขาขึ้น Bullish"; month_desc = "แนวโน้มระยะยาวแข็งแกร่ง (Golden Cross bias)"
        elif ema50_val < ema200 * 0.99:
            month_trend = "🔴"; month_text = "ขาลง Bearish"; month_desc = "แนวโน้มระยะยาวอ่อนแอร์ (Death Cross bias)"
        else:
            month_trend = "⚪️"; month_text = "ทรงตัว Neutral"; month_desc = "แนวโน้มระยะยาวยังคงแข็งแกร่ง แต่โมเมนตัมไม่ชัดเจน"
    else:
        if current_price > ema50_val * 1.03:
            month_trend = "🟢"; month_text = "ขาขึ้น Bullish"; month_desc = "ราคายืนเหนือเส้นค่าเฉลี่ยระยะยาว"
        elif current_price < ema50_val * 0.97:
            month_trend = "🔴"; month_text = "ขาลง Bearish"; month_desc = "ราคาหลุดเส้นค่าเฉลี่ยระยะยาว"
        else:
            month_trend = "⚪️"; month_text = "ทรงตัว Neutral"; month_desc = "แนวโน้มระยะยาวยังคงแข็งแกร่ง แต่โมเมนตัมไม่ชัดเจน"

    if rsi > 70:
        rsi_status = "🔴"
    elif rsi < 30:
        rsi_status = "🟢"
    else:
        rsi_status = "⚪️"

    vol = df['Volume'].iloc[-1]
    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
    vol_status = "📈 Inflow" if vol > vol_avg else "📉 Outflow"

    score = 50
    if current_price > ema20_val: score += 12
    if ema20_val > ema50_val: score += 12
    if month_trend == "🟢": score += 8
    elif month_trend == "🔴": score -= 5
    if macd_val > signal_val: score += 10
    if 40 < rsi < 70: score += 10
    else: score -= 5
    if vol > vol_avg: score += 8
    score = max(0, min(100, score))

    try:
        news_items = fetch_news_items(stock)
    except Exception:
        news_items = []

    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName", ticker.upper()) if info else ticker.upper(),
        "price": current_price,
        "change": ((current_price - prev_close) / prev_close) * 100,
        "ema20": ema20_val, "ema50": ema50_val, "ema200": ema200,
        "rsi": rsi, "rsi_status": rsi_status,
        "macd": macd_val, "signal": signal_val,
        "bb_upper": bb_upper_val, "bb_lower": bb_lower_val,
        "poc": poc,
        "support": support, "resistance": resistance,
        "entry_low": entry_low, "entry_high": entry_high,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": rr1, "rr2": rr2,
        "position_entry_low": position_entry_low,
        "position_entry_high": position_entry_high,
        "position_tp": position_tp,
        "trailing_stop": trailing_stop,
        "day_trend": day_trend, "day_text": day_text, "day_desc": day_desc,
        "week_trend": week_trend, "week_text": week_text, "week_desc": week_desc,
        "month_trend": month_trend, "month_text": month_text, "month_desc": month_desc,
        "volume": vol, "volume_avg": vol_avg,
        "volume_status": vol_status, "score": score,
        "currency": info.get("currency", "USD") if info else "USD",
        "news_items": news_items,
        "df": df,
        "setup": setup
    }


def format_report(data, portfolio_qty=0, portfolio_avg=0):
    """
    สร้างรายงานโดยใช้ TradeSetup จาก data['setup']
    คำนวณ % จาก Entry Price (baseline) อย่างสอดคล้องกับกราฟ
    """
    ticker = data["ticker"]
    price = data["price"]
    setup = data.get("setup")

    emoji_conviction = "🔥" if data["score"] >= 80 else "🚀" if data["score"] >= 60 else "⚡️"
    confidence = "🟢 สูง" if data["rr1"] >= 3 else "🟡 ปานกลาง" if data["rr1"] >= 2 else "🔴 ต่ำ"

    portfolio_text = ""
    if portfolio_qty > 0:
        pnl_pct = ((price - portfolio_avg) / portfolio_avg) * 100
        emoji_pnl = "🟢" if pnl_pct >= 0 else "🔴"
        portfolio_text = f"💼 คุณถืออยู่: {portfolio_qty:.0f} หุ้น @ {portfolio_avg:.2f}  {emoji_pnl} {pnl_pct:+.2f}%"

    poc_text = ""
    if data.get("poc") is not None:
        poc_val = data['poc']
        if poc_val >= price * 0.6:
            poc_text = f"• 🟡 POC (โซนคนกระจุก): {poc_val:.2f}"
        else:
            poc_text = f"• 🟡 POC: ไม่สมเหตุสมผล ({poc_val:.2f}) — ใช้ VWAP แทน"

    # === คำนวณ % จาก Entry Price (baseline) - สอดคล้องกับกราฟ ===
    if setup:
        baseline = setup.entry
        entry_pct = setup.get_pct_from_entry(data['entry_low'])
        sl_pct = setup.get_pct_from_entry(data['sl'])
        tp1_pct = setup.get_pct_from_entry(data['tp1'])
        tp2_pct = setup.get_pct_from_entry(data['tp2'])
        pos_entry_pct = setup.get_pct_from_entry(data['position_entry_low'])
        pos_tp_pct = setup.get_pct_from_entry(data['position_tp'])
        trail_pct = setup.get_pct_from_entry(data['trailing_stop'])
    else:
        baseline = data['entry_low']
        entry_pct = 0.0
        sl_pct = ((data['sl'] - baseline) / baseline) * 100 if baseline != 0 else 0
        tp1_pct = ((data['tp1'] - baseline) / baseline) * 100 if baseline != 0 else 0
        tp2_pct = ((data['tp2'] - baseline) / baseline) * 100 if baseline != 0 else 0
        pos_entry_pct = ((data['position_entry_low'] - baseline) / baseline) * 100 if baseline != 0 else 0
        pos_tp_pct = ((data['position_tp'] - baseline) / baseline) * 100 if baseline != 0 else 0
        trail_pct = ((data['trailing_stop'] - baseline) / baseline) * 100 if baseline != 0 else 0

    position_insight = ""
    poc_val = data.get("poc")
    if poc_val and poc_val >= price * 0.6:
        if price > poc_val:
            position_insight = f"หากราคายืนเหนือ POC {poc_val:.2f} ได้ อาจพิจารณาถือต่อเพื่อเป้าหมายถัดไป"
        else:
            position_insight = f"ราคาอยู่ต่ำกว่า POC {poc_val:.2f} อาจเป็นโอกาสสะสมในโซนคนกระจุก"
    else:
        if data['month_trend'] == "🟢":
            position_insight = "แนวโน้มระยะยาวเป็นขาขึ้น พิจารณาสะสมเมื่อราคาลงมาใกล้แนวรับ"
        elif data['month_trend'] == "🔴":
            position_insight = "แนวโน้มระยะยาวอ่อนแอ ควรระวังและรอสัญญาณกลับตัว"
        else:
            position_insight = "พิจารณาถือต่อหากแนวโน้มระยะยาวยังเป็นขาขึ้น"

    if portfolio_qty > 0:
        pnl_pct = ((price - portfolio_avg) / portfolio_avg) * 100
        short_term = "แสดงความอ่อนแอ" if data['day_trend'] == "🔴" else "แสดงความแข็งแกร่ง" if data['day_trend'] == "🟢" else "ยังไม่มีทิศทางชัดเจน"
        week_trend_status = "ขาขึ้น" if "ขาขึ้น" in data['week_text'] else "ขาลง" if "ขาลง" in data['week_text'] else "ทรงตัว"
        insight_text = f"🧠 Apexify Insight: หุ้นที่คุณถืออยู่ตอนนี้{'มีกำไร' if pnl_pct >= 0 else 'ขาดทุน'}ประมาณ {abs(pnl_pct):.2f}% การเคลื่อนไหวระยะสั้น{short_term} แต่แนวโน้มรายสัปดาห์ยังคงเป็น{week_trend_status}."
    else:
        insight_text = "🧠 Apexify Insight: ยังไม่มีตำแหน่งในพอร์ต พิจารณาสะสมเมื่อราคาเข้าใกล้แนวรับ และรอการยืนยันตัวของแนวโน้มรายสัปดาห์."

    news_text = ""
    if data.get("news_items"):
        first_title = data['news_items'][0].replace("• ", "").split(" [")[0]
        summary = first_title[:90] + "..." if len(first_title) > 90 else first_title
        news_lines = [
            "📰 ข่าวล่าสุด — Apexify ว่ายังไง ⚪ เป็นกลาง",
            f"  💬 {summary}"
        ]
        for item in data['news_items'][:3]:
            news_lines.append(f"  {item}")
        news_text = "\n".join(news_lines) + "\n"
    else:
        news_text = ""

    report = f"""━━━━━━━━━━━━━━━━━
👑 Apexify - PRO Report
🤖 {ticker} ({data['name']})
━━━━━━━━━━━━━━━━━
💵 ราคา: {price:.2f} ({data['change']:+.2f}%)
📡 Trend: •  Day {data['day_trend']} 
                  •  Week {data['week_trend']} 
                  •  Month {data['month_trend']}
🎯 Conviction: {emoji_conviction} {data['score']:.0f}/100
─ ─ ─ ─ ─ ─ ─ ─ ─
📊 วิเคราะห์สุขภาพหุ้น
• 🌊 Trend: {data['day_trend']} {((price - data['ema20'])/data['ema20']*100):+.2f}% vs EMA20
• 🌡 RSI: {data['rsi_status']} {data['rsi']:.2f}
• ⚡️ MACD: {"🟢 โมเมนตัมบวก" if data['macd'] > data['signal'] else "🔴 โมเมนตัมลบ"}  ({data['macd']:.2f} / {data['signal']:.2f})
• 💧 Volume: {data['volume_status']}
         Vol {data['volume']/1e6:.2f}M | Avg {data['volume_avg']/1e6:.2f}M
─ ─ ─ ─ ─ ─ ─ ─ ─
🗺 โซนราคาสำคัญ
• 🟢 แนวรับ: {data['support']:.2f}
• 🔴 แนวต้าน: {data['resistance']:.2f}
{poc_text}
• 🔵 Bollinger Band: {data['bb_lower']:.2f} — {data['bb_upper']:.2f}
{portfolio_text}
🎯 Apexify Confidence: {confidence}
  R:R TP1: {data['rr1']:.2f}x | TP2: {data['rr2']:.2f}x

🔭 Apexify Trend Radar — 3 ระยะ
• ⏱ วัน: {data['day_trend']} {data['day_text']} — {data['day_desc']}
• 📅 สัปดาห์: {data['week_trend']} {data['week_text']} — {data['week_desc']}
• 🔭 เดือน: {data['month_trend']} {data['month_text']} — {data['month_desc']}

🏃 สายสั้น (Swing) — กรอบ 1-4 สัปดาห์
  💡 เข้าซื้อ: {data['entry_low']:.2f} - {data['entry_high']:.2f}
  🎯 TP1: {data['tp1']:.2f} ({tp1_pct:+.1f}%)
  🎯 TP2: {data['tp2']:.2f} ({tp2_pct:+.1f}%)
  🛑 SL: {data['sl']:.2f} ({sl_pct:+.1f}%)
  ⚖️ R:R: ✅ {data['rr1']:.2f} — {"ดีเยี่ยม" if data['rr1'] >= 3 else "ดี" if data['rr1'] >= 2 else "ระวัง"} เสี่ยง 1 ส่วน แลกผลตอบแทน {data['rr1']:.1f} ส่วน

🧘 สายยาว (Position) — กรอบ 3-12 เดือน
  💡 {position_insight}
  📍 สะสมเพิ่ม: {data['position_entry_low']:.2f} - {data['position_entry_high']:.2f} ({pos_entry_pct:+.1f}%)
  🎯 เป้าระยะยาว: {data['position_tp']:.2f} ({pos_tp_pct:+.1f}%)
  🛡 Trailing Stop: {data['trailing_stop']:.2f} ({trail_pct:+.1f}%)

🧭 เงื่อนไข Plan
  ✅ ยืนยันเข้า: ราคาปิดเหนือ {data['resistance']:.2f} พร้อมปริมาณการซื้อขายที่เพิ่มขึ้น และ MACD รายวันกลับมาเป็นบวก.
  ❌ ยกเลิก Plan: ราคาหลุดแนวรับ {data['support']:.2f} อย่างมีนัยสำคัญ จะทำให้แผนการฟื้นตัวไม่เป็นไปตามคาด.
  👀 Watch Next: การทดสอบแนวรับ {data['support']:.2f} หรือการกลับมายืนเหนือ {data['resistance']:.2f} เพื่อยืนยันการฟื้นตัว.

{insight_text}

{news_text}
💡 Position Sizing: เสี่ยงไม่เกิน {RISK_PERCENT:.1f}% ของพอร์ตต่อไม้

⚠️ ข้อมูลเพื่อประกอบการพิจารณา ไม่ใช่คำแนะนำลงทุน · การลงทุนมีความเสี่ยง
"""
    return report


# ─── News Helper ───────────────────────────────
def fetch_news_items(stock, max_items=3):
    try:
        raw = stock.news
        if not raw or not isinstance(raw, list):
            return []
        items = []
        now = datetime.now().astimezone()
        for item in raw[:max_items]:
            title = item.get('title') or item.get('headline') or item.get('summary') or item.get('content', {}).get('title')
            if not title or not isinstance(title, str) or title.strip() in ('', 'Untitled'):
                continue
            title = title.strip()
            ts = None
            for key in ('providerPublishTime', 'published', 'pubDate', 'publish_time', 'time'):
                if key in item and item[key]:
                    ts = item[key]
                    break
            if not ts and isinstance(item.get('content'), dict):
                for key in ('providerPublishTime', 'published', 'pubDate', 'publish_time', 'time'):
                    if key in item['content'] and item['content'][key]:
                        ts = item['content'][key]
                        break
            time_str = "?"
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        pub_dt = datetime.fromtimestamp(ts, tz=now.tzinfo)
                    else:
                        pub_dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
                    hours_ago = int((now - pub_dt).total_seconds() / 3600)
                    if hours_ago < 1:
                        time_str = "now"
                    elif hours_ago < 24:
                        time_str = f"{hours_ago}h"
                    else:
                        days_ago = hours_ago // 24
                        time_str = f"{days_ago}d"
                except Exception:
                    time_str = "?"
            items.append(f"• {title} [{time_str}]")
        return items
    except Exception as e:
        logger.error(f"News fetch error: {e}")
        return []


# ─── Level / Breakout Helpers ──────────────────
def update_levels(symbol, df):
    current_price = float(df["Close"].iloc[-1])
    setup = analyzer.find_trade_setup(df, current_price)
    atr = setup.atr

    state_mgr.set_setup(symbol, setup)

    levels = []
    for h in setup.resistance_levels:
        levels.append({
            "type": "resistance",
            "price": round(h.price, 2),
            "strength": h.strength,
            "zone_top": h.price + atr * analyzer.ZONE_WIDTH,
            "zone_bottom": h.price - atr * analyzer.ZONE_WIDTH
        })
    for l in setup.support_levels:
        levels.append({
            "type": "support",
            "price": round(l.price, 2),
            "strength": l.strength,
            "zone_top": l.price + atr * analyzer.ZONE_WIDTH,
            "zone_bottom": l.price - atr * analyzer.ZONE_WIDTH
        })
    levels = sorted(levels, key=lambda x: x["price"], reverse=True)
    state_mgr.set_levels(symbol, levels)
    return levels


def detect_breakout(symbol, df):
    levels = state_mgr.get_levels(symbol)
    if not levels:
        return []

    df_atr = analyzer.calculate_atr(df)
    close_price = float(df["Close"].iloc[-1])
    atr = df_atr["atr"].iloc[-1]

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
                signals.append(
                    f"🚀 {symbol}\nBreak Resistance\n"
                    f"Level: {lvl['price']:.2f}\n"
                    f"Current: {close_price:.2f}\n"
                    f"Strength: {lvl['strength']}"
                )
                breakout_count += 1
        elif (lvl["type"] == "support" and
              close_price < lvl["zone_bottom"] - buffer and
              breakout_move >= analyzer.MIN_BREAKOUT_MOVE_ATR):
            if breakout_count < analyzer.MAX_BREAKOUT_SIGNALS:
                signals.append(
                    f"🔻 {symbol}\nBreak Support\n"
                    f"Level: {lvl['price']:.2f}\n"
                    f"Current: {close_price:.2f}\n"
                    f"Strength: {lvl['strength']}"
                )
                breakout_count += 1
        else:
            remaining.append(lvl)

    state_mgr.set_levels(symbol, remaining)
    return signals



# ─── Telegram Handlers ───────────────────────────

SYMBOL, ENTRY, TP_SL = range(3)
user_data = {}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🚀 ยินดีต้อนรับสู่ Apexify Trading Bot!\n\n"
        "พิมพ์ชื่อสัญลักษณ์หุ้น/สินค้าโภคภัณฑ์ เพื่อดูกราฟการเทรดทันที\n"
        "หรือใช้คำสั่งต่อไปนี้:\n\n"
        "📌 /chart <symbol> - ดูกราฟพร้อมตั้งค่า TP/SL\n"
        "📌 /quick <symbol> - ดูกราฟแบบเร็ว (Smart Entry)\n"
        "📌 /portfolio หรือ /port - ดูพอร์ต\n"
        "📌 /watch <TICKER> - เพิ่ม Watchlist\n"
        "📌 /setalert <TICKER> <ราคา/%> - ตั้งเตือน\n"
        "📌 /track - สถิติ Track Record\n"
        "📌 /help - ดูคำแนะนำทั้งหมด\n\n"
        "*ตัวอย่างสัญลักษณ์:*\n"
        "• GC=F (ทองคำ)\n"
        "• SI=F (เงิน)\n"
        "• CL=F (น้ำมัน WTI)\n"
        "• AAPL, TSLA, NVDA (หุ้น US)\n\n"
        "พิมพ์ชื่อสัญลักษณ์ได้เลย! 👇"
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 คู่มือการใช้งาน Apexify Bot\n\n"
        "*คำสั่งพื้นฐาน:*\n"
        "• /start - เริ่มต้นใช้งาน\n"
        "• /chart <symbol> - สร้างกราฟพร้อมตั้งค่า TP/SL\n"
        "• /quick <symbol> - ดูกราฟเร็วๆ (Smart Entry)\n"
        "• /help - ดูคำแนะนำ\n\n"
        "*โหมด Smart Entry:*\n"
        "• ระบบจะหา Swing Low ใกล้ EMA200 อัตโนมัติ\n"
        "• SL คำนวณจาก ATR (2.5x)\n"
        "• TP คำนวณจาก Risk:Reward (1:2 และ 1:4)\n\n"
        "*จัดการพอร์ต:*\n"
        "• /add <TICKER> <จำนวน> <ราคา> — บันทึกซื้อ\n"
        "• /portfolio หรือ /port — ดูพอร์ต\n"
        "• /pnl — ดูกำไร/ขาดทุน\n"
        "• /watch <TICKER> — เพิ่ม Watchlist\n"
        "• /setalert <TICKER> <ราคา/%> — ตั้งเตือน\n"
        "• /track — สถิติ Track Record\n"
        "• /glossary — คำศัพท์เทคนิค\n"
        "• /settings — ตั้งค่า\n"
        "• /health — เช็คสถานะระบบ"
    )
    await update.message.reply_text(help_text)


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ ระบบทำงานปกติ\n"
        f"🕐 เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📊 Mode: Polling (No Webhook)\n"
        f"💾 Data Dir: {DATA_DIR}\n"
        f"⚙️ SL: {SL_SUPPORT_FACTOR} | TP2: {TP2_RESISTANCE_FACTOR} | Position TP: {POSITION_TP_FACTOR} | Risk: {RISK_PERCENT}%"
    )


async def glossary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 คำศัพท์เทคนิค (อ่านง่ายๆ)\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🌡️ RSI (Relative Strength Index)\n"
        "วัดความ 'ตึง' ของราคา 0-100\n"
        "• > 70 = ตึงเกิน (Overbought) — อาจปรับลง\n"
        "• < 30 = ของถูก (Oversold) — อาจรีบาวด์\n"
        "• 40-60 = ปกติ\n\n"
        "⚡ MACD\n"
        "ดูแรง momentum เปรียบเทียบ EMA สั้น vs ยาว\n"
        "• MACD > Signal = มีแรง buy\n"
        "• MACD < Signal = แรง buy หาย\n\n"
        "✨ Golden Cross / 💀 Death Cross\n"
        "• Golden: EMA50 ตัด EMA200 ขึ้น → bullish\n"
        "• Death: EMA50 ตัด EMA200 ลง → bearish\n\n"
        "🟢 แนวรับ (Support) = ราคาที่มักจะหยุดลง\n"
        "🔴 แนวต้าน (Resistance) = ราคาที่มักจะหยุดขึ้น\n\n"
        "🟡 POC (Point of Control)\n"
        "ราคาที่มีการซื้อขายหนาแน่นที่สุด → เป็น 'แม่เหล็ก' ของราคา\n\n"
        "🛑 SL (Stop Loss) = ราคาตัดขาดทุน\n"
        "🎯 TP (Take Profit) = ราคาเป้าหมาย\n\n"
        "⚖️ R:R Ratio (Risk-Reward)\n"
        "• > 2:1 = ดี (เสี่ยง 1 ได้ 2)\n"
        "• < 1:1 = แย่ (เสี่ยงเกินกำไร)\n\n"
        "🔵 Bollinger Bands\n"
        "ช่องราคาบน/ล่าง วัด volatility\n"
        "• ทะลุ band บน = pump/overextended\n"
        "• ทะลุ band ล่าง = panic/oversold\n\n"
        "🧘 Position Trading = ถือยาว 3-12 เดือน\n"
        "สะสมเมื่อราคาลึก รอเป้าหมายใหญ่\n\n"
        "🎯 Conviction Score\n"
        "คะแนน 0-100% ความมั่นใจของระบบ"
    )


# ─── Portfolio ───────────────────────────────────

async def add_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if len(context.args) < 3:
        await update.message.reply_text("❌ รูปแบบ:\n/add <TICKER> <จำนวน> <ราคาเฉลี่ย>\nตัวอย่าง:\n/add AAPL 10 150")
        return
    ticker = context.args[0].upper()
    try:
        qty = float(context.args[1])
        avg_price = float(context.args[2])
    except ValueError:
        await update.message.reply_text("❌ จำนวนและราคาต้องเป็นตัวเลข")
        return
    portfolio = await db_portfolio.get_user(user_id)
    portfolio[ticker] = {"qty": qty, "avg_price": avg_price, "added_at": datetime.now().isoformat()}
    await db_portfolio.set_user(user_id, portfolio)
    await update.message.reply_text(f"✅ บันทึกพอร์ตสำเร็จ\n📈 {ticker}: {qty:.0f} หุ้น @ ${avg_price:.2f}")


async def edit_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if len(context.args) < 3:
        await update.message.reply_text("❌ รูปแบบ: /edit <TICKER> <จำนวนใหม่> <ราคาเฉลี่ยใหม่>")
        return
    ticker = context.args[0].upper()
    qty = float(context.args[1])
    avg_price = float(context.args[2])
    portfolio = await db_portfolio.get_user(user_id)
    if ticker not in portfolio:
        await update.message.reply_text(f"❌ ไม่พบ {ticker} ในพอร์ต ใช้ /add เพื่อเพิ่มใหม่")
        return
    portfolio[ticker] = {"qty": qty, "avg_price": avg_price, "updated_at": datetime.now().isoformat()}
    await db_portfolio.set_user(user_id, portfolio)
    await update.message.reply_text(f"✅ แก้ไข {ticker} สำเร็จ: {qty:.0f} หุ้น @ ${avg_price:.2f}")


async def del_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("❌ รูปแบบ: /del <TICKER>")
        return
    ticker = context.args[0].upper()
    portfolio = await db_portfolio.get_user(user_id)
    if ticker in portfolio:
        del portfolio[ticker]
        await db_portfolio.set_user(user_id, portfolio)
        await update.message.reply_text(f"✅ ลบ {ticker} ออกจากพอร์ตแล้ว")
    else:
        await update.message.reply_text(f"❌ ไม่พบ {ticker} ในพอร์ต")


async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    portfolio = await db_portfolio.get_user(user_id)
    if not portfolio:
        await update.message.reply_text("💼 พอร์ตว่างเปล่า\nใช้\n/add <TICKER> <จำนวน> <ราคา>\nเพื่อเพิ่ม")
        return
    lines = ["💼 พอร์ตการลงทุนของคุณ\n", "━━━━━━━━━━━━━━━━━\n"]
    total_value = 0
    total_cost = 0
    for ticker, data in portfolio.items():
        qty = data["qty"]
        avg = data["avg_price"]
        cost = qty * avg
        total_cost += cost
        current = None
        try:
            df, _ = get_stock_data(ticker, period="5d")
            if df is not None and not df.empty:
                current = float(df['Close'].iloc[-1])
        except Exception:
            pass
        if current:
            value = qty * current
            total_value += value
            total_net = (current - avg) * qty
            pnl = ((current - avg) / avg) * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"{emoji} {ticker}: {qty:.0f} หุ้น @ ${avg:.2f} → ${current:.2f}\n                  {total_net:.2f} ({pnl:+.2f}%)\n\n")
        else:
            lines.append(f"• {ticker}: {qty:.0f} หุ้น @ ${avg:.2f} (ไม่สามารถดึงราคา)")
    if total_value > 0:
        total_profit = total_value - total_cost
        total_pnl = (total_profit / total_cost) * 100
        emoji_total = "🟢 กำไรรวม" if total_pnl >= 0 else "🔴 ขาดทุนรวม"
        lines.append(f"━━━━━━━━━━━━━━━━━\n")
        lines.append(f"📊 มูลค่ารวม: ${total_value:,.2f}\n")
        lines.append(f"💰 ต้นทุนรวม: ${total_cost:,.2f}\n")
        lines.append(f"{emoji_total}: {total_profit:+.2f} ({total_pnl:+.2f}%)")
    await update.message.reply_text("".join(lines))


async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    portfolio = await db_portfolio.get_user(user_id)
    if not portfolio:
        await update.message.reply_text("💼 พอร์ตว่างเปล่า")
        return
    lines = ["📊 P&L Card", "━━━━━━━━━━━━━━━━━━━━━"]
    total_pnl_usd = 0
    for ticker, data in portfolio.items():
        qty = data["qty"]
        avg = data["avg_price"]
        try:
            df, _ = get_stock_data(ticker, period="5d")
            if df is not None and not df.empty:
                current = float(df['Close'].iloc[-1])
                pnl_usd = (current - avg) * qty
                pnl_pct = ((current - avg) / avg) * 100
                total_pnl_usd += pnl_usd
                emoji = "🟢" if pnl_usd >= 0 else "🔴"
                lines.append(f"{emoji} {ticker}: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
        except Exception:
            lines.append(f"⚪️ {ticker}: ไม่สามารถคำนวณได้")
    emoji_total = "🟢" if total_pnl_usd >= 0 else "🔴"
    lines.append(f"{emoji_total} Total Unrealized P&L: ${total_pnl_usd:+.2f}")
    await update.message.reply_text("\n".join(lines))


# ─── Watchlist ───────────────────────────────────

async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("❌ รูปแบบ: /watch <TICKER>")
        return
    ticker = context.args[0].upper()
    watchlist = await db_watchlist.get_user(user_id)
    if not isinstance(watchlist, list):
        watchlist = []
    if ticker not in watchlist:
        watchlist.append(ticker)
        await db_watchlist.set_user(user_id, watchlist)
        await update.message.reply_text(f"⭐ เพิ่ม {ticker} เข้า Watchlist แล้ว")
    else:
        await update.message.reply_text(f"⚠️ {ticker} อยู่ใน Watchlist แล้ว")


async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("❌ รูปแบบ: /unwatch <TICKER>")
        return
    ticker = context.args[0].upper()
    watchlist = await db_watchlist.get_user(user_id)
    if not isinstance(watchlist, list):
        watchlist = []
    if ticker in watchlist:
        watchlist.remove(ticker)
        await db_watchlist.set_user(user_id, watchlist)
        await update.message.reply_text(f"🗑 ลบ {ticker} ออกจาก Watchlist แล้ว")
    else:
        await update.message.reply_text(f"❌ ไม่พบ {ticker} ใน Watchlist")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    watchlist = await db_watchlist.get_user(user_id)
    if not isinstance(watchlist, list) or not watchlist:
        await update.message.reply_text("⭐ Watchlist ว่างเปล่า\nใช้ /watch <TICKER> เพื่อเพิ่ม")
        return
    lines = ["⭐ Watchlist ของคุณ", "━━━━━━━━━━━━━━━━━━━━━"]
    for ticker in watchlist:
        try:
            df, _ = get_stock_data(ticker, period="5d")
            if df is not None and not df.empty:
                price = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2])
                change = ((price - prev) / prev) * 100
                emoji = "🟢" if change >= 0 else "🔴"
                lines.append(f"• {ticker}: ${price:.2f} {emoji} {change:+.2f}%")
            else:
                lines.append(f"• {ticker}: ไม่พบข้อมูล")
        except Exception:
            lines.append(f"• {ticker}: ไม่สามารถดึงข้อมูล")
    await update.message.reply_text("\n".join(lines))


# ─── Alerts ─────────────────────────────────────

async def setalert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ รูปแบบ:\n"
            "  /setalert AAPL 200 — แจ้งเตือนที่ $200\n"
            "  /setalert AAPL +5% — แจ้งเตือนเมื่อขึ้น 5%\n"
            "  /setalert AAPL -3% — แจ้งเตือนเมื่อลง 3%"
        )
        return
    ticker = context.args[0].upper()
    alert_input = context.args[1]
    try:
        df, _ = get_stock_data(ticker, period="5d")
        if df is None or df.empty:
            await update.message.reply_text(f"❌ ไม่พบข้อมูลหุ้น {ticker}")
            return
        current_price = float(df['Close'].iloc[-1])
    except Exception:
        await update.message.reply_text(f"❌ ไม่สามารถดึงราคา {ticker}")
        return
    alerts = await db_alerts.get_user(user_id)
    if not isinstance(alerts, list):
        alerts = []
    if alert_input.endswith('%'):
        pct = float(alert_input.replace('%', ''))
        target_price = current_price * (1 + pct / 100)
        condition = f"{pct:+.0f}%"
    else:
        target_price = float(alert_input)
        condition = f"reach ${target_price:.2f}"
    alerts = [a for a in alerts if a["ticker"] != ticker]
    alerts.append({
        "ticker": ticker, "target_price": target_price, "condition": condition,
        "created_at": datetime.now().isoformat(), "triggered": False
    })
    await db_alerts.set_user(user_id, alerts)
    await update.message.reply_text(
        f"🔔 ตั้งเตือน {ticker} สำเร็จ\n"
        f"📍 ราคาปัจจุบัน: ${current_price:.2f}\n"
        f"🎯 เป้าหมาย: {condition} (${target_price:.2f})"
    )


async def myalerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    alerts = await db_alerts.get_user(user_id)
    if not isinstance(alerts, list) or not alerts:
        await update.message.reply_text("🔔 ไม่มีการแจ้งเตือน\nใช้ /setalert <TICKER> <ราคา/%> เพื่อตั้ง")
        return
    lines = ["🔔 การแจ้งเตือนของคุณ", "━━━━━━━━━━━━━━━━━━━━━"]
    for a in alerts:
        status = "✅ แจ้งแล้ว" if a.get("triggered") else "⏳ รอ"
        lines.append(f"• {a['ticker']}: {a['condition']} — {status}")
    await update.message.reply_text("\n".join(lines))


async def delalert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("❌ รูปแบบ: /delalert <TICKER>")
        return
    ticker = context.args[0].upper()
    alerts = await db_alerts.get_user(user_id)
    if not isinstance(alerts, list):
        alerts = []
    new_alerts = [a for a in alerts if a["ticker"] != ticker]
    await db_alerts.set_user(user_id, new_alerts)
    await update.message.reply_text(f"🗑 ลบการแจ้งเตือน {ticker} แล้ว")


# ─── Track Record ───────────────────────────────

async def track_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    track = await db_track.get_user(user_id)
    if not isinstance(track, list) or not track:
        await update.message.reply_text(
            "📊 Track Record\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "ยังไม่มีประวัติการวิเคราะห์\n"
            "พิมพ์ชื่อหุ้นเพื่อวิเคราะห์และบันทึก Track Record อัตโนมัติ\n\n"
            "💡 ระบบจะบันทึกทุกครั้งที่คุณวิเคราะห์หุ้น\n"
            "และติดตามว่า TP1/TP2 ถูกต้องหรือไม่"
        )
        return
    total = len(track)
    hits_tp1 = sum(1 for t in track if t.get("hit_tp1"))
    hits_tp2 = sum(1 for t in track if t.get("hit_tp2"))
    lines = [
        "📊 Track Record (ย้อนหลัง 90 วัน)",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"📈 จำนวน Plans: {total}",
        f"🎯 Hit TP1: {hits_tp1}/{total} ({hits_tp1/total*100:.0f}%)",
        f"🎯🎯 Hit TP2: {hits_tp2}/{total} ({hits_tp2/total*100:.0f}%)",
        "",
        "📝 Plans ล่าสุด:"
    ]
    for t in track[-5:]:
        status = "✅" if t.get("hit_tp1") else "⏳"
        lines.append(f"{status} {t['ticker']} — TP1: ${t['tp1']:.2f} | SL: ${t['sl']:.2f}")
    await update.message.reply_text("\n".join(lines))


# ─── Settings ────────────────────────────────────

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    settings = await db_settings.get_user(user_id)
    if not isinstance(settings, dict):
        settings = {}
    lang = settings.get("language", "th")
    tz = settings.get("timezone", "Asia/Bangkok")
    await update.message.reply_text(
        "⚙️ การตั้งค่า\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌐 ภาษา: {lang.upper()}\n"
        f"🕐 Timezone: {tz}\n\n"
        "💡 ฟีเจอร์ที่วางแผน:\n"
        "• แจ้งเตือน Morning Briefing\n"
        "• ช่วงเวลาเงียบ (Quiet Hours)\n"
        "• ความถี่ Digest\n\n"
        "(ตั้งค่าละเอียดผ่าน Web Dashboard เร็วๆ นี้)"
    )


# ─── Chart / Quick Commands ──────────────────────

async def quick_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ กรุณาระบุสัญลักษณ์\nตัวอย่าง: /quick GC=F")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"⏳ กำลังวิเคราะห์ {symbol} ด้วย Smart Entry...")
    df, error = data_fetcher.get_stock_data(symbol)
    if error:
        await update.message.reply_text(f"❌ {error}")
        return
    #chart_buf = chart_gen.generate_trading_chart(   #5555555555555555
    #    df, symbol,
    #    use_smart_entry=True,
    #    entry_price=None,
    #    tp1_pct=None,
    #    tp2_pct=None,
    #    sl_price=None
    #)
    chart_buf = chart_gen.generate_trading_chart(
        df, symbol,
        use_smart_entry=True
    )
    current_price = df['Close'].iloc[-1]
    await update.message.reply_photo(
        photo=chart_buf,
        caption=(
            f"📊 {symbol} | Apexify Smart Chart\n"
            f"ราคาปัจจุบัน: ${current_price:,.2f}\n\n"
            f"💡 Smart Entry ระบบหาจุดเข้าอัตโนมัติ\n"
            f"ใช้ /chart {symbol} เพื่อตั้งค่าเอง"
        )
    )


async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ กรุณาระบุสัญลักษณ์\nตัวอย่าง: /chart GC=F")
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
        [InlineKeyboardButton("🤖 Smart Entry (แนะนำ)", callback_data="mode_smart")],
        [InlineKeyboardButton(f"ใช้ราคาปัจจุบัน (${current_price:,.2f})", callback_data="mode_current")],
        [InlineKeyboardButton("ระบุราคาเอง", callback_data="mode_manual")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"📊 {symbol}\n"
        f"ราคาปัจจุบัน: ${current_price:,.2f}\n\n"
        f"เลือกโหมด Entry:",
        reply_markup=reply_markup
    )


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await query.edit_message_text("พิมพ์ราคา Entry ที่ต้องการ (เช่น 4555.80):")
        return ENTRY


async def generate_smart_chart(update, context, user_id):
    data = user_data[user_id]
    symbol = data['symbol']
    df = data['df']
    #chart_buf = chart_gen.generate_trading_chart(  #5555555555555555
    #    df, symbol,
    #    use_smart_entry=True,
    #    entry_price=None,
    #    tp1_pct=None,
    #    tp2_pct=None,
    #    sl_price=None
    #)
    chart_buf = chart_gen.generate_trading_chart(
        df, symbol,
        use_smart_entry=True
    )
    current_price = df['Close'].iloc[-1]
    caption = (
        f"📊 {symbol} | Apexify Smart Trading Plan\n\n"
        f"🤖 ใช้ Smart Entry (Swing Low + ATR)\n"
        f"📈 ราคาปัจจุบัน: ${current_price:,.2f}\n\n"
        f"ดูกราฟสำหรับระดับ TP/SL ที่คำนวณอัตโนมัติ"
    )
    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.message.reply_photo(photo=chart_buf, caption=caption)
    else:
        await update.message.reply_photo(photo=chart_buf, caption=caption)


async def ask_tp_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    keyboard = [
        [InlineKeyboardButton(
            f"✅ ใช้ค่าเริ่มต้น (TP1 +{DEFAULT_TP1_PCT}%, TP2 +{DEFAULT_TP2_PCT}%, SL {DEFAULT_SL_PCT}%)",
            callback_data="tp_default"
        )],
        [InlineKeyboardButton("🛠 ตั้งค่าเอง", callback_data="tp_custom")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        f"ราคา Entry: ${user_data[user_id]['entry_price']:,.2f}\n\n"
        f"เลือกการตั้งค่า TP/SL:",
        reply_markup=reply_markup
    )


async def tp_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    if query.data == "tp_default":
        await generate_manual_chart(update, context, user_id)
    elif query.data == "tp_custom":
        await query.edit_message_text(
            "พิมพ์ค่า TP1 TP2 SL คั่นด้วยช่องว่าง\n"
            "ตัวอย่าง: 5.6 22.6 -3.5"
        )
        return TP_SL


async def custom_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            f"ราคา Entry: ${entry_price:,.2f}\n\nเลือก TP/SL:",
            reply_markup=reply_markup
        )
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("❌ ราคาไม่ถูกต้อง กรุณาพิมพ์ตัวเลข")
        return ENTRY


async def custom_tp_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await update.message.reply_text("❌ รูปแบบไม่ถูกต้อง\nตัวอย่าง: 5.6 22.6 -3.5")
        return TP_SL


async def generate_manual_chart(update, context, user_id):
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
    #chart_buf = chart_gen.generate_trading_chart(  #5555555555555555
    #    df, symbol,
    #    use_smart_entry=False,
    #    entry_price=entry,
    #    tp1_pct=tp1,
    #    tp2_pct=tp2,
    #    sl_price=sl_price
    #)
    setup = TradeSetup(
        entry=entry,
        sl=sl_price,
        tp1=entry * (1 + tp1/100),
        tp2=entry * (1 + tp2/100),
        atr=(df["High"] - df["Low"]).mean()
    )
    chart_buf = chart_gen.generate_trading_chart(
        df, symbol,
        setup=setup,
        use_smart_entry=False
    )
    current = df['Close'].iloc[-1]
    change = ((current - entry) / entry) * 100
    caption = (
        f"📊 {symbol} | Manual Trading Plan\n\n"
        f"💰 Entry: ${entry:,.2f}\n"
        f"🎯 TP1: ${entry * (1 + tp1/100):,.2f} (+{tp1}%)\n"
        f"🎯 TP2: ${entry * (1 + tp2/100):,.2f} (+{tp2}%)\n"
        f"🛑 SL: ${entry * (1 + sl_pct/100):,.2f} ({sl_pct}%)\n\n"
        f"📈 ราคาปัจจุบัน: ${current:,.2f} ({change:+.2f}%)"
    )
    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.message.reply_photo(photo=chart_buf, caption=caption)
    else:
        await update.message.reply_photo(photo=chart_buf, caption=caption)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ ยกเลิกการดำเนินการ")
    return ConversationHandler.END


# ─── Text Handler (Main Flow) ────────────────────

TICKER_PATTERN = re.compile(r'^[A-Z0-9]+(\.[A-Z]+)?(-USD)?$', re.IGNORECASE)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    user_id = update.effective_user.id
    if text.startswith('/'):
        return
    if not TICKER_PATTERN.match(text):
        return
    ticker = text.split()[0]
    portfolio = await db_portfolio.get_user(user_id)
    qty = portfolio.get(ticker, {}).get("qty", 0) if isinstance(portfolio, dict) else 0
    avg = portfolio.get(ticker, {}).get("avg_price", 0) if isinstance(portfolio, dict) else 0
    status_msg = await update.message.reply_text(f"🔍 กำลังวิเคราะห์ {ticker}...")
    try:
        data = analyze_stock(ticker)
        if data is None:
            await status_msg.edit_text(f"❌ ไม่พบข้อมูลหุ้น {ticker}\nกรุณาตรวจสอบชื่อ Ticker อีกครั้ง")
            return
        # สร้างกราฟ
        #chart_buf = chart_gen.generate_trading_chart(  #5555555555555555
        #    data["df"], ticker,
        #    use_smart_entry=True,
        #    entry_price=None,
        #    tp1_pct=None,
        #    tp2_pct=None,
        #    sl_price=None
        #)
        chart_buf = chart_gen.generate_trading_chart(
            data["df"], ticker,
            setup=data.get("setup"),  # ใช้ TradeSetup จาก analyze_stock
            use_smart_entry=True
        )
        # Track record
        track = await db_track.get_user(user_id)
        if not isinstance(track, list):
            track = []
        track.append({
            "ticker": ticker, "date": datetime.now().isoformat(),
            "entry_low": data["entry_low"], "tp1": data["tp1"], "tp2": data["tp2"], "sl": data["sl"],
            "hit_tp1": False, "hit_tp2": False
        })
        track = track[-100:]
        await db_track.set_user(user_id, track)
        # ส่งกราฟก่อน
        await update.message.reply_photo(photo=chart_buf)
        # ส่งรายงานข้อความ
        report = format_report(data, portfolio_qty=qty, portfolio_avg=avg)
        await status_msg.edit_text(report)
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ {ticker}")


# ─── Error Handler ─────────────────────────────

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("❌ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง")


# ─── Background Alert Checker ───────────────────

async def check_alerts(application):
    while True:
        try:
            all_alerts = await db_alerts.load()
            for user_id_str, alerts in all_alerts.items():
                if not isinstance(alerts, list):
                    continue
                user_id = int(user_id_str)
                updated = False
                for alert in alerts:
                    if alert.get("triggered"):
                        continue
                    ticker = alert["ticker"]
                    target = alert["target_price"]
                    try:
                        df, _ = get_stock_data(ticker, period="5d")
                        if df is None or df.empty:
                            continue
                        current = float(df['Close'].iloc[-1])
                    except Exception:
                        continue
                    if alert["condition"].startswith('+'):
                        if current >= target:
                            alert["triggered"] = True
                            updated = True
                            await application.bot.send_message(
                                chat_id=user_id,
                                text=f"🔔 Alert Triggered!\n📈 {ticker} ถึงเป้าหมาย {alert['condition']}\n💵 ราคาปัจจุบัน: ${current:.2f}"
                            )
                    elif alert["condition"].startswith('-'):
                        if current <= target:
                            alert["triggered"] = True
                            updated = True
                            await application.bot.send_message(
                                chat_id=user_id,
                                text=f"🔔 Alert Triggered!\n📉 {ticker} ถึงเป้าหมาย {alert['condition']}\n💵 ราคาปัจจุบัน: ${current:.2f}"
                            )
                    else:
                        if (target >= current * 0.99) and (target <= current * 1.01):
                            alert["triggered"] = True
                            updated = True
                            await application.bot.send_message(
                                chat_id=user_id,
                                text=f"🔔 Alert Triggered!\n🎯 {ticker} ถึงราคาเป้าหมาย ${target:.2f}\n💵 ราคาปัจจุบัน: ${current:.2f}"
                            )
                if updated:
                    await db_alerts.set_user(user_id, alerts)
        except Exception as e:
            logger.error(f"Alert checker error: {e}")
        await asyncio.sleep(300)


# ─── FastAPI ─────────────────────────────────────

app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "bot": "Apexify Unified Bot",
        "mode": "polling",
        "features": [
            "smart-entry-pivot-sr", "chart-generation", "position-plan",
            "POC", "3-timeframe-trend", "portfolio", "watchlist",
            "alerts", "track_record", "glossary"
        ],
        "version": "6.1-improved"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "polling"}


# ─── Main ────────────────────────────────────────

async def run_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    chart_conv = ConversationHandler(
        entry_points=[CommandHandler('chart', chart_command)],
        states={
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_entry)],
            TP_SL: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_tp_sl)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(CommandHandler('start', start_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('manual', help_command))
    application.add_handler(CommandHandler('health', health_command))
    application.add_handler(CommandHandler('glossary', glossary_command))
    application.add_handler(CommandHandler('terms', glossary_command))

    application.add_handler(CommandHandler('add', add_portfolio))
    application.add_handler(CommandHandler('edit', edit_portfolio))
    application.add_handler(CommandHandler('del', del_portfolio))
    application.add_handler(CommandHandler('portfolio', show_portfolio))
    application.add_handler(CommandHandler('port', show_portfolio))
    application.add_handler(CommandHandler('pnl', pnl_command))

    application.add_handler(CommandHandler('watch', watch_command))
    application.add_handler(CommandHandler('unwatch', unwatch_command))
    application.add_handler(CommandHandler('watchlist', watchlist_command))

    application.add_handler(CommandHandler('setalert', setalert_command))
    application.add_handler(CommandHandler('myalerts', myalerts_command))
    application.add_handler(CommandHandler('delalert', delalert_command))

    application.add_handler(CommandHandler('track', track_command))
    application.add_handler(CommandHandler('settings', settings_command))

    application.add_handler(CommandHandler('quick', quick_chart))
    application.add_handler(chart_conv)
    application.add_handler(CallbackQueryHandler(mode_callback, pattern='^mode_'))
    application.add_handler(CallbackQueryHandler(tp_callback, pattern='^tp_'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)

    logger.info("🤖 Starting Apexify Bot v6.1 (Improved) in polling mode...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)

    asyncio.create_task(check_alerts(application))

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Shutting down bot...")
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set! Please set the environment variable.")
        exit(1)
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    asyncio.run(run_bot())
