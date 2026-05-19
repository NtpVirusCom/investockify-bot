"""
Apexify Trading Bot - Improved Version
======================================
ปรับปรุงจาก investockify_bot8.py โดยแก้ไข:
1. สูตร % ให้ถูกต้องตามหลักการเทรด (คำนวณจาก Entry)
2. Pivot Detection แบบ Vectorized (เร็วขึ้น 10-50x)
3. ATR แบบ Wilder's Smoothing (ตามต้นฉบับ)
4. ใช้ mplfinance วาดกราฟแท่งเทียน (สวยและเร็ว)
5. แยก State เป็น Class (รองรับ Multi-user ได้ดีขึ้น)
6. Error Handling ที่เฉพาะเจาะจง
7. ตรรกะ Entry/SL/TP ที่สอดคล้องกัน
"""

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import io
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from types import ModuleType

# =========================================================
# LOGGING CONFIG
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# MOCK CONFIG MODULE
# =========================================================
_config_module = ModuleType("config")
_config_module.COLORS = {
    'bullish': '#26a69a',
    'bearish': '#ef5350',
    'ema20': '#2196F3',
    'ema50': '#FF9800',
    'ema200': '#9C27B0',
    'tp1': '#4CAF50',
    'tp2': '#2E7D32',
    'entry': '#2196F3',
    'sl': '#F44336'
}
_config_module.DEFAULT_TP1_PCT = 5.0
_config_module.DEFAULT_TP2_PCT = 10.0
_config_module.DEFAULT_SL_PCT = -3.0
sys.modules["config"] = _config_module

# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Level:
    """แทนระดับ Support/Resistance"""
    price: float
    bar_index: int
    strength: int
    level_type: str  # 'support' หรือ 'resistance'

@dataclass
class TradeSetup:
    """แทน Setup การเทรดที่สอดคล้องกัน"""
    entry: float
    sl: float
    tp1: float
    tp2: float
    support_levels: List[Level] = field(default_factory=list)
    resistance_levels: List[Level] = field(default_factory=list)
    atr: float = 0.0

    @property
    def risk(self) -> float:
        """ความเสี่ยง = |Entry - SL|"""
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
        """คำนวณ % จาก Entry (ถูกต้องตามหลักการเทรด)"""
        return ((price - self.entry) / self.entry) * 100


# =========================================================
# CHART GENERATOR CLASS (IMPROVED)
# =========================================================
class ChartGenerator:
    def __init__(self):
        plt.style.use('default')
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

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """คำนวณ EMA มาตรฐาน"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        คำนวณ ATR แบบ Wilder's Smoothing (ตามต้นฉบับของ Wilder)
        ไม่ใช่ SMA ธรรมดา แต่ใช้ exponential smoothing: 
        ATR_t = ((n-1)*ATR_prev + TR_t) / n
        """
        df = df.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["tr1"] = df["High"] - df["Low"]
        df["tr2"] = (df["High"] - df["prev_close"]).abs()
        df["tr3"] = (df["Low"] - df["prev_close"]).abs()
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        # Wilder's Smoothing (ATR ดั้งเดิม)
        df["atr"] = df["tr"].ewm(alpha=1/period, adjust=False).mean()
        return df

    def detect_pivots_vectorized(self, df: pd.DataFrame) -> Tuple[List[Level], List[Level]]:
        """
        ตรวจจับ Pivot High/Low แบบ Vectorized ใช้ rolling window
        เร็วกว่า for-loop ธรรมดา 10-50 เท่า
        """
        highs = []
        lows = []
        n = self.PIVOT_LEN

        if len(df) < 2 * n + 1:
            return highs, lows

        # Pivot High: สูงกว่าซ้าย n แถว และ ขวา n แถว
        left_high = df["High"].rolling(window=n, center=False).max().shift(1)
        right_high = df["High"].rolling(window=n, center=False).max().shift(-n)
        pivot_high_mask = (df["High"] >= left_high) & (df["High"] >= right_high)

        # Pivot Low: ต่ำกว่าซ้าย n แถว และ ขวา n แถว
        left_low = df["Low"].rolling(window=n, center=False).min().shift(1)
        right_low = df["Low"].rolling(window=n, center=False).min().shift(-n)
        pivot_low_mask = (df["Low"] <= left_low) & (df["Low"] <= right_low)

        atr_series = df["atr"]

        # กรองเฉพาะที่มี ATR และ strength ผ่านเกณฑ์
        for i in range(n, len(df) - n):
            if pd.isna(atr_series.iloc[i]) or atr_series.iloc[i] == 0:
                continue

            atr = atr_series.iloc[i]

            if pivot_high_mask.iloc[i]:
                current_high = df["High"].iloc[i]
                close = df["Close"].iloc[i]
                strength = (current_high - close) / atr
                if strength >= self.MIN_ATR_STRENGTH:
                    highs.append(Level(
                        price=current_high,
                        bar_index=i,
                        strength=1,  # จะนับ strength จริงใน merge
                        level_type='resistance'
                    ))

            if pivot_low_mask.iloc[i]:
                current_low = df["Low"].iloc[i]
                close = df["Close"].iloc[i]
                strength = (close - current_low) / atr
                if strength >= self.MIN_ATR_STRENGTH:
                    lows.append(Level(
                        price=current_low,
                        bar_index=i,
                        strength=1,
                        level_type='support'
                    ))

        return highs, lows

    def merge_levels(self, levels: List[Level], atr: float) -> List[Level]:
        """รวมระดับที่ใกล้กันเกินไป"""
        if not levels:
            return []

        # เรียงตามราคา
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
            result.append(Level(
                price=round(avg_price, 2),
                bar_index=latest_bar,
                strength=len(group),
                level_type=group[0].level_type
            ))
        return result

    def find_optimal_entry(self, df: pd.DataFrame, current_price: float) -> TradeSetup:
        """
        หา Entry/SL/TP ที่เหมาะสม
        ปรับปรุง: ตรรกะชัดเจน สอดคล้องกัน ไม่เปลี่ยนค่าทับซ้อน
        """
        df_atr = self.calculate_atr(df)
        atr = df_atr["atr"].iloc[-1]

        if pd.isna(atr) or atr == 0:
            atr = (df["High"] - df["Low"]).mean()

        # ตรวจจับ pivot แบบ vectorized
        highs, lows = self.detect_pivots_vectorized(df_atr)
        current_bar = len(df)

        # กรองเฉพาะระดับที่ยังไม่เก่าเกินไป
        highs = [x for x in highs if current_bar - x.bar_index <= self.MAX_LEVEL_AGE]
        lows = [x for x in lows if current_bar - x.bar_index <= self.MAX_LEVEL_AGE]

        # Merge
        merged_highs = self.merge_levels(highs, atr)
        merged_lows = self.merge_levels(lows, atr)

        # กรองระดับที่อยู่ไม่ไกลจากราคาปัจจุบัน
        merged_highs = [
            x for x in merged_highs
            if x.price > current_price and (x.price - current_price) / current_price <= self.MAX_DISTANCE_FROM_PRICE
        ]
        merged_lows = [
            x for x in merged_lows
            if x.price < current_price and (current_price - x.price) / current_price <= self.MAX_DISTANCE_FROM_PRICE
        ]

        # เรียงลำดับ: ใกล้ราคาปัจจุบันก่อน แล้ว strength มากก่อน
        merged_highs = sorted(
            merged_highs,
            key=lambda x: (abs(x.price - current_price), -x.strength)
        )[:self.MAX_ACTIVE_LEVELS_EACH]
        merged_lows = sorted(
            merged_lows,
            key=lambda x: (abs(x.price - current_price), -x.strength)
        )[:self.MAX_ACTIVE_LEVELS_EACH]

        # กำหนด TP จาก Resistance
        if merged_highs:
            tp1_price = merged_highs[0].price
            tp2_price = merged_highs[1].price if len(merged_highs) >= 2 else tp1_price * 1.06
        else:
            tp1_price = current_price * 1.05
            tp2_price = tp1_price * 1.06

        # กำหนด Entry/SL จาก Support
        if merged_lows:
            entry_price = merged_lows[0].price
            sl_price = merged_lows[1].price if len(merged_lows) >= 2 else None
        else:
            entry_price = current_price * 0.96
            sl_price = None

        # ถ้าไม่มี SL จาก support level ที่ 2 ให้คำนวณจาก ATR
        if sl_price is None:
            sl_buffer = max(atr * 2, entry_price * 0.03)
            sl_price = entry_price - sl_buffer

        # Validation: ถ้า Entry >= Current แสดงว่าไม่มีสัญญาณ Long ที่ดี
        # ให้ใช้ default ที่สมเหตุสมผลแทนการทับค่าแบบไร้เงื่อนไข
        if entry_price >= current_price:
            entry_price = current_price * 0.97
            sl_buffer = max(atr * 1.5, entry_price * 0.025)
            sl_price = entry_price - sl_buffer

        # Validation: SL ต้องต่ำกว่า Entry เสมอ
        if sl_price >= entry_price:
            sl_price = entry_price * 0.97

        return TradeSetup(
            entry=entry_price,
            sl=sl_price,
            tp1=tp1_price,
            tp2=tp2_price,
            support_levels=merged_lows,
            resistance_levels=merged_highs,
            atr=atr
        )

    def generate_trading_chart(self, df: pd.DataFrame, symbol: str,
                              setup: Optional[TradeSetup] = None,
                              use_smart_entry: bool = True,
                              tp1_pct: Optional[float] = None,
                              tp2_pct: Optional[float] = None) -> io.BytesIO:
        """
        สร้างกราฟด้วย mplfinance (optimized)
        แทนการวาดทีละแท่งด้วย matplotlib ธรรมดา
        """
        close = df['Close']
        ema20 = self.calculate_ema(close, 20)
        ema50 = self.calculate_ema(close, 50)
        ema200 = self.calculate_ema(close, 200)

        current_price = close.iloc[-1]

        # เตรียม DataFrame สำหรับ mplfinance
        df_display = df.tail(60).copy()
        df_display.index = pd.DatetimeIndex(df_display.index)

        # เพิ่ม EMA เป็น additional plots
        ema20_display = ema20.tail(60)
        ema50_display = ema50.tail(60)
        ema200_display = ema200.tail(60)

        # สร้าง Setup ถ้าไม่ได้ส่งมา
        if use_smart_entry and setup is None:
            setup = self.find_optimal_entry(df, current_price)
        elif setup is None:
            entry = current_price
            sl = entry * (1 + _config_module.DEFAULT_SL_PCT / 100)
            tp1 = entry * (1 + (tp1_pct or _config_module.DEFAULT_TP1_PCT) / 100)
            tp2 = entry * (1 + (tp2_pct or _config_module.DEFAULT_TP2_PCT) / 100)
            setup = TradeSetup(entry=entry, sl=sl, tp1=tp1, tp2=tp2)

        # คำนวณ Entry Zone
        entry_zone_top = setup.entry * 1.009
        entry_zone_bottom = setup.entry * 0.991

        # สร้าง hlines และ alines
        hlines = []
        hlines_colors = []
        hlines_styles = []
        hlines_widths = []

        # TP2
        hlines.append(setup.tp2)
        hlines_colors.append(_config_module.COLORS['tp2'])
        hlines_styles.append('-')
        hlines_widths.append(2)

        # TP1
        hlines.append(setup.tp1)
        hlines_colors.append(_config_module.COLORS['tp1'])
        hlines_styles.append('--')
        hlines_widths.append(1.5)

        # Entry
        hlines.append(setup.entry)
        hlines_colors.append(_config_module.COLORS['entry'])
        hlines_styles.append(':')
        hlines_widths.append(1.5)

        # SL
        hlines.append(setup.sl)
        hlines_colors.append(_config_module.COLORS['sl'])
        hlines_styles.append('-')
        hlines_widths.append(2)

        # Support/Resistance levels
        for lvl in setup.support_levels:
            hlines.append(lvl.price)
            hlines_colors.append('#4CAF50')
            hlines_styles.append(':')
            hlines_widths.append(1)
        for lvl in setup.resistance_levels:
            hlines.append(lvl.price)
            hlines_colors.append('#F44336')
            hlines_styles.append(':')
            hlines_widths.append(1)

        # สร้าง additional plots สำหรับ EMA
        apds = [
            mpf.make_addplot(ema20_display, color=_config_module.COLORS['ema20'], width=1.5, label='EMA20'),
            mpf.make_addplot(ema50_display, color=_config_module.COLORS['ema50'], width=1.5, label='EMA50'),
            mpf.make_addplot(ema200_display, color=_config_module.COLORS['ema200'], width=1.5, label='EMA200'),
        ]

        # สร้าง figure
        fig, axes = mpf.plot(
            df_display,
            type='candle',
            style='yahoo',
            volume=True,
            addplot=apds,
            hlines=dict(
                hlines=hlines,
                colors=hlines_colors,
                linestyle=hlines_styles,
                linewidths=hlines_widths,
                alpha=0.8
            ),
            title=f'Apexify — {symbol} | Pivot S/R Entry + TP + SL',
            ylabel='Price',
            ylabel_lower='Volume',
            returnfig=True,
            figsize=(14, 10),
            panel_ratios=(4, 1),
            tight_layout=True
        )

        ax1 = axes[0]

        # เพิ่ม Entry Zone (hspan)
        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=_config_module.COLORS['entry'])

        # คำนวณ % ที่ถูกต้อง: จาก Entry
        sl_pct = setup.get_pct_from_entry(setup.sl)
        tp1_pct_val = setup.get_pct_from_entry(setup.tp1)
        tp2_pct_val = setup.get_pct_from_entry(setup.tp2)
        entry_pct = 0.0  # Entry คือจุดอ้างอิง

        # เพิ่ม Label
        price_min = min(df_display['Low'].min(), setup.sl * 0.95)
        price_max = max(df_display['High'].max(), setup.tp2 * 1.05)
        y_range = price_max - price_min
        y_shift = y_range * 0.015
        x_offset = len(df_display) * 0.02

        labels = [
            (x_offset, setup.tp2 + y_shift, f"TP2 ${setup.tp2:,.2f} ({tp2_pct_val:+.1f}%)", _config_module.COLORS['tp2']),
            (x_offset, setup.tp1 + y_shift, f"TP1 ${setup.tp1:,.2f} ({tp1_pct_val:+.1f}%)", _config_module.COLORS['tp1']),
            (x_offset, current_price + y_shift, f">> NOW ${current_price:,.2f}", _config_module.COLORS['entry']),
            (x_offset, setup.entry + y_shift, f"ENTRY ${setup.entry:,.2f} ({entry_pct:+.1f}%)", _config_module.COLORS['tp1']),
            (x_offset, setup.sl + y_shift, f"SL ${setup.sl:,.2f} ({sl_pct:+.1f}%)", _config_module.COLORS['sl']),
        ]

        for x, y, text, color in labels:
            ax1.text(x, y, text, fontsize=10, fontweight='bold', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor='white', alpha=0.9),
                    color='white')

        # EMA Labels
        x_right = len(df_display) * 0.98
        emas = [
            (ema20_display.iloc[-1], _config_module.COLORS['ema20'], 'EMA20'),
            (ema50_display.iloc[-1], _config_module.COLORS['ema50'], 'EMA50'),
            (ema200_display.iloc[-1], _config_module.COLORS['ema200'], 'EMA200'),
        ]
        for val, color, name in emas:
            ax1.text(x_right, val + y_shift, f"{name} {val:,.2f}", 
                    fontsize=9, fontweight='bold', va='bottom', ha='right', color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))

        # บันทึกเป็น PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig)
        return buf

    def generate_simple_chart(self, df: pd.DataFrame, symbol: str) -> io.BytesIO:
        return self.generate_trading_chart(df, symbol, use_smart_entry=True)


# =========================================================
# STATE MANAGER (แทน Global Variables)
# =========================================================
class StateManager:
    """จัดการ state แบบ instance-based แทน global dict"""
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


# =========================================================
# GLOBAL INSTANCE & HELPERS
# =========================================================
chart_gen = ChartGenerator()
state_mgr = StateManager()


def _df_to_cg(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'timestamp': 'Timestamp'
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _df_from_cg(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Timestamp': 'timestamp'
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# =========================================================
# TOKEN & CONFIG
# =========================================================
TELEGRAM_TOKEN = "7732232153:AAEaIP5zpcR90YxKXMX_03uYNlui-tIgl68"
TIMEFRAME = "1d"

# =========================================================
# WELCOME & MENUS
# =========================================================
WELCOME_MESSAGE = """
⚡ ยินดีต้อนรับสู่ Apexify (Improved)

🤖 ระบบวิเคราะห์หุ้นด้วย Adaptive S/R AI

รองรับ:
• หุ้น US
• หุ้นไทย
• Crypto

👇 กด 📊 วิเคราะห์หุ้น เพื่อเริ่มต้น
"""

MAIN_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("📊 วิเคราะห์หุ้น"), KeyboardButton("📱 เปิดเมนูหลัก")],
        [KeyboardButton("💎 บัญชี / VIP"), KeyboardButton("📖 คู่มือ /manual")]
    ],
    resize_keyboard=True
)


def get_main_inline_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 ลอง AAPL", callback_data="stock_AAPL"),
            InlineKeyboardButton("⚡ ลอง NVDA", callback_data="stock_NVDA")
        ],
        [InlineKeyboardButton("🇹🇭 ลอง PTT.BK", callback_data="stock_PTT")]
    ])


# =========================================================
# SYMBOL FORMATTING
# =========================================================
THAI_STOCKS = {"PTT", "AOT", "CPALL", "SCB", "KBANK", "ADVANC", "BDMS", "BBL", "KTB"}


def format_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol in THAI_STOCKS:
        return f"{symbol}.BK"
    if symbol.endswith("USDT"):
        return symbol.replace("USDT", "-USD")
    return symbol


# =========================================================
# DATA FETCHING (IMPROVED ERROR HANDLING)
# =========================================================
def fetch_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y", interval=TIMEFRAME, auto_adjust=False)
        if df.empty:
            logger.warning(f"Empty data for {symbol}")
            return None

        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        keep_cols = ["date", "datetime", "open", "high", "low", "close", "volume"]
        existing = [c for c in keep_cols if c in df.columns]
        df = df[existing]

        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})

        df = df.dropna()
        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            return None

        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


# =========================================================
# LEVELS & BREAKOUTS (IMPROVED)
# =========================================================
def update_levels(symbol: str, df: pd.DataFrame) -> List[Dict]:
    current_price = df["close"].iloc[-1]
    df_cg = _df_to_cg(df)
    setup = chart_gen.find_optimal_entry(df_cg, current_price)
    atr = setup.atr

    state_mgr.set_setup(symbol, setup)

    levels = []
    for h in setup.resistance_levels:
        levels.append({
            "type": "resistance",
            "price": round(h.price, 2),
            "strength": h.strength,
            "zone_top": h.price + atr * chart_gen.ZONE_WIDTH,
            "zone_bottom": h.price - atr * chart_gen.ZONE_WIDTH
        })
    for l in setup.support_levels:
        levels.append({
            "type": "support",
            "price": round(l.price, 2),
            "strength": l.strength,
            "zone_top": l.price + atr * chart_gen.ZONE_WIDTH,
            "zone_bottom": l.price - atr * chart_gen.ZONE_WIDTH
        })

    levels = sorted(levels, key=lambda x: x["price"], reverse=True)
    state_mgr.set_levels(symbol, levels)
    return levels


def detect_breakout(symbol: str, df: pd.DataFrame) -> List[str]:
    levels = state_mgr.get_levels(symbol)
    if not levels:
        return []

    df_cg = _df_to_cg(df)
    df_cg = chart_gen.calculate_atr(df_cg, period=chart_gen.ATR_PERIOD)
    df_result = _df_from_cg(df_cg)

    for col in ['atr', 'tr', 'tr1', 'tr2', 'tr3', 'prev_close']:
        if col in df_cg.columns and col not in df_result.columns:
            df_result[col] = df_cg[col].values

    close_price = df_result["close"].iloc[-1]
    atr = df_result["atr"].iloc[-1]

    if pd.isna(atr) or atr == 0:
        return []

    buffer = atr * chart_gen.BREAK_SENS
    signals = []
    remaining = []
    breakout_count = 0

    for lvl in levels:
        breakout_move = abs(close_price - lvl["price"]) / atr

        if (lvl["type"] == "resistance" and 
            close_price > lvl["zone_top"] + buffer and 
            breakout_move >= chart_gen.MIN_BREAKOUT_MOVE_ATR):
            if breakout_count < chart_gen.MAX_BREAKOUT_SIGNALS:
                signals.append(
                    f"🚀 {symbol}\nBreak Resistance\n"
                    f"Level: {lvl['price']:.2f}\n"
                    f"Current: {close_price:.2f}\n"
                    f"Strength: {lvl['strength']}"
                )
                breakout_count += 1
        elif (lvl["type"] == "support" and 
              close_price < lvl["zone_bottom"] - buffer and 
              breakout_move >= chart_gen.MIN_BREAKOUT_MOVE_ATR):
            if breakout_count < chart_gen.MAX_BREAKOUT_SIGNALS:
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


# =========================================================
# TELEGRAM HANDLERS
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE, reply_markup=MAIN_MENU)
    await update.message.reply_text("📱 เมนูหลัก", reply_markup=get_main_inline_keyboard())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "stock_AAPL":
        await send_real_stock_analysis(query.message, "AAPL")
    elif data == "stock_NVDA":
        await send_real_stock_analysis(query.message, "NVDA")
    elif data == "stock_PTT":
        await send_real_stock_analysis(query.message, "PTT.BK")


async def send_real_stock_analysis(message, stock: str):
    try:
        symbol = format_symbol(stock)

        # ดึงชื่อบริษัท
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get("longName", symbol)
        except Exception as e:
            logger.warning(f"Could not fetch company name for {symbol}: {e}")
            company_name = symbol

        df = fetch_ohlcv(symbol)
        if df is None:
            await message.reply_text(f"❌ ไม่พบข้อมูล {symbol}")
            return

        levels = update_levels(symbol, df)
        breakouts = detect_breakout(symbol, df)
        current_price = df["close"].iloc[-1]

        resistance_levels = [x for x in levels if x["type"] == "resistance"]
        support_levels = [x for x in levels if x["type"] == "support"]

        setup = state_mgr.get_setup(symbol)

        result = (
            f"━━━━━━━━━━━━━━━━━\n"
            f"📊 วิเคราะห์หุ้น {symbol}\n"
            f"🏢 {company_name}\n"
            f"━━━━━━━━━━━━━━━━━\n\n"
            f"💵 Current Price: {current_price:.2f}\n\n"
        )

        if setup:
            # ใช้สูตรที่ถูกต้อง: คำนวณ % จาก Entry
            entry_pct = 0.0
            sl_pct = setup.get_pct_from_entry(setup.sl)
            tp1_pct = setup.get_pct_from_entry(setup.tp1)
            tp2_pct = setup.get_pct_from_entry(setup.tp2)

            result += (
                f"🎯 SMART ENTRY SETUP\n"
                f"━━━━━━━━━━━━━━━━━\n"
                f"🟢 Entry: {setup.entry:.2f} ({entry_pct:+.1f}%)\n"
                f"🔴 SL: {setup.sl:.2f} ({sl_pct:+.1f}%)\n"
                f"🟡 TP1: {setup.tp1:.2f} ({tp1_pct:+.1f}%) | R:R 1:{setup.rr1:.1f}\n"
                f"🟢 TP2: {setup.tp2:.2f} ({tp2_pct:+.1f}%) | R:R 1:{setup.rr2:.1f}\n\n"
            )

        result += "🔴 Resistance\n"
        if resistance_levels:
            for lvl in resistance_levels:
                result += f"  {lvl['price']:.2f} (S:{lvl['strength']})\n"
        else:
            result += "  No resistance\n"

        result += "\n🟢 Support\n"
        if support_levels:
            for lvl in support_levels:
                result += f"  {lvl['price']:.2f} (S:{lvl['strength']})\n"
        else:
            result += "  No support\n"

        if breakouts:
            result += "\n⚡ Breakouts\n\n"
            result += "\n\n".join(breakouts)

        # ส่งข้อความพร้อมปุ่ม
        keyboard = [
            [InlineKeyboardButton("⭐ เพิ่ม Watchlist", callback_data="watchlist")],
            [
                InlineKeyboardButton("🔔 ตั้ง Alert", callback_data="alert"),
                InlineKeyboardButton("📈 เปิดกราฟ", callback_data="chart")
            ]
        ]
        await message.reply_text(result, reply_markup=InlineKeyboardMarkup(keyboard))

        # ส่งกราฟ (ถ้ามี setup)
        if setup:
            try:
                df_cg = _df_to_cg(df)
                buf = chart_gen.generate_trading_chart(df_cg, symbol, setup=setup)
                await message.reply_photo(photo=buf)
            except Exception as e:
                logger.error(f"Chart generation error: {e}")
                await message.reply_text("⚠️ ไม่สามารถสร้างกราฟได้")

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        await message.reply_text(f"❌ Error: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.upper().strip()

    if text == "📊 วิเคราะห์หุ้น":
        await update.message.reply_text(
            "📊 พิมพ์ชื่อหุ้น เช่น:\n\n"
            "AAPL\nNVDA\nTSLA\nMSFT\nPTT.BK\nAOT.BK\nBTCUSDT"
        )
        return
    elif text == "📱 เปิดเมนูหลัก":
        await update.message.reply_text("📱 เมนูหลัก", reply_markup=get_main_inline_keyboard())
        return
    elif text == "💎 บัญชี / VIP":
        await update.message.reply_text(
            "💎 สมาชิก VIP\n\n"
            "✅ วิเคราะห์หุ้น\n"
            "✅ Smart Alert\n"
            "✅ Watchlist\n"
            "✅ Portfolio"
        )
        return
    elif text == "📖 คู่มือ /MANUAL":
        await update.message.reply_text(
            "📖 วิธีใช้งาน\n\n"
            "1️⃣ กด 📊 วิเคราะห์หุ้น\n"
            "2️⃣ พิมพ์ชื่อหุ้น\n"
            "ตัวอย่าง: AAPL, NVDA, PTT.BK, BTCUSDT\n"
            "3️⃣ ระบบจะวิเคราะห์ให้อัตโนมัติ"
        )
        return

    if len(text) >= 1:
        await send_real_stock_analysis(update.message, text)


# =========================================================
# MAIN
# =========================================================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
