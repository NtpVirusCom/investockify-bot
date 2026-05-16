import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import io
from config import COLORS, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT, DEFAULT_SL_PCT

class ChartGenerator:
    def __init__(self):
        plt.style.use('default')

    def calculate_ema(self, data: pd.Series, period: int):
        return data.ewm(span=period, adjust=False).mean()

    def find_support_resistance(self, df: pd.DataFrame, window: int = 10):
        """
        หาแนวรับและแนวต้านจาก Pivot Points
        """
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()

        # แนวรับ = จุดต่ำสุดที่เกิดขึ้นหลายครั้ง
        # แนวต้าน = จุดสูงสุดที่เกิดขึ้นหลายครั้ง
        support = lows.quantile(0.25)  # 25th percentile of lows
        resistance = highs.quantile(0.75)  # 75th percentile of highs

        return support, resistance

    def find_poc(self, df: pd.DataFrame, bins: int = 50):
        """
        หา POC (Point of Control) = ราคาที่มี Volume สูงสุด
        """
        price_volume = []
        for idx, row in df.iterrows():
            mid_price = (row['High'] + row['Low']) / 2
            price_volume.append((mid_price, row['Volume']))

        # คำนวณ weighted average price by volume
        total_vol = sum(v for p, v in price_volume)
        if total_vol > 0:
            poc = sum(p * v for p, v in price_volume) / total_vol
        else:
            poc = df['Close'].mean()

        return poc

    def find_optimal_entry(self, df: pd.DataFrame, ema200: pd.Series, current_price: float):
        """
        Apexify TRUE Logic (จาก Report จริง):

        Entry Zone:
        - ใช้แนวรับ (Support) เป็น baseline
        - ปรับลงมาเล็กน้อย (-1-2%)
        - ไม่ใช่ Swing Low เก่า แต่เป็น "โซนที่น่าซื้อถ้าย่อ"

        SL:
        - ต่ำกว่า POC (Point of Control)
        - หรือต่ำกว่า Swing Low ที่แท้จริงในช่วง 4-6 สัปดาห์

        TP1:
        - แนวต้านที่มีอยู่จริง (Resistance)
        - ไม่ใช่คำนวณจาก Risk:Reward

        TP2:
        - แนวต้านถัดไป หรือ +10-15% จาก Entry
        """

        # หาแนวรับและแนวต้าน
        support, resistance = self.find_support_resistance(df.tail(60))

        # หา POC
        poc = self.find_poc(df.tail(60))

        # หา Swing Low ที่แท้จริงในช่วง 4-6 สัปดาห์
        recent_df = df.tail(30)
        swing_lows = []
        lows = recent_df['Low'].values

        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append(lows[i])

        # SL = ต่ำกว่า POC หรือ Swing Low ที่แท้จริง (เลือกอันที่สูงกว่า)
        if swing_lows:
            true_swing_low = min(swing_lows)
            sl_price = max(poc * 0.995, true_swing_low * 0.99)
        else:
            sl_price = poc * 0.99

        # Entry = แนวรับที่ปรับลงมาเล็กน้อย
        entry_price = support * 0.99

        # ถ้า Entry สูงกว่าราคาปัจจุบัน ปรับให้ต่ำกว่า
        if entry_price >= current_price:
            entry_price = current_price * 0.96

        # ถ้า Entry ต่ำกว่า SL มากเกินไป ปรับให้เหมาะสม
        if entry_price < sl_price * 1.05:
            entry_price = sl_price * 1.05

        entry_date = df.index[-1]

        return entry_price, entry_date, sl_price, support, resistance, poc

    def calculate_atr(self, df: pd.DataFrame, period: int = 14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def generate_trading_chart(self, df: pd.DataFrame, symbol: str,
                              entry_price: float = None,
                              tp1_price: float = None,
                              tp2_price: float = None,
                              sl_price: float = None,
                              use_smart_entry: bool = True):

        close = df['Close']
        ema20 = self.calculate_ema(close, 20)
        ema50 = self.calculate_ema(close, 50)
        ema200 = self.calculate_ema(close, 200)

        ema20_last = ema20.iloc[-1]
        ema50_last = ema50.iloc[-1]
        ema200_last = ema200.iloc[-1]

        current_price = close.iloc[-1]

        # Apexify: แสดงกราฟ 3 เดือน (90 วัน)
        df_display = df.tail(60).copy()
        ema20_display = ema20.tail(60)
        ema50_display = ema50.tail(60)
        ema200_display = ema200.tail(60)

        df = df_display

        # === กำหนด Entry, SL, TP ตาม Apexify ===
        if use_smart_entry and entry_price is None:
            entry_price, entry_date, auto_sl, support, resistance, poc = self.find_optimal_entry(df, ema200, current_price)

            if sl_price is None:
                sl_price = auto_sl

            # TP1 = แนวต้าน (Resistance)
            if tp1_price is None:
                tp1_price = resistance

            # TP2 = แนวต้านถัดไป หรือ +10-15% จาก Entry
            if tp2_price is None:
                tp2_price = tp1_price * 1.06  # +6% จาก TP1
        else:
            if entry_price is None:
                entry_price = current_price

            if sl_price is None:
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

            if tp1_price is None:
                tp1_price = entry_price * (1 + (DEFAULT_TP1_PCT if DEFAULT_TP1_PCT is not None else 5.0) / 100)

            if tp2_price is None:
                tp2_price = entry_price * (1 + (DEFAULT_TP2_PCT if DEFAULT_TP2_PCT is not None else 10.0) / 100)

        # คำนวณเปอร์เซ็นต์
        sl_pct = ((sl_price - entry_price) / entry_price) * 100
        tp1_pct_display = ((tp1_price - entry_price) / entry_price) * 100
        tp2_pct_display = ((tp2_price - entry_price) / entry_price) * 100

        # Entry Zone
        entry_zone_top = entry_price * 1.009
        entry_zone_bottom = entry_price * 0.991

        # === สร้างกราฟ ===
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [4, 1]},
                                       sharex=True)

        # --- Candlestick ---
        for i, (idx, row) in enumerate(df.iterrows()):
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

        # --- EMA Lines ---
        x_range = range(len(df))
        ax1.plot(x_range, ema20_display.values, color=COLORS['ema20'], linewidth=2, label='EMA 20', alpha=0.8)
        ax1.plot(x_range, ema50_display.values, color=COLORS['ema50'], linewidth=2, label='EMA 50', alpha=0.8)
        ax1.plot(x_range, ema200_display.values, color=COLORS['ema200'], linewidth=2, label='EMA 200', alpha=0.8)

        # --- Horizontal Lines ---
        ax1.axhline(y=tp2_price, color=COLORS['tp2'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhline(y=tp1_price, color=COLORS['tp1'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=entry_price, color=COLORS['entry'], linestyle='dotted', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=sl_price, color=COLORS['sl'], linestyle='-', linewidth=2, alpha=0.9)

        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=COLORS['entry'])

        change_pct = ((current_price - entry_price) / entry_price) * 100

        price_min = min(df['Low'].min(), sl_price * 0.95)
        price_max = max(df['High'].max(), tp2_price * 1.05)
        ax1.set_ylim(price_min, price_max)
        ax1.set_xlim(-1, len(df))

        y_range = price_max - price_min
        y_shift = y_range * 0.008
        x_offset = len(df) * 0.02

        # Labels
        ax1.text(x_offset, tp2_price + y_shift,
                f"TP2 ${tp2_price:,.2f} (+{tp2_pct_display:.1f}%)",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp2'],
                         edgecolor='white', alpha=0.9), color='white')

        ax1.text(x_offset, tp1_price + y_shift,
                f"TP1 ${tp1_price:,.2f} (+{tp1_pct_display:.1f}%)",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp1'],
                         edgecolor='white', alpha=0.9), color='white')

        ax1.text(x_offset, current_price + y_shift,
                f">> NOW ${current_price:,.2f}",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['entry'],
                         edgecolor='white', alpha=0.9), color='white')

        if use_smart_entry and entry_price != current_price:
            entry_text = f"ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f} ({change_pct:+.1f}%)"
        else:
            entry_text = f"ENTRY: ${entry_price:,.2f} ({change_pct:+.1f}%)"

        ax1.text(x_offset, entry_price + y_shift,
                entry_text,
                fontsize=10, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['tp1'],
                         edgecolor='white', alpha=0.8), color='white')

        ax1.text(x_offset, sl_price + y_shift,
                f"SL ${sl_price:,.2f} ({sl_pct:.1f}%)",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['sl'],
                         edgecolor='white', alpha=0.9), color='white')

        # EMA Labels
        x_right = len(df) * 0.98

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

        ax1.set_title(f'Apexify — {symbol}  |  Swing Entry (1-4W) + TP + SL\n'
                     f'EMA: 20(Blue) 50(Orange) 200(Purple)',
                     fontsize=14, fontweight='bold', pad=20)

        ax1.grid(True, alpha=0.3, linestyle='-')
        ax1.set_axisbelow(True)

        # Volume
        volumes = df['Volume'].values
        max_vol = volumes.max()

        for i, (idx, row) in enumerate(df.iterrows()):
            x = i
            color = COLORS['bullish'] if row['Close'] >= row['Open'] else COLORS['bearish']
            ax2.bar(x, row['Volume'], color=color, alpha=0.7, width=0.8)

        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylim(0, max_vol * 1.5)
        ax2.grid(True, alpha=0.3, linestyle='-')

        date_labels = [d.strftime('%b %d') for d in df.index[::5]]
        x_ticks = range(0, len(df), 5)
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
        return self.generate_trading_chart(df, symbol,
                                          use_smart_entry=True,
                                          entry_price=None,
                                          tp1_price=None,
                                          tp2_price=None,
                                          sl_price=None)