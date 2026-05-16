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

    def find_optimal_entry(self, df: pd.DataFrame, ema200: pd.Series, current_price: float):
        """
        Apexify Swing Trading Logic (1-4 weeks timeframe):
        - หา Swing Low ในกรอบ 4-6 สัปดาห์ล่าสุด (20-30 วัน)
        - Swing Low ต้องอยู่ใกล้ EMA200 (<= 2-3%)
        - ต้องเป็น Bounce จริง (Close วันถัดไป > Low)
        - Risk:Reward ขั้นต่ำ 1:2
        - ถ้าไม่มี Swing Low ที่ valid ในกรอบ ใช้ EMA50 แทน
        """
        # Apexify: ใช้ข้อมูล 4-6 สัปดาห์ล่าสุด (20-30 วัน) สำหรับหา Swing Low
        swing_lookback = 25  # ~4-5 สัปดาห์
        recent_df = df.tail(swing_lookback).reset_index()

        if len(recent_df) < 5:
            # ถ้าข้อมูลน้อยเกินไป ใช้ EMA50
            ema50 = self.calculate_ema(df['Close'], 50)
            entry_price = ema50.iloc[-1]
            entry_date = df.index[-1]
            sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)
            return entry_price, entry_date, sl_price, "ema50_fallback"

        lows = recent_df['Low'].values
        highs = recent_df['High'].values
        closes = recent_df['Close'].values
        swing_lows = []

        ema200_current = ema200.iloc[-1]

        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                idx = recent_df.index[i]
                original_idx = recent_df.iloc[i]['Date'] if 'Date' in recent_df.columns else recent_df.index[i]

                # EMA200 ณ วันนั้น
                ema_val = ema200.loc[original_idx] if original_idx in ema200.index else ema200_current

                # ระยะห่างจาก EMA200 ณ วันนั้น (%)
                ema200_dist = abs(lows[i] - ema_val) / ema_val * 100

                # ตรวจสอบ Bounce (Close วันถัดไป > Low วันนั้น)
                is_bounce = closes[i+1] > lows[i] if i+1 < len(closes) else False

                # ระยะห่างจากราคาปัจจุบัน
                dist_from_current = abs(lows[i] - current_price) / current_price * 100

                # Apexify: คำนวณ Risk:Reward ที่เป็นไปได้
                atr = self.calculate_atr(df).iloc[-1]
                sl_buffer = max(atr * 1.5, lows[i] * 0.015)
                potential_sl = lows[i] - sl_buffer
                risk = lows[i] - potential_sl
                potential_reward = current_price - lows[i]
                rr_ratio = potential_reward / risk if risk > 0 else 0

                swing_lows.append({
                    'price': lows[i],
                    'date': original_idx,
                    'ema200_dist': ema200_dist,
                    'ema_val': ema_val,
                    'is_bounce': is_bounce,
                    'dist_from_current': dist_from_current,
                    'rr_ratio': rr_ratio,
                    'score': ema200_dist + (0 if is_bounce else 15) + (10 if rr_ratio < 2 else 0)
                })

        if swing_lows:
            # Apexify Criteria:
            # 1. ใกล้ EMA200 <= 3%
            # 2. เป็น Bounce จริง
            # 3. Risk:Reward >= 1:2 (ถ้าซื้อที่ Swing Low แล้วขายที่ราคาปัจจุบัน)
            valid_swing_lows = [
                s for s in swing_lows 
                if s['ema200_dist'] <= 3.0 
                and s['is_bounce']
                and s['rr_ratio'] >= 2.0
            ]

            if valid_swing_lows:
                # เลือก Swing Low ที่มี score ต่ำสุด
                best = min(valid_swing_lows, key=lambda x: x['score'])
                entry_price = best['price']

                # SL = ต่ำกว่า Swing Low
                atr = self.calculate_atr(df).iloc[-1]
                sl_buffer = max(atr * 1.5, entry_price * 0.015)
                sl_price = entry_price - sl_buffer

                return entry_price, best['date'], sl_price, "apexify_swing"
            else:
                # ผ่อนเกณฑ์: ไม่ตรวจสอบ RR แต่ต้อง Bounce + ใกล้ EMA200
                relaxed = [s for s in swing_lows if s['ema200_dist'] <= 5.0 and s['is_bounce']]
                if relaxed:
                    best = min(relaxed, key=lambda x: x['ema200_dist'])
                    entry_price = best['price']
                    atr = self.calculate_atr(df).iloc[-1]
                    sl_buffer = max(atr * 1.5, entry_price * 0.015)
                    sl_price = entry_price - sl_buffer
                    return entry_price, best['date'], sl_price, "swing_relaxed"

        # Apexify Fallback: ใช้ EMA50 ถ้าไม่มี Swing Low ที่ valid
        ema50 = self.calculate_ema(df['Close'], 50)
        entry_price = ema50.iloc[-1]
        entry_date = df.index[-1]
        sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

        return entry_price, entry_date, sl_price, "ema50_fallback"

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
                              tp1_pct: float = None,
                              tp2_pct: float = None,
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

        # Apexify: แสดงกราฟ 3 เดือน (90 วัน) แต่ดึงข้อมูล 6 เดือน - 1 ปี
        df_display = df.tail(60).copy()
        ema20_display = ema20.tail(60)
        ema50_display = ema50.tail(60)
        ema200_display = ema200.tail(60)

        df = df_display

        # === กำหนด Entry, SL, TP ===
        if use_smart_entry and entry_price is None:
            entry_price, entry_date, auto_sl, method = self.find_optimal_entry(df, ema200, current_price)

            if sl_price is None:
                sl_price = auto_sl

            risk = entry_price - sl_price
            if risk > 0:
                # Apexify: TP จาก Risk:Reward (1:2 และ 1:4)
                if tp1_pct is None:
                    tp1_price = entry_price + (risk * 2)
                    tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
                else:
                    tp1_price = entry_price * (1 + tp1_pct / 100)

                if tp2_pct is None:
                    tp2_price = entry_price + (risk * 4)
                    tp2_pct = ((tp2_price - entry_price) / entry_price) * 100
                else:
                    tp2_price = entry_price * (1 + tp2_pct / 100)
            else:
                tp1_price = entry_price * 1.05
                tp2_price = entry_price * 1.10
                tp1_pct = 5.0
                tp2_pct = 10.0
        else:
            if entry_price is None:
                entry_price = current_price

            if sl_price is None:
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

            tp1_price = entry_price * (1 + (tp1_pct if tp1_pct is not None else DEFAULT_TP1_PCT) / 100)
            tp2_price = entry_price * (1 + (tp2_pct if tp2_pct is not None else DEFAULT_TP2_PCT) / 100)

        # === FIX: TP1 ต้องสูงกว่าราคาปัจจุบัน ===
        if tp1_price <= current_price:
            tp1_price = current_price * 1.05
            tp2_price = current_price * 1.10
            tp1_pct = 5.0
            tp2_pct = 10.0

            if use_smart_entry and entry_price is not None and entry_price < current_price * 0.95:
                entry_price = current_price * 0.98
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

        entry_zone_top = entry_price * 1.009
        entry_zone_bottom = entry_price * 0.991

        sl_pct = ((sl_price - entry_price) / entry_price) * 100
        tp1_pct_display = ((tp1_price - entry_price) / entry_price) * 100
        tp2_pct_display = ((tp2_price - entry_price) / entry_price) * 100

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
                                          tp1_pct=None,
                                          tp2_pct=None,
                                          sl_price=None)