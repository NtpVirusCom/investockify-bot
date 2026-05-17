import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import io
from config import COLORS, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT, DEFAULT_SL_PCT

class ChartGenerator:
    def __init__(self):
        plt.style.use('default')
        # Constants from investockify_bot5
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
        """ATR calculation from investockify_bot5"""
        df = df.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["tr1"] = df["High"] - df["Low"]
        df["tr2"] = abs(df["High"] - df["prev_close"])
        df["tr3"] = abs(df["Low"] - df["prev_close"])
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["tr"].rolling(period).mean()
        return df

    def detect_pivots(self, df):
        """Pivot detection from investockify_bot5"""
        highs = []
        lows = []

        for i in range(self.PIVOT_LEN, len(df) - self.PIVOT_LEN):
            current_high = df["High"].iloc[i]
            current_low = df["Low"].iloc[i]

            left_high = df["High"].iloc[i - self.PIVOT_LEN:i]
            right_high = df["High"].iloc[i + 1:i + self.PIVOT_LEN + 1]
            left_low = df["Low"].iloc[i - self.PIVOT_LEN:i]
            right_low = df["Low"].iloc[i + 1:i + self.PIVOT_LEN + 1]

            atr = df["atr"].iloc[i]
            if pd.isna(atr):
                continue

            # Resistance pivot
            if (current_high >= left_high.max() and current_high >= right_high.max()):
                strength = (current_high - df["Close"].iloc[i]) / atr
                if strength >= self.MIN_ATR_STRENGTH:
                    highs.append({"price": current_high, "bar_index": i})

            # Support pivot
            if (current_low <= left_low.min() and current_low <= right_low.min()):
                strength = (df["Close"].iloc[i] - current_low) / atr
                if strength >= self.MIN_ATR_STRENGTH:
                    lows.append({"price": current_low, "bar_index": i})

        return highs, lows

    def merge_levels(self, levels, atr):
        """Merge levels from investockify_bot5"""
        if not levels:
            return []

        levels = sorted(levels, key=lambda x: x["price"])
        merged = []
        current = [levels[0]]

        for lvl in levels[1:]:
            avg_price = np.mean([x["price"] for x in current])
            if abs(lvl["price"] - avg_price) <= atr * self.MERGE_THRESHOLD:
                current.append(lvl)
            else:
                merged.append(current)
                current = [lvl]

        merged.append(current)

        result = []
        for group in merged:
            avg_price = np.mean([x["price"] for x in group])
            latest_bar = max([x["bar_index"] for x in group])
            result.append({
                "price": round(avg_price, 2),
                "bar_index": latest_bar,
                "strength": len(group)
            })

        return result

    def find_optimal_entry(self, df: pd.DataFrame, current_price: float):
        """
        ใช้ Pivot Detection จาก investockify_bot5 ในการหา:
        - Support = Entry Zone
        - Resistance = TP1
        - Next Resistance = TP2
        - Swing Low = SL
        """

        # Calculate ATR
        df_atr = self.calculate_atr(df)
        atr = df_atr["atr"].iloc[-1]

        if pd.isna(atr):
            atr = (df["High"] - df["Low"]).mean()

        # Detect pivots
        highs, lows = self.detect_pivots(df_atr)

        current_bar = len(df)

        # Filter by age
        highs = [x for x in highs if current_bar - x["bar_index"] <= self.MAX_LEVEL_AGE]
        lows = [x for x in lows if current_bar - x["bar_index"] <= self.MAX_LEVEL_AGE]

        # Merge levels
        merged_highs = self.merge_levels(highs, atr)
        merged_lows = self.merge_levels(lows, atr)

        # Filter by distance from current price
        merged_highs = [
            x for x in merged_highs
            if x["price"] > current_price and (x["price"] - current_price) / current_price <= self.MAX_DISTANCE_FROM_PRICE
        ]

        merged_lows = [
            x for x in merged_lows
            if x["price"] < current_price and (current_price - x["price"]) / current_price <= self.MAX_DISTANCE_FROM_PRICE
        ]

        # Sort by distance and strength
        merged_highs = sorted(
            merged_highs,
            key=lambda x: (abs(x["price"] - current_price), -x["strength"])
        )[:self.MAX_ACTIVE_LEVELS_EACH]

        merged_lows = sorted(
            merged_lows,
            key=lambda x: (abs(x["price"] - current_price), -x["strength"])
        )[:self.MAX_ACTIVE_LEVELS_EACH]

        # === DETERMINE LEVELS ===

        # TP1 = Nearest Resistance (strongest)
        if merged_highs:
            tp1_price = merged_highs[0]["price"]
            tp1_strength = merged_highs[0]["strength"]
        else:
            tp1_price = current_price * 1.05
            tp1_strength = 0

        # TP2 = Next Resistance or TP1 + 6%
        if len(merged_highs) >= 2:
            tp2_price = merged_highs[1]["price"]
        else:
            tp2_price = tp1_price * 1.06

        # Entry = Nearest Support (strongest)
        if merged_lows:
            entry_price = merged_lows[0]["price"]
            entry_strength = merged_lows[0]["strength"]
        else:
            entry_price = current_price * 0.96
            entry_strength = 0

        # SL = Next Support or Entry - ATR buffer
        if len(merged_lows) >= 2:
            sl_price = merged_lows[1]["price"]
        else:
            sl_buffer = max(atr * 2, entry_price * 0.03)
            sl_price = entry_price - sl_buffer

        # Adjust if entry is above current price
        if entry_price >= current_price:
            entry_price = current_price * 0.96
            sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

        # Ensure SL is below entry
        if sl_price >= entry_price:
            sl_price = entry_price * 0.97

        return {
            "entry": entry_price,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "support_levels": merged_lows,
            "resistance_levels": merged_highs,
            "atr": atr
        }

    def generate_trading_chart(self, df: pd.DataFrame, symbol: str,
                              entry_price: float = None,
                              tp1_price: float = None,
                              tp2_price: float = None,
                              sl_price: float = None,
                              use_smart_entry: bool = True,
                              tp1_pct: float = None,
                              tp2_pct: float = None):

        close = df['Close']
        ema20 = self.calculate_ema(close, 20)
        ema50 = self.calculate_ema(close, 50)
        ema200 = self.calculate_ema(close, 200)

        ema20_last = ema20.iloc[-1]
        ema50_last = ema50.iloc[-1]
        ema200_last = ema200.iloc[-1]

        current_price = close.iloc[-1]

        # แสดงกราฟ 3 เดือน (60 วัน)
        df_display = df.tail(60).copy()
        ema20_display = ema20.tail(60)
        ema50_display = ema50.tail(60)
        ema200_display = ema200.tail(60)

        df_plot = df_display

        # Backward compatibility
        if tp1_pct is not None and tp1_price is None:
            tp1_price = entry_price * (1 + tp1_pct / 100) if entry_price else None
        if tp2_pct is not None and tp2_price is None:
            tp2_price = entry_price * (1 + tp2_pct / 100) if entry_price else None

        # กำหนด Entry, SL, TP ด้วย investockify logic
        if use_smart_entry and entry_price is None:
            levels = self.find_optimal_entry(df, current_price)

            entry_price = levels["entry"]
            sl_price = levels["sl"]
            tp1_price = levels["tp1"]
            tp2_price = levels["tp2"]
            support_levels = levels["support_levels"]
            resistance_levels = levels["resistance_levels"]
            atr = levels["atr"]
        else:
            if entry_price is None:
                entry_price = current_price

            if sl_price is None:
                sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

            if tp1_price is None:
                tp1_price = entry_price * (1 + (DEFAULT_TP1_PCT if DEFAULT_TP1_PCT is not None else 5.0) / 100)

            if tp2_price is None:
                tp2_price = entry_price * (1 + (DEFAULT_TP2_PCT if DEFAULT_TP2_PCT is not None else 10.0) / 100)

            support_levels = []
            resistance_levels = []
            atr = (df["High"] - df["Low"]).mean()

        # คำนวณเปอร์เซ็นต์
        sl_pct = ((sl_price - current_price) / current_price) * 100
        tp1_pct_display = ((tp1_price - current_price) / current_price) * 100
        tp2_pct_display = ((tp2_price - current_price) / current_price) * 100

        entry_zone_top = entry_price * 1.009
        entry_zone_bottom = entry_price * 0.991

        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [4, 1]},
                                       sharex=True)

        # Candlestick
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

        # EMA Lines
        x_range = range(len(df_plot))
        ax1.plot(x_range, ema20_display.values, color=COLORS['ema20'], linewidth=2, label='EMA 20', alpha=0.8)
        ax1.plot(x_range, ema50_display.values, color=COLORS['ema50'], linewidth=2, label='EMA 50', alpha=0.8)
        ax1.plot(x_range, ema200_display.values, color=COLORS['ema200'], linewidth=2, label='EMA 200', alpha=0.8)

        # === PLOT SUPPORT/RESISTANCE LEVELS ===
        for lvl in support_levels:
            ax1.axhline(y=lvl["price"], color='#4CAF50', linestyle=':', linewidth=1, alpha=0.5)

        for lvl in resistance_levels:
            ax1.axhline(y=lvl["price"], color='#F44336', linestyle=':', linewidth=1, alpha=0.5)

        # Horizontal Lines (TP/SL/Entry)
        ax1.axhline(y=tp2_price, color=COLORS['tp2'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhline(y=tp1_price, color=COLORS['tp1'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=entry_price, color=COLORS['entry'], linestyle='dotted', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=sl_price, color=COLORS['sl'], linestyle='-', linewidth=2, alpha=0.9)

        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=COLORS['entry'])

        change_pct = ((entry_price - current_price) / current_price) * 100

        price_min = min(df_plot['Low'].min(), sl_price * 0.95)
        price_max = max(df_plot['High'].max(), tp2_price * 1.05)
        ax1.set_ylim(price_min, price_max)
        ax1.set_xlim(-1, len(df_plot))

        y_range = price_max - price_min
        y_shift = y_range * 0.008
        x_offset = len(df_plot) * 0.02
        x_center = len(df_plot) / 2

        # Labels
        ax1.text(x_offset, tp2_price,
                #f"\U0001f3af TP2 ${tp2_price:,.2f} (+{tp2_pct_display:.1f}%)",
                f"\u25C9 TP2 ${tp2_price:,.2f} (+{tp2_pct_display:.1f}%)",
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp2'],
                         edgecolor='white', alpha=0.9), color='white')

        ax1.text(x_offset, tp1_price,
                #f"\U0001f3af TP1 ${tp1_price:,.2f} (+{tp1_pct_display:.1f}%)",
                f"\u25C9 TP1 ${tp1_price:,.2f} (+{tp1_pct_display:.1f}%)",
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp1'],
                         edgecolor='white', alpha=0.9), color='white')

        ax1.text(x_offset, current_price,
                f"\u25b6 NOW ${current_price:,.2f}",
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['now'],
                         edgecolor='white', alpha=0.9), color='white')

        if use_smart_entry and entry_price != current_price:
            #entry_text = f"\U0001f6e1 ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f} ({change_pct:+.1f}%)"
            entry_text = f"\u25A3 ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f} ({change_pct:+.1f}%)"
        else:
            #entry_text = f"\U0001f6e1 ENTRY: ${entry_price:,.2f} ({change_pct:+.1f}%)"
            entry_text = f"\u25A3 ENTRY: ${entry_price:,.2f} ({change_pct:+.1f}%)"

        ax1.text(x_offset, entry_price,
                entry_text,
                fontsize=10, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['entry'],
                         edgecolor='white', alpha=0.8), color='white')

        ax1.text(x_offset, sl_price,
                #f"\U0001f6d1 SL ${sl_price:,.2f} ({sl_pct:.1f}%)",
                f"\u25CD SL ${sl_price:,.2f} ({sl_pct:.1f}%)",
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['sl'],
                         edgecolor='white', alpha=0.9), color='white')

        # EMA Labels
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

        ax1.set_title(f'Apexify \u2014 {symbol}  |  Pivot S/R Entry + TP + SL\n'
                     f'EMA: 20(Blue) 50(Orange) 200(Purple)',
                     fontsize=14, fontweight='bold', pad=20)

        ax1.grid(True, alpha=0.3, linestyle='-')
        ax1.set_axisbelow(True)

        # Volume
        volumes = df_plot['Volume'].values
        max_vol = volumes.max()

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
        return self.generate_trading_chart(df, symbol,
                                          use_smart_entry=True,
                                          entry_price=None,
                                          tp1_price=None,
                                          tp2_price=None,
                                          sl_price=None)
