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

    def find_optimal_entry(self, df: pd.DataFrame, ema200: pd.Series):
        recent_df = df.tail(30)
        lows = recent_df['Low'].values
        swing_lows = []

        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                idx = recent_df.index[i]
                ema_val = ema200.loc[idx] if idx in ema200.index else ema200.iloc[-1]
                swing_lows.append({
                    'price': lows[i],
                    'date': idx,
                    'ema200_dist': abs(lows[i] - ema_val) / ema_val * 100
                })

        if swing_lows:
            best = min(swing_lows, key=lambda x: x['ema200_dist'])
            entry_price = best['price']
            atr = self.calculate_atr(df).iloc[-1]
            sl_price = entry_price - (atr * 2.5)
            return entry_price, best['date'], sl_price, "swing_low"

        entry_price = df['Low'].tail(20).min()
        entry_date = df['Low'].tail(20).idxmin()
        sl_price = entry_price * (1 + DEFAULT_SL_PCT / 100)

        return entry_price, entry_date, sl_price, "recent_low"

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

        # à¸à¹à¸² EMA à¸¥à¹à¸²à¸ªà¸¸à¸
        ema20_last = ema20.iloc[-1]
        ema50_last = ema50.iloc[-1]
        ema200_last = ema200.iloc[-1]

        current_price = close.iloc[-1]

        # === à¹à¸ªà¸à¸à¸à¸£à¸²à¸à¹à¸à¸à¸²à¸° 3 à¹à¸à¸·à¸­à¸à¸¥à¹à¸²à¸ªà¸¸à¸ (90 à¸§à¸±à¸) à¹à¸à¹à¸à¸³à¸à¸§à¸à¸à¸²à¸à¸à¹à¸­à¸¡à¸¹à¸¥ 3 à¸à¸µ ===
        df_display = df.tail(30).copy()
        ema20_display = ema20.tail(30)
        ema50_display = ema50.tail(30)
        ema200_display = ema200.tail(30)

        # à¹à¸à¹ df_display à¹à¸à¸ df à¹à¸à¸à¸²à¸£ plot à¸à¸£à¸²à¸
        df = df_display

        # === à¸à¸³à¸«à¸à¸ Entry, SL, TP ===
        if use_smart_entry and entry_price is None:
            entry_price, entry_date, auto_sl, method = self.find_optimal_entry(df, ema200)

            if sl_price is None:
                sl_price = auto_sl

            risk = entry_price - sl_price
            if risk > 0:
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

        sl_pct = ((sl_price - entry_price) / entry_price) * 100
        tp1_pct_display = ((tp1_price - entry_price) / entry_price) * 100
        tp2_pct_display = ((tp2_price - entry_price) / entry_price) * 100

        entry_zone_top = entry_price * 1.009
        entry_zone_bottom = entry_price * 0.991

        # === à¸ªà¸£à¹à¸²à¸à¸à¸£à¸²à¸ ===
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

        # --- Horizontal Lines (TP/SL/Entry) ---
        ax1.axhline(y=tp2_price, color=COLORS['tp2'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhline(y=tp1_price, color=COLORS['tp1'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=entry_price, color=COLORS['entry'], linestyle='dotted', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=sl_price, color=COLORS['sl'], linestyle='-', linewidth=2, alpha=0.9)

        # Entry Zone
        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=COLORS['entry'])

        change_pct = ((current_price - entry_price) / entry_price) * 100

        # === à¸à¸³à¸«à¸à¸à¸à¸­à¸à¹à¸à¸à¹à¸à¸ Y à¸à¹à¸­à¸à¸§à¸²à¸ Label ===
        price_min = min(df['Low'].min(), sl_price * 0.95)
        price_max = max(df['High'].max(), tp2_price * 1.05)
        ax1.set_ylim(price_min, price_max)
        ax1.set_xlim(-1, len(df))

        # === à¸à¸³à¸à¸§à¸ y_shift à¸«à¸¥à¸±à¸à¸à¸³à¸«à¸à¸ ylim ===
        y_range = price_max - price_min
        y_shift = y_range * 0.008
        x_offset = len(df) * 0.02

        # ============================================================
        # LABELS - à¸§à¸²à¸à¹à¸«à¹à¸à¸£à¸à¸£à¸°à¸à¸±à¸à¹à¸ªà¹à¸ (à¹à¸«à¸¡à¸·à¸­à¸ Apexify)
        # ============================================================

        # TP2 Label
        ax1.text(x_offset, tp2_price + y_shift,
                f"TP2 ${tp2_price:,.2f} (+{tp2_pct_display:.1f}%)",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp2'],
                         edgecolor='white', alpha=0.9), color='white')

        # TP1 Label
        ax1.text(x_offset, tp1_price + y_shift,
                f"TP1 ${tp1_price:,.2f} (+{tp1_pct_display:.1f}%)",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tp1'],
                         edgecolor='white', alpha=0.9), color='white')

        # NOW Label
        ax1.text(x_offset, current_price + y_shift,
                f">> NOW ${current_price:,.2f}",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['entry'],
                         edgecolor='white', alpha=0.9), color='white')

        # ENTRY Label
        if use_smart_entry and entry_price != current_price:
            entry_text = f"ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f} ({change_pct:+.1f}%)"
        else:
            entry_text = f"ENTRY: ${entry_price:,.2f} ({change_pct:+.1f}%)"

        ax1.text(x_offset, entry_price + y_shift,
                entry_text,
                fontsize=10, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['tp1'],
                         edgecolor='white', alpha=0.8), color='white')

        # SL Label
        ax1.text(x_offset, sl_price + y_shift,
                f"SL ${sl_price:,.2f} ({sl_pct:.1f}%)",
                fontsize=11, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['sl'],
                         edgecolor='white', alpha=0.9), color='white')

        # ============================================================
        # EMA VALUE LABELS - à¹à¸ªà¸à¸à¸à¹à¸² EMA à¸¥à¹à¸²à¸ªà¸¸à¸à¸à¸µà¹à¸à¹à¸²à¸à¸à¸§à¸²à¸à¸­à¸à¸à¸£à¸²à¸
        # ============================================================
        x_right = len(df) * 0.98  # à¸à¹à¸²à¸à¸à¸§à¸² 98%

        # EMA 20 Label
        ax1.text(x_right, ema20_display.iloc[-1] + y_shift,
                f"EMA20 {ema20_last:,.2f}",
                fontsize=9, fontweight='bold', va='bottom', ha='right',
                color=COLORS['ema20'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['ema20'], alpha=0.9))

        # EMA 50 Label
        ax1.text(x_right, ema50_display.iloc[-1] + y_shift,
                f"EMA50 {ema50_last:,.2f}",
                fontsize=9, fontweight='bold', va='bottom', ha='right',
                color=COLORS['ema50'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['ema50'], alpha=0.9))

        # EMA 200 Label
        ax1.text(x_right, ema200_display.iloc[-1] + y_shift,
                f"EMA200 {ema200_last:,.2f}",
                fontsize=9, fontweight='bold', va='bottom', ha='right',
                color=COLORS['ema200'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['ema200'], alpha=0.9))

        # ============================================================

        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

        ax1.set_title(f'Apexify \u2014 {symbol}  |  Entry Zone + TP + SL Plan\n'
                     f'EMA: 20(Blue) 50(Orange) 200(Purple)',
                     fontsize=14, fontweight='bold', pad=20)

        ax1.grid(True, alpha=0.3, linestyle='-')
        ax1.set_axisbelow(True)

        # --- Volume ---
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