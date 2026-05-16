import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import io
from config import COLORS, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT, DEFAULT_SL_PCT

class ChartGenerator:
    def __init__(self):
        plt.style.use('default')
        # ตั้งค่า font สำหรับ emoji
        self._setup_emoji_font()

    def _setup_emoji_font(self):
        """หา font ที่รองรับ emoji"""
        # ลองหา font ที่รองรับ emoji ตามลำดับ
        emoji_font_candidates = [
            'Segoe UI Emoji',      # Windows
            'Segoe UI Symbol',     # Windows fallback
            'Noto Color Emoji',    # Linux
            'Noto Emoji',          # Linux fallback
            'Twemoji',             # Cross-platform
            'EmojiOne',            # Cross-platform
        ]

        self.emoji_font = None
        for font_name in emoji_font_candidates:
            try:
                fp = fm.FontProperties(family=font_name)
                # ทดสอบว่า font นี้ใช้ได้จริง
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, "\U0001f3af", fontproperties=fp, fontsize=20)
                plt.close()
                self.emoji_font = fp
                print(f"Using emoji font: {font_name}")
                break
            except:
                continue

        if self.emoji_font is None:
            print("Warning: No emoji font found, using text fallback")

    def _emoji_text(self, text_with_emoji):
        """
        สร้าง text ที่มี emoji โดยใช้ font ที่เหมาะสม
        ถ้าไม่มี emoji font จะใช้ text แทน
        """
        if self.emoji_font is None:
            # Fallback: แทนที่ emoji ด้วย text
            replacements = {
                '\U0001f3af': 'TP ',      # 🎯
                '\u25b6': '>> ',           # ▶
                '\U0001f6e1': 'ENTRY ',    # 🛡
                '\U0001f6d1': 'SL ',       # 🛑
            }
            result = text_with_emoji
            for emoji, text in replacements.items():
                result = result.replace(emoji, text)
            return result, None  # None = ใช้ default font

        return text_with_emoji, self.emoji_font

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

    def _add_label(self, ax, x, y, text, color, fontsize=11, fontweight='bold'):
        """เพิ่ม label พร้อมจัดการ emoji font"""
        text_clean, font_prop = self._emoji_text(text)

        if font_prop:
            ax.text(x, y, text_clean,
                   fontsize=fontsize, fontweight=fontweight, va='bottom',
                   fontproperties=font_prop,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                            edgecolor='white', alpha=0.9), color='white')
        else:
            ax.text(x, y, text_clean,
                   fontsize=fontsize, fontweight=fontweight, va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                            edgecolor='white', alpha=0.9), color='white')

    def generate_trading_chart(self, df: pd.DataFrame, symbol: str,
                              entry_price: float = None,
                              tp1_pct: float = None,
                              tp2_pct: float = None,
                              sl_price: float = None,
                              use_smart_entry: bool = True):

        ema20 = self.calculate_ema(df['Close'], 20)
        ema50 = self.calculate_ema(df['Close'], 50)
        ema200 = self.calculate_ema(df['Close'], 200)

        current_price = df['Close'].iloc[-1]

        # === กำหนด Entry, SL, TP ===
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
        ax1.plot(x_range, ema20, color=COLORS['ema20'], linewidth=2, label='EMA 20', alpha=0.8)
        ax1.plot(x_range, ema50, color=COLORS['ema50'], linewidth=2, label='EMA 50', alpha=0.8)
        ax1.plot(x_range, ema200, color=COLORS['ema200'], linewidth=2, label='EMA 200', alpha=0.8)

        # --- Horizontal Lines ---
        ax1.axhline(y=tp2_price, color=COLORS['tp2'], linestyle='-', linewidth=2, alpha=0.9)
        ax1.axhline(y=tp1_price, color=COLORS['tp1'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=entry_price, color=COLORS['entry'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.axhline(y=sl_price, color=COLORS['sl'], linestyle='-', linewidth=2, alpha=0.9)

        ax1.axhspan(entry_zone_bottom, entry_zone_top, alpha=0.15, color=COLORS['entry'])

        change_pct = ((current_price - entry_price) / entry_price) * 100

        # ============================================================
        # LABELS - วางตรงระดับเส้น พร้อม emoji
        # ============================================================

        x_offset = len(df) * 0.02
        y_shift = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.008

        # TP2 Label
        self._add_label(ax1, x_offset, tp2_price + y_shift,
                       f"\U0001f3af TP2 ${tp2_price:,.2f} (+{tp2_pct_display:.1f}%)",
                       COLORS['tp2'])

        # TP1 Label
        self._add_label(ax1, x_offset, tp1_price + y_shift,
                       f"\U0001f3af TP1 ${tp1_price:,.2f} (+{tp1_pct_display:.1f}%)",
                       COLORS['tp1'])

        # NOW Label
        self._add_label(ax1, x_offset, current_price + y_shift,
                       f"\u25b6 NOW ${current_price:,.2f}",
                       COLORS['entry'])

        # ENTRY Label
        if use_smart_entry and entry_price != current_price:
            entry_text = f"\U0001f6e1 ENTRY ${entry_zone_bottom:,.2f}-${entry_zone_top:,.2f} ({change_pct:+.1f}%)"
        else:
            entry_text = f"\U0001f6e1 ENTRY: ${entry_price:,.2f} ({change_pct:+.1f}%)"

        self._add_label(ax1, x_offset, entry_price + y_shift,
                       entry_text, COLORS['tp1'], fontsize=10)

        # SL Label
        self._add_label(ax1, x_offset, sl_price + y_shift,
                       f"\U0001f6d1 SL ${sl_price:,.2f} ({sl_pct:.1f}%)",
                       COLORS['sl'])

        # ============================================================

        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

        ax1.set_title(f'Apexify -- {symbol}  |  Entry Zone + TP + SL Plan\n'
                     f'EMA: 20(Blue) 50(Orange) 200(Purple)',
                     fontsize=14, fontweight='bold', pad=20)

        price_min = min(df['Low'].min(), sl_price * 0.95)
        price_max = max(df['High'].max(), tp2_price * 1.05)
        ax1.set_ylim(price_min, price_max)
        ax1.set_xlim(-1, len(df))

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
