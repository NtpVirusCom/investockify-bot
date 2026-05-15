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

# =========================================================
# TOKEN
# =========================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# =========================================================
# CONFIG
# =========================================================

TIMEFRAME = "1d"

PIVOT_LEN = 5

MIN_ATR_STRENGTH = 0.03

MAX_LEVEL_AGE = 500

MAX_ACTIVE_LEVELS_EACH = 10

MERGE_THRESHOLD = 0.8

ZONE_WIDTH = 0.25

BREAK_SENS = 0.1

MAX_BREAKOUT_SIGNALS = 4

ATR_PERIOD = 14

MAX_DISTANCE_FROM_PRICE = 0.5

MIN_BREAKOUT_MOVE_ATR = 0.1

# =========================================================
# GLOBAL STATE
# =========================================================

active_levels = {}

broken_levels = {}

last_break = {}

# =========================================================
# WELCOME MESSAGE
# =========================================================

WELCOME_MESSAGE = """
⚡ ยินดีต้อนรับสู่ Apexify

🤖 ระบบวิเคราะห์หุ้นด้วย Adaptive S/R AI

รองรับ:
• หุ้น US
• หุ้นไทย
• Crypto

👇 กด 📊 วิเคราะห์หุ้น เพื่อเริ่มต้น
"""

# =========================================================
# MAIN MENU
# =========================================================

MAIN_MENU = ReplyKeyboardMarkup(
    [
        [
            KeyboardButton("📊 วิเคราะห์หุ้น"),
            KeyboardButton("📱 เปิดเมนูหลัก")
        ],
        [
            KeyboardButton("💎 บัญชี / VIP"),
            KeyboardButton("📖 คู่มือ /manual")
        ]
    ],
    resize_keyboard=True
)

# =========================================================
# INLINE MENU
# =========================================================

def get_main_inline_keyboard():

    keyboard = [

        [
            InlineKeyboardButton(
                "📊 ลอง AAPL",
                callback_data="stock_AAPL"
            ),

            InlineKeyboardButton(
                "⚡ ลอง NVDA",
                callback_data="stock_NVDA"
            )
        ],

        [
            InlineKeyboardButton(
                "🇹🇭 ลอง PTT.BK",
                callback_data="stock_PTT"
            )
        ]

    ]

    return InlineKeyboardMarkup(keyboard)

# =========================================================
# FORMAT SYMBOL
# =========================================================

def format_symbol(symbol):

    symbol = symbol.upper().strip()

    thai_stocks = [

        "PTT",
        "AOT",
        "CPALL",
        "SCB",
        "KBANK",
        "ADVANC",
        "BDMS",
        "BBL",
        "KTB"

    ]

    if symbol in thai_stocks:

        return f"{symbol}.BK"

    if symbol.endswith("USDT"):

        return symbol.replace(
            "USDT",
            "-USD"
        )

    return symbol

# =========================================================
# FETCH DATA
# =========================================================

def fetch_ohlcv(symbol):

    try:

        ticker = yf.Ticker(symbol)

        df = ticker.history(

            period="2y",

            interval=TIMEFRAME,

            auto_adjust=False

        )

        if df.empty:

            return None

        df = df.reset_index()

        df.columns = [

            str(c).lower().replace(" ", "_")

            for c in df.columns

        ]

        keep_cols = [

            "date",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume"

        ]

        existing = [

            c for c in keep_cols

            if c in df.columns

        ]

        df = df[existing]

        if "date" in df.columns:

            df.rename(
                columns={"date": "timestamp"},
                inplace=True
            )

        if "datetime" in df.columns:

            df.rename(
                columns={"datetime": "timestamp"},
                inplace=True
            )

        df.dropna(inplace=True)

        if len(df) < 100:

            return None

        return df

    except:

        return None

# =========================================================
# ATR
# =========================================================

def calculate_atr(df):

    df = df.copy()

    df["prev_close"] = (
        df["close"].shift(1)
    )

    df["tr1"] = (
        df["high"] - df["low"]
    )

    df["tr2"] = abs(
        df["high"] - df["prev_close"]
    )

    df["tr3"] = abs(
        df["low"] - df["prev_close"]
    )

    df["tr"] = df[
        ["tr1", "tr2", "tr3"]
    ].max(axis=1)

    df["atr"] = df["tr"].rolling(
        ATR_PERIOD
    ).mean()

    return df

# =========================================================
# PIVOT DETECTION
# =========================================================

def detect_pivots(df):

    highs = []

    lows = []

    for i in range(
        PIVOT_LEN,
        len(df) - PIVOT_LEN
    ):

        current_high = (
            df["high"].iloc[i]
        )

        current_low = (
            df["low"].iloc[i]
        )

        left_high = df["high"].iloc[
            i - PIVOT_LEN:i
        ]

        right_high = df["high"].iloc[
            i + 1:i + PIVOT_LEN + 1
        ]

        left_low = df["low"].iloc[
            i - PIVOT_LEN:i
        ]

        right_low = df["low"].iloc[
            i + 1:i + PIVOT_LEN + 1
        ]

        atr = df["atr"].iloc[i]

        if pd.isna(atr):

            continue

        # =================================================
        # RESISTANCE
        # =================================================

        if (
            current_high >= left_high.max()
            and
            current_high >= right_high.max()
        ):

            strength = (
                current_high -
                df["close"].iloc[i]
            ) / atr

            if (
                strength >=
                MIN_ATR_STRENGTH
            ):

                highs.append({
                    "price": current_high,
                    "bar_index": i
                })

        # =================================================
        # SUPPORT
        # =================================================

        if (
            current_low <= left_low.min()
            and
            current_low <= right_low.min()
        ):

            strength = (
                df["close"].iloc[i] -
                current_low
            ) / atr

            if (
                strength >=
                MIN_ATR_STRENGTH
            ):

                lows.append({
                    "price": current_low,
                    "bar_index": i
                })

    return highs, lows

# =========================================================
# MERGE LEVELS
# =========================================================

def merge_levels(levels, atr):

    if not levels:

        return []

    levels = sorted(
        levels,
        key=lambda x: x["price"]
    )

    merged = []

    current = [levels[0]]

    for lvl in levels[1:]:

        avg_price = np.mean(
            [x["price"] for x in current]
        )

        if abs(
            lvl["price"] - avg_price
        ) <= atr * MERGE_THRESHOLD:

            current.append(lvl)

        else:

            merged.append(current)

            current = [lvl]

    merged.append(current)

    result = []

    for group in merged:

        avg_price = np.mean(
            [x["price"] for x in group]
        )

        latest_bar = max(
            [x["bar_index"] for x in group]
        )

        result.append({
            "price": round(avg_price, 2),
            "bar_index": latest_bar,
            "strength": len(group)
        })

    return result

# =========================================================
# UPDATE LEVELS
# =========================================================

def update_levels(symbol, df):

    global active_levels

    df = calculate_atr(df)

    atr = df["atr"].iloc[-1]

    current_price = df["close"].iloc[-1]

    highs, lows = detect_pivots(df)

    current_bar = len(df)

    highs = [
        x for x in highs
        if current_bar - x["bar_index"]
        <= MAX_LEVEL_AGE
    ]

    lows = [
        x for x in lows
        if current_bar - x["bar_index"]
        <= MAX_LEVEL_AGE
    ]

    merged_highs = merge_levels(
        highs,
        atr
    )

    merged_lows = merge_levels(
        lows,
        atr
    )

    merged_highs = [
        x for x in merged_highs
        if (
            x["price"] > current_price
            and
            (
                x["price"] -
                current_price
            ) / current_price
            <= MAX_DISTANCE_FROM_PRICE
        )
    ]

    merged_lows = [
        x for x in merged_lows
        if (
            x["price"] < current_price
            and
            (
                current_price -
                x["price"]
            ) / current_price
            <= MAX_DISTANCE_FROM_PRICE
        )
    ]

    merged_highs = sorted(
        merged_highs,
        key=lambda x: (
            abs(
                x["price"] -
                current_price
            ),
            -x["strength"]
        )
    )[:MAX_ACTIVE_LEVELS_EACH]

    merged_lows = sorted(
        merged_lows,
        key=lambda x: (
            abs(
                x["price"] -
                current_price
            ),
            -x["strength"]
        )
    )[:MAX_ACTIVE_LEVELS_EACH]

    levels = []

    # =====================================================
    # RESISTANCE
    # =====================================================

    for h in merged_highs:

        levels.append({

            "type": "resistance",

            "price": round(
                h["price"],
                2
            ),

            "strength": h["strength"],

            "zone_top": (
                h["price"] +
                atr * ZONE_WIDTH
            ),

            "zone_bottom": (
                h["price"] -
                atr * ZONE_WIDTH
            )
        })

    # =====================================================
    # SUPPORT
    # =====================================================

    for l in merged_lows:

        levels.append({

            "type": "support",

            "price": round(
                l["price"],
                2
            ),

            "strength": l["strength"],

            "zone_top": (
                l["price"] +
                atr * ZONE_WIDTH
            ),

            "zone_bottom": (
                l["price"] -
                atr * ZONE_WIDTH
            )
        })

    levels = sorted(
        levels,
        key=lambda x: x["price"],
        reverse=True
    )

    active_levels[symbol] = levels

    return levels

# =========================================================
# BREAKOUTS
# =========================================================

def detect_breakout(symbol, df):

    global broken_levels

    if symbol not in active_levels:

        return []

    if symbol not in broken_levels:

        broken_levels[symbol] = []

    df = calculate_atr(df)

    close_price = df["close"].iloc[-1]

    atr = df["atr"].iloc[-1]

    buffer = atr * BREAK_SENS

    signals = []

    remaining = []

    breakout_count = 0

    for lvl in active_levels[symbol]:

        breakout_move = (
            abs(
                close_price -
                lvl["price"]
            ) / atr
        )

        # =================================================
        # BREAK RESISTANCE
        # =================================================

        if (
            lvl["type"] == "resistance"
            and
            close_price >
            lvl["zone_top"] + buffer
            and
            breakout_move >=
            MIN_BREAKOUT_MOVE_ATR
        ):

            if (
                breakout_count <
                MAX_BREAKOUT_SIGNALS
            ):

                signals.append(
                    f"🚀 {symbol}\n"
                    f"Break Resistance\n"
                    f"Level: {lvl['price']:.2f}\n"
                    f"Current: {close_price:.2f}\n"
                    f"Strength: {lvl['strength']}"
                )

                breakout_count += 1

        # =================================================
        # BREAK SUPPORT
        # =================================================

        elif (
            lvl["type"] == "support"
            and
            close_price <
            lvl["zone_bottom"] - buffer
            and
            breakout_move >=
            MIN_BREAKOUT_MOVE_ATR
        ):

            if (
                breakout_count <
                MAX_BREAKOUT_SIGNALS
            ):

                signals.append(
                    f"🔻 {symbol}\n"
                    f"Break Support\n"
                    f"Level: {lvl['price']:.2f}\n"
                    f"Current: {close_price:.2f}\n"
                    f"Strength: {lvl['strength']}"
                )

                breakout_count += 1

        else:

            remaining.append(lvl)

    active_levels[symbol] = remaining

    return signals

# =========================================================
# START
# =========================================================

async def start(update: Update, context):

    await update.message.reply_text(
        WELCOME_MESSAGE,
        reply_markup=MAIN_MENU
    )

    await update.message.reply_text(
        "📱 เมนูหลัก",
        reply_markup=get_main_inline_keyboard()
    )

# =========================================================
# BUTTON HANDLER
# =========================================================

async def button_handler(update, context):

    query = update.callback_query

    await query.answer()

    data = query.data

    if data == "stock_AAPL":

        await send_real_stock_analysis(
            query.message,
            "AAPL"
        )

    elif data == "stock_NVDA":

        await send_real_stock_analysis(
            query.message,
            "NVDA"
        )

    elif data == "stock_PTT":

        await send_real_stock_analysis(
            query.message,
            "PTT.BK"
        )

# =========================================================
# ANALYSIS
# =========================================================

async def send_real_stock_analysis(
    message,
    stock
):

    try:

        symbol = format_symbol(stock)

        # =================================================
        # COMPANY NAME
        # =================================================

        try:

            ticker = yf.Ticker(symbol)

            info = ticker.info

            company_name = info.get(
                "longName",
                symbol
            )

        except:

            company_name = symbol

        # =================================================
        # FETCH DATA
        # =================================================

        df = fetch_ohlcv(symbol)

        if df is None:

            await message.reply_text(
                f"❌ ไม่พบข้อมูล {symbol}"
            )

            return

        # =================================================
        # LEVELS
        # =================================================

        levels = update_levels(
            symbol,
            df
        )

        # =================================================
        # BREAKOUTS
        # =================================================

        breakouts = detect_breakout(
            symbol,
            df
        )

        current_price = (
            df["close"].iloc[-1]
        )

        resistance_levels = [
            x for x in levels
            if x["type"] == "resistance"
        ]

        support_levels = [
            x for x in levels
            if x["type"] == "support"
        ]

        result = (
            f"━━━━━━━━━━━━━━━━━\n"
            f"📊 วิเคราะห์หุ้น {symbol}\n"
            f"🏢 {company_name}\n"
            f"━━━━━━━━━━━━━━━━━\n\n"
            f"💵 Current Price: "
            f"{current_price:.2f}\n\n"
        )

        # =================================================
        # RESISTANCE
        # =================================================

        result += "🔴 Resistance\n"

        if resistance_levels:

            for lvl in resistance_levels:

                result += (
                    f"{lvl['price']:.2f} "
                    f"(S:{lvl['strength']})\n"
                )

        else:

            result += "No resistance\n"

        # =================================================
        # SUPPORT
        # =================================================

        result += "\n🟢 Support\n"

        if support_levels:

            for lvl in support_levels:

                result += (
                    f"{lvl['price']:.2f} "
                    f"(S:{lvl['strength']})\n"
                )

        else:

            result += "No support\n"

        # =================================================
        # BREAKOUTS
        # =================================================

        if breakouts:

            result += (
                "\n⚡ Breakouts\n\n"
            )

            result += "\n\n".join(
                breakouts
            )

        # =================================================
        # BUTTONS
        # =================================================

        keyboard = [

            [
                InlineKeyboardButton(
                    "⭐ เพิ่ม Watchlist",
                    callback_data="watchlist"
                )
            ],

            [
                InlineKeyboardButton(
                    "🔔 ตั้ง Alert",
                    callback_data="alert"
                ),

                InlineKeyboardButton(
                    "📈 เปิดกราฟ",
                    callback_data="chart"
                )
            ]

        ]

        await message.reply_text(
            result,
            reply_markup=InlineKeyboardMarkup(
                keyboard
            )
        )

    except Exception as e:

        await message.reply_text(
            f"❌ Error: {str(e)}"
        )

# =========================================================
# MESSAGE HANDLER
# =========================================================

async def handle_message(update, context):

    text = (
        update.message.text
        .upper()
        .strip()
    )

    # =====================================================
    # MENU
    # =====================================================

    if text == "📊 วิเคราะห์หุ้น":

        await update.message.reply_text(
            "📊 พิมพ์ชื่อหุ้น เช่น:\n\n"
            "AAPL\n"
            "NVDA\n"
            "TSLA\n"
            "MSFT\n"
            "PTT.BK\n"
            "AOT.BK\n"
            "BTCUSDT"
        )

        return

    elif text == "📱 เปิดเมนูหลัก":

        await update.message.reply_text(
            "📱 เมนูหลัก",
            reply_markup=get_main_inline_keyboard()
        )

        return

    elif text == "💎 บัญชี / VIP":

        await update.message.reply_text(
            """
💎 สมาชิก VIP

✅ วิเคราะห์หุ้น
✅ Smart Alert
✅ Watchlist
✅ Portfolio
"""
        )

        return

    elif text == "📖 คู่มือ /MANUAL":

        await update.message.reply_text(
            """
📖 วิธีใช้งาน

1️⃣ กด 📊 วิเคราะห์หุ้น

2️⃣ พิมพ์ชื่อหุ้น

ตัวอย่าง:
AAPL
NVDA
PTT.BK
BTCUSDT

3️⃣ ระบบจะวิเคราะห์ให้อัตโนมัติ
"""
        )

        return

    # =====================================================
    # ANALYZE
    # =====================================================

    if len(text) >= 1:

        await send_real_stock_analysis(
            update.message,
            text
        )

# =========================================================
# MAIN
# =========================================================

def main():

    app = Application.builder().token(
        TELEGRAM_TOKEN
    ).build()

    app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    app.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Bot running...")

    app.run_polling()

# =========================================================
# START APP
# =========================================================

if __name__ == "__main__":

    main()
