from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)

from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)

import os

# ==========================================
# TOKEN
# ==========================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")


# ==========================================
# START MESSAGE
# ==========================================

WELCOME_TEXT = """
🤖 บอท AI วิเคราะห์หุ้น TH/US บน Telegram

📊 พิมพ์ชื่อหุ้น → AI ตอบใน 10 วิ
(RSI, MACD, แนวรับ-ต้าน)

🔔 ตั้งเตือนราคา + Smart Alerts 24/7 (PRO)

📈 Trade Plan ตัวเลข Entry/TP/SL พร้อมใช้

🌐 Web Dashboard + Heatmap + Tax Export

🎁 ทดลอง PRO ฟรี 7 วัน

💎 VIP 79฿ • PRO 109฿/เดือน
"""

# ==========================================
# /START
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    keyboard = [

        [
            InlineKeyboardButton(
                "🚀 ทดลอง PRO ฟรี",
                callback_data="trial"
            )
        ],

        [
            InlineKeyboardButton(
                "📊 วิเคราะห์หุ้น",
                callback_data="analyze"
            )
        ],

        [
            InlineKeyboardButton(
                "💎 สมัคร VIP",
                callback_data="vip"
            )
        ]

    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        WELCOME_TEXT,
        reply_markup=reply_markup
    )

# ==========================================
# BUTTON HANDLER
# ==========================================

async def button_click(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    data = query.data

    # ทดลองฟรี
    if data == "trial":

        await query.message.reply_text(
            "🎁 ทดลองใช้ฟรี 7 วัน\n\nกรุณาพิมพ์อีเมลของคุณ"
        )

    # วิเคราะห์หุ้น
    elif data == "analyze":

        await query.message.reply_text(
            "📊 พิมพ์ชื่อหุ้น เช่น:\n\nAAPL\nNVDA\nTSLA\nPTT"
        )

    # สมัคร VIP
    elif data == "vip":

        await query.message.reply_text(
            "💎 แพ็กเกจสมาชิก\n\nVIP 79฿\nPRO 109฿"
        )

# ==========================================
# STOCK ANALYSIS
# ==========================================

async def analyze_stock(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = update.message.text.upper()

    stocks = [
        "AAPL",
        "NVDA",
        "TSLA",
        "MSFT",
        "GOOGL",
        "PTT"
    ]

    if text in stocks:

        result = f"""
━━━━━━━━━━━━━━━━━
👑 PRO Report
🤖 Apexify — {text}
━━━━━━━━━━━━━━━━━

💵 ราคา: 220.78

📡 Trend:
⏱️ Day 🟢
📅 Week 🟢
🔭 Month ⚪️

🎯 Conviction: 🔥 91/100

━━━━━━━━━━━━━━━━━

🟢 Support
218.50
215.20

🔴 Resistance
225.00
230.00

━━━━━━━━━━━━━━━━━

📈 Trade Plan

Entry: 221.00
TP1: 225.00
TP2: 230.00
SL: 217.00
"""

        await update.message.reply_text(result)

    else:

        await update.message.reply_text(
            "❌ ไม่พบหุ้นนี้\n\nลองเช่น:\nAAPL\nNVDA\nTSLA"
        )

# ==========================================
# MAIN
# ==========================================

def main():

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))

    # Buttons
    app.add_handler(
        CallbackQueryHandler(button_click)
    )

    # Messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            analyze_stock
        )
    )

    print("Bot running...")

    # POLLING
    app.run_polling()

# ==========================================
# START
# ==========================================

if __name__ == "__main__":
    main()