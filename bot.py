##   ***main.py แยกไฟล์***
import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, CallbackQueryHandler,
    filters, ContextTypes
)
from data_fetcher import DataFetcher
from chart_generator import ChartGenerator
from config import TELEGRAM_TOKEN, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT, DEFAULT_SL_PCT

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SYMBOL, ENTRY, TP_SL = range(3)
user_data = {}

data_fetcher = DataFetcher()
chart_generator = ChartGenerator()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "\U0001f680 *ยินดีต้อนรับสู่ Apexify Trading Bot!*\n\n"
        "พิมพ์ชื่อสัญลักษณ์หุ้น/สินค้าโภคภัณฑ์ เพื่อดูกราฟการเทรดทันที\n"
        "หรือใช้คำสั่งต่อไปนี้:\n\n"
        "\U0001f4cc */chart <symbol>* - ดูกราฟพร้อมตั้งค่า TP/SL\n"
        "\U0001f4cc */quick <symbol>* - ดูกราฟแบบเร็ว (Smart Entry)\n"
        "\U0001f4cc */help* - ดูคำแนะนำทั้งหมด\n\n"
        "*ตัวอย่างสัญลักษณ์:*\n"
        "\u2022 GC=F (ทองคำ)\n"
        "\u2022 SI=F (เงิน)\n"
        "\u2022 CL=F (น้ำมัน WTI)\n"
        "\u2022 AAPL, TSLA, NVDA (หุ้น US)\n\n"
        "พิมพ์ชื่อสัญลักษณ์ได้เลย! \U0001f447"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "\U0001f4d6 *คู่มือการใช้งาน Apexify Bot*\n\n"
        "*คำสั่งพื้นฐาน:*\n"
        "\u2022 `/start` - เริ่มต้นใช้งาน\n"
        "\u2022 `/chart <symbol>` - สร้างกราฟพร้อมตั้งค่า TP/SL\n"
        "\u2022 `/quick <symbol>` - ดูกราฟเร็วๆ (Smart Entry)\n"
        "\u2022 `/help` - ดูคำแนะนำ\n\n"
        "*โหมด Smart Entry:*\n"
        "\u2022 ระบบจะหา Swing Low ใกล้ EMA200 อัตโนมัติ\n"
        "\u2022 SL คำนวณจาก ATR (2.5x)\n"
        "\u2022 TP คำนวณจาก Risk:Reward (1:2 และ 1:4)\n\n"
        "*โหมด Manual:*\n"
        "\u2022 ใช้ `/chart` เพื่อระบุ Entry และ TP/SL เอง"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def quick_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """คำสั่ง /quick - ใช้ Smart Entry เหมือน Apexify"""
    if not context.args:
        await update.message.reply_text(
            "\u274c กรุณาระบุสัญลักษณ์\nตัวอย่าง: `/quick GC=F`",
            parse_mode='Markdown'
        )
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"\u23f3 กำลังวิเคราะห์ {symbol} ด้วย Smart Entry...")

    df, error = data_fetcher.get_stock_data(symbol)
    if error:
        await update.message.reply_text(f"\u274c {error}")
        return

    # ใช้ Smart Entry (เหมือน Apexify)
    chart_buf = chart_generator.generate_trading_chart(
        df, symbol,
        use_smart_entry=True,
        entry_price=None,
        tp1_pct=None,
        tp2_pct=None,
        sl_price=None
    )

    current_price = df['Close'].iloc[-1]

    await update.message.reply_photo(
        photo=chart_buf,
        caption=(
            f"\U0001f4ca *{symbol}* | Apexify Smart Chart\n"
            f"ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
            f"\U0001f4a1 *Smart Entry* ระบบหาจุดเข้าอัตโนมัติ\n"
            f"ใช้ `/chart {symbol}` เพื่อตั้งค่าเอง"
        ),
        parse_mode='Markdown'
    )

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """คำสั่ง /chart - ตั้งค่าเอง"""
    if not context.args:
        await update.message.reply_text(
            "\u274c กรุณาระบุสัญลักษณ์\nตัวอย่าง: `/chart GC=F`",
            parse_mode='Markdown'
        )
        return

    symbol = context.args[0].upper()
    user_id = update.effective_user.id

    user_data[user_id] = {'symbol': symbol}

    df, error = data_fetcher.get_stock_data(symbol)
    if error:
        await update.message.reply_text(f"\u274c {error}")
        return

    current_price = df['Close'].iloc[-1]
    user_data[user_id]['current_price'] = current_price
    user_data[user_id]['df'] = df

    keyboard = [
        [InlineKeyboardButton(
            f"\U0001f916 Smart Entry (แนะนำ)",
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
        f"\U0001f4ca *{symbol}*\n"
        f"ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
        f"เลือกโหมด Entry:",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการ callback จากปุ่มเลือกโหมด"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    data = query.data

    if data == "mode_smart":
        # ใช้ Smart Entry - ไม่ต้องถามอะไรเพิ่ม
        await query.edit_message_text("\u23f3 กำลังวิเคราะห์ด้วย Smart Entry...")
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

    chart_buf = chart_generator.generate_trading_chart(
        df, symbol,
        use_smart_entry=True,
        entry_price=None,
        tp1_pct=None,
        tp2_pct=None,
        sl_price=None
    )

    current_price = df['Close'].iloc[-1]

    caption = (
        f"\U0001f4ca *{symbol}* | Apexify Smart Trading Plan\n\n"
        f"\U0001f916 ใช้ Smart Entry (Swing Low + ATR)\n"
        f"\U0001f4c8 ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
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

async def ask_tp_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ถาม TP/SL สำหรับ Manual Entry"""
    user_id = update.effective_user.id

    keyboard = [
        [InlineKeyboardButton(
            (f"\u2705 ใช้ค่าเริ่มต้น "
             f"(TP1 +{DEFAULT_TP1_PCT}%, TP2 +{DEFAULT_TP2_PCT}%, SL {DEFAULT_SL_PCT}%)"),
            callback_data="tp_default"
        )],
        [InlineKeyboardButton("\U0001f527 ตั้งค่าเอง", callback_data="tp_custom")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_text(
        f"ราคา Entry: `${user_data[user_id]['entry_price']:,.2f}`\n\n"
        f"เลือกการตั้งค่า TP/SL:",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def entry_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการ callback จากปุ่ม Entry (เก่า - ยังคงไว้เพื่อ compatibility)"""
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
        await update.message.reply_text("\u274c ราคาไม่ถูกต้อง กรุณาพิมพ์ตัวเลข")
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
            "\u274c รูปแบบไม่ถูกต้อง\nตัวอย่าง: `5.6 22.6 -3.5`"
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
        await update.callback_query.edit_message_text("\u23f3 กำลังสร้างกราฟ...")
    else:
        await update.message.reply_text("\u23f3 กำลังสร้างกราฟ...")

    chart_buf = chart_generator.generate_trading_chart(
        df, symbol,
        use_smart_entry=False,
        entry_price=entry,
        tp1_pct=tp1,
        tp2_pct=tp2,
        sl_price=sl_price
    )

    current = df['Close'].iloc[-1]
    change = ((current - entry) / entry) * 100

    caption = (
        f"\U0001f4ca *{symbol}* | Manual Trading Plan\n\n"
        f"\U0001f4b0 Entry: `${entry:,.2f}`\n"
        f"\U0001f3af TP1: `${entry * (1 + tp1/100):,.2f}` (+{tp1}%)\n"
        f"\U0001f3af TP2: `${entry * (1 + tp2/100):,.2f}` (+{tp2}%)\n"
        f"\U0001f6d1 SL: `${entry * (1 + sl_pct/100):,.2f}` ({sl_pct}%)\n\n"
        f"\U0001f4c8 ราคาปัจจุบัน: `${current:,.2f}` ({change:+.2f}%)"
    )

    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.message.reply_photo(
            photo=chart_buf, caption=caption, parse_mode='Markdown'
        )
    else:
        await update.message.reply_photo(
            photo=chart_buf, caption=caption, parse_mode='Markdown'
        )

async def handle_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """จัดการเมื่อผู้ใช้พิมพ์ชื่อสัญลักษณ์โดยตรง - ใช้ Smart Entry"""
    symbol = update.message.text.strip().upper()

    if symbol.startswith('/'):
        return

    await update.message.reply_text(
        f"\U0001f50d กำลังวิเคราะห์ {symbol} ด้วย Smart Entry..."
    )

    df, error = data_fetcher.get_stock_data(symbol)
    if error:
        await update.message.reply_text(
            f"\u274c {error}\n\nลองใช้ `/chart {symbol}` เพื่อตั้งค่าเอง"
        )
        return

    # ใช้ Smart Entry เหมือน Apexify
    chart_buf = chart_generator.generate_trading_chart(
        df, symbol,
        use_smart_entry=True,
        entry_price=None,
        tp1_pct=None,
        tp2_pct=None,
        sl_price=None
    )

    current_price = df['Close'].iloc[-1]

    await update.message.reply_photo(
        photo=chart_buf,
        caption=(
            f"\U0001f4ca *{symbol}* | Apexify Smart Chart\n"
            f"ราคาปัจจุบัน: `${current_price:,.2f}`\n\n"
            f"\U0001f916 Smart Entry: ระบบหาจุดเข้าอัตโนมัติ\n"
            f"ใช้ `/chart {symbol}` เพื่อตั้งค่าเอง"
        ),
        parse_mode='Markdown'
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\u274c ยกเลิกการดำเนินการ")
    return ConversationHandler.END

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "\u274c เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง"
        )

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    chart_conv = ConversationHandler(
        entry_points=[CommandHandler('chart', chart_command)],
        states={
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_entry)],
            TP_SL: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_tp_sl)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('quick', quick_chart))
    application.add_handler(chart_conv)
    # ปุ่มใหม่
    application.add_handler(CallbackQueryHandler(mode_callback, pattern='^mode_'))
    # ปุ่มเก่า (compatibility)
    application.add_handler(CallbackQueryHandler(entry_callback, pattern='^entry_'))
    application.add_handler(CallbackQueryHandler(tp_callback, pattern='^tp_'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_symbol))
    application.add_error_handler(error_handler)

    print("\U0001f916 Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
