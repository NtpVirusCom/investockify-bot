import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        pass
    
    def get_stock_data(self, symbol: str, period: str = "3mo", interval: str = "1d"):
        """
        ดึงข้อมูลราคาหุ้น/สินค้าโภคภัณฑ์
        รองรับ: AAPL, GC=F (ทอง), SI=F (เงิน), CL=F (น้ำมัน), ฯลฯ
        """
        try:
            # เพิ่ม .NS สำหรับตลาดอินเดีย, .BK สำหรับตลาดไทย ถ้าจำเป็น
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None, "ไม่พบข้อมูลสำหรับสัญลักษณ์นี้"
            
            return df, None
            
        except Exception as e:
            return None, f"เกิดข้อผิดพลาด: {str(e)}"
    
    def get_current_price(self, symbol: str):
        """ดึงราคาล่าสุด"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current = info.get('regularMarketPrice') or info.get('currentPrice')
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
            
            if current and prev_close:
                change_pct = ((current - prev_close) / prev_close) * 100
                return {
                    'price': current,
                    'change_pct': change_pct,
                    'prev_close': prev_close
                }
            return None
        except:
            return None