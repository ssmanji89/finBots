#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
import os
import time
import random
import logging
import datetime
import threading
import pandas as pd
import numpy as np
from collections import deque
from scipy.stats import norm, zscore
import pyotp
import pytz
import robin_stocks as rs
import schedule
import tweepy
import mpu
from time import sleep 
import schedule
import datetime
import logging
from collections import deque
from datetime import datetime
from scipy.stats import norm, zscore
from threading import Lock
import numpy as np
import os
import pandas as pd
import pandas_ta as ta 

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WATCHLIST_NAMES = ["100 Most Popular", "Popular Recurring Investments", "Upcoming Earnings"]
 

def cancel_all_stockOrders(): return print(rs.robinhood.cancel_all_stock_orders())

class RateLimitHandler:
    def __init__(self, rate, per, allow_burst=False):
        self.rate = rate
        self.per = per
        self.allow_burst = allow_burst
        self.time_queue = deque()
        self.lock = Lock()
        self.maxlen = rate if allow_burst else None
        return
    
    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            with self.lock:
                current_time = time()
                while self.time_queue and self.time_queue[0] < current_time - self.per:
                    self.time_queue.popleft()
                if len(self.time_queue) < self.rate:
                    self.time_queue.append(current_time)
                    return f(*args, **kwargs)
                else:
                    sleep_duration = 1 + self.time_queue[0] + self.per - current_time
                    logging.warning(f"Rate limit exceeded. Sleeping for {sleep_duration} seconds.")
                    if sleep_duration > 0:
                        sleep(sleep_duration)
                    return f(*args, **kwargs)
        return wrapped_f

def _1_init():
    cur_user, cur_pass, cur_totp_secret = fetch_env_vars()
    totp = pyotp.TOTP(cur_totp_secret).now()
    print(rs.robinhood.authentication.login(cur_user, cur_pass, mfa_code=totp))
    return

def _helper_sellStock(ticker, quantity, last_trade_price):
    if quantity > 0.000000001:
        try:
            sleep(random.randint(3, 5))
            logging.info(rs.robinhood.orders.order(symbol=ticker, quantity=round(quantity, 5), side="sell", timeInForce='gfd'))
        except Exception as e: 
            logging.error(f"{ticker}: {e}")
    return 

def _helper_buyStock(ticker, last_trade_price):
    trade_size = 1.11
    try:
        sleep(random.randint(3, 5))
        logging.info(rs.robinhood.orders.order_buy_fractional_by_price(symbol=ticker, amountInDollars=1.11, timeInForce='gfd', extendedHours=False))
    except Exception as e: 
        logging.error(f"{ticker}: {e}")
    return

def fetch_env_vars():
    """Fetch environment variables."""
    cur_user = os.environ.get('CURUSER')
    cur_pass = os.environ.get('CURPASS')
    cur_totp_secret = os.environ.get('CURTOTP')
    if not cur_user or not cur_pass or not cur_totp_secret:
        raise ValueError("Environment variables not set correctly.")    
    return cur_user, cur_pass, cur_totp_secret

def generate_totp(totp_secret):
    """Generate TOTP code."""
    return pyotp.TOTP(totp_secret).now()

def fetch_open_positions():
    return pd.DataFrame(rs.robinhood.get_open_stock_positions())

def fetchCustomWatchlist(watchlist_name):
    watchlist_dfs = [pd.DataFrame(rs.robinhood.account.get_watchlist_by_name(name=name, info='results')) for name in [watchlist_name]]
    return pd.concat(watchlist_dfs).drop_duplicates(subset='object_id').sort_values(by='created_at', ascending=False).fillna(value=0, axis=1)

def fetch_watchlist():
    watchlist_dfs = [pd.DataFrame(rs.robinhood.account.get_watchlist_by_name(name=name, info='results')) for name in WATCHLIST_NAMES]
    return pd.concat(watchlist_dfs).drop_duplicates(subset='object_id').sort_values(by='created_at', ascending=False).fillna(value=0, axis=1)

def fetch_fundamentals(rh_symbol):
    return pd.DataFrame(rs.robinhood.stocks.get_fundamentals(rh_symbol, info=None))


def get_stock_historicals(rhSymbol, interval, span, logon):
    try:
        # replace any "-USD" suffix with ""
        rhSymbol = rhSymbol.replace("-USD","")
        # fetch historical data
        df = historical_data = pd.DataFrame(rs.robinhood.get_stock_historicals(inputSymbols=rhSymbol,interval=interval,span=span))
        df[['open_price','close_price','high_price','low_price','volume']] = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    except Exception as exception:
        logging.error(f"> {rhSymbol} {interval}:{span} - Error assembling dataframe; do not continue.")
        return
    # list of lengths for multiple calculations
    lengths = [2, 3, 5, 7, 9, 14]
    try:
        # perform multiple calculations using loop for efficiency
        for length in lengths:
            df = df.join(ta.mom(close=df['close_price'], length=length))
            df = df.join(ta.rsi(close=df['close_price'], length=length))
            df = df.join(ta.ema(close=df['close_price'], length=length))
        # perform other calculations
        try: df = df.join(ta.ema(close=df['close_price'], length=20))
        except: pass
        try: df = df.join(ta.ema(close=df['close_price'], length=50))
        except: pass
        try: df = df.join(ta.ema(close=df['close_price'], length=70))
        except: pass
        try: df = df.join(ta.sma(close=df['close_price'], length=50))
        except: pass
        df = df.join(ta.adx(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=3))
        df = df.join(ta.macd(close=df['close_price'], fast=12, slow=26, signal=9))
        df = df.join(ta.psar(high=df['high_price'], low=df['low_price'], close=df['close_price']))
        df = df.join(ta.bbands(close=df['close_price'], length=5))
        df = df.join(ta.atr(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=14))
        df = df.join(ta.kc(df['high_price'], df['low_price'], df['close_price'], 3))
        # replace any NaN values with 0
        df = df.fillna(value=0,axis=1)
        # cast prices into float type
        # List of all potential columns for conversion
        potential_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 
                            'MOM_2', 'RSI_2', 'EMA_2', 'MOM_3', 'RSI_3', 'EMA_3', 'MOM_5', 'RSI_5', 'EMA_5', 
                            'MOM_7', 'RSI_7', 'EMA_7', 'MOM_9', 'RSI_9', 'EMA_9', 'MOM_14', 'RSI_14', 'EMA_14', 
                            'EMA_20', 'EMA_50', 'EMA_70', 'SMA_50', 'ADX_3', 'DMP_3', 'DMN_3', 
                            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 
                            'PSARaf_0.02_0.2', 'PSARr_0.02_0.2', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 
                            'BBB_5_2.0', 'BBP_5_2.0', 'ATRr_14', 'KCLe_3_2', 'KCBe_3_2', 'KCUe_3_2']
        # Filter the list to include only columns that exist in the DataFrame
        columns_to_convert = [col for col in potential_columns if col in df.columns]
        # Perform the type conversion on the filtered list of columns
        df[columns_to_convert] = df[columns_to_convert].astype(float)
    except Exception as e:
        logging.error(f"Error performing calculations: {e}")
        return
    return df


def analyze_stock(rhSymbol, df, risk_tolerance=0.02):
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import logging
    """
    Analyze stock data focusing on trend following for capital appreciation and limiting losses.
    
    :param rhSymbol: str, Robinhood symbol for the stock
    :param df: DataFrame containing historical stock data and technical indicators
    :param risk_tolerance: float, maximum allowed loss as a fraction of position value
    :return: dict, containing 'action' (Buy, Sell, or Hold) and 'stop_loss' price
    """
    try:
        # Define possible column names for each indicator
        column_mappings = {
            'EMA_9': ['EMA_9', 'EMA_10'],
            'EMA_20': ['EMA_20', 'EMA_21'],
            'EMA_50': ['EMA_50', 'EMA_55'],
            'EMA_200': ['EMA_200', 'SMA_200'],
            'RSI_14': ['RSI_14', 'RSI_13', 'RSI_15'],
            'ADX_14': ['ADX_14', 'ADX_13', 'ADX_15', 'ADX_3'],
            'MACD_12_26_9': ['MACD_12_26_9', 'MACD'],
            'MACDs_12_26_9': ['MACDs_12_26_9', 'MACDs'],
            'BBL_20_2.0': ['BBL_20_2.0', 'BBL_5_2.0', 'BBL'],
            'BBM_20_2.0': ['BBM_20_2.0', 'BBM_5_2.0', 'BBM'],
            'BBU_20_2.0': ['BBU_20_2.0', 'BBU_5_2.0', 'BBU']
        }

        # Function to get the first available column from the mapping
        def get_column(mapping):
            return next((col for col in mapping if col in df.columns), None)

        # Ensure all relevant columns are float type
        float_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']
        float_columns.extend([get_column(mapping) for mapping in column_mappings.values() if get_column(mapping)])

        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)

        action = "Hold"
        current_time = datetime.now().strftime('%B %d, %Y')
        message = f"{current_time}; {action} {rhSymbol}: "

        if len(df) < 200:
            return {"action": "Hold", "stop_loss": None, "message": "Insufficient data for analysis"}

        latest = df.iloc[-1]
        
        # 1. Trend Analysis
        ema_9 = get_column(column_mappings['EMA_9'])
        ema_20 = get_column(column_mappings['EMA_20'])
        ema_50 = get_column(column_mappings['EMA_50'])
        ema_200 = get_column(column_mappings['EMA_200'])

        short_term_trend = ema_9 and ema_20 and latest[ema_9] > latest[ema_20]
        medium_term_trend = ema_20 and ema_50 and latest[ema_20] > latest[ema_50]
        long_term_trend = ema_50 and ema_200 and latest[ema_50] > latest[ema_200]

        # 2. Momentum
        rsi_col = get_column(column_mappings['RSI_14'])
        rsi = latest[rsi_col] if rsi_col else None
        
        # 3. Trend Strength
        adx_col = get_column(column_mappings['ADX_14'])
        adx = latest[adx_col] if adx_col else None

        # 4. MACD
        macd_col = get_column(column_mappings['MACD_12_26_9'])
        signal_col = get_column(column_mappings['MACDs_12_26_9'])
        if macd_col and signal_col:
            macd_line = latest[macd_col]
            signal_line = latest[signal_col]
            macd_histogram = macd_line - signal_line
        else:
            macd_line = signal_line = macd_histogram = None

        # 5. Bollinger Bands
        bbl_col = get_column(column_mappings['BBL_20_2.0'])
        bbm_col = get_column(column_mappings['BBM_20_2.0'])
        bbu_col = get_column(column_mappings['BBU_20_2.0'])
        lower_bb = latest[bbl_col] if bbl_col else None
        middle_bb = latest[bbm_col] if bbm_col else None
        upper_bb = latest[bbu_col] if bbu_col else None

        # 6. Volume
        volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_trend = latest['volume'] > volume_sma * 1.5

        # Decision Making
        buy_signals = 0
        sell_signals = 0

        # Trend signals
        if short_term_trend and medium_term_trend and long_term_trend:
            buy_signals += 1
        elif not short_term_trend and not medium_term_trend and not long_term_trend:
            sell_signals += 1

        # RSI signals
        if rsi is not None:
            if rsi < 30:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 1

        # ADX signal
        if adx is not None and adx > 25:
            if short_term_trend and medium_term_trend:
                buy_signals += 1
            elif not short_term_trend and not medium_term_trend:
                sell_signals += 1

        # MACD signals
        if macd_line is not None and signal_line is not None:
            if macd_line > signal_line and macd_histogram > 0:
                buy_signals += 1
            elif macd_line < signal_line and macd_histogram < 0:
                sell_signals += 1

        # Bollinger Bands signals
        if lower_bb is not None and upper_bb is not None:
            if latest['close_price'] < lower_bb:
                buy_signals += 1
            elif latest['close_price'] > upper_bb:
                sell_signals += 1

        # Volume confirmation
        if volume_trend:
            if buy_signals > sell_signals:
                buy_signals += 1
            elif sell_signals > buy_signals:
                sell_signals += 1

        # Final decision
        if buy_signals >= 3 and buy_signals > sell_signals:
            action = "Buy"
            stop_loss = lower_bb if lower_bb else latest['close_price'] * (1 - risk_tolerance)
            message = f"{current_time}; {action} {rhSymbol}: Multiple buy signals detected; stop loss at {stop_loss:.2f}"
            logging.info(message)
            post_message_to_moneyBots_stockls(message)
            threading.Thread(target=execute_stock_buy_order, args=(rhSymbol, stop_loss, 1.11)).start()
        elif sell_signals >= 3 and sell_signals > buy_signals:
            action = "Sell"
            message = f"{current_time}; {action} {rhSymbol}: Multiple sell signals detected"
            logging.info(message)
        else:
            message = f"{current_time}; Hold {rhSymbol}: No clear direction"
            logging.info(message)
        return {"action": action, "stop_loss": stop_loss if action == "Buy" else None, "message": message}
    except Exception as e:
        error_message = f"Error in analyze_stock for {rhSymbol}: {str(e)}"
        logging.error(error_message)
        return {"action": "Hold", "stop_loss": None, "message": error_message}

# The rest of the code (get_stock_recommendation, etc.) remains the same
def get_stock_recommendation(rhSymbol, interval, span, logon, risk_tolerance=0.05):
    """
    Get a stock recommendation based on trend following and risk management.
    
    :param rhSymbol: str, Robinhood symbol for the stock
    :param interval: str, time interval for historical data
    :param span: str, time span for historical data
    :param logon: object, Robinhood logon instance
    :param risk_tolerance: float, maximum allowed loss as a fraction of position value
    :return: dict, containing 'action' (Buy, Sell, or Hold), 'stop_loss' price, and 'message'
    """
    try:
        df = get_stock_historicals(rhSymbol, interval, span, logon)
        
        if df is None or df.empty:
            return {"action": "Hold", "stop_loss": None, "message": "Insufficient data"}
        
        recommendation = analyze_stock(rhSymbol, df, risk_tolerance)    
        return recommendation
    
    except Exception as e:
        error_message = f"Error in get_stock_recommendation for {rhSymbol}: {str(e)}"
        logging.error(error_message)
        return {"action": "Hold", "stop_loss": None, "message": error_message}


def main_open_positions():
    logon=_1_init()
    try:
        open_positions = fetch_open_positions()
        # full_watchlist = fetch_watchlist()
        watchlist_name = "sullysDividendList"
        full_watchlist = fetchCustomWatchlist(watchlist_name)
        merged_df = full_watchlist.merge(open_positions, left_on='object_id', right_on='instrument_id', how='left')
        all_data = []
        for index, row in merged_df.iterrows():
            try:
                # Extract common data fields
                instrument_id = row.get('object_id') or row.get('instrument_id')
                instrument = row.get('id')
                quantity = float(row.get('open_positions') or row.get('quantity'))
                average_buy_price = float(row.get('price') or row.get('average_buy_price'))
                # Fetch additional data
                stock_quote = rs.robinhood.get_stock_quote_by_id(instrument_id)
                last_trade_price = float(stock_quote.get('last_trade_price'))
                previous_close = float(stock_quote.get('previous_close'))
                rh_symbol = str(stock_quote.get('symbol'))
                fundamentals = pd.DataFrame() # fetch_fundamentals(rh_symbol)
                threading.Thread(target=get_stock_recommendation, args=(rh_symbol, "day", "year", logon, 0.05)).start()
                # trade(rh_symbol, quantity, last_trade_price, fundamentals, logon)
                # trade_backtest(hr_df, day_df, fundamentals, rh_symbol, last_trade_price)
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
        logging.info("Completed processing all rows.")
    except Exception as e:
        logging.error(f"An error occurred in main_open_positions: {e}")
        exit






# Initialize the rate limit queue
rate_limit_queue = deque(maxlen=11)

#df = get_stock_historicals("AAPL", "hour", "month", logon)
#chicago_tz = pytz.timezone('America/Chicago')

logon=_1_init()


from apscheduler.schedulers.background import BackgroundScheduler
import pytz

scheduler = BackgroundScheduler()
trading_hours = pytz.timezone('US/Central')
scheduler.add_job(main_open_positions, trigger='cron', day_of_week='mon-fri', hour='10,12,14', minute='30', timezone=trading_hours)
scheduler.start()

from time import sleep
# main_open_positions() 
# cancel_all_stockOrders()
try:
    # Simulate application activity
    while True:
        time.sleep(6)
        logging.info(f"Handler Alive. ")
except Exception as e: print(e)
 