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

def fetch_watchlist():
    watchlist_dfs = [pd.DataFrame(rs.robinhood.account.get_watchlist_by_name(name=name, info='results')) for name in WATCHLIST_NAMES]
    return pd.concat(watchlist_dfs).drop_duplicates(subset='object_id').sort_values(by='created_at', ascending=False).fillna(value=0, axis=1)

def fetch_fundamentals(rh_symbol):
    return pd.DataFrame(rs.robinhood.stocks.get_fundamentals(rh_symbol, info=None))

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model MSE: {mse}')
    return model


def optimize_lookback(df, eval_metric):
    best_metric = float('-inf')
    best_lookback = None
    # Iterate over a range of lookback periods to find the best one
    for lookback in range(1, len(df) // 2):  # Adjust the range as needed
        df['Action'] = evaluate_macd_signals(df, lookback_period=lookback)
        current_metric = eval_metric(df['Action'], df['close_price'])
        if current_metric > best_metric:
            best_metric = current_metric
            best_lookback = lookback
    return best_lookback

def calculate_volatility(df, short_window=14, long_window=252):
    """
    Calculate the volatility using two windows: a short window for short-term volatility
    and a long window for long-term volatility. The 'long_window' defaults to 252, 
    which is the typical number of trading days in a year, to capture annual volatility.
    """
    short_term_volatility = df['close_price'].pct_change().rolling(window=short_window).std(ddof=0)
    long_term_volatility = df['close_price'].pct_change().rolling(window=long_window).std(ddof=0)
    return short_term_volatility, long_term_volatility

def calculate_volume_ratio(df, short_window=14, long_window=28):
    short_vol = df['volume'].rolling(window=short_window).mean()
    long_vol = df['volume'].rolling(window=long_window).mean()
    return short_vol / long_vol

def evaluate_macd_signals(df):
    # Calculate volatility and volume ratio
    df['Short_Term_Volatility'], df['Long_Term_Volatility'] = calculate_volatility(df)
    df['Volume_Ratio'] = calculate_volume_ratio(df)
    # Determine dynamic lookback_period based on long-term volatility
    if df['Long_Term_Volatility'].iloc[-1] > df['Long_Term_Volatility'].median():
        lookback_period = max(int(df['Long_Term_Volatility'].count() * 0.1), 1)  # Shorter in high volatility
    else:
        lookback_period = min(int(df['Long_Term_Volatility'].count() * 0.2), len(df) - 1)  # Longer in low volatility
    # Adjust lookback_period based on volume ratio
    if df['Volume_Ratio'].iloc[-1] > 1:
        lookback_period = max(int(lookback_period * 0.75), 1)
    elif df['Volume_Ratio'].iloc[-1] < 1:
        lookback_period = min(int(lookback_period * 1.25), len(df) - 1)
    # Select the MACD and signal line values based on the dynamic lookback_period
    macd_line = df['MACD_3_9_7'].iloc[-lookback_period:]
    macd_signal = df['MACDs_3_9_7'].iloc[-lookback_period:]
    macd_histogram = df['MACDh_3_9_7'].iloc[-lookback_period:]
    # Initialize the action as 'Hold'
    action = 'Hold'
    # Check for Sell signal
    if (macd_line.iloc[-1] < macd_signal.iloc[-1] and
        macd_line.iloc[-2] > macd_signal.iloc[-2] and
        all(macd_histogram > 0) and
        macd_histogram.iloc[-1] < 0 and
        df['RSI_2'].iloc[-1] > 70 and df['RSI_9'].iloc[-1] > 65):
        action = 'Sell'
    # Check for Buy signal
    elif (macd_line.iloc[-1] > macd_signal.iloc[-1] and
          macd_line.iloc[-2] < macd_signal.iloc[-2] and
          all(macd_histogram < 0) and
          macd_histogram.iloc[-1] > 0 and
          df['RSI_2'].iloc[-1] < 30 and df['RSI_9'].iloc[-1] < 55):
        action = 'Buy'
    # Return the action recommendation
    return action


def trade(rhSymbol, quantity, last_trade_price, pdFundementals, logon):
    # Fetch historical data
    rs.robinhood.get_stock_historicals()
    # buydf = get_stock_historicals(rhSymbol, "hour", "3month", logon)
    df = buydf = get_stock_historicals(rhSymbol, "day", "year", logon)
    pricebook = rs.robinhood.stocks.get_pricebook_by_symbol(rhSymbol)
    asks = pricebook.get("asks", [])
    bids = pricebook.get("bids", [])
    action = "Hold"
    # _helper_buyStock(ticker=rhSymbol, last_trade_price=last_trade_price)
    # 
    # Apply the function to the DataFrame with a specified lookback period
    trading_action = evaluate_macd_signals(df)
    if (trading_action == 'Sell'): 
        _helper_sellStock(ticker=rhSymbol, quantity=quantity, last_trade_price=last_trade_price)
        message = f"Recommended action: {trading_action} for {rhSymbol}. "; 
        logging.info(message)
    elif (trading_action == 'Buy'): 
        _helper_buyStock(ticker=rhSymbol, last_trade_price=last_trade_price) 
        message = f"Recommended action: {trading_action} for {rhSymbol}. "; 
        logging.info(message)
    else: 
        message = f"Recommended action: {trading_action} for {rhSymbol}. "; 
        logging.info(message)
    return action

def main_open_positions():
    logon=_1_init()
    try:
        open_positions = fetch_open_positions()
        full_watchlist = fetch_watchlist()
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
                threading.Thread(target=trade, args=(rh_symbol, quantity, last_trade_price, fundamentals, logon)).start()
                # trade(rh_symbol, quantity, last_trade_price, fundamentals, logon)
                # trade_backtest(hr_df, day_df, fundamentals, rh_symbol, last_trade_price)
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
        logging.info("Completed processing all rows.")
    except Exception as e:
        logging.error(f"An error occurred in main_open_positions: {e}")
        exit


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
        try: df = df.join(ta.sma(close=df['close_price'], length=200))
        except: pass
        try: df = df.join(ta.sma(close=df['close_price'], length=100))
        except: pass
        try: df = df.join(ta.sma(close=df['close_price'], length=50))
        except: pass
        df = df.join(ta.adx(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=3))
        df = df.join(ta.macd(close=df['close_price'], fast=3, slow=9, signal=7))
        df = df.join(ta.psar(high=df['high_price'], low=df['low_price'], close=df['close_price']))
        df = df.join(ta.bbands(close=df['close_price'], length=5))
        df = df.join(ta.atr(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=14))
        df = df.join(ta.kc(df['high_price'], df['low_price'], df['close_price'], 3))
        # replace any NaN values with 0
        df = df.fillna(value=0,axis=1)
        # cast prices into float type
        df[['open_price','close_price','high_price','low_price','volume','MOM_2','RSI_2','EMA_2','MOM_3','RSI_3','EMA_3','MOM_5','RSI_5','EMA_5','MOM_7','RSI_7','EMA_7','MOM_9','RSI_9','EMA_9','MOM_14','RSI_14','EMA_14','EMA_20','EMA_50','EMA_70','SMA_200','SMA_100','SMA_50','ADX_3','DMP_3','DMN_3','MACD_3_9_7','MACDh_3_9_7','MACDs_3_9_7','PSARl_0.02_0.2','PSARs_0.02_0.2','PSARaf_0.02_0.2','PSARr_0.02_0.2','BBL_5_2.0','BBM_5_2.0','BBU_5_2.0','BBB_5_2.0','BBP_5_2.0','ATRr_14','KCLe_3_2','KCBe_3_2','KCUe_3_2']] = df[['open_price','close_price','high_price','low_price','volume','MOM_2','RSI_2','EMA_2','MOM_3','RSI_3','EMA_3','MOM_5','RSI_5','EMA_5','MOM_7','RSI_7','EMA_7','MOM_9','RSI_9','EMA_9','MOM_14','RSI_14','EMA_14','EMA_20','EMA_50','EMA_70','SMA_200','SMA_100','SMA_50','ADX_3','DMP_3','DMN_3','MACD_3_9_7','MACDh_3_9_7','MACDs_3_9_7','PSARl_0.02_0.2','PSARs_0.02_0.2','PSARaf_0.02_0.2','PSARr_0.02_0.2','BBL_5_2.0','BBM_5_2.0','BBU_5_2.0','BBB_5_2.0','BBP_5_2.0','ATRr_14','KCLe_3_2','KCBe_3_2','KCUe_3_2']].astype(float)
    except Exception as e:
        logging.error(f"Error performing calculations: {e}")
        return
    return df

def cancel_all_stockOrders(): return print(rs.robinhood.cancel_all_stock_orders())


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
 