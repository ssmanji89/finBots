#!
#/bash 
# Set the default encoding to UTF-8
import robin_stocks as rs
import numpy as np
import pandas as pd
import pandas_ta as ta
import threading
import pyotp 
import subprocess
import random
from scipy.stats import norm
from datetime import datetime 
from scipy.stats import zscore
from collections import deque
from time import * 
from time import sleep, time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import logging

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

def _1_init():
    cur_user, cur_pass, cur_totp_secret = fetch_env_vars()
    totp = generate_totp(cur_totp_secret)
    return print(rs.robinhood.authentication.login(cur_user, cur_pass, mfa_code=totp))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_and_import_from_requirements(requirements_file):
    import pip  
    with open(requirements_file, 'r') as file:
        modules = [line.strip() for line in file]
    # install modules
    for module in modules:
        pip.main(['install', module])
    # import modules
    imported_modules = {}
    for module in modules:
        module_name = module.split('==')[0]  # remove version if it's there
        try:
            imported_modules[module_name] = __import__(module_name)
            logging.info(f'Successfully imported {module_name}')
        except ImportError:
            logging.error(f'Could not import {module_name}. Is it installed?')
    return imported_modules

def botOrders_Crypto_Sell(ticker,trade_size,mark_price): 
    try: 
        trade_size = round(trade_size,3)
        logging.info(f'Selling {trade_size} of {ticker}.')
        print(rs.robinhood.order_sell_crypto_by_quantity(symbol=ticker,quantity=round((trade_size/mark_price),2)))
        logging.info(f'{rs.robinhood.order_sell_crypto_by_price(rhSymbol,1.11)}')
    except Exception as exception: 
        logging.error(f"Error encountered executing botStuffOrders.botOrders_Stock_SellStock; {ticker} {exception}")
    return

def botOrders_Crypto_Buy(ticker,trade_size,mark_price): 
    try:  
        trade_size = round(trade_size,3)
        logging.info(f'Buying {trade_size} of {ticker}.')
        #print(rs.robinhood.order_buy_crypto_by_quantity(symbol=ticker,quantity=round(trade_size/mark_price,2)))
        # logging.info(f'{rs.robinhood.order_buy_crypto_by_price(rhSymbol,1.11)}')
    except Exception as exception: 
        logging.error(f"Error encountered executing botStuffOrders.botOrders_Stock_SellStock; {ticker} {exception}")
    return

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


def trade(rhSymbol, quantity, last_trade_price, logon):
    # Fetch historical data
    # buydf = get_stock_historicals(rhSymbol, "hour", "3month", logon)
    df = buydf = get_stock_historicals(rhSymbol, "hour", "month", logon)
    action = "Hold"
    # _helper_buyStock(ticker=rhSymbol, last_trade_price=last_trade_price)
    # 
    # Apply the function to the DataFrame with a specified lookback period
    trading_action = evaluate_macd_signals(df)
    if (trading_action == 'Sell'): 
        message = f"Recommended action: {trading_action} for {rhSymbol}. "; 
        logging.info(message)
    elif (trading_action == 'Buy'): 
        message = f"Recommended action: {trading_action} for {rhSymbol}. "; 
        logging.info(message)
    else: 
        message = f"Recommended action: {trading_action} for {rhSymbol}. "; 
        logging.info(message)
    return action

import pandas as pd
import logging
import pandas_ta as ta  # Assuming 'ta' is a technical analysis library

def fetch_historical_data(rhSymbol, interval, span):
    rhSymbol = rhSymbol.replace("-USD", "")
    historical_data = pd.DataFrame(rs.robinhood.get_crypto_historicals(rhSymbol, interval, span))
    return historical_data

def calculate_technical_indicators(df, lengths):
    for length in lengths:
        df[f'MOM_{length}'] = ta.mom(close=df['close_price'], length=length)
        df[f'RSI_{length}'] = ta.rsi(close=df['close_price'], length=length)
        df[f'EMA_{length}'] = ta.ema(close=df['close_price'], length=length)
    return df

def add_additional_indicators(df):
    lengths = [20, 50, 70, 100, 200]
    for length in lengths:
        df = df.join(ta.ema(close=df['close_price'], length=length), how='left', rsuffix=f'_{length}')
    df = df.join(ta.adx(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=3), how='left')
    df = df.join(ta.macd(close=df['close_price'], fast=3, slow=9, signal=7), how='left')
    df = df.join(ta.psar(high=df['high_price'], low=df['low_price'], close=df['close_price']), how='left')
    df = df.join(ta.bbands(close=df['close_price'], length=5), how='left')
    df = df.join(ta.atr(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=14), how='left')
    df = df.join(ta.kc(df['high_price'], df['low_price'], df['close_price'], 3), how='left')
    return df

def get_stock_historicals(rhSymbol, interval, span, logon):
    try:
        df = fetch_historical_data(rhSymbol, interval, span)
        df[['open_price', 'close_price', 'high_price', 'low_price', 'volume']] = df[
            ['open_price', 'close_price', 'high_price', 'low_price', 'volume']
        ].astype(float)
        lengths = [2, 3, 5, 7, 9, 14]
        df = calculate_technical_indicators(df, lengths)
        df = add_additional_indicators(df)
        df.fillna(0, inplace=True)
        df = df.apply(pd.to_numeric, errors='ignore')
        return df
    except Exception as e:
        logging.error(f"> {rhSymbol} {interval}:{span} - Error: {e}")
        return

def rate_limit_handler():
    global rate_limit_queue
    # Add the current time to the queue
    rate_limit_queue.append(time())
    # Remove all timestamps older than 60 seconds (or your rate limit window)
    while rate_limit_queue and rate_limit_queue[0] < time() - 60:
        rate_limit_queue.popleft()
    # If the queue is full, sleep until the oldest item is older than 60 seconds
    if len(rate_limit_queue) == rate_limit_queue.maxlen:
        sleep(1 + rate_limit_queue[0] + 60 - time())
    return

def countdown_timer(seconds):
    for i in range(seconds, 0, -1):
        #logging.info(f"Next iteration in {i} seconds...")
        sleep(1)
    #logging.info("Executing next iteration...")
    return

def main_cryptos():
    logon=_1_init()
    try:
        merged_df = rhCryptos = pd.DataFrame(rs.robinhood.get_crypto_positions())
        all_data = []
        for index, row in merged_df.iterrows():
            try:
                rhSymbol = row['currency']['code'].replace("-USD", "")
                quantity = float(0.00)
                average_buy_price = float(pd.DataFrame(row['cost_bases'])['direct_cost_basis'][0])
                df3 = rs.robinhood.crypto.get_crypto_quote(symbol=rhSymbol)
                last_trade_price = float(df3['mark_price'])
                threading.Thread(target=trade, args=(rhSymbol, quantity, last_trade_price, logon)).start()
                # trade(rh_symbol, quantity, last_trade_price, fundamentals, logon)
                # trade_backtest(hr_df, day_df, fundamentals, rh_symbol, last_trade_price)
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
        logging.info("Completed processing all rows.")
    except Exception as e:
        logging.error(f"An error occurred in main_cryptos: {e}")
        exit

# Initialize the rate limit queue
rate_limit_queue = deque(maxlen=5)

# logon=_1_init()
# main_cryptos() 

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import pytz
import datetime
import random

# Function to generate a random next run time
def schedule_next_run():
    now = datetime.datetime.now(tz=trading_hours)
    # Ensure we run at some random minute within the hour but not in the first 15 minutes
    next_run_minute = random.randint(15, 59)
    next_run_time = now.replace(minute=next_run_minute)
    # If the minute is already past, schedule for the next hour
    if now.minute >= next_run_minute:
        next_run_time += datetime.timedelta(hours=1)
    next_run_time = next_run_time.replace(second=0, microsecond=0)  # Zero out seconds and microseconds
    return next_run_time

def reschedule_job():
    next_run_time = schedule_next_run()
    scheduler.add_job(main_cryptos, DateTrigger(run_date=next_run_time), misfire_grace_time=120)


# Function to reschedule the next run after a job is executed
def job_listener(event):
    if event.code == apscheduler.events.EVENT_JOB_EXECUTED:
        reschedule_job()

trading_hours = pytz.timezone('US/Central')
scheduler = BackgroundScheduler()

# Initial scheduling
reschedule_job()

# Start the scheduler
scheduler.start()
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
import apscheduler.events
# Adding the listener to the scheduler
scheduler.add_listener(job_listener, apscheduler.events.EVENT_JOB_EXECUTED)
