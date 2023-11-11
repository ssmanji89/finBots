#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
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
from sklearn.linear_model import LinearRegression
from google.oauth2.service_account import Credentials
import pyotp
import pytz
import robin_stocks as rs
import gspread
import schedule
import tweepy
import mpu
from time import sleep
from sklearn.preprocessing import StandardScaler
import schedule
import time
import datetime
import logging
from collections import deque
from datetime import datetime
from scipy.stats import norm, zscore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from threading import Lock
import numpy as np
import os
import pandas as pd
import pandas_ta as ta 

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WATCHLIST_NAMES = ["100 Most Popular", "Popular Recurring Investments", "Upcoming Earnings"]

# Function to calculate key metrics like EMA, RSI, etc.
def calculate_metrics(data, short_window, long_window, rsi_window):
    # Calculate short and long-term Exponential Moving Averages
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return short_ema, long_ema, rsi

# Function to calculate key metrics like EMA, RSI, etc.
def calculate_metrics(data, short_window, long_window, rsi_window):
    # Calculate short and long-term Exponential Moving Averages
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return short_ema, long_ema, rsi

# Function to calculate MACD and Signal line
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Function to calculate Simple Moving Average (SMA)
def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean()


# Function to calculate regression slope for close prices over a given window
def calculate_regression_slope(data, window):
    slopes = []
    for i in range(len(data) - window + 1):
        y = data['Close'].iloc[i:i+window].values.reshape(-1, 1)
        X = np.array(range(window)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0][0]
        slopes.append(slope)
        
    # Pad with NaN for missing values at the beginning
    slopes = [np.nan] * (window - 1) + slopes
    return pd.Series(slopes, index=data.index)





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
    totp = generate_totp(cur_totp_secret)
    print(rs.robinhood.authentication.login(cur_user, cur_pass, mfa_code=totp))
    return


def _helper_sellStock(ticker, quantity, last_trade_price):
    sleep(random.randint(3, 5))
    if quantity > 0:
        try:
            logging.info(rs.robinhood.orders.order(symbol=ticker, quantity=round(quantity, 5), side="sell", timeInForce='gfd'))
        except Exception as e: 
            logging.error(f"{ticker}: {e}")
    return 

def _helper_buyStock(ticker, last_trade_price):
    trade_size = 1.11
    sleep(random.randint(3, 5))
    #try:
    #   logging.info(rs.robinhood.orders.order_buy_fractional_by_price(symbol=ticker, amountInDollars=1.11, timeInForce='gfd', extendedHours=False))
    #except Exception as e: 
    #    logging.error(f"{ticker}: {e}")
    return


def fetch_env_vars():
    """Fetch environment variables."""
    cur_user = os.getenv('curUser')
    cur_pass = os.getenv('curPass')
    cur_totp_secret = os.getenv('curTotp')
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

def get_gpt_sentiment(text):
    import openai
    # Initialize OpenAI API
    api_key = os.getenv('openAiKey') 
    openai.api_key = api_key
    # Define conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Is the news '{text}' positive, negative, or neutral? Only respond with the requested response of positive, negative, or neutral"}
    ]
    # Get response from GPT-3.5 Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=33
    )
    # Extract and return sentiment
    sentiment = response.choices[0].message['content'].strip()
    return sentiment


# Define a mock trade helper function for backtesting
def _helper_mockTrade(ticker, action, last_trade_price):
    return {"ticker": ticker, "action": action, "price": last_trade_price}

# Define the trade function adapted for backtesting
def trade_backtest(df, buydf, pdFundementals, rhSymbol, last_trade_price):
    # Initialize action and details for logging
    action = "Hold"
    
    # Compute MACD conditions
    macd_condition_buy = buydf['MACDs_12_26_9'].iloc[-1] < buydf['MACD_12_26_9'].iloc[-1] or (buydf['MACDs_12_26_9'].iloc[-1] >= buydf['MACDs_12_26_9'].iloc[-3])
    macd_condition_sell = df['MACDs_12_26_9'].iloc[-1] > df['MACD_12_26_9'].iloc[-1]
    
    # Compute recent price slope
    recent_slope = (df['close_price'].iloc[-1] - df['close_price'].iloc[-2])
    
    # Compute fundamental indicators
    pdFundementals = pdFundementals.infer_objects().convert_dtypes()
    pdFundementals['pe_ratio'] = pd.to_numeric(pdFundementals['pe_ratio'], errors='coerce')
    pdFundementals['average_volume'] = pd.to_numeric(pdFundementals['average_volume'], errors='coerce')
    
    dynamic_rsi_upper = buydf['RSI_14'].rolling(window=20).max().iloc[-1]
    dynamic_rsi_lower = buydf['RSI_14'].rolling(window=20).min().iloc[-1]
    
    # Calculate adaptive MACD settings based on recent price action volatility
    volatility = buydf['high_price'].astype(float).iloc[-10:].std()
    
    if volatility < 0.5:
        macd_short, macd_long, macd_signal = 12, 26, 9
    else:
        macd_short, macd_long, macd_signal = 5, 35, 5
    
    # Make a decision based on Priority 1 indicators
    if macd_condition_buy and (df['RSI_2'].iloc[-1] < 10 and df['RSI_9'].iloc[-1] < 55):
        action = "Buy"
    elif macd_condition_sell or (df['RSI_2'].iloc[-1] > dynamic_rsi_lower) or (df['RSI_2'].iloc[-1] > 65 and df['RSI_9'].iloc[-1] > 65):
        action = "Sell"
        
    # Simulate the trade for backtesting
    trade_result = _helper_mockTrade(ticker=rhSymbol, action=action, last_trade_price=last_trade_price)
    
    return trade_result


def trade(rhSymbol, quantity, last_trade_price, pdFundementals):
    # Fetch historical data
    buydf = get_stock_historicals(rhSymbol, "hour", "3month", logon)
    df = get_stock_historicals(rhSymbol, "day", "5year", logon)
    df = pd.DataFrame(df)
    buydf = pd.DataFrame(buydf)
    # Compute MACD conditions
    macd_condition_buy = buydf['MACDs_12_26_9'].iloc[-1] < buydf['MACD_12_26_9'].iloc[-1]
    macd_condition_sell = df['MACDs_12_26_9'].iloc[-1] > df['MACD_12_26_9'].iloc[-1]
    # Dynamic RSI bounds
    dynamic_rsi_upper = buydf['RSI_14'].rolling(window=20).max().iloc[-1]
    dynamic_rsi_lower = buydf['RSI_14'].rolling(window=20).min().iloc[-1]
    # Volume indicators
    pricebook = rs.robinhood.stocks.get_pricebook_by_symbol(rhSymbol)
    asks = pricebook.get("asks", [])
    bids = pricebook.get("bids", [])
    total_ask_volume = sum(ask["quantity"] for ask in asks)
    total_bid_volume = sum(bid["quantity"] for bid in bids)
    # Initialize action
    action = "Hold"
    # Buy Conditions: MACD bullish and higher bid volume
    if macd_condition_buy and (total_bid_volume > total_ask_volume) and (df['RSI_2'].iloc[-1] < 10 and df['RSI_9'].iloc[-1] < 55):
        action = "Buy"
        _helper_buyStock(ticker=rhSymbol, last_trade_price=last_trade_price)
    # Sell Conditions: MACD bearish and RSI overbought and higher ask volume
    elif (macd_condition_sell and (total_ask_volume > total_bid_volume)) or (df['RSI_2'].iloc[-1] > 65 and df['RSI_9'].iloc[-1] > 65 and total_ask_volume > total_bid_volume):
        action = "Sell"
        _helper_sellStock(ticker=rhSymbol, quantity=quantity, last_trade_price=last_trade_price)
    message = f"{action} signal for {rhSymbol}. RSI Upper: {dynamic_rsi_upper}, RSI Lower: {dynamic_rsi_lower}. Bid Volume: {total_bid_volume}, Ask Volume: {total_ask_volume}"
    logging.info(message)
    return action

def main_open_positions(logon):
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
                rh_symbol = row.get('symbol')
                
                # Fetch additional data
                stock_quote = rs.robinhood.get_stock_quote_by_id(instrument_id)
                last_trade_price = float(stock_quote.get('last_trade_price'))
                previous_close = float(stock_quote.get('previous_close'))
                fundamentals = fetch_fundamentals(rh_symbol)
                # Starting a new thread for trading (assuming trade is a defined function)
                threading.Thread(target=trade, args=(rh_symbol, quantity, last_trade_price, fundamentals)).start()
                # trade(rh_symbol, quantity, last_trade_price, fundamentals)
                # trade_backtest(hr_df, day_df, fundamentals, rh_symbol, last_trade_price)
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
        logging.info("Completed processing all rows.")
    except Exception as e:
        logging.error(f"An error occurred in main_open_positions: {e}")


def get_stock_historicals(rhSymbol, interval, span, logon):
    try:
        # replace any "-USD" suffix with ""
        rhSymbol = rhSymbol.replace("-USD","")
        # fetch historical data
        historical_data = rs.robinhood.get_stock_historicals(inputSymbols=rhSymbol,interval=interval,span=span)
        # convert the fetched data into a pandas DataFrame
        df = pd.DataFrame(historical_data)
        # cast prices into float type
        df[['close_price', 'high_price', 'low_price']] = df[['close_price', 'high_price', 'low_price']].astype(float)
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
        df = df.join(ta.macd(close=df['close_price'], fast=12, slow=26, signal=9))
        df = df.join(ta.psar(high=df['high_price'], low=df['low_price'], close=df['close_price']))
        df = df.join(ta.bbands(close=df['close_price'], length=5))
        df = df.join(ta.atr(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=14))
        df = df.join(ta.kc(df['high_price'], df['low_price'], df['close_price'], 3))
        # replace any NaN values with 0
        df = df.fillna(value=0,axis=1)
    except Exception as e:
        logging.error(f"Error performing calculations: {e}")
        return
    return df

def log_alive_message():
    logging.info('Application is running normally')
    
# Scheduling Logic
def job():
    weekday = datetime.datetime.now().strftime('%A')
    if weekday not in ['Saturday', 'Sunday']:
        main_open_positions()

# Function for logging worker status
def log_status():
    logging.info('Worker is running. Checking scheduled tasks...')
    for job in schedule.jobs:
        logging.info(f"Job: {job}")

def job_wrapper():
    current_time = datetime.datetime.now().time()
    start_time = datetime.time(10, 15)
    end_time = datetime.time(14, 45)
    if start_time <= current_time <= end_time:
        main_open_positions(logon)
    return

# Initialize the rate limit queue
rate_limit_queue = deque(maxlen=5)

logon=_1_init()
chicago_tz = pytz.timezone('America/Chicago')


main_open_positions(logon)


