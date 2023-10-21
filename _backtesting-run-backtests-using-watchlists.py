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


# Initialize the insights DataFrame
columns = [
    "timestamp",
    "session_id",
    "ticker",
    "action",
    "outcome",
    "price",
    "volume",
    "volatility",
    "macd",
    "rsi_14",
    "bollinger_upper",
    "bollinger_lower",
    "parabolic_sar",
    "adx",
    "pe_ratio",
    "dividend_yield",
    "market_cap",
    "signal_strength",
    "price_slope",
    "sentiment_score",
    "sector",
    "industry",
    "time_of_day",
    "day_of_week",
    "market_events",
    "notes"
]
insights_df = pd.DataFrame(columns=columns)
# Create the directory if it doesn't exist
output_directory = "csv"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


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
                hr_df = get_stock_historicals(rh_symbol,"hour","3month",logon)
                day_df = get_stock_historicals(rh_symbol,"day","5year",logon)
                # Starting a new thread for trading (assuming trade is a defined function)
                # threading.Thread(target=trade, args=(rh_symbol, quantity, last_trade_price, fundamentals)).start()
                trade_backtest(hr_df, day_df, fundamentals, rh_symbol, last_trade_price)
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
        logging.info("Completed processing all rows.")
    except Exception as e:
        logging.error(f"An error occurred in main_open_positions: {e}")
    return


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

def fetch_env_vars():
    """Fetch environment variables."""
    cur_user = os.getenv('curUser')
    cur_pass = os.getenv('curPass')
    cur_totp_secret = os.getenv('curTotp')
    if not cur_user or not cur_pass or not cur_totp_secret:
        raise ValueError("Environment variables not set correctly.")    
    return cur_user, cur_pass, cur_totp_secret

def _1_init():
    cur_user, cur_pass, cur_totp_secret = fetch_env_vars()
    totp = generate_totp(cur_totp_secret)
    print(rs.robinhood.authentication.login(cur_user, cur_pass, mfa_code=totp))
    return


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
    
    # Add insights into the DataFrame
    insights_row = {
        "timestamp": timestamp,
        "session_id": session_id,
        "ticker": rhSymbol,
        "action": action,
        "outcome": "",  # This needs to be filled out based on the results of the trade
        "price": last_trade_price,
        "volume": df['volume'].iloc[-1],
        "volatility": volatility,
        "macd": df['MACD_12_26_9'].iloc[-1],
        "rsi_14": df['RSI_14'].iloc[-1],
        "bollinger_upper": df['BBU_5_2.0'].iloc[-1],
        "bollinger_lower": df['BBL_5_2.0'].iloc[-1],
        "parabolic_sar": df['PSARs_0.02_0.2'].iloc[-1],
        "adx": df['ADX_3'].iloc[-1],
        "pe_ratio": pdFundementals['pe_ratio'].iloc[0],
        "dividend_yield": pdFundementals['dividend_yield'].iloc[0],
        "market_cap": pdFundementals['market_cap'].iloc[0],
        "signal_strength": "",  # Derived metric, can be a function of other metrics
        "price_slope": recent_slope,
        "sentiment_score": "",  # Derived from news or social media analysis
        "sector": pdFundementals['sector'].iloc[0],
        "industry": pdFundementals['industry'].iloc[0],
        "time_of_day": "",  # This should be derived from the timestamp
        "day_of_week": "",  # This should be derived from the timestamp
        "market_events": "",  # Any market events on that day
        "notes": ""  # Any additional notes
    }
    
    insights_df = insights_df.append(insights_row, ignore_index=True)
    
    # Export insights to CSV
    csv_path = os.path.join('csv', f"insights.csv")
    insights_df.to_csv(csv_path, index=False)
    
    return trade_result

logon=_1_init()
chicago_tz = pytz.timezone('America/Chicago')


main_open_positions(logon)

