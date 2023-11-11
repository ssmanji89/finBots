import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import ta  # Technical Analysis library
import logging
import robin_stocks as rs  # Assuming this is the library you are using

class LSTMStockDataProcessor:
    def __init__(self, rhSymbol, interval, span, logon):
        self.rhSymbol = rhSymbol.replace("-USD", "")
        self.interval = interval
        self.span = span
        self.logon = logon
        self.df = None
        self.scaler = None
    def get_stock_historicals(self):
        try:
            historical_data = rs.robinhood.get_stock_historicals(
                inputSymbols=self.rhSymbol, interval=self.interval, span=self.span
            )
            self.df = pd.DataFrame(historical_data)
            self.df[['close_price', 'high_price', 'low_price']] = self.df[['close_price', 'high_price', 'low_price']].astype(float)
        except Exception as exception:
            logging.error(f"> {self.rhSymbol} {self.interval}:{self.span} - Error assembling dataframe; do not continue.")
            return

        lengths = [2, 3, 5, 7, 9, 14]
        try:
            for length in lengths:
                self.df = self.df.join(ta.mom(close=self.df['close_price'], length=length))
                self.df = self.df.join(ta.rsi(close=self.df['close_price'], length=length))
                self.df = self.df.join(ta.ema(close=self.df['close_price'], length=length))
            # ... (rest of your technical analysis calculations)
            self.df = self.df.fillna(value=0, axis=1)
        except Exception as e:
            logging.error(f"Error performing calculations: {e}")
            return

        self.df, self.scaler = self.normalize_features(self.df)

    def normalize_features(self, df):
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled, scaler

    def select_features(self, target_column):
        if self.df is None:
            logging.error("Data not loaded. Call get_stock_historicals first.")
            return None

        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        model = RandomForestRegressor()
        model.fit(X, y)
        sfm = SelectFromModel(model, prefit=True)
        selected_features = self.df.columns[sfm.get_support()]
        return self.df[selected_features.union([target_column])]

"""
    # Usage:
    processor = LSTMStockDataProcessor(rhSymbol='AAPL', interval='day', span='year', logon=None)
    processor.get_stock_historicals()
    selected_features_df = processor.select_features('close_price')
"""