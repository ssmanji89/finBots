import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class StockPredictionModel:
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.best_params_ = None

    def build_model(self, lstm_layers, units, activation, dropout_rate):
        model = Sequential()
        for i in range(lstm_layers):
            return_sequences = True if i < lstm_layers - 1 else False
            model.add(LSTM(units=units, activation=activation, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def grid_search(self, param_grid, X_train, y_train):
        model = KerasRegressor(build_fn=self.build_model, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
        grid_result = grid.fit(X_train, y_train)
        self.best_params_ = grid_result.best_params_
        return grid_result.best_params_, grid_result.best_score_

    def train_best_model(self, X_train, y_train, ticker, epochs=100, batch_size=32):
        self.model = self.build_model(**self.best_params_)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        # Log the best hyperparameters and model performance metrics
        metrics = self.evaluate_model(X_train, y_train)
        self.logger.log_hyperparams_and_metrics(ticker, self.best_params_, metrics)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {'mse': mse}

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
