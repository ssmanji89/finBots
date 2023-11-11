import StockPredictionModel

class AutomatedModelTuning:
    def __init__(self, tickers, data_loader, logger):
        self.tickers = tickers
        self.data_loader = data_loader
        self.logger = logger

    def run(self, param_grid):
        for ticker in self.tickers:
            print(f'Processing {ticker}...')
            data = self.data_loader(ticker)
            X_train, y_train = self.prepare_data(data)
            model = StockPredictionModel(self.logger)
            best_params, best_score = model.grid_search(param_grid, X_train, y_train)
            model.train_best_model(X_train, y_train, ticker)
            model.save_model(f'models/{ticker}_model.h5')

    def prepare_data(self, data):
        # Assume this function prepares your data
        pass

# Usage example:
# logger = ModelLogger()
# automated_tuner = AutomatedModelTuning(tickers, data_loader, logger)
# param_grid = {
#     'lstm_layers': [1, 2, 3],
#     'units': [30, 50, 70],
#     'activation': ['relu', 'tanh'],
#     'dropout_rate': [0.2, 0.5]
# }
# automated_tuner.run(param_grid)
