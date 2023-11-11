import json
import os

class ModelLogger:
    def __init__(self, log_dir='model_logs'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log_hyperparams_and_metrics(self, ticker, hyperparams, metrics):
        log_data = {
            'hyperparameters': hyperparams,
            'metrics': metrics
        }
        log_file_path = os.path.join(self.log_dir, f'{ticker}_log.json')
        with open(log_file_path, 'w') as log_file:
            json.dump(log_data, log_file, indent=4)
