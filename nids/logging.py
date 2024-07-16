# nids/logging.py

import logging

def setup_logging():
    # Configure logging
    logging.basicConfig(filename='nids_logs.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def log_prediction(data, prediction):
    logging.info(f'Data: {data}, Prediction: {prediction.item()}')
