# nids/logging.py

import logging
from scapy.all import IP, IPv6
import os

def setup_logging():
    # Clear the log file if it exists
    log_file = 'nids_logs.log'
    if os.path.exists(log_file):
        with open(log_file, 'w'):
            pass
    
    # Setup logging configuration
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def log_prediction(packet, prediction, original_data, traffic_type, src_ip, dst_ip):
    summary = packet.summary() if IP in packet or IPv6 in packet else "Non-IP packet"
    features_str = ', '.join([f'{k}: {v}' for k, v in original_data.items()])
    log_message = f'Packet: {summary}, Prediction: {prediction.item()} ({traffic_type}), Source IP: {src_ip}, Destination IP: {dst_ip}, Features: [{features_str}]'
    logging.info(log_message)