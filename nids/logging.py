# nids/logging.py

import logging
from logging.handlers import RotatingFileHandler
from scapy.all import IP, IPv6
import os

def setup_logging():
    log_file = 'nids_logs.log'
    max_log_size = 5 * 1024 * 1024  # 5 MB
    backup_count = 3

    # Setup logging configuration
    handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        handlers=[handler])

def log_prediction(packet, prediction, original_data, traffic_type, src_ip, dst_ip):
    summary = packet.summary() if IP in packet or IPv6 in packet else "Non-IP packet"
    features_str = ', '.join([f'{k}: {v}' for k, v in original_data.items()])
    log_message = f'Packet: {summary}, Prediction: {prediction.item()} ({traffic_type}), Source IP: {src_ip}, Destination IP: {dst_ip}, Features: [{features_str}]'
    logging.info(log_message)
