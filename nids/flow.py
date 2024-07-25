# nids/flow.py

import time
import pandas as pd
from scapy.all import IP, IPv6, TCP, UDP

class Flow:
    def __init__(self, packet):
        self.packets = [packet]
        self.start_time = time.time()
        self.end_time = time.time()
        if IP in packet:
            self.src_ip = packet[IP].src
            self.dst_ip = packet[IP].dst
            self.src_port = packet[IP].sport if TCP in packet or UDP in packet else 0
            self.dst_port = packet[IP].dport if TCP in packet or UDP in packet else 0
        elif IPv6 in packet:
            self.src_ip = packet[IPv6].src
            self.dst_ip = packet[IPv6].dst
            self.src_port = packet[IPv6].sport if TCP in packet or UDP in packet else 0
            self.dst_port = packet[IPv6].dport if TCP in packet or UDP in packet else 0
        self.total_fwd_packets = 1 if self.src_ip == (packet[IP].src if IP in packet else packet[IPv6].src) else 0
        self.total_bwd_packets = 1 if self.src_ip == (packet[IP].dst if IP in packet else packet[IPv6].dst) else 0
        self.total_fwd_bytes = len(packet) if self.src_ip == (packet[IP].src if IP in packet else packet[IPv6].src) else 0
        self.total_bwd_bytes = len(packet) if self.src_ip == (packet[IP].dst if IP in packet else packet[IPv6].dst) else 0

    def update(self, packet):
        self.packets.append(packet)
        self.end_time = time.time()
        if self.src_ip == (packet[IP].src if IP in packet else packet[IPv6].src):
            self.total_fwd_packets += 1
            self.total_fwd_bytes += len(packet)
        else:
            self.total_bwd_packets += 1
            self.total_bwd_bytes += len(packet)

    def get_duration(self):
        return (self.end_time - self.start_time) * 1e6  # duration in microseconds

    def get_features(self):
        features = {
            'Destination Port': self.dst_port,
            'Flow Duration': self.get_duration(),
            'Total Fwd Packets': self.total_fwd_packets,
            'Total Backward Packets': self.total_bwd_packets,
            'Total Length of Fwd Packets': self.total_fwd_bytes,
            'Total Length of Bwd Packets': self.total_bwd_bytes,
            'Fwd Packet Length Max': max([len(p) for p in self.packets if self.src_ip == (p[IP].src if IP in p else p[IPv6].src)], default=0),
            'Fwd Packet Length Min': min([len(p) for p in self.packets if self.src_ip == (p[IP].src if IP in p else p[IPv6].src)], default=0),
            'Fwd Packet Length Mean': sum([len(p) for p in self.packets if self.src_ip == (p[IP].src if IP in p else p[IPv6].src)]) / self.total_fwd_packets if self.total_fwd_packets > 0 else 0,
            'Fwd Packet Length Std': pd.Series([len(p) for p in self.packets if self.src_ip == (p[IP].src if IP in p else p[IPv6].src)]).std(),
            'Bwd Packet Length Max': max([len(p) for p in self.packets if self.src_ip == (p[IP].dst if IP in p else p[IPv6].dst)], default=0),
            'Bwd Packet Length Min': min([len(p) for p in self.packets if self.src_ip == (p[IP].dst if IP in p else p[IPv6].dst)], default=0),
            'Bwd Packet Length Mean': sum([len(p) for p in self.packets if self.src_ip == (p[IP].dst if IP in p else p[IPv6].dst)]) / self.total_bwd_packets if self.total_bwd_packets > 0 else 0,
            'Bwd Packet Length Std': pd.Series([len(p) for p in self.packets if self.src_ip == (p[IP].dst if IP in p else p[IPv6].dst)]).std(),
            'Flow Bytes/s': (self.total_fwd_bytes + self.total_bwd_bytes) / self.get_duration() * 1e6 if self.get_duration() > 0 else 0,
            'Flow Packets/s': (self.total_fwd_packets + self.total_bwd_packets) / self.get_duration() * 1e6 if self.get_duration() > 0 else 0,
        }
        return features
