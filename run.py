# run.py

import torch
import pickle
from scapy.all import sniff, IP, IPv6
from nids import setup_logging
from nids.model import Net
from nids.prediction import run_prediction

# Setup logging
setup_logging()

# Load the model and scaler
with open('nids/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

model = Net(input_size=metadata['num_features'], num_classes=metadata['num_classes'])
model.load_state_dict(torch.load('nids/model.pth'))
model.eval()

with open('nids/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def packet_handler(packet):
    if IP or IPv6 in packet:
        run_prediction(packet, model, scaler)
    else:
        print("Non-IP/IPv6 packet ignored")

if __name__ == '__main__':
    print("Starting packet capture...")
    sniff(prn=packet_handler, store=0)  # prn specifies the function to apply to each packet
