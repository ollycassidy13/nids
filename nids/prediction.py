# nids/prediction.py

from kafka import KafkaConsumer
import torch
import pandas as pd
import json
from nids.model import Net
from nids.logging import log_prediction
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, scaler):
    data = pd.DataFrame([data])
    data = pd.get_dummies(data)
    data = scaler.transform(data)
    return torch.tensor(data, dtype=torch.float32)

def run_prediction(model, scaler):
    # Initialize Kafka consumer
    consumer = KafkaConsumer('network_traffic',
                             bootstrap_servers='localhost:9092',
                             value_deserializer=lambda v: json.loads(v.decode('utf-8')))
    
    model.eval()
    # Real-time prediction loop
    for message in consumer:
        data = message.value
        data_tensor = preprocess_data(data, scaler)
        
        # Make prediction
        with torch.no_grad():
            output = model(data_tensor)
            _, prediction = torch.max(output, 1)
            log_prediction(data, prediction)
            print(f'Prediction: {prediction.item()}')
