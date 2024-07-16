# run.py

import torch
import pickle
from nids import Net, setup_logging, run_prediction

if __name__ == '__main__':
    # Setup logging
    setup_logging()

    # Load the number of features and classes
    with open('nids/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        num_features = metadata['num_features']
        num_classes = metadata['num_classes']
    
    # Load the model and scaler
    model = Net(input_size=num_features, num_classes=num_classes)
    model.load_state_dict(torch.load('nids/model.pth'))
    
    with open('nids/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Run real-time prediction
    run_prediction(model, scaler)
