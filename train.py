# train.py

import torch
import pickle
from nids.data_preprocessing import load_and_preprocess_data
from nids.model import train_model

if __name__ == '__main__':
    # Set number of epochs for training
    epochs = 50

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('dataset/MachineLearningCVE/*.csv')
    
    # Train model
    model, num_classes = train_model(X_train, y_train, X_test, y_test, epochs)
    
    # Save model and scaler
    torch.save(model.state_dict(), 'nids/model.pth')
    with open('nids/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save number of features and classes
    with open('nids/model_metadata.pkl', 'wb') as f:
        metadata = {
            'num_features': X_train.shape[1],
            'num_classes': num_classes
        }
        pickle.dump(metadata, f)

    print("Model, scaler, and metadata (number of features and classes) saved.")
