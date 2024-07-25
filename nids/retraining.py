# nids/retraining.py

import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nids import Net

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['timestamp'])
    X = data.drop(columns=['label'])
    y = data['label']
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

def retrain_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):  # Fewer epochs for incremental training
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def retrain(csv_files_path):
    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data(csv_files_path)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Load the existing model
    with open('nids/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        num_features = metadata['num_features']
        num_classes = metadata['num_classes']
    
    model = Net(num_features, num_classes)
    model.load_state_dict(torch.load('nids/model.pth'))
    
    # Retrain the model
    model = retrain_model(model, train_loader)
    
    # Save the updated model
    torch.save(model.state_dict(), 'nids/updated_model.pth')
    
    # Save the scaler and metadata
    with open('nids/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('nids/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("Model, scaler, and metadata (number of features and classes) updated successfully.")
