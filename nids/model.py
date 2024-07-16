# nids/model.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(X_train, y_train):
    # Convert data to numpy arrays before creating tensors
    X_train_array = X_train.to_numpy()
    y_train_array = y_train.to_numpy()
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Get the number of unique classes
    num_classes = len(pd.unique(y_train))
    
    # Initialize the model, loss function, and optimizer
    model = Net(X_train.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(20):  # Number of epochs
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{20}, Loss: {loss.item()}')
    
    return model, num_classes
