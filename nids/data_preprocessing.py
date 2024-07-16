# nids/data_preprocessing.py

import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_preprocess_data(csv_files_path):
    # Load all CSV files
    csv_files = glob.glob(csv_files_path)
    print(f"Found CSV files: {csv_files}")  # Debug print
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the path: {csv_files_path}")
    
    dataframes = [pd.read_csv(file) for file in csv_files]
    
    # Concatenate all dataframes
    data = pd.concat(dataframes, ignore_index=True)
    print(f"Concatenated Data Shape: {data.shape}")  # Debug print

    # Handle missing values
    data = data.dropna()

    # Check for the target label column
    possible_label_columns = ['label', 'Label', 'class', 'Class', ' Label']
    label_column = None
    for col in possible_label_columns:
        if col in data.columns:
            label_column = col
            break

    if label_column is None:
        raise ValueError("The target label column is not found in the dataset.")
    
    # Encode categorical variables
    data[label_column] = data[label_column].astype('category').cat.codes

    # Print unique values of the target labels
    print(f"Unique target labels: {data[label_column].unique()}")

    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Convert all columns to numeric, forcing non-numeric to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Normalize numerical features
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    # Split features and labels
    X = data.drop(columns=[label_column])
    y = data[label_column]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler
