# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:00:34 2024

@author: anush
"""
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def get_data(data_directory):

    data_splits = {'train': [], 'val': [], 'test': []}
    metadata_splits = {'train': [], 'val': [], 'test': []}
    
    # List all files in the specified directory
    pa_files = os.listdir(data_directory)
    
    # Read each CSV file and concatenate them into one DataFrame
    data = pd.concat([pd.read_csv(os.path.join(data_directory, f)).assign(epa_sensor_id=f.split('~')[0][3:], pa_sensor_id=f.split('~')[1][2:-4]) for f in pa_files if "~" in f and f.lower().endswith('.csv')])
    
    print(f'total number of records after concatenating {len(data)}')
    print(data.columns)
    print(data.head())
    
    # Convert the 'datetime' column to a datetime object
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Set the 'datetime' column as the index
    data.set_index('datetime', inplace=True)
    
    # Ensure columns are numeric, coercing errors to NaN
    numeric_columns = ['epa_pm25', 'pm25_cf_1', 'temperature', 'humidity']
    
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        
    # Aggregate the data to hourly granularity and drop rows with missing values
    hourly_data = data[numeric_columns].groupby(pd.Grouper(freq='h')).mean().dropna(subset=['epa_pm25', 'pm25_cf_1'])

    
    X = hourly_data[['pm25_cf_1', 'temperature', 'humidity']].reset_index(drop=True)
    y = hourly_data['epa_pm25'].reset_index(drop=True)
        
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y.values, test_size=0.2, random_state=42)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42)
    
    # Apply scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    # X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # # Convert target arrays to pandas DataFrames
    # y_train_df = pd.DataFrame(y_train,columns=['epa_pm25'])
    # y_val_df = pd.DataFrame(y_val, columns=['epa_pm25'])
    # y_test_df = pd.DataFrame(y_test, columns=['epa_pm25'])
    
    # Assuming X_train, X_val, X_test, y_train, y_val, y_test are your datasets
    with open('train_data.pkl', 'wb') as f:
        pickle.dump((X_train_scaled, y_train), f)
    
    with open('val_data.pkl', 'wb') as f:
        pickle.dump((X_val_scaled, y_val), f)
    
    with open('test_data.pkl', 'wb') as f:
        pickle.dump((X_test_scaled, y_test), f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Store the dataframes in data_splits instead of numpy arrays
    # data_splits['train'] = (X_train_df, y_train_df)
    # data_splits['val'] = (X_val_df, y_val_df)
    # data_splits['test'] = (X_test_df, y_test_df)
    
    # # Extract and store metadata (sensor IDs and timestamps)
    # metadata_splits['train'].append(X_train.index)
    # metadata_splits['val'].append(X_val.index)
    # metadata_splits['test'].append(X_test.index)


if __name__ == "__main__":
    data_dir = '/TrainData'
    get_data(data_dir)