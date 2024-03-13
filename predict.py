# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:52:49 2024

@author: anush
"""
# Modified to work with API by Tayven Stover 3/13/24
import os
from ray import tune, train
from ray.tune import ResultGrid
from train import train_model, test_accuracy
from ray.train import Result
import matplotlib.pyplot as plt
from RegressionNet import RegressionNet
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


storage_path="/Users/tayvenstover/ray_results/logs"
exp_name = 'pm25_exp8'
experiment_path = os.path.join(storage_path, exp_name)
#print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_model)
result_grid = restored_tuner.get_results()
best_result = result_grid.get_best_result("loss", "min")
config = best_result.config
l1 = config['l1']
batch_size = config['batch_size']

net = RegressionNet(3, 1, l1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net.to(device)
checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
model_state, optimizer_state = torch.load(checkpoint_path)
net.load_state_dict(model_state)
    


data_dir = 'TestData'
file_name = 'EPA1336~PA16541.csv' #TODO this is hardcoded, eventually, using colocation this will be automated.
data = pd.read_csv(os.path.join(data_dir, file_name))

with open('./scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)
    
# Convert the 'datetime' column to a datetime object
data['datetime'] = pd.to_datetime(data['datetime'])
time_data = data['datetime'].tolist()
    
# Set the 'datetime' column as the index
data.set_index('datetime', inplace=True)
    
# Ensure columns are numeric, coercing errors to NaN
numeric_columns = ['epa_pm25', 'pm25_cf_1', 'temperature', 'humidity']
    
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
        
# Aggregate the data to hourly granularity and drop rows with missing values
hourly_data = data[numeric_columns].groupby(pd.Grouper(freq='h')).mean().dropna(subset=['epa_pm25', 'pm25_cf_1'])
#print(hourly_data.head())
    
X = hourly_data[['pm25_cf_1', 'temperature', 'humidity']].reset_index(drop=True)
y = hourly_data['epa_pm25'].reset_index(drop=True)

    
X_scaled = loaded_scaler.transform(X)


sensor_pair_set = CustomDataset(X_scaled, y.values)
sensor_pair_loader = DataLoader(sensor_pair_set, batch_size=batch_size, shuffle=False)

net.eval()
predictions = []
actuals = []

with torch.no_grad():  # Inference mode, no gradients needed
    for batch in sensor_pair_loader:
        X_batch, y_batch = batch
        y_pred = net(X_batch)  # Perform the forward pass
        #print(y_pred.detach().numpy(), y_batch.detach().numpy())
        predictions.extend(y_pred.detach().numpy())  # Collect predictions
        actuals.extend(y_batch.detach().numpy())  # Collect actual values

# Convert to numpy arrays for ease of use
predictions = np.array(predictions).flatten()
actuals = np.array(actuals).flatten()
uncalibrate = hourly_data['pm25_cf_1'].values.flatten()

def format_results():
    results = {
        'prediction_data': [],
    }
    predicted = predictions.tolist()
    uncalibrated = uncalibrate.tolist()

    for i in range(len(time_data)):
        results['prediction_data'].append({'datetime': time_data[i], 'raw': uncalibrated[i], 'prediction': predicted[i]})
    
    return results
