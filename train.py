# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:24:31 2024

@author: anush
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from LSTMNet import LSTMNet
from RegressionNet import RegressionNet
import matplotlib.pyplot as plt
from torcheval.metrics import R2Score

import numpy as np
import math, os, shutil
import tempfile

import ray
from ray import train, tune
from ray.train import Checkpoint

import pickle

from CustomDataset import CustomDataset

import pandas as pd

input_size = 3
hidden_size = 32
num_layers = 5
output_size = 1


def train_model(config):
    input_size = 3
    net = RegressionNet(input_size = input_size, output_size = 1, l1 = config['l1'])
    
    # 2. Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            
    net.to(device)
    
    # 3. Criterion
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    # Define weights for each loss component
    weight_mse = 0.01  # example weight for MSE
    weight_mae = 0.01  # example weight for MAE
        
    # 4. optimizer
    if config["optimizer"] == "Adam":
       optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
       optimizer = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD":
       optimizer = torch.optim.SGD(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
   
    # 5. checkpoint 
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            
    # 6. load data
    with open('/train_data.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)

    with open('/val_data.pkl', 'rb') as f:
        X_val, y_val = pickle.load(f)
        
    # X_train = pd.DataFrame(X_train)
    # y_train = pd.DataFrame(y_train)
    # X_val = pd.DataFrame(X_val)
    # y_val = pd.DataFrame(y_val)
        
    train_set = CustomDataset(X_train, y_train)
    val_set = CustomDataset(X_val, y_val)

    # 8. Train loader
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True, num_workers = 8)
    
    # 9. Validation loader
    val_loader  = DataLoader(val_set, batch_size=int(config["batch_size"]), shuffle=True, num_workers = 8)

    # 10. Model Traiing
    for epoch in range(config['epochs']):

        # Initialize metrics parameters
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        train_r2_score_metric = R2Score()

        net.train()
        # Loop through data loader
        for i, data in enumerate(train_loader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            train_input, train_target = data

            
            # train_input = train_input.unsqueeze(dim = 1)
            train_input, train_target = train_input.to(device), train_target.to(device)  
            
            # zero the parameter gradients 
            optimizer.zero_grad()
            
            # forward + backward + optimize
            out = net(train_input)
            print(out.detach().numpy())

            out = out.squeeze()
            train_target = train_target.squeeze()
            # loss = criterion(out, train_target)
            loss_mse = criterion_mse(out, train_target)
            loss_mae = criterion_mae(out, train_target)
            
            # Combine losses
            combined_loss = (weight_mse * loss_mse) + (weight_mae * loss_mae)

            combined_loss.backward()
            optimizer.step()
            
            # print statistics
            train_loss += combined_loss.item() * train_input.size(0)
            # train_mse += ((out - train_target) ** 2).sum().item()
            train_mse += loss_mse.item() * train_input.size(0)
            train_mae += loss_mae.item() * train_input.size(0)
            train_r2_score_metric.update(out, train_target)
            
            # Update R-squared variables
            # total_variance += ((train_target - train_target.mean()) ** 2).sum().item()
            # explained_variance += ((train_target - out) ** 2).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)  # Final MSE for training
        train_mae /= len(train_loader.dataset)
        train_r2score = train_r2_score_metric.compute()
        
        net.eval()
        # Validation loss
        val_loss = 0.0
        val_mse = 0.0
        val_mae = 0.0
        val_r2_score_metric = R2Score()
        
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data
                # inputs = inputs.unsqueeze(dim = 1)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs).squeeze()
                labels = labels.squeeze()
                
                # loss = criterion(outputs, labels)
                loss_mse = criterion_mse(outputs, labels)
                loss_mae = criterion_mae(outputs, labels)
                
                combined_loss = (weight_mse * loss_mse) + (weight_mae * loss_mae)
                val_loss += combined_loss.item() * inputs.size(0)
                    
                # val_mse += ((outputs - labels) ** 2).sum().item() 
                val_mse += loss_mse.item() * inputs.size(0)
                val_mae += loss_mae.item() * inputs.size(0)
                val_r2_score_metric.update(outputs, labels)
                
        val_loss /= len(val_loader.dataset)  # Average validation loss
        val_mse /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_r2score = val_r2_score_metric.compute().item()
        
        metrics = {
                    "loss": val_loss,
                    "mse": val_mse,
                    "rmse": math.sqrt(val_mse),
                    "r2score": val_r2score,
                        }
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            print(f'temp check point dir {temp_checkpoint_dir}')
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                metrics,
                checkpoint=checkpoint,
            )

    
    print('Finished training...')

def test_accuracy(best_result, plot=True):
    # best_trained_model = LSTMNet(input_size = input_size, hidden_size = best_result.config["hidden_size"], num_layers = best_result.config["num_layers"],\
    #                          output_size = 1, l1 = best_result.config["l1"], l2 = best_result.config["l2"], l3 = best_result.config["l3"])
        
    best_trained_model =  RegressionNet(input_size = input_size, output_size = 1, l1 = best_result.config['l1'])
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    
    print('Loading the test data...')

    with open('/test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
        
    # X_test = pd.DataFrame(X_test)
    # y_test = pd.DataFrame(y_test)

    test_set = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=int(best_result.config["batch_size"]), shuffle=True)
    
    best_trained_model.eval()
    
    test_loss = 0.0
    test_mse = 0.0
    test_mae = 0.0
    # Define weights for each loss component
    weight_mse = 0.01  # example weight for MSE
    weight_mae = 0.01  # example weight for MAE

    test_r2_score_metric = R2Score()
    actuals, predictions = [],[]
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
        
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data

            # inputs = inputs.unsqueeze(dim = 1)
            inputs, labels = inputs.to(device), labels.to(device)
            outs = best_trained_model(inputs)
            outs = outs.squeeze()
 
            labels = labels.squeeze()
            actuals.extend(labels.cpu().numpy())
            predictions.extend(outs.cpu().numpy())
            
            # # For regression, outputs are directly the predicted values
            # loss = criterion(outs, labels)
            # test_loss += loss.item() * inputs.size(0)
            
            # # MSE calculation
            # test_mse += ((outs - labels) ** 2).sum().item()
            # Compute both losses
            loss_mse = criterion_mse(outs, labels)
            loss_mae = criterion_mae(outs, labels)
            
            combined_loss = (weight_mse * loss_mse) + (weight_mae * loss_mae)
            test_loss += combined_loss.item() * inputs.size(0)
            
            # Update individual metrics
            test_mse += loss_mse.item() * inputs.size(0)
            test_mae += loss_mae.item() * inputs.size(0)

            test_r2_score_metric.update(outs, labels)
            
    test_loss /= len(test_loader.dataset)
    test_mse /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    test_rmse = torch.sqrt(torch.tensor(test_mse))
    test_r2score = test_r2_score_metric.compute()
    
        
    return test_loss, test_mse, test_rmse, test_r2score
