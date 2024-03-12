# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:00:19 2024

@author: anush
"""
import torch
import torch.nn as nn
import torchbnn as bnn
import torch.nn.functional as F

# Define the neural network class using PyTorch
# class RegressionNet(nn.Module):
#     def __init__(self, input_size, output_size, l1, l2, l3):
#         super(RegressionNet, self).__init__()
#         self.hidden1 = nn.Linear(input_size, l1)
#         self.act1 = nn.LeakyReLU()
#         self.hidden2 = nn.Linear(l1, l2)
#         self.act2 = nn.LeakyReLU()
#         self.hidden3 = nn.Linear(l2, l3)
#         self.act3 = nn.LeakyReLU()
#         self.output = nn.Linear(l3, output_size)  # For regression, we typically have a single output
        
#     def forward(self, x):
#         x = self.act1(self.hidden1(x))
#         x = self.act2(self.hidden2(x))
#         x = self.act3(self.hidden3(x))
#         x = self.output(x)  # No activation (linear) for the output layer in regression
#         return x

# 1. model initialization
# net = LSTMNet(
#     input_size=input_size,  
#     hidden_size=config["hidden_size"],  
#     num_layers=config['num_layers'],  
#     output_size=1,
#     l1=config["l1"],
#     l2=config["l2"],
#     l3=config["l3"]
# )


def get_activation_function(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "Sigmoid":
        return nn.Sigmoid()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == 'PReLU':
        return nn.PReLU()
    # Add more as needed
    else:
        raise ValueError("Unknown activation function")
    
class RegressionNet(nn.Module):
    def __init__(self, input_size, output_size, l1):
        super(RegressionNet, self).__init__()

        l2 = l1 * 2  # l2 is double the size of l1
        l3 = l1 * 4  # l3 is four times the size of l1, or double the size of l2

        # Using nn.Sequential to define the model layers in a compact form
        self.model = nn.Sequential(
            nn.Linear(input_size, l1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Linear(l1, l2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Linear(l2, l3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Linear(l3, output_size)
        )
        
    def forward(self, x):
        # Pass the input through the sequential model
        return self.model(x)
    

class BayesianRegressionNet(nn.Module):
    def __init__(self, input_size, output_size, l1, prior_mu, prior_sigma, activation_name):
        super(BayesianRegressionNet, self).__init__()
        # Get the actual activation function from the string identifier
        self.activation_fn = get_activation_function(activation_name)
        l2 = l1 * 2
        l3 = l1 * 4
        l4 = l1 * 8
        # Define Bayesian linear layers
        self.layer1 = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=input_size, out_features=l1)
        self.layer2 = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=l1, out_features=l2)
        self.layer3 = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=l2, out_features=l3)
        self.layer4 = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=l3, out_features=l4)
        self.layer5 = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=l4, out_features=1)

        
    def forward(self, x):
        # Apply the Bayesian layers with the selected activation function
        x = self.activation_fn(self.layer1(x))
        x = self.activation_fn(self.layer2(x))
        x = self.activation_fn(self.layer3(x))
        x = self.activation_fn(self.layer4(x))
        x = self.layer5(x)
        return x
