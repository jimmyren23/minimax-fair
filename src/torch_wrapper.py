# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

import torch
from torch import nn

# Explore this https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html for GPU
class TorchMLP(nn.Module, ABC):
    """
    A feedforward NN in pytorch using ReLU activiation functions between all layers but the last
    which uses a sigmoid activiation function. Supports an arbitrary number of hidden layers.
    """

    def __init__(self, h_sizes, out_size=1, task='classification'):
        """
        :param h_sizes: input sizes for each hidden layer (including the first)
        :param out_size: defaults to 1 for binary and represents the (positive class probability?)
        :param task: 'classification' or 'regression'
        """
        super(TorchMLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        self.relu = torch.nn.ReLU()


    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = self.relu(layer(x))
        output = self.out(x)  # Sigmoid applied later in BCEWithLogitLoss, and applied automatically in predict_proba
        return output.double()


class MLPRegressor:
    """
    Wrapper class so our MLP looks like an sklearn model
    """

    def __init__(self, h_sizes, lr=0.0001, momentum=0.9, weight_decay=0, task='regression'):
        self.model = TorchMLP(h_sizes)
        self.model.double() # Sets model to double
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def fit(self, X, y, sample_weights, batchsize, minibatch, n_epochs, loss_type='MSE'):
        """
        Fits the model using the entire sample data as the batch size
        """
        # Reformates from numpy to torch
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).double()
        # Puts model in training mode
        self.model.train()
        
        for epoch in range(n_epochs):   
            if minibatch:
                permutation = torch.randperm(X.size()[0])
                for i in range(0, X.size()[0], batchsize):
                    self.optimizer.zero_grad()  # Set gradients to 0 before back propagation for this epoch
                    indices = permutation[i:i+batchsize] # get the indices of the samples that will be included in the batch
                    # Adds samples to batches that are under the batchsize (X.size mod batchsize)
                    indices = indices if (len(indices) == batchsize) else torch.cat((indices, permutation[0: batchsize - len(indices)])) 
                    # print("this indices", indices)
                    batch_x, batch_y = X[indices], y[indices]

                    # Forward pass
                    y_pred = self.model(batch_x)
                    # Compute Loss using Weighted MSE Criterion defined in the class
                    loss = self.weighted_mse_loss(y_pred.squeeze(), batch_y, torch.from_numpy(sample_weights))
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
            else:
                self.optimizer.zero_grad()  # Set gradients to 0 before back propagation for this epoch
                # Forward pass
                y_pred = self.model(X)
                # Compute Loss
                loss = self.weighted_mse_loss(y_pred.squeeze(), y, torch.from_numpy(sample_weights))
                # print(f'Epoch {epoch}: train loss: {loss.item()}')
                # Backward pass
                loss.backward()
                self.optimizer.step()
        return self
    
    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of predicted values, one for each row (instance) in X
        """
        self.model.eval()  # Puts the model in evaluation mode so calls to forward do not update it
        with torch.no_grad():  # Disables automatic gradient updates from pytorch since we are just evaluating
            return self.model(torch.from_numpy(X)).numpy().squeeze()  # Return the prediction, y_hat

    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).mean()

class MLPClassifier:
    """
    Wrapper class so our MLP looks like an sklearn model
    """

    def __init__(self, h_sizes, lr=0.0001, momentum=0.9, weight_decay=0, task='classification'):
        self.model = TorchMLP(h_sizes)
        self.model.double()  # set model type to double
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def fit(self, X, y, sampleweights, n_epochs, loss_type='BCE'):
        """
        Fits the model using the entire sample data as the batch size
        """
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).double()
        self.model.train()  # Puts model in training mode so it updates itself

        # Binary Cross-Entropy Loss with sample weights
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(sampleweights))  # convert weights to tensor

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()  # Set gradients to 0 before back propagation for this epoch
            # Forward pass
            y_pred = self.model(X)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y)
            # print(f'Epoch {epoch}: train loss: {loss.item()}')
            # Backward pass
            loss.backward()
            self.optimizer.step()

        return self

    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        self.model.eval()  # Puts the model in evaluation mode so calls to forward do not update it
        with torch.no_grad():  # Disables automatic gradient updates from pytorch since we are just evaluating
            return torch.sigmoid(self.model(torch.from_numpy(X))).numpy().squeeze()  # Apply sigmoid manually

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return self.predict_proba(X) > 0.5  # Converts probabilistic predictions into binary ones


    