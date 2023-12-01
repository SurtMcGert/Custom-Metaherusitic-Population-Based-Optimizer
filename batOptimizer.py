import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from batLibrary import *
import time


class batOptimizer(torch.optim.Optimizer):
    def __init__(self, device, model, lossFn):
        
        params = model.last_layer.parameters()
        super(batOptimizer, self).__init__(params, defaults={})

        #Counter
        self.counter = 0

        #Store parameter values
        self.model = model

        self.device = device
        self.model = model
        self.lossFn = lossFn

        
    def step(self):
        # Set GPU usage
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)
        # Working only on last layer so only 1 group
        for group in self.param_groups:

            # First p is weights, second p is the biases
            for index, p in enumerate(group['params']):
                self.counter += 1
                # Call algorithm for weights
                if self.counter % 2 != 0:
                    bats = BatAlgorithm(500, 100, 20, 0.7, 0.9, 0, 100, -1, 1, self.calculateFitness)
                # Call algorithm for biases
                else:
                    bats = BatAlgorithm(10, 100, 20, 0.7, 0.9, 0, 100, -1, 1, self.calculateFitness)
                best_bat = bats.move_bat()
                self.setWeights(index, torch.tensor(best_bat))

    def calculateFitness(self, dimensions, weights):
        """function to calculate the fitness of an individual for the given index parameter (weights or biases) then save the loss in currentFitness"""
        #Check the size of weights to determine whether to set weights or bias
        if len(weights) == 500:
            self.setWeights(0, weights)
        else:
            self.setWeights(1, weights)
        model = self.model
        # calculate the individuals fitness
        # set the model in evaluation mode
        model.eval()
        x = model.input
        y = model.y
        y_pred = model(x)
        loss = self.lossFn(y_pred, y)
        loss = loss.cpu().detach().item()
        if loss == 0:
            loss = 0.000001
        return loss
    
    
    def setWeights(self, index, weights):
        with torch.no_grad():
            count = 0
            for param in self.model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights.to(param.dtype)))
                    break
                count += 1