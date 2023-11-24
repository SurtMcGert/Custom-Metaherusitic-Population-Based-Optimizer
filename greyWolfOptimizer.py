import threading
from queue import Queue

import numpy as np
import torch
import torch.nn as nn


# An individual
class Wolf():
    def __init__(self, position):
        self.position = position # A solution
        self.fitness = 0

class GreyWolfOptimizer(torch.optim.Optimizer):
    def __init__(self, device, model, lossFn, pop=10, max_iters=10):
        if pop <= 0:
            raise ValueError("Population size must be positive")
        
        params = model.last_layer.parameters()
        super(GreyWolfOptimizer, self).__init__(params, defaults={'pop' : pop})

        self.pop = pop
        self.lossFn= lossFn
     
        self.device = device
        self.model = model
     
        self.max_iters = max_iters
        
        # Get the state of the inputs
        for group in self.param_groups:
            for p in group['params']:
                arr = list()
                for _ in range(pop):
                    arr.append(torch.rand_like(p.data))
                self.state[p] = arr

    """
    Calculates the fitness of an individual
    wolf: the individual to calculate, sets its `fitness` field when complete
    index: the index of the group the wolf belongs to
    """
    def calculateFitness(self, wolf, index):
        # Set the weight in the final layer to the solution carried by this individual
        self.setWeights(index, wolf.position)
        
        # Compute the output
        self.model.eval()
        x = self.model.input
        y = self.model.y
        
        y_pred = self.model(x)
        
        # Calculate loss
        loss = self.lossFn(y_pred, y)
        loss = loss.cpu().detach().item()
        
        # Set loss in the individual
        wolf.fitness = loss
        return wolf

    """
    Sets the weights in a specific group
    index: the index of the group
    weights: weights to set to
    """
    def setWeights(self, index, weights):
        with torch.no_grad():
            count = 0
            for param in self.model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1

    """
    Grey Wolf algorithm equations
    """
    def calculateWolf(self, wolf, alpha_pos, beta_pos, delta_pos, p):
        a = alpha_pos - wolf.position.cuda() * torch.rand_like(p)
        b = beta_pos - wolf.position.cuda() * torch.rand_like(p)
        c = delta_pos - wolf.position.cuda() * torch.rand_like(p)

        updated_p = (a + b + c) / 3

        wolf.position = updated_p
        return wolf

    """
    MAIN FUNCTION
    """
    def step(self):
        # Set GPU usage
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)

        # Loop over groups (first weights, then biases)
        for group in self.param_groups:
            # For every weight/bias
            for index, p in enumerate(group['params']):
                # Create population of individuals (wolves)
                wolves = []
                # Set initial state of the wolves based on the output of the previous layer
                for s in self.state[p]:
                    wolves.append(Wolf(s))
                # Calculate fitness of each wolf
                wolves = [self.calculateFitness(wolf, index) for wolf in wolves]
                
                # Sort wolves by fitness
                wolves = sorted(wolves, key=lambda wolf: wolf.fitness)

                # Main algorithm loop
                for _ in range(self.max_iters):
                    alpha_pos = wolves[0].position.cuda()
                    beta_pos = wolves[1].position.cuda()
                    delta_pos = wolves[2].position.cuda()
                        
                    # Apply Grey Wolf algorithm
                    wolves = [self.calculateWolf(wolf, alpha_pos, beta_pos, delta_pos, p) for wolf in wolves]
                    
                    # Calculate new fitnesses
                    wolves = [self.calculateFitness(wolf, index) for wolf in wolves]

                    # Sort wolves by fitness
                    wolves = sorted(wolves, key=lambda wolf: wolf.fitness)

                # Set the weight of the layer to the best solution
                #print(f"Best of {p}: {wolves[0].position}")
                self.setWeights(index, wolves[0].position)
