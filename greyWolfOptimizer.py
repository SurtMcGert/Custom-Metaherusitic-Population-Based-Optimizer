import math
import threading
from queue import Queue

import numpy as np
import torch
import torch.nn as nn


class Wolf():
    def __init__(self, position):
        """
        Represents a solution

        Args:
            position (torch.Tensor): the position of the wolf. Equivalent to the solution the indivudal represents
        """

        self.position = position # A solution
        self.fitness = 0

        print(type(self.position))

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

    def calculateFitness(self, wolf, index):
        """
        Calculates the fitness of a wolf

        Args:
            wolf (Wolf): the wolf to calculate
            index (int): the index of the group the wolf belongs to

        Returns:
            wolf (Wolf): the updated wolf
        """
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

    def setWeights(self, index, weights):
        """
        Sets the weights in a specific group

        Args:
            index (int): the index of the group
            weights (torch.Tensor): weights to set to
        """
        with torch.no_grad():
            count = 0
            for param in self.model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1

    def calculateWolf(self, wolf, alpha_pos, beta_pos, delta_pos, p):
        """
        Applies the Grey Wolf algorithm equations to a wolf, updating its position

        Args:
            wolf (Wolf): the wolf to update the position of
            alpha_pos (torch.Tensor): the position of the alpha wolf
            beta_pos (torch.Tensor): the position of the beta wolf
            delta_pos (torch.Tensor): the position of the delta wolf
            p (torch.nn.paramter.Parameter): the current parameter. This is used to create a randomly-generated tensor of the same size

        Returns:
            wolf (Wolf): the updated wolf

        """
        d_a = alpha_pos - wolf.position.cuda() * torch.rand_like(p)
        d_b = beta_pos - wolf.position.cuda() * torch.rand_like(p)
        d_c = delta_pos - wolf.position.cuda() * torch.rand_like(p)
        
        a = wolf.position.cuda() - alpha_pos * d_a
        b = wolf.position.cuda() - beta_pos * d_b
        c = wolf.position.cuda() - delta_pos * d_c

        updated_p = (a + b + c) / 3

        wolf.position = updated_p
        return wolf
    
    def calculateFitnessProportionate(self, wolves):
        """
        Calculates the fitness proportionate of each wolf

        Args:
            wolves (Wolf[]): the population of wolves
        
        Returns:
            wolves (Wolf[]): the updated population of wolves
        """
        denominator = 0
        
        for wolf in wolves:
            wolf.fitness = np.reciprocal(wolf.fitness)
            denominator += wolf.fitness
            
        wolves = [self.calculateFitnessProportionateHelper(wolf, denominator) for wolf in wolves]    

        return wolves

    def calculateFitnessProportionateHelper(self, wolf, denominator):
        """
        Calculates the fitness proportion of a wolf

        Args:
            wolf (Wolf): the wolf to update
            denominator (torch.Tensor): the sum of the fitnesses of the population

        Returns:
            wolf (Wolf): the updated wolf
        
        """
        wolf.fitness = (1/wolf.fitness)/denominator
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

                print(np.shape(p))
                # Main algorithm loop
                for _ in range(self.max_iters):
                                    
                    # Calculate the fitness of each wolf
                    wolves = [self.calculateFitness(wolf, index) for wolf in wolves]
                    # Calculate the fitness proportionate
                    wolves = self.calculateFitnessProportionate(wolves)
                    
                    # Sort wolves by fitness proportionate
                    wolves = sorted(wolves, key=lambda wolf: wolf.fitness)
                    
                    # Get positions of the best three solutions
                    alpha_pos = wolves[0].position.cuda()
                    beta_pos = wolves[1].position.cuda()
                    delta_pos = wolves[2].position.cuda()
                        
                    # Apply Grey Wolf algorithm
                    wolves = [self.calculateWolf(wolf, alpha_pos, beta_pos, delta_pos, p) for wolf in wolves]

                # Calculate the fitness of each wolf
                wolves = [self.calculateFitness(wolf, index) for wolf in wolves]
                # Calculate the fitness proportionate
                wolves = self.calculateFitnessProportionate(wolves)
                
                # Set the weight of the layer to the best solution
                self.setWeights(index, wolves[0].position)