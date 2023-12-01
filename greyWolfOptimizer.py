import math
import threading
from queue import Queue

import numpy as np
import torch
import torch.nn as nn


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

    def calculateFitness(self, solution, index, i, fitnesses):
        """
        Calculates the fitness of a solution

        Args:
            wolf (torch.Tensor): the solution to calculate from
            index (int): the index of the group the solution belongs to

        Returns:
            loss (float): the fitness of the solution
        """
        # Set the weight in the final layer to the solution carried by this individual
        self.setWeights(index, torch.tensor(solution))
        
        # Compute the output
        self.model.eval()
        x = self.model.input
        y = self.model.y
        
        y_pred = self.model(x)
        
        # Calculate loss
        loss = self.lossFn(y_pred, y)
        loss = loss.cpu().detach().item()
        
        # Set loss in the individual
        fitnesses[i] = loss

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

    def calculateWolf(self, position, alpha_pos, beta_pos, delta_pos, p):
        """
        Applies the Grey Wolf algorithm equations to a wolf, updating its position

        Args:
            position (numpy.Array): position to calculate from
            alpha_pos (numpy.Array): the position of the alpha wolf
            beta_pos (numpy.Array): the position of the beta wolf
            delta_pos (numpy.Array): the position of the delta wolf
            p (torch.nn.paramter.Parameter): the current parameter. This is used to create a randomly-generated tensor of the same size

        Returns:
            updated_p (numpy.Array): newly-calculated position

        """
        d_a = alpha_pos - position * np.random.rand(*np.shape(p[1]['position']))
        d_b = beta_pos - position * np.random.rand(*np.shape(p[1]['position']))
        d_c = delta_pos - position * np.random.rand(*np.shape(p[1]['position']))
        
        a = position - alpha_pos * d_a
        b = position - beta_pos * d_b
        c = position - delta_pos * d_c

        updated_p = (a + b + c) / 3

        return updated_p
    
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
                fitnesses = np.zeros(self.pop)
                for i in range(self.pop):
                    dic = {}
                    dic['id'] = i
                    dic['position'] = np.random.uniform(-1, 1, size=list(np.shape(p.data)))
                    wolves.append(dic)
                    self.calculateFitness(dic['position'], index, i, fitnesses)

                self.state[p] = np.array(wolves)
                state = self.state[p]
                # Main algorithm loop
                for _ in range(self.max_iters):
                    wolvesSortedIndexes = np.argsort(fitnesses)

                    # Get positions of the best three solutions
                    alpha_pos = wolves[wolvesSortedIndexes[0]]['position']
                    beta_pos = wolves[wolvesSortedIndexes[1]]['position']
                    delta_pos = wolves[wolvesSortedIndexes[2]]['position']

                    for i, wolf in enumerate(state):
                        # Apply Grey Wolf algorithm
                        wolf['position'] = self.calculateWolf(wolf['position'], alpha_pos, beta_pos, delta_pos, state)
                        self.calculateFitness(wolf['position'], index, i, fitnesses)
                    
                # Set the weight of the layer to the best solution
                wolvesSortedIndexes = np.argsort(fitnesses)
                self.setWeights(index, torch.tensor(wolves[wolvesSortedIndexes[0]]['position']))