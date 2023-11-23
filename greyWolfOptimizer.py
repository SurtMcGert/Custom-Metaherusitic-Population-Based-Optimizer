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
    def __init__(self, device, model, lossFn, lr=0.01, pop=10, max_iters=10):
        if pop <= 0:
            raise ValueError("Population size must be positive")
        
        params = model.last_layer.parameters()
        super(GreyWolfOptimizer, self).__init__(params, defaults={'pop' : pop})

        self.pop = pop
        self.lossFn= lossFn
        self.lr = lr
     
        self.device = device
        self.model = model
     
        self.max_iters = max_iters
        
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
    def calculateWolf(self, wolf, alpha_pos, beta_pos, delta_pos, p, i, newWolves):
        a = alpha_pos - wolf.position.cuda() * torch.rand_like(p)
        b = beta_pos - wolf.position.cuda() * torch.rand_like(p)
        c = delta_pos - wolf.position.cuda() * torch.rand_like(p)

        updated_p = (a + b + c) / 3

        wolf.position = updated_p
        newWolves[i] = wolf

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
                for wolf in wolves:
                   self.calculateFitness(wolf, index)

                # Multiprocessing setup
                threads = list()
                # Setup space for new wolves, where the results of the calculations will go
                newWolves = np.full((self.pop), Wolf(0))

                # Main algorithm loop
                for _ in range(self.max_iters):
                    for i, wolf in enumerate(wolves):
                        # Grey Wolf calculations
                        alpha_pos = wolves[0].position.cuda()
                        beta_pos = wolves[1].position.cuda()
                        delta_pos = wolves[2].position.cuda()

                        # Splits the calculation of each individual into separate threads
                        # `newWolves` and `i` are passed through
                        # `newWolves` is a numpy array shared between the threads
                        # `i` represents the specific thread/individual
                        # Each individual will write their result of their calculations to their own element in the array
                        t = threading.Thread(target=self.calculateWolf, args=(wolf, alpha_pos, beta_pos, delta_pos, p, i, newWolves))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    # End of multiprocessing
                    
                    # Calculate new fitnesses
                    for wolf in newWolves:
                        self.calculateFitness(wolf, index)

                    # Sort wolves by fitness (this is done slightly differently than before since this one is a numpy array)
                    indices = np.argsort([Wolf.fitness for Wolf in newWolves])
                    wolves = newWolves[indices]
                
                # Set the weight of the layer to the best solution
                print(f"Best of {p}: {newWolves[0].position}")
                self.setWeights(index, newWolves[0].position)
