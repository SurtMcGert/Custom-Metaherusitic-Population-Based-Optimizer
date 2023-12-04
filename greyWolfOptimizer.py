import copy
import math
import random as rnd
import threading
from queue import Queue

import numpy as np
import torch
import torch.nn as nn


class GreyWolfOptimizer(torch.optim.Optimizer):
    def __init__(self, device, model, lossFn, numOfIters, pop=10, debug=False):
        if pop < 3:
            raise ValueError("Population size must be greater than 3")

        params = model.last_layer.parameters()
        super(GreyWolfOptimizer, self).__init__(params, defaults={'pop': pop})

        self.pop = pop
        self.lossFn = lossFn
        self.numOfIters = numOfIters
        self.currentIter = 0
        self.device = device
        self.model = model
        self.debug = debug

        # Get the state of the inputs
        for group in self.param_groups:
            for p in group['params']:
                stddev = 1. / math.sqrt(p.size(-1))
                arr = list()
                for _ in range(pop):
                    numpyData = np.random.uniform(
                        -stddev, stddev, size=list(np.shape(p.data)))
                    arr.append(numpyData)
                self.state[p] = arr

    def step(self):
        # incrase iteration counter
        self.currentIter += 1
        # Set GPU usage
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)
        # Loop over groups (first weights, then biases)
        for group in self.param_groups:
            # For every weight/bias
            for index, p in enumerate(group['params']):
                if self.debug == True:
                    print("=================ORIGINAL POPULATION=================")
                    print(self.state[p])
                # Calculate the fitness of each wolf
                # make an array to store all the current fitness values
                currentFitness = np.zeros(self.pop)
                threads = list()  # create a list for storing threads
                # loop over all of the wolves in the population and generate their fitness
                for wolf, weights in enumerate(self.state[p]):
                    t = threading.Thread(target=self.calculateFitness, args=(
                        copy.deepcopy(self.model), wolf, index, weights, currentFitness))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                # Sort wolves
                orderedIndices = np.argsort(currentFitness)
                wolves = [(self.state[p])[i] for i in orderedIndices]
                currentFitness = np.sort(currentFitness)

                if self.debug == True:
                    print("=================SORTED=================")
                    print(wolves)
                    print("=================FITNESSES=================")
                    print(currentFitness)

                # Get positions of the best three solutions
                alpha_pos = wolves[0]
                beta_pos = wolves[1]
                delta_pos = wolves[2]

                if self.debug == True:
                    print(
                        "=================ALPHA, BETA, DELTA=================")
                    print("ALPHA: ")
                    print(alpha_pos)
                    print("BETA: ")
                    print(beta_pos)
                    print("DELTA: ")
                    print(delta_pos)

                # Apply Grey Wolf algorithm
                updatedWolves = np.zeros(np.shape(wolves))
                newFitnesses = np.zeros(np.shape(currentFitness))
                threads = list()  # create a list for storing threads
                for wolf, position in enumerate(wolves):
                    random = rnd.randint(0, wolf)
                    t = threading.Thread(target=self.calculateWolf, args=(
                        copy.deepcopy(self.model), index, wolf, position, alpha_pos, beta_pos, delta_pos, np.shape(alpha_pos), updatedWolves, newFitnesses, random))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                # Sort wolves
                orderedIndices = np.argsort(newFitnesses)
                wolves = [(updatedWolves)[i] for i in orderedIndices]
                newFitnesses = np.sort(newFitnesses)

                if self.debug == True:
                    print(
                        "=================UPDATED WOLVES=================")
                    print(updatedWolves)
                    print(
                        "=================UPDATED fitnesses=================")
                    print(newFitnesses)
                    print(
                        "=================BEST=================")
                    print(wolves[0])

                # set the wolf population to the new set of wolves
                self.state[p] = wolves

                # Set the weight of the layer to the best solution
                self.setWeights(self.model, index, wolves[0])

    def calculateFitness(self, model, wolf, index, weights, currentFitness, returnLoss=False):
        """
        Calculates the fitness of a wolf

        Args:
            model (pytorch.module): the model to calculate the fitness on
            wolf (int): the index of the wolf to test
            index (int): the index of the group the at the method is in (weights/biases)
            weights (numpy.ndarray): the weights of this wolf to calculate the fitness of
            currentFitness (numpy.ndarray): the array to store the fitness in
            returnLoss (boolean): set to True to return the loss instead of setting it in currentFitness
        Returns:
            loss (float): the calculated loss value
        """
        model = model.to(self.device)

        # Set the weight in the final layer to the solution carried by this individual
        self.setWeights(model, index, weights)

        # Compute the output
        self.lossFn = self.lossFn.to(self.device)
        model.eval()
        x = (model.input).to(self.device)
        y = (model.y).to(self.device)

        y_pred = model(x)

        # Calculate loss
        loss = self.lossFn(y_pred, y)
        loss = loss.cpu().detach().item()
        if returnLoss == True:
            return loss
        currentFitness[wolf] = loss

    def setWeights(self, model, index, weights):
        """
        Sets the weights in a specific group

        Args:
            model (pytorch.module): the model to set the weights on
            index (int): the index of the group
            weights (numpy.ndarray): weights to set to
        """
        weights = torch.tensor(weights)
        with torch.no_grad():
            count = 0
            for param in model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1

    def calculateWolf(self, model, index, wolf, wolfs_pos, alpha_pos, beta_pos, delta_pos, shape, updatedWolves, newFitnesses, random):
        """
        Applies the Grey Wolf algorithm equations to a wolf, updating its position

        Args:
            model (pytorch.module): the model to calculate the fitness on
            index (int): the index of the group the at the method is in (weights/biases)
            wolf (int): the index of the wolf we are updating
            alpha_pos (np.ndarray): the position of the alpha wolf
            beta_pos (np.ndarray): the position of the beta wolf
            delta_pos (np.ndarray): the position of the delta wolf
            shape (tuple): the shape of the wolfs weights
            updatedWolves (np.ndarray): an array to put the updated wolf position into
            newFitnesses (np.ndarray): an array to put the new fitness of the wolf into
            random (int): a randomly-generated number. This cannot be generated in the function as the function is threaded
        """
        a = 2 - ((2 / self.numOfIters) * self.currentIter)
        r1 = np.random.uniform(0, 1, size=tuple(shape))
        A1 = ((2 * a) * r1) - a
        r1 = np.random.uniform(0, 1, size=tuple(shape))
        A2 = ((2 * a) * r1) - a
        r1 = np.random.uniform(0, 1, size=tuple(shape))
        A3 = ((2 * a) * r1) - a

        r2 = np.random.uniform(0, 1, size=tuple(shape))
        C1 = 2 * r2
        r2 = np.random.uniform(0, 1, size=tuple(shape))
        C2 = 2 * r2
        r2 = np.random.uniform(0, 1, size=tuple(shape))
        C3 = 2 * r2

        Da = (C1 * alpha_pos) - wolfs_pos
        Db = (C2 * beta_pos) - wolfs_pos
        Dd = (C3 * delta_pos) - wolfs_pos

        X1 = alpha_pos - (A1 * Da)
        X2 = beta_pos - (A2 * Db)
        X3 = delta_pos - (A3 * Dd)

        updated_p = (X1 + X2 + X3) / 3

        newLoss = self.calculateFitness(model, wolf, index, updated_p, newFitnesses, True)
        if newLoss < newFitnesses[wolf]:
            # calculate the fitness of the new wolf
            newFitnesses[wolf] = newLoss
            updatedWolves[wolf] = updated_p
        else:
            updatedWolves[wolf] = updatedWolves[random]
            newFitnesses[wolf] = self.calculateFitness(model, wolf, index, updatedWolves[random], newFitnesses, True)