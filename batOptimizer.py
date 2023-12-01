import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import threading
import copy
import random


class BatOptimizer(torch.optim.Optimizer):
    def __init__(self, device, model, lossFn, populationSize=100, Amin=0, gamma=0.9, fmin=0, fmax=100, rmin=0, rmax=1, debug=False):
        if populationSize < 0:
            raise ValueError("Population size must be positive")

        params = model.last_layer.parameters()
        super(BatOptimizer, self).__init__(
            params, defaults={'populationSize': populationSize})

        # Counter
        self.counter = 0

        # Store parameter values
        self.populationSize = populationSize
        self.model = model
        self.Amin = Amin
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        self.rmin = rmin
        self.rmax = rmax

        self.device = device
        self.model = model
        self.lossFn = lossFn
        self.debug = debug

        # Working only on last layer so only 1 group
        for group in self.param_groups:
            # First p is weights, second p is the biases
            for index, p in enumerate(group['params']):
                bats = list()  # make a list to store the bats
                loudnesses = np.random.uniform(
                    1, 2, size=(self.populationSize, 1))
                stddev = 1. / math.sqrt(p.size(-1))
                dic = {}
                # Save bat's id
                dic['id'] = np.array(range(self.populationSize))
                # Initialize bat's position
                shape = (self.populationSize, )+tuple(list(np.shape(p.data)))
                dic['x'] = np.random.uniform(-stddev, stddev, size=shape)
                # Initialize bat's velocity
                dic['v'] = np.random.uniform(-stddev, stddev, size=shape)
                # Initialize bat's frequency
                dic['f'] = self.fmin + ((self.fmax - self.fmin) * np.random.uniform(
                    0, 1, size=self.populationSize))
                # Initialize bat's pulse rate
                dic['r'] = np.random.uniform(
                    self.rmin, self.rmax, size=self.populationSize)
                # Save bat's initial pulse rate
                dic['r0'] = dic['r']
                # Save bat's loudness
                dic['a'] = loudnesses
                # Save bat's potential new solution
                dic['xnew'] = np.zeros(shape=shape)
                self.state[p] = dic

    def step(self):
        # Set GPU usage
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)
        # Working only on last layer so only 1 group
        for group in self.param_groups:
            # First p is weights, second p is the biases
            for index, p in enumerate(group['params']):
                if self.debug == True:
                    print(
                        "=================ORIGINAL POPULATION POSITIONS=================")
                    print((self.state[p])['x'])
                    print(
                        "=================ORIGINAL POPULATION VELOCITIES=================")
                    print((self.state[p])['v'])
                    print(
                        "=================ORIGINAL POPULATION FREQUENCIES=================")
                    print((self.state[p])['f'])
                    print(
                        "=================ORIGINAL POPULATION PULES RATES=================")
                    print((self.state[p])['r'])
                self.counter += 1

                # Calculate bat's fitness
                # Current fitnesses of bats stored in order of bat id
                currentFitness = np.zeros(self.populationSize)
                threads = list()  # create a list for storing threads
                # loop over all of the bats in the population and generate their fitness
                for bat, weights in enumerate((self.state[p])['x']):
                    t = threading.Thread(target=self.calculateFitness, args=(
                        copy.deepcopy(self.model), bat, index, weights, currentFitness))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                if self.debug == True:
                    print("=================FITNESSES=================")
                    print(currentFitness)

                # Sort each bat's index by its fitness
                orderedIndices = np.argsort(currentFitness)
                currentFitness = np.sort(currentFitness)
                (self.state[p])['id'] = [((self.state[p])['id'])[i]
                                         for i in orderedIndices]
                (self.state[p])['x'] = [((self.state[p])['x'])[i]
                                        for i in orderedIndices]
                (self.state[p])['v'] = [((self.state[p])['v'])[i]
                                        for i in orderedIndices]
                (self.state[p])['f'] = [((self.state[p])['f'])[i]
                                        for i in orderedIndices]
                (self.state[p])['r'] = [((self.state[p])['r'])[i]
                                        for i in orderedIndices]
                (self.state[p])['r0'] = [((self.state[p])['r0'])[i]
                                         for i in orderedIndices]
                (self.state[p])['a'] = [((self.state[p])['a'])[i]
                                        for i in orderedIndices]

                # loop over all of the bats in the population and update their positions and get their new fitnesses
                newFitnesses = np.zeros(self.populationSize)
                for bat, id in enumerate((self.state[p])['id']):
                    t = threading.Thread(target=self.updateBats, args=(
                        copy.deepcopy(self.model), index, bat, (self.state[p])['f'], (self.state[p])['v'], (self.state[p])['x'], (self.state[p])['r'], (self.state[p])['r0'], (self.state[p])['a'], ((self.state[p])['x'])[0], currentFitness, newFitnesses))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                # Sort each bat's index by its fitness
                orderedIndices = np.argsort(newFitnesses)
                currentFitness = np.sort(newFitnesses)
                (self.state[p])['id'] = [((self.state[p])['id'])[i]
                                         for i in orderedIndices]
                (self.state[p])['x'] = [((self.state[p])['x'])[i]
                                        for i in orderedIndices]
                (self.state[p])['v'] = [((self.state[p])['v'])[i]
                                        for i in orderedIndices]
                (self.state[p])['f'] = [((self.state[p])['f'])[i]
                                        for i in orderedIndices]
                (self.state[p])['r'] = [((self.state[p])['r'])[i]
                                        for i in orderedIndices]
                (self.state[p])['r0'] = [((self.state[p])['r0'])[i]
                                         for i in orderedIndices]
                (self.state[p])['a'] = [((self.state[p])['a'])[i]
                                        for i in orderedIndices]

                if self.debug == True:
                    print("=================NEW FITNESSES=================")
                    print(newFitnesses)
                    print(
                        "=================NEW POPULATION POSITIONS=================")
                    print((self.state[p])['x'])
                    print(
                        "=================NEW POPULATION VELOCITIES=================")
                    print((self.state[p])['v'])
                    print(
                        "=================NEW POPULATION FREQUENCIES=================")
                    print((self.state[p])['f'])
                    print(
                        "=================NEW POPULATION PULES RATES=================")
                    print((self.state[p])['r'])

                bestBat = ((self.state[p])['id'])[0]
                bestWeights = ((self.state[p])['x'])[0]

                if self.debug == True:
                    print(
                        "=================BEST=================")
                    print("id: ", bestBat)
                    print(bestWeights)

            # Update weights to best bat position
            self.setWeights(self.model, index, bestWeights)

    def calculateFitness(self, model, bat, index, weights, currentFitness):
        """
        Calculates the fitness of a wolf

        Args:
            model (pytorch.module): the model to calculate the fitness on
            wolf (int): the index of the wolf to test
            index (int): the index of the group the at the method is in (weights/biases)
            weights (numpy.ndarray): the weights of this wolf to calculate the fitness of
            currentFitness (numpy.ndarray): the array to store the fitness in
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
        currentFitness[bat] = loss

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

    def updateBats(self, model, index, bat, f, v, x, r, r0, a, best, currentFitnesses, newFitnesses):
        """
        function to update the position of a bat

        Args:
            model (pytorch.module): the model to calculate the fitness on
            index (int): the index of the group the at the method is in (weights/biases)
            bat (int): the index of the bat we are updating
            f (numpy.ndarray): the array of bat frequencies
            v (numpy.ndarray): the array of bat velocities
            x (numpy.ndarray): the array of bat positions
            r (numpy.ndarray): the array of bat pulse rates
            r0 (numpy.ndarray): tha array of the bats initial pulse rates
            a (numpy.ndarray): the array of bat loudnesses
            best (numpy.ndarray): the position of the best bat
            currentFitnesses (numpy.ndarray): an array of the current fitnesses of the bats
            newFitnesses (numpy.ndarray): an array to put the new fitness of the bat into
        """
        beta = np.random.uniform(0, 1, np.shape(f[bat]))
        newV = v[bat] + ((x[bat] - best) * f[bat])
        newX = x[bat] + newV
        f[bat] = self.fmin + ((self.fmax - self.fmin) * beta)
        v[bat] = newV

        if np.random.rand() < r[bat]:
            newX = best + (0.01 * np.random.randn(*list(np.shape(newX))))

        # calculate the fitness of the bats new position
        self.calculateFitness(model, bat, index, newX, newFitnesses)

        if ((newFitnesses[bat] < currentFitnesses[bat]) & (np.random.rand() < a[bat])):
            alpha = 0.999  # if this is bigger, the loudness gets reduced by less each time, meaning more new fitnesses get accepted
            y = 0.1  # if this is smaller, the pulse rate gets closer to the original pulse rate faster
            x[bat] = newX
            a[bat] = a[bat] * alpha
            r[bat] = r0[bat] * (1 - (math.exp(-y)))
        else:
            newFitnesses[bat] = currentFitnesses[bat]
