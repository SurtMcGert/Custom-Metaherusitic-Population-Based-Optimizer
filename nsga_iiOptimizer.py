# Import the necessary libraries
import torch
import torch.nn as nn
import random
import numpy as np
import threading
import itertools
import copy
import array
from scipy import stats
import math
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import functools


class NSGAIIOptimizer(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, device, model, lossFn, weightLowerBound, weightUpperBound, pop=20, numOfBits=4):
        params = model.last_layer.parameters()
        super(NSGAIIOptimizer, self).__init__(
            params, defaults={'pop': pop})
        self.numOfBits = numOfBits  # save the number of bits for each decision variable
        self.popSize = pop  # save the population size
        self.device = device  # save the device
        self.model = model  # save the model that this optimizer is for
        self.lossFn = lossFn  # save the loss function for this optimizer
        self.state = {}  # a dictionary to store the populations
        self.toolbox = base.Toolbox()  # the deap toolbox
        self.CXPB = 0.9
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d',
                       fitness=creator.FitnessMin)
        # register all of the toolbox methods
        self.toolbox.register("evaluate", self.calcFitness)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=weightLowerBound, up=weightUpperBound, eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=weightLowerBound, up=weightUpperBound, eta=20.0)
        self.toolbox.register("select", tools.selNSGA2)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        # the lower bound for the weight values
        self.weightLowerBound = weightLowerBound
        # the upper bound for the weight values
        self.weightUpperBound = weightUpperBound
        # loop over the param groups
        for group in self.param_groups:
            # loop over first the weights then the bias
            for p in group['params']:
                self.toolbox.register(
                    "attr_float", self.uniform, self.weightLowerBound, self.weightUpperBound, p.numel())
                stddev = 1. / math.sqrt(p.size(-1))
                self.toolbox.register(
                    "individual", self.generateIndividual, p.numel(), stddev)
                self.toolbox.register(
                    "population", tools.initRepeat, list, self.toolbox.individual)
                # create a population
                population = self.toolbox.population(n=self.popSize)

                # This is just to assign the crowding distance to the individuals
                # no actual selection is done
                population = self.toolbox.select(population, len(population))
                self.state[p] = population

    # Step Method
    def step(self):
        # move the model and the loss function to the device
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)
        for group in self.param_groups:
            # loop over each group i.e. the weights, then the biases
            for index, p in enumerate(group['params']):

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in self.state[p]
                               if not ind.fitness.valid]
                # Create a partial function with the constant value bound
                partial_function = functools.partial(
                    self.toolbox.evaluate, model=self.model, index=index, shape=tuple(list(np.shape(p))))
                fitnesses = self.toolbox.map(
                    partial_function, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Vary the population
                offspring = tools.selTournamentDCD(
                    self.state[p], len(self.state[p]))
                # selTournamentDCD means Tournament selection based on dominance (D)
                # followed by crowding distance (CD). This selection requires the
                # individuals to have a crowding_dist attribute
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    # make pairs of all (even,odd) in offspring
                    if random.random() <= self.CXPB:
                        self.toolbox.mate(ind1, ind2)

                    self.toolbox.mutate(ind1, indpb=1.0/p.numel())
                    self.toolbox.mutate(ind2, indpb=1.0/p.numel())
                    del ind1.fitness.values, ind2.fitness.values

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [
                    ind for ind in offspring if not ind.fitness.valid]
                partial_function = functools.partial(
                    self.toolbox.evaluate, model=self.model, index=index, shape=tuple(list(np.shape(p))))
                fitnesses = self.toolbox.map(
                    partial_function, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Select the next generation population
                pop = self.toolbox.select(
                    self.state[p] + offspring, self.popSize)
                # sort the generation
                pop.sort(key=lambda x: x.fitness.values)
                self.state[p] = pop
                best = np.reshape(pop[0], np.shape(p))
                # set the weights of the network to the best individual
                self.setWeights(self.model, index, best)

    def calcFitness(self, individual, model, index, shape):
        """
        function to calculate the fitness of an individual

        Args:
            individual (numpy.ndarray): an array of decision variables
            model (pytorch.module): the model to set the weights of
            index (int): the index of the group the at the method is in (weights/biases)
            shape (tuple): the shape that this individual needs to be
        """
        # convert the individual to weights
        weights = np.reshape(individual, shape)
        # set the weights of the network to this individual
        self.setWeights(model, index, weights)
        # set the model in evaluation mode
        model.eval()
        x = model.input
        y = model.y
        y_pred = model(x)
        loss = self.lossFn(y_pred, y)
        loss = loss.cpu().detach().item()
        if loss == 0:
            loss = 0.000001
        f1 = loss
        f2 = np.sum(weights**2)

        return f1, f2

    def setWeights(self, model, index, weights):
        """
        function to set the weights for the given indexed parameter

        Args:
            model (pytorch.module): the model to set the weights of
            index (int): the index of the group the at the method is in (weights/biases)
            weights (numpy.ndarray): the weights to set into the model
        """
        weights = torch.tensor(weights)
        with torch.no_grad():
            count = 0
            for param in model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1

    def generateIndividual(self, length, stddev):
        """
        function to generate a new individual

        Args:
            length (int): the number of decision variables
            stddev (float): the standard deviation for the weight value generation

        Returns:
            individual: the new individual
        """
        numpyData = np.random.uniform(
            -stddev, stddev, size=(length,))
        individual = creator.Individual(numpyData)
        return individual

    def uniform(self, low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
