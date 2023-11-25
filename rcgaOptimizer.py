# Import the necessary libraries
import torch
import torch.nn as nn
import random
import numpy as np
import threading
import itertools
import copy
from scipy import stats
import math


class RCGAOptimizer(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, device, model, lossFn, weightLowerBound, weightUpperBound, pop=20, elites=0, debug=False):
        params = model.last_layer.parameters()
        super(RCGAOptimizer, self).__init__(params, defaults={'pop': pop})
        self.popSize = pop  # save the population size
        self.device = device  # save the device
        self.model = model  # save the model that this optimizer is for
        self.lossFn = lossFn  # save the loss function for this optimizer
        self.state = {}  # a dictionary to store the populations
        self.elites = elites  # store the number of elites
        self.debug = debug  # store whether to debug the optimizer
        # determin the parent pool size
        # coefficients = [1, -1, 0 - pop]
        # self.parentPoolSize = roots = np.round([x for x in np.roots(coefficients) if x > 0]
        #                                        [0]).astype(np.int32)
        if (elites > pop):
            raise ("you cant have more elites than the population")
        # the lower bound for the weight values
        self.weightLowerBound = weightLowerBound
        # the upper bound for the weight values
        self.weightUpperBound = weightUpperBound
        mean = 0
        stddev = 1. / math.sqrt(self.model.last_layer.weight.size(1))
        # loop over the param groups
        for group in self.param_groups:
            # loop over first the weights then the bias
            for p in group['params']:
                arr = list()
                for i in range(pop):
                    numpyData = np.random.uniform(
                        -stddev, stddev, size=list(np.shape(p.data)))

                    arr.append(numpyData)
                self.state[p] = np.array(arr)

    # Step Method
    def step(self):
        # move the model and the loss function to the device
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)
        for group in self.param_groups:
            # loop over each group i.e. the weights, then the biases
            for index, p in enumerate(group['params']):
                # make an array to store all the current fitness values
                currentFitness = np.zeros(self.popSize)

                threads = list()  # create a list for storing threads
                if self.debug == True:
                    print("=================ORIGINAL POPULATION=================")
                    print(self.state[p])

                # loop over all of the individuals in the population and generate their fitness
                for individual, weights in enumerate(self.state[p]):
                    t = threading.Thread(target=self.calculateFitness, args=(
                        copy.deepcopy(self.model), individual, index, weights, currentFitness))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                if self.debug == True:
                    print("=================ORIGINAL FITNESSES=================")
                    print(currentFitness)

                # now we have the fitness of each individual, we can perform crossover and mutate
                # calculate the fitness proportionate
                # fitnessProportionates = self.calculateFitnessProportionate(
                #     np.copy(currentFitness))

                fitnessProportionates = self.calculateRankProportionate(
                    np.copy(currentFitness))

                if self.debug == True:
                    print(
                        "=================ORIGINAL FITNESSES PROPORTIONATES=================")
                    print(fitnessProportionates)

                eliteIndices = np.argpartition(currentFitness, self.elites)[
                    :self.elites]
                eliteIndividuals = (self.state[p])[eliteIndices]
                eliteFitnesses = np.partition(currentFitness, self.elites)[
                    :self.elites]

                if self.debug == True:
                    print(
                        "=================ELITES=================")
                    print("indexes: ", eliteIndices)
                    print(eliteIndividuals)

                # generate the parent pairs
                choices = np.arange(self.popSize)
                parents = self.generatePairs(
                    choices, fitnessProportionates, self.popSize - self.elites)

                if self.debug == True:
                    print("=================PARENTS=================")
                    print(parents)

                # set the array for the offspring and their fitness
                numOfOffspring = 2 * len(parents)
                offspring = np.zeros(
                    ((numOfOffspring,) + np.shape((self.state[p])[0])))
                offspringFitness = np.zeros(numOfOffspring)

                threads = list()  # create a list for storing threads
                # for each pair of parents, crossover, mutate the offspring
                for pair, parents in enumerate(parents):
                    t = threading.Thread(target=self.generateOffspring, args=(
                        pair, parents, index, self.state[p], offspring, offspringFitness))
                    threads.append(t)
                    t.start()

                # wait for the threads to finish
                for t in threads:
                    t.join()

                if self.debug == True:
                    print("=================NEW OFFSPRING=================")
                    print(offspring)

                    print("=================NEW FITNESSES=================")
                    print(offspringFitness)

                newPop = np.concatenate((eliteIndividuals, offspring), axis=0)
                newPopFitnesses = np.concatenate((
                    eliteFitnesses, offspringFitness), axis=0)
                newPop = newPop[:self.popSize]
                newPopFitnesses = newPopFitnesses[:self.popSize]

                if self.debug == True:
                    print("=================NEW POP=================")
                    print(newPop)
                    print("=================NEW POP FITNESSES=================")
                    print(newPopFitnesses)

                # calculate who was the best
                best = np.argmin(newPopFitnesses)
                if self.debug == True:
                    print("=================BEST OF POP=================")
                    print("index: ", best)
                    print("weights: ", newPop[best])
                # set the networks weights to that of the best offspring
                self.setWeights(self.model, index, newPop[best])
                # save all the offspring
                self.state[p] = newPop

    def generateOffspring(self, threadID, parents, index, popWeights, offspring, offspringFitness):
        """function to generate an offspring for an individual"""
        # crossover the two parents
        o1, o2 = self.blendCrossover(
            popWeights[parents[0]], popWeights[parents[1]])

        # mutate the offspring
        o1 = self.mutate(o1)
        o2 = self.mutate(o2)

        # calculate the fitness of both offspring
        fitness = np.zeros(2)
        self.calculateFitness(copy.deepcopy(self.model), 0, index, o1, fitness)
        self.calculateFitness(copy.deepcopy(self.model), 1, index, o2, fitness)

        # add the offspring and their fitnesses to the lists
        offspring[threadID * 2] = o1
        offspring[(threadID * 2) + 1] = o2
        offspringFitness[threadID * 2] = fitness[0]
        offspringFitness[(threadID * 2) + 1] = fitness[1]

    def calculateFitness(self, model, individual, index, weights, currentFitness):
        """function to calculate the fitness of an individual for the given index parameter (weights or biases) then save the loss in currentFitness"""
        # assign these weights to the last layer
        self.setWeights(model, index, weights)
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
        currentFitness[individual] = loss

    def setWeights(self, model, index, weights):
        """function to set the weights for the given indexed parameter"""
        weights = torch.tensor(weights)
        with torch.no_grad():
            count = 0
            for param in model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1

    def calculateFitnessProportionate(self, fitnesses):
        """function to calculate the fitness proportionate of each individual"""
        fitnessProprtionates = fitnesses
        reciprocals = np.reciprocal(fitnesses)
        denominator = reciprocals.sum()
        for i, fitness in enumerate(fitnesses):
            fitnessProprtionates[i] = (1 / fitness) / denominator
        return fitnessProprtionates

    def calculateRankProportionate(self, fitnesses):
        """function to use rank selection to get the proportions for parent selection"""
        vectorized = np.vectorize(self.giveRank)
        length = np.full((np.shape(fitnesses)), len(fitnesses))
        ranks = vectorized(fitnesses, length)
        denominator = ranks.sum()
        rankProportionates = ranks/denominator
        return rankProportionates

    def giveRank(self, arr, len):
        return len - arr

    def blendCrossover(self, p1, p2):
        """function to perform blend crossover on two parents and return the two children"""
        p1R = np.ravel(p1)
        p2R = np.ravel(p2)
        a = random.uniform(0, 1)
        a = np.full(len(p1R), a)
        p = np.vstack((p1R, p2R)).T
        minMax = np.array(list(map(self.calcMinMax, p, a)))

        o1 = [random.uniform(lower, upper) for lower, upper in minMax]
        o1 = np.reshape(o1, np.shape(p1))
        o2 = [random.uniform(lower, upper) for lower, upper in minMax]
        o2 = np.reshape(o2, np.shape(p2))
        return o1, o2

    def calcMinMax(self, p, a):
        # calculate b first
        d = np.absolute(np.diff(p))[0]
        # calculate the min and max
        min = np.min(p) - (a * d)
        max = np.max(p) + (a * d)
        return (min, max)

    def mutate(self, o):
        """function to mutate i with a probability of p"""
        oR = np.ravel(o)
        vectorizedFunction = np.vectorize(self.mutateDecisionVariable)
        mutant = vectorizedFunction(oR)
        mutant = np.reshape(mutant, np.shape(o))
        return mutant

    def mutateDecisionVariable(self, v):
        """function to mutate a decision variable"""
        u = np.random.uniform(0, 1)
        nm = 100
        if (v <= 0.5):
            L = ((2 * u) ** (1/(1 + nm))) - 1
            mutated = v + (L * (v - self.weightLowerBound))
        else:
            R = 1 - ((2 * (1 - u)) ** (1/(1 + nm)))
            mutated = v + (R * (self.weightUpperBound - v))
        return mutated

    def generatePairs(self, parents, proportions, popSize):
        """function to generate pairs of parents"""
        numOfPairs = math.ceil(popSize/2)
        pairs = list()
        usedPairs = list()
        for i in range(numOfPairs):
            population = parents
            weights = proportions
            isUnique = False
            while isUnique == False:
                # Perform weighted random selection to choose the first individual
                firstIndividual = random.choices(
                    population, weights=weights, k=1)[0]
                # Perform weighted random selection to choose the second individual
                secondIndividual = firstIndividual
                while secondIndividual == firstIndividual:
                    secondIndividual = random.choices(
                        population, weights=weights, k=1)[0]
                pair = (firstIndividual, secondIndividual)
                if (len(pairs) == 0):
                    isUnique = True
                else:
                    isUnique = pair not in usedPairs
            pairs.append(pair)
            usedPairs += list(itertools.permutations(pair, 2))
        return pairs

    def pairIsUnique(self, newPair, existingPairs):
        # Convert the new pair into a set for efficient comparison
        new_pair_set = set(newPair)

        # Convert the existing list of pairs into a set of sets for faster lookups
        existing_pairs_set = set(set(p) for p in existingPairs)

        # Check if the new pair set exists in the set of existing pair sets
        return new_pair_set in existing_pairs_set
