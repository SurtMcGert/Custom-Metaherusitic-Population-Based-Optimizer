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
        self.counter = 0
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
        # loop over the param groups
        for group in self.param_groups:
            # loop over first the weights then the bias
            for p in group['params']:
                stddev = 1. / math.sqrt(p.size(-1))
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
                self.counter +=1
                print(f"Index: {index}")
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

                # order the offspring
                orderedIndices = np.argsort(offspringFitness)
                offspring = [offspring[i] for i in orderedIndices]
                offspringFitness = np.sort(offspringFitness)

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
        """
        function to generate two offspring from two parents

        Args:
            threadID (int): the id of the thread that this method is running in
            parents (tuple): the indexes of the two parents to mate
            index (int): the index of the group the at the method is in (weights/biases)
            popWeights (numpy.ndarray): the weights of the current population
            offspring (numpy.ndarray): The array to store the offspring in
            offspringFitness (numpy.ndarray): the array to store the fitness of the two offspring in

        """
        # crossover the two parents
        # o1, o2 = self.blendCrossover(
        #     popWeights[parents[0]], popWeights[parents[1]])

        # first perform crossover with the two parents to get two offspring
        o1, o2 = self.simulatedBinaryCrossover(
            popWeights[parents[0]], popWeights[parents[1]])

        # mutate the offspring
        o1 = self.mutate(o1, self.weightLowerBound, self.weightUpperBound)
        o2 = self.mutate(o2, self.weightLowerBound, self.weightUpperBound)

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
        """
        function to calculate the fitness of an individual for the given index parameter (weights or biases) then save the loss in currentFitness

        Args:
            model (pytorch.module): the model to calculate the fitness on
            individual (int): the index of the individual to test
            index (int): the index of the group the at the method is in (weights/biases)
            weights (numpy.ndarray): the weights to calculate the fitness of
            currentFitness (numpy.ndarray): the array to store the fitness in
        """
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

    def calculateFitnessProportionate(self, fitnesses):
        """
        function to calculate the fitness proportionate of each individual

        Args:
            fitnesses (numpy.ndarray): the fitnesses of all the individuals

        Returns:
            fitnessProportionates (numpy.ndarray): the fitness proportionate of each individual
        """
        fitnessProprtionates = fitnesses
        reciprocals = np.reciprocal(fitnesses)
        denominator = reciprocals.sum()
        for i, fitness in enumerate(fitnesses):
            fitnessProprtionates[i] = (1 / fitness) / denominator
        return fitnessProprtionates

    def calculateRankProportionate(self, fitnesses):
        """
        function to use rank selection to get the proportions for parent selection

        Args:
            fitnesses (numpy.ndarray): the fitnesses of all the individuals

        Returns:
            rankProportionates (numpy.ndarray): the fitness proportionate of each individual
        """
        # give each item in the array a rank
        uniques = np.unique(fitnesses)
        sortedIndices = np.argsort(uniques)[::-1]
        valueToIndex = dict(zip(uniques, sortedIndices + 1))
        ranks = [valueToIndex[x] for x in fitnesses]
        denominator = np.sum(ranks)
        rankProportionates = ranks/denominator
        return rankProportionates

    def blendCrossover(self, p1, p2):
        """
        function to perform blend crossover on two parents and return the two children

        Args:
            p1 (numpy.ndarray): the weight values of the first parent
            p2 (numpy.ndarray): the weight values of the second parent

        Returns:
            o1 o2 (tuple): the two offspring weight values
        """
        a = random.uniform(0, 1)
        minMax = np.full(np.shape(p1), a)
        minMax = list(((np.min((p1, p2)) - (minMax * np.absolute(p2 - p1))),
                       (np.max((p1, p2)) + (minMax * np.absolute(p2 - p1)))))

        # arrange the min and max correctly
        minMax = np.stack((minMax[0], minMax[1]), axis=-1)
        # create the two new offspring
        o1 = np.random.uniform(low=minMax[..., 0], high=minMax[..., 1])
        o2 = np.random.uniform(low=minMax[..., 0], high=minMax[..., 1])
        return o1, o2

    def simulatedBinaryCrossover(self, p1, p2):
        """
        function to perform simulated binary crossover on two parents and return the two children

        Args:
            p1 (numpy.ndarray): the weight values of the first parent
            p2 (numpy.ndarray): the weight values of the second parent

        Returns:
            o1 o2 (tuple): the two offspring weight values

        """
        u = random.uniform(0, 1)
        n = 8
        if u <= 0.5:
            B = (2 * u) ** (1 / (n + 1))
        else:
            B = (2*(1 - u)) ** (1 / (n + 1))
        o1 = (1/2) * (((1-B) * p1) + ((1+B) * p2))
        o2 = (1/2) * (((1+B) * p1) + ((1-B) * p2))
        return o1, o2

    def mutate(self, o, lowerBound, upperBound):
        """
        function to mutate an offspring

        Args:
            o (numpy.ndarray): the offspring to mutate
            lowerBound (float): the lower bound for each decision variable
            upperBound (float): the upper bound for each decision variable

        Returns:
            m (numpy.ndarray): the mutated offspring
        """
        m = np.random.uniform(0, 1, size=o.shape)
        nm = 125
        L = (((2 * m) ** (1/(1 + nm))) - 1)
        R = 1 - ((2 * (1 - m)) ** (1/(1 + nm)))
        m = np.where(m <= 0.5, o + (L * (o - lowerBound)),
                     o + (R * (upperBound - o)))
        return m

    def generatePairs(self, parents, proportions, popSize):
        """
        function to generate pairs of parents

        Args:
            parents (numpy.ndarray): the indices of all the possible parents to make pairs from
            proportions (numpy.ndarray): the selection proportions for each parent
            popSize (int): the size of the population to generate

        Returns:
            pairs (numpy.ndarray): a list of parent pairs
        """
        # calculate the number of pairs that are needed to make the population
        numOfPairs = math.ceil(popSize/2)
        pairs = list()
        usedPairs = list()
        # for each pair
        for i in range(numOfPairs):
            population = parents
            weights = proportions
            # if the generated pair already exists, keep looping and make another
            isUnique = False
            while isUnique == False:
                # Perform weighted random selection to choose the first individual
                firstIndividual = random.choices(
                    population, weights=weights, k=1)[0]
                # Perform weighted random selection to choose the second individual making sure it is a different individual
                secondIndividual = firstIndividual
                while secondIndividual == firstIndividual:
                    secondIndividual = random.choices(
                        population, weights=weights, k=1)[0]
                # set the pair
                pair = (firstIndividual, secondIndividual)
                # check if the pair is unique
                if (len(pairs) == 0):
                    isUnique = True
                else:
                    isUnique = pair not in usedPairs
            # add the pair to the list and make a note of it so it doesnt get generated again
            pairs.append(pair)
            usedPairs += list(itertools.permutations(pair, 2))
        return pairs

    def pairIsUnique(self, newPair, existingPairs):
        """
        function to tell if a pair of parents is unique

        Args:
            newPair (tuple): the pair to check for uniqueness
            existsingPairs (numpy.ndarray): the list of alrady existing pairs

        Returns
            boolean: if the new pair is unique
        """
        # Convert the new pair into a set for efficient comparison
        new_pair_set = set(newPair)

        # Convert the existing list of pairs into a set of sets for faster lookups
        existing_pairs_set = set(set(p) for p in existingPairs)

        # Check if the new pair set exists in the set of existing pair sets
        return new_pair_set in existing_pairs_set
