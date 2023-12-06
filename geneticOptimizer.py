# Import the necessary libraries
import torch
import torch.nn as nn
import random
import numpy as np
import threading
from queue import Queue
import copy
import math
import itertools


class GeneticOptimizer(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, device, model, lossFn, weightLowerBound, weightUpperBound, numOfBits=32, pop=20, elites=0, debug=False):
        params = model.last_layer.parameters()
        super(GeneticOptimizer, self).__init__(params, defaults={'pop': pop})
        self.popSize = pop  # save the population size
        self.elites = elites  # save the number of elites to take to the next population
        self.device = device  # save the device
        self.model = model  # save the model that this optimizer is for
        self.lossFn = lossFn  # save the loss function for this optimizer
        self.state = {}  # a dictionary to store the populations
        self.numOfBits = numOfBits  # the number of bits for each weight
        self.debug = debug
        # the lower bound for the weight values
        self.weightLowerBound = weightLowerBound
        # the upper bound for the weight values
        self.weightUpperBound = weightUpperBound
        if (elites > pop):
            raise ("you cant have more elites than in your population")
        # loop over the param groups
        for group in self.param_groups:
            # loop over first the weights then the bias
            for p in group['params']:
                stddev = 1. / math.sqrt(p.size(-1))
                arr = list()
                for i in range(pop):
                    ind = self.generateIndividual(
                        stddev, p.numel(), self.numOfBits)
                    arr.append(ind)
                self.state[p] = arr

    # Step Method
    def step(self):
        # move the model and the loss function to the device
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)
        for group in self.param_groups:
            # loop over each group i.e. the weights, then the biases
            for index, p in enumerate(group['params']):
                if self.debug == True:
                    print("=================ORIGINAL POPULATION=================")
                    print(self.state[p])
                # make an array to store all the current fitness values
                currentFitness = np.zeros(self.popSize)
                # decode the individuals
                decoded = self.decodeIndividuals(
                    np.copy(self.state[p]), np.shape(p), numOfBits=self.numOfBits)

                threads = list()  # create a list for storing threads
                # loop over all of the individuals in the population
                for individual, weights in enumerate(decoded):
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
                fitnessProportionates = self.calculateFitnessProprtionate(
                    np.copy(currentFitness))

                if self.debug == True:
                    print(
                        "=================ORIGINAL FITNESSES PROPORTIONATES=================")
                    print(fitnessProportionates)

                # make an array to store the offspring in
                offspring = np.full(np.shape(self.state[p]), '0')

                # get the elites
                eliteIndices = np.argpartition(currentFitness, self.elites)[
                    :self.elites]
                eliteIndividuals = [(self.state[p])[i] for i in eliteIndices]
                eliteFitnesses = np.partition(currentFitness, self.elites)[
                    :self.elites]

                if self.debug == True:
                    print(
                        "=================ELITES=================")
                    print("indexes: ", eliteIndices)
                    print(eliteIndividuals)

                # generate pairs of parents to breed
                choices = np.arange(self.popSize)
                pairs = self.generatePairs(
                    choices, fitnessProportionates, self.popSize - self.elites)
                numOfOffspring = 2 * len(pairs)
                offspringFitness = np.zeros(numOfOffspring)

                if self.debug == True:
                    print("=================PARENTS=================")
                    print(pairs)

                threads = list()  # create a list for storing threads
                # for each pair, generate two offspring
                for pair, parents in enumerate(pairs):
                    t = threading.Thread(target=self.generateOffspring, args=(
                        pair, parents, index, self.state[p], list(np.shape(p)), offspring, offspringFitness))
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

                try:
                    newPop = np.concatenate(
                        (eliteIndividuals, offspring), axis=0)
                except:
                    newPop = np.array(offspring)
                newPopFitnesses = np.concatenate((
                    eliteFitnesses, offspringFitness), axis=0)
                newPop = newPop[:self.popSize]
                newPopFitnesses = newPopFitnesses[:self.popSize]

                if self.debug == True:
                    print("=================NEW POP=================")
                    print(newPop)
                    print("=================NEW POP FITNESSES=================")
                    print(newPopFitnesses)

                # set the population to the new pop
                self.state[p] = newPop

                # set the weights to that of the best in the population
                best = np.argmin(newPopFitnesses)
                bestWeights = list()
                bestWeights.append(newPop[0])
                bestWeights = np.array(bestWeights)
                bestWeights = self.decodeIndividuals(
                    bestWeights, np.shape(p), numOfBits=self.numOfBits)
                bestWeights = bestWeights[0]

                if self.debug == True:
                    print("=================BEST OF POP=================")
                    print("index: ", best)
                    print("weights: ", bestWeights)

                self.setWeights(self.model, index, bestWeights)

    def generateOffspring(self, threadID, parents, index, popWeights, shape, offspring, offspringFitness):
        """
        function to generate two offspring from two parents

        Args:
            threadID (int): the id of the thread that this method is running in
            parents (tuple): the indexes of the two parents to mate
            index (int): the index of the group the at the method is in (weights/biases)
            popWeights (numpy.ndarray): the weights of the current population
            shape (tuple): the shape that the offspring needs to be
            offspring (numpy.ndarray): The array to store the offspring in
            offspringFitness (numpy.ndarray): the array to store the fitness of the two offspring in

        """
        shape = tuple(shape)

        # first perform crossover with the two parents to get two offspring
        o1, o2 = self.onePointCrossover(
            popWeights[parents[0]], popWeights[parents[1]])

        # mutate the offspring
        p = 1 / len(o1)
        o1 = self.mutate(o1, p)
        o2 = self.mutate(o2, p)
        offs = list()
        offs.append(o1)
        offs.append(o2)
        offs = np.array(offs)

        # decode the individuals to weights
        decoded = self.decodeIndividuals(
            offs, shape, numOfBits=self.numOfBits)
        # calculate the fitness of both offspring
        fitness = np.zeros(2)
        self.calculateFitness(copy.deepcopy(self.model),
                              0, index, decoded[0], fitness)
        self.calculateFitness(copy.deepcopy(self.model),
                              1, index, decoded[1], fitness)

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

    def grayCode(self, n):
        """
        function to gray code a number n

        Args:
            n (int): the integer to gray code

        Returns:
            n (int): the gray coded integer
        """
        # gray code the number
        n ^= np.right_shift(n, 1)
        return n

    def encodeIndividual(self, i, numOfBits=4):
        """
        function to encode an individual with graycoding

        Args:
            i (numpy.ndarray): an array of the individuals weights
            numOfBits (int): the number of bits to encode each decision variable with

        Returns:
            indiv (numpy.ndarray): the encoded individual
        """
        numpyArr = i
        numpyArr = self.encodeRealValue(
            numpyArr, lower=self.weightLowerBound, upper=self.weightUpperBound, numOfBits=numOfBits)
        indiv = numpyArr
        return indiv

    def decodeIndividuals(self, i, shape, numOfBits=4):
        """
        function to decode all the individuals back into weights

        Args:
            i (numpy.ndarray): an array of encoded weight values for individuals
            shape (tuple): the shape each individual should be
            numOfBits (int): the number of bits each decision variable is encoded with

        Returns:
            indiv (numpy.ndarray): an array of the decoded individuals
        """
        numpyArr = np.ravel(i)
        # Convert the array to a list for easier manipulation
        arrayList = numpyArr.tolist()
        # Divide the array into groups of numOfBits using slicing
        grouped_segments = [arrayList[i:i + numOfBits]
                            for i in range(0, len(arrayList), numOfBits)]
        # Convert each group of elements into a single string
        groupedStrings = [''.join(segment) for segment in grouped_segments]
        # Convert the list of grouped strings back to a NumPy array
        groupedArray = np.array(groupedStrings)
        # convert to an array of ints
        integerArray = np.array([int(element, 2) for element in groupedArray])
        # decode the ints as real values
        numpyArr = self.decodeRealValue(
            integerArray, lower=self.weightLowerBound, upper=self.weightUpperBound, numOfBits=numOfBits)
        newShape = (-1,) + tuple(shape)
        individuals = np.reshape(numpyArr, newShape)
        return individuals

    def encodeRealValue(self, x, lower, upper, numOfBits=4):
        """
        function to encode a real value x as a graycoded integer

        Args:
            x (float): a real value to encode as binary
            lower (float): a lower bound for the real value
            upper (float): a upper bound for the real value
            numOfBits (int): the number of bits to encode the real value with

        Returns:
            grayCoded (int): the encoded and graycoded value
        """
        integer = ((x - (lower)) * ((2 ** numOfBits) - 1))/(upper - (lower))
        integer = np.round(integer, 0)
        integer = integer.astype(np.int64)
        grayCoded = self.grayCode(integer)
        return grayCoded

    def decodeRealValue(self, n, lower, upper, numOfBits=4):
        """
        function to decode a graycoded integer n to a real value

        Args:
            n (int): an integer to decode
            lower (float): the lower bound for the real value
            upper (float): a upper bound for the real value
            numOfBits (int): the number of bits the real value was encoded with

        Returns:
            decoded (float): the decoded real value
        """
        # convert the binary to an integer
        decoded = (lower) + (((upper - (lower)) / ((2 ** numOfBits) - 1)) * n)
        return decoded

    def individualToBinaryArray(self, individual, weights, outputArray, numOfBits=4):
        """
        function to turn an individual to a binary array and store it in the output array

        Args:
            individual (int): the index of the individual to convert to binary
            weights (numpy.ndarray): the weights for this individual encoded as integers
            outputArray (numpy.ndarray): the array to store the binary in
            numOfBits (int): the number of bits per decision variable
        """
        binary = np.vectorize(np.binary_repr)(weights, numOfBits)
        flattened = np.ravel(binary)
        flattened = [i for ele in flattened for i in ele]
        flattened = np.array(flattened).astype(int)
        outputArray[individual] = flattened.astype(str)

    def binaryArraysToIndividuals(self, b, shape, numOfBits=4):
        """
        function to convert a dictionary of binary arrays to a dictionary of pytorch tensors (individuals)

        Args:
            b (numpy.ndarray): an array of binary strings to convert to individuals
            shape (tuple): the shape that each individual needs to be
            numOfBits (int): the number of bits per decision variable

        Returns:
            individuals (numpy.ndarray): an array of all the individuals encoded as integers
        """
        binary = np.ravel(b)
        binary = np.reshape(binary, (-1, numOfBits))
        binary = np.apply_along_axis(lambda row: ''.join(row), 1, binary)
        integers = np.array([int(binary_string, 2)
                            for binary_string in binary])
        shape = list([len(b)]) + shape
        shape = tuple((shape))
        individuals = np.reshape(integers, shape)
        individuals = [torch.from_numpy(row) for row in individuals]
        return individuals

    def calculateFitnessProprtionate(self, fitnesses):
        """
        function to calculate the fitness proportionate of each individual

        Args:
            fitnesses (numpy.ndarray): an array of fitnesses

        Returns:
            fitnessProportionates (numpy.ndarray): an array of proportions
        """
        reciprocals = np.reciprocal(fitnesses)
        denominator = reciprocals.sum()
        fitnessProprtionates = reciprocals/denominator
        return fitnessProprtionates

    def onePointCrossover(self, p1, p2):
        """
        function to perform 1 point crossover on two parents and return one of the children

        Args:
            p1 (numpy.ndarray): a binary string representing an individual
            p2 (numpy.ndarray): a binary string representing another individual

        Returns:
            offspring (numpy.ndarray): a binary string representing the new offspring
        """
        point = random.randint(1, len(p1) - 1)
        o1 = np.concatenate((p1[0:point], p2[point:len(p2)]))
        o2 = np.concatenate((p2[0:point], p1[point:len(p2)]))
        return p1, p2

    def mutate(self, i, p):
        """
        function to mutate i with a probability of p

        Args:
            i (numpy.ndarray): a binary string representing an individual
            p (float): a probability to flip a bit

        Returns
            flipped_array (numpy.ndarray): a binary string representing the mutated individual
        """
        newInt = np.vectorize(int)
        random_floats = np.random.random(len(i))
        flipped_array = np.where(random_floats < p,
                                 (1 - newInt(np.copy(i))).astype(str), i)
        return flipped_array

    def generateIndividual(self, stddev, length, numOfBits):
        """
        function to generate an individual

        Args:
            stddev (float): the standard deviation to use for generating decision variables
            length (int): the number of decision variables to generate
            numOfBits (int): the number of bits to encode each decision variable with

        Returns
            individual (numpy.ndarray): an array of bits representing an individual
        """
        numpyData = np.random.uniform(-stddev, stddev, size=(length,))
        encoded = self.encodeIndividual(numpyData, numOfBits=numOfBits)
        outputArray = np.full((1, length * numOfBits), '0')
        self.individualToBinaryArray(
            0, encoded, outputArray, numOfBits=numOfBits)
        return outputArray[0]

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
