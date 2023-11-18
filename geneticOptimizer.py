# Import the necessary libraries
import torch
import torch.nn as nn
import random
import numpy as np


class GeneticOptimizer(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, device, model, lossFn, pop=20):
        # TODO - change this to model.last_layer.parameters()
        params = model.last_layer.parameters()
        super(GeneticOptimizer, self).__init__(params, defaults={'pop': pop})
        self.popSize = pop  # save the population size
        self.device = device  # save the device
        self.model = model  # save the model that this optimizer is for
        self.lossFn = lossFn  # save the loss function for this optimizer
        self.state = {}  # a dictionary to store the populations
        self.fitness = {}  # a dictionary to store the fitnesses
        self.numOfBits = 64  # the number of bits for each weight
        # loop over the param groups
        for group in self.param_groups:
            # loop over first the weights then the bias
            for p in group['params']:
                dict = {}
                for i in range(pop):
                    # add the gray coded weights/biases to a dictionary
                    dict[i] = self.encodeIndividual(
                        torch.rand_like(p.data), numOfBits=self.numOfBits)
                self.state[p] = dict

    # Step Method
    def train(self, trainingDataLoader):
        self.model = self.model.to(self.device)
        self.model.eval()
        for (x, y) in trainingDataLoader:
            (x, y) = (x.to(self.device), y.to(self.device))
            for group in self.param_groups:
                for index, p in enumerate(group['params']):
                    # a dictionary to hold the binary array for each individual, ready for crossover and mutation
                    binaryArrays = {}
                    if p not in self.state:
                        dict = {}
                        for i in range(self.popSize):
                            dict[i] = self.encodeIndividual(
                                torch.rand_like(p.data), numOfBits=self.numOfBits)
                            self.state[p] = dict

                    currentFitness = list()
                    # loop over all of the individuals in the population
                    for individual in self.state[p]:
                        # decode the individual
                        decoded = self.decodeIndividual(
                            (self.state[p])[individual], numOfBits=self.numOfBits)
                        # now assign these weights to the last layer
                        with torch.no_grad():
                            count = 0
                            for param in self.model.last_layer.parameters():
                                if (count == index):
                                    param.copy_(nn.Parameter(decoded))
                                    break

                        # calculate the individuals fitness
                        # set the model in evaluation mode
                        self.model.eval()
                        y_pred = self.model(x)
                        loss = self.lossFn(
                            y_pred, y).cpu().data.numpy().argmax()
                        if loss == 0:
                            loss = 0.000001
                        currentFitness.append(loss)

                        # convert the individual to a binary array
                        binaryArray = self.individualToBinaryArray(
                            (self.state[p])[individual], self.numOfBits)
                        binaryArrays[individual] = binaryArray

                    # now we have the fitness of each individual, we can perform crossover and mutate
                    # calculate the fitness proportionate
                    fitnessProportionates = self.calculateFitnessProprtionate(
                        currentFitness)
                    newBinaryArrays = {}
                    for individual, binary in enumerate(binaryArrays):
                        choices = np.arange(self.popSize)
                        choices = np.delete(choices, individual)
                        proportions = np.delete(
                            fitnessProportionates, individual)
                        # get the parents
                        p1 = individual
                        p2 = random.choices(choices, proportions)[0]
                        # perform one point crossover
                        offspring = self.onePointCrossover(
                            binaryArrays[p1], binaryArrays[p2])
                        newBinaryArrays[p1] = offspring
                        # now we can mutate this new offspring
                        probability = 1 / len(newBinaryArrays[p1])
                        newBinaryArrays[p1] = self.mutate(
                            newBinaryArrays[p1], probability)

                    # now we can convert the binary arrays back into weights, then encode them and set them as the new population
                    for child, binary in enumerate(newBinaryArrays):
                        tmp = self.binaryArrayToIndividual(newBinaryArrays[child], np.shape(
                            (self.state[p])[child]), numOfBits=self.numOfBits)
                        tmp = self.grayCode(tmp)
                        (self.state[p])[child] = tmp

    def grayCode(self, n):
        """function to gray code a number n"""
        # gray code the number
        n ^= np.right_shift(n, 1)
        return n

    def encodeIndividual(self, i, numOfBits=4):
        """function to encode an individual with graycoding"""
        numpyArr = i.detach().cpu().numpy()
        numpyArr = self.encodeRealValue(numpyArr, numOfBits=numOfBits)
        indiv = torch.tensor(numpyArr)
        return indiv

    def decodeIndividual(self, i, numOfBits=4):
        """function to decode an individual back into weights"""
        numpyArr = i.detach().cpu().numpy()
        numpyArr = self.decodeRealValue(numpyArr, numOfBits=numOfBits)
        indiv = torch.tensor(numpyArr)
        return indiv

    def encodeRealValue(self, x, numOfBits=4):
        """function to encode a real value x as a graycoded integer"""
        integer = ((x - (-1)) * ((2 ** numOfBits) - 1))/(1 - (-1))
        np.round(integer, 0)
        integer = integer.astype(int)
        grayCoded = self.grayCode(integer)
        return grayCoded

    def decodeRealValue(self, n, numOfBits=4):
        """function to decode a graycoded integer n to a real value"""
        # convert the binary to an integer
        decoded = (-1) + (((1 - (-1)) / ((2 ** numOfBits) - 1)) * n)
        return decoded

    def individualToBinaryArray(self, i, numOfBits=4):
        """function to convert an individual i to a long list of binary"""
        binary = np.vectorize(np.binary_repr)(i, numOfBits)
        binary = np.ravel(binary)
        characters_list = []
        for string in binary:
            for character in string:
                characters_list.append(character)
        binary = np.array(characters_list)
        return binary

    def binaryArrayToIndividual(self, b, shape, numOfBits=4):
        """function to convert a binary array into an individual"""
        b = np.reshape(b, (-1, numOfBits))
        b = np.apply_along_axis(lambda row: ''.join(row), 1, b)
        b = np.array([int(binary_string, 2) for binary_string in b])
        b = np.reshape(b, shape)
        return torch.from_numpy(b)

    def calculateFitnessProprtionate(self, fitnesses):
        """function to calculate the fitness proportionate of each individual"""
        fitnessProprtionates = fitnesses
        reciprocals = np.reciprocal(fitnesses)
        denominator = reciprocals.sum()
        for i, fitness in enumerate(fitnesses):
            fitnessProprtionates[i] = (1 / fitness) / denominator
        return fitnessProprtionates

    def onePointCrossover(self, p1, p2):
        """function to perform 1 point crossover on two parents and return one of the children"""
        point = random.randint(1, len(p1) - 1)
        firstHalf = p1[0:point]
        secondHalf = p2[point:len(p2)]
        offspring = np.concatenate((firstHalf, secondHalf))
        return offspring

    def mutate(self, i, p):
        """function to mutate i with a probability of p"""
        newInt = np.vectorize(int)
        i = newInt(i)
        if random.random() < p:
            i = 1 - i
        return i.astype(str)
