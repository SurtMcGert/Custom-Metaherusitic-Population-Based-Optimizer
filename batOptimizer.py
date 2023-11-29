import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class batOptimizer(torch.optim.Optimizer):
    def __init__(self, device, model, lossFn, populationSize=100, max_iters=10, alpha=0.9, Amin = 0, gamma=0.9, fmin = 0, fmax = 100, rmin=0, rmax=1):
        if populationSize <0:
            raise ValueError("Population size must be positive")
        
        params = model.last_layer.parameters()
        super(batOptimizer, self).__init__(params, defaults={'populationSize' : populationSize})

        #Counter
        self.counter = 0

        #Store parameter values
        self.populationSize = populationSize
        self.model = model
        self.max_iters = max_iters
        self.alpha = alpha
        self.Amin = Amin
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        self.rmin = rmin
        self.rmax = rmax

        self.device = device
        self.model = model
        self.lossFn = lossFn

        
    def step(self):
        # Set GPU usage
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)

        # Working only on last layer so only 1 group
        for group in self.param_groups:

            # First p is weights, second p is the biases
            for index, p in enumerate(group['params']):
                self.counter += 1
                print(f"Counter: {self.counter}")
                #Current fitnesses of bats stored in order of bat id
                currentFitness = np.zeros(self.populationSize)
                loudnesses = np.random.uniform(1, 2, size=(self.populationSize,1))
                bats = list()
                for i in range(self.populationSize):
                    dic = {}
                    # Save bat's id
                    dic['id'] = i
                    # Initialize bat's position
                    dic['x'] = np.random.uniform(-1, 1, size=list(np.shape(p.data)))
                    # Initialize bat's velocity
                    dic['v'] = np.random.uniform(-1, 1, size=list(np.shape(p.data)))
                    # Initialize bat's frequency
                    dic['f'] = (self.fmax - self.fmin) * np.random.rand() + self.fmin
                    # Initialize bat's pulse rate
                    dic['r'] = (self.rmax - self.rmin) * np.random.rand() + self.rmin
                    # Save bat's initial pulse rate
                    dic['r0'] = dic['r']
                    # Save bat's loudness
                    dic['a'] = loudnesses[i]
                    # Save bat's potential new solution
                    dic['xnew'] = np.zeros(shape = np.shape(p.data))
                    # Store each bat in bats array
                    bats.append(dic)
                    # Calculate bat's fitness
                    self.calculateFitness(self.model, i, index, torch.tensor(dic['x']), currentFitness)

                self.state[p] = np.array(bats)
                # Sort each bat's index by its fitness
                fitnessSortedIndexes = np.argsort(currentFitness)
                state = self.state[p]
                for t in range(self.max_iters): 
                    for bat in state:
                        # Random values for loudness and pulse rate
                        beta = np.random.rand()
                        rand = np.random.rand()

                        # Update frequency and velocity
                        bat['f'] = self.fmin + (self.fmax-self.fmin) * beta
                        bat['v'] += bat['v'] + (bat['x']-bats[fitnessSortedIndexes[0]]['x']) * bat['f']

                        # Calculate new potential solution
                        bat['xnew'] = bat['x'] + bat['v']

                        # Chance to move towards top 3 bats 
                        if rand > bat['r']:
                            # Get random bat from top 3 bats
                            randomBatNumber = int(np.random.uniform(0, 2))
                            randomSolution = bats[randomBatNumber]
                            # Perform local search around the chosen bat
                            bat['xnew'] = randomSolution['x'] + np.random.uniform(-1,1, size=list(np.shape(p.data)))*bat['a']

                        # Check if new solution is within bounds
                        bat['xnew'] = np.clip(bat['xnew'], -1, 1)
                        
                        # Calculate new fitness
                        newFitnessArray = np.zeros(1)
                        self.calculateFitness(self.model, 0, index, torch.tensor(bat['xnew']), newFitnessArray)

                        # Accept new solution
                        if rand < bat['a'] and newFitnessArray[0] < currentFitness[fitnessSortedIndexes[0]]:
                            bat['x'] = bat['xnew']

                        # Update loudness
                            bat['a'] *= self.alpha

                        # Minimum loudness set to Amin
                            if bat['a'] < self.Amin:
                                bat['a'] = self.Amin
                        
                        # Update pulse rate
                            bat['r'] = bat['r0'] * (1- np.exp(-self.gamma*t))
                        
                        # Calculate fitness and store it in current fitness array
                        self.calculateFitness(self.model, bat['id'], index, torch.tensor(bat['x']), currentFitness)
                    fitnessSortedIndexes = np.argsort(currentFitness)
                # Find bat with best fitness
                fitnessSortedIndexes = np.argsort(currentFitness)
                bestBat = bats[fitnessSortedIndexes[0]]
                # Update weights to best bat position
                self.setWeights(index, torch.tensor(bestBat['x']))


    def calculateFitness(self, model, individual, index, weights, currentFitness):
        """function to calculate the fitness of an individual for the given index parameter (weights or biases) then save the loss in currentFitness"""
        # assign these weights to the last layer
        self.setWeights(index, weights)
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
    
    
    def setWeights(self, index, weights):
        with torch.no_grad():
            count = 0
            for param in self.model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1