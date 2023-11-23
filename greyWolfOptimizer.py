import threading
from queue import Queue

import numpy as np
import torch
import torch.nn as nn


class Wolf():
    def __init__(self, position):
        self.position = position
        self.fitness = 0

class GreyWolfOptimizer(torch.optim.Optimizer):
    def __init__(self, device, model, lossFn, lr=0.01, pop=10, max_iters=100):
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
            
    def initialize_wolves_fitness(self):
        for wolf in self.wolves:
            params = wolf['params']
            fitness = self.loss_fn(params)
            wolf['fitness'] = fitness

    def calculateFitness(self, wolf, index):
        self.setWeights(index, wolf.position)
        
        self.model.eval()
        x = self.model.input
        y = self.model.y
        
        y_pred = self.model(x)
        
        loss = self.lossFn(y_pred, y)
        loss = loss.cpu().detach().item()
        
        wolf.fitness = loss

    def setWeights(self, index, weights):
        with torch.no_grad():
            count = 0
            for param in self.model.last_layer.parameters():
                if (count == index):
                    param.copy_(nn.Parameter(weights))
                    break
                count += 1

    def calculateWolf(self, wolf, alpha_pos, beta_pos, delta_pos, p, i, newWolves):
        a = alpha_pos - wolf.position.cuda() * torch.rand_like(p)
        b = beta_pos - wolf.position.cuda() * torch.rand_like(p)
        c = delta_pos - wolf.position.cuda() * torch.rand_like(p)

        updated_p = (a + b + c) / 3

        wolf.position = updated_p
        newWolves[i] = wolf


    def worker_thread(self, queue):
        while not queue.empty():
            function, *args = queue.get()
            function(*args)
        
    def step(self):
        self.model = self.model.to(self.device)
        self.lossFn = self.lossFn.to(self.device)

        # Loop over groups (first weights, then biases)
        for group in self.param_groups:
            for index, p in enumerate(group['params']):

                # Create population of wolves
                wolves = []
                for s in self.state[p]:
                    wolves.append(Wolf(s))

                # Loop over wolves (individuals)
                for wolf in wolves:
                   self.calculateFitness(wolf, index)

                threads = list()
                newWolves = np.full((self.pop), Wolf(0))
                for _ in range(self.max_iters):
                    for i, wolf in enumerate(wolves):
                        alpha_pos = wolves[0].position.cuda()
                        beta_pos = wolves[1].position.cuda()
                        delta_pos = wolves[2].position.cuda()
                        t = threading.Thread(target=self.calculateWolf, args=(wolf, alpha_pos, beta_pos, delta_pos, p, i, newWolves))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    
                    # Loop over wolves (individuals)
                    for wolf in newWolves:
                        self.calculateFitness(wolf, index)

                    indices = np.argsort([Wolf.fitness for Wolf in newWolves])
                    wolves = newWolves[indices]
                
                print(f"Best of {p}: {newWolves[0].position}")
                self.setWeights(index, newWolves[0].position)
