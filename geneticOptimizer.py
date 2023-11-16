# Import the necessary libraries
import torch
import torch.nn as nn


class GeneticOptimizer(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super(GeneticOptimizer, self).__init__(params, defaults={'lr': lr})
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            print(group)
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))

    # Step Method
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = {'mom': torch.zeros_like(p)}
                mom = self.state[p]['mom']
                with torch.no_grad():
                    if p.grad is not None:
                        new_mom = self.momentum * mom - group['lr'] * p.grad
                        p.add_(new_mom)  # Update parameter
                        self.state[p]['mom'] = new_mom  # Update momentum
