import numpy as np
import torch


class SimpleSGD():
    def __init__(self, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.hyperparameters = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay, 'nesterov': nesterov,
                                'dampening': dampening}
        # print(self.hyperparameters)
        self.buffer = None

    def step(self, params, params_grad):

        if self.hyperparameters['weight_decay'] != 0:
            params_grad = params_grad.add(params, alpha=self.hyperparameters['weight_decay'])

        if self.hyperparameters['momentum'] != 0:
            if self.buffer is None:
                print('self.hyperparameters', self.hyperparameters)
                self.buffer = torch.clone(params_grad).detach()
            else:
                self.buffer.mul_(self.hyperparameters['momentum']).add_(params_grad,
                                                                        alpha=1 - self.hyperparameters['dampening'])

            if self.hyperparameters['nesterov']:
                params_grad = params_grad.add(self.buffer, alpha=self.hyperparameters['momentum'])
            else:
                params_grad = self.buffer

        return torch.add(params, torch.mul(params_grad, -self.hyperparameters['lr']))


class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # TODO: initialize momentum
        # TODO: consider adding validity checks
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = None
        # hyperparameters = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon, 'momentum': None}
        # self.hyperparameters = hyperparameters

    def step(self, objective, gradient):
        if self.momentum is None:
            self.momentum = np.zeros(objective.shape())
            # see if momentum is initialized
        pass
