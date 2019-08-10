
import numpy as np
from model import Model
import random


class AlwaysLeftPolicy:

    def get_action(self, state):
        return np.array([0, 0, 1, 0])


class RandomPolicy:

    def get_action(self, state):
        r = random.randint(0, 3)
        a = np.array([0, 0, 0, 0])
        a[r] = 1
        return a



class ModelPolicy:

    def __init__(self, model):
        self.m = model

    def get_action(self, state):
        ypred = self.m.predict(state)
        #return ypred
        b = ypred.astype(np.float64)
        ypred = b / b.sum()  # normalize due to float32 imprecision
        action = np.random.multinomial(1, ypred)
        return action


OPPOSITE = [1, 0, 3, 2]

class EpsilonGreedyPolicy:

    def __init__(self, model, eps, strict=False):
        self.strict = strict
        self.m = model
        self.epsilon = eps
        self.last_action = -1

    def get_action(self, state):
        action = None
        while action is None:
            r = random.random()
            if r > self.epsilon:
                # replay
                ypred = self.m.predict(state)
                b = ypred.astype(np.float64)
                ypred = b / b.sum()  # normalize due to float32 imprecision
                if self.strict:
                    action = ypred
                else:
                    action = np.random.multinomial(1, ypred)
            else:
                action = np.zeros(4, dtype=np.float32)
                action[random.randint(0, 3)] = 1
            if action.argmax() == self.last_action:
                action = None
        self.last_action = OPPOSITE[action.argmax()]
        return action

