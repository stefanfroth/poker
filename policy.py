import numpy as np
import random

class EpsilonGreedyPolicy:

    def __init__(self, model, eps, strict=False):
        self.strict = strict
        self.m = model
        self.epsilon = eps

    def get_action(self, state):
        r = random.random()
        if r > self.epsilon:
            # replay
            ypred = self.m.predict(state)
            b = ypred.astype(np.float64)
            ypred = b / b.sum()  # normalize due to float32 imprecision
            if self.strict:
                action = ypred
            else:
                action = np.random.choice(3)
            return action
        else:
            z = np.zeros(4)
            z[random.randint(0, 3)] = 1
            return z
