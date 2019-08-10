
from tron import Environment
from model import Model
from sim import Simulator, RandomPolicy
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
                action = np.random.multinomial(1, ypred)
            return action
        else:
            z = np.zeros(4)
            z[random.randint(0, 3)] = 1
            return z



if __name__ == '__main__':
    from pg_model import PolicyGradientModel
    m = PolicyGradientModel()
    m.create()
    i = 0
    eps = 1.0
    while True:
        sim = Simulator(verbose=False)
        print("ROUND: ", i + 1)
        if i == 0:
            policy = RandomPolicy()
        else:
            policy = EpsilonGreedyPolicy(m, eps)
        sim.simulate(policy, 50)

        states, actions, rewards = sim.get_memory()

        m.train(states, actions, rewards)
        if eps > 0.10:
            eps -= 0.01
        i += 1
        m.save()
