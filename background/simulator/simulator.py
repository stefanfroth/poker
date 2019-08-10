
from environment import Environment
from policy import AlwaysLeftPolicy, RandomPolicy, ModelPolicy, EpsilonGreedyPolicy
import numpy as np
from model import Model
from policy_gradient import PolicyGradientModel


class Episode:
    """Collects data from one game"""

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []


    def calc_discounts(self, discount=0.99):
        discounted_rewards = []
        tail = 0
        for r in self.rewards[::-1]:
            tail = r + tail * discount
            discounted_rewards.append(tail)
        discounted_rewards.reverse()
        self.rewards = discounted_rewards

    def add(self, state, reward, action):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def play(self, policy, verbose=True):
        env = Environment()
        while not env.finished:
            state = env.get_state()
            score_before = env.get_score()
            action = policy.get_action(state)
            env.perform_action(action)
            score_after = env.get_score()
            reward = score_after - score_before
            #if score_after != 0:  # do not record deadly moves
            self.add(state, reward, action)
            if verbose:
                print('.', end='')
        print()


class Simulator:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.states = []
        self.rewards = []
        self.actions = []
        self.durations = []

    def add(self, episode):
        episode.calc_discounts()
        self.states += episode.states
        self.rewards += episode.rewards
        self.actions += episode.actions
        self.durations.append(len(episode.states))

    def __repr__(self):
        md = sum(self.durations) / len(self.durations)
        mr = sum(self.rewards) / len(self.rewards)
        return f"""Memory:
    states        {len(self.states):6} 
    mean duration {md}
    mean reward   {mr}
    """

    def simulate(self, policy, episodes):
        for i in range(episodes):
            if i % 10 == 0 and self.verbose:
                print(f'simulating episode {i}')
            episode = Episode()
            episode.play(policy, self.verbose)
            self.add(episode)

    def get_memory(self):
        states = np.array(self.states).reshape((len(self.states), 25))
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        rewards = rewards / rewards.max()  # normalize
        return states, actions, rewards


if __name__ == '__main__':
    sim = Simulator()
    m = PolicyGradientModel()
    #m.load()
    m.create()
    m.build_train_fn()
    policy = EpsilonGreedyPolicy(m, 0.05)

    for i in range(1000):
        print('SIMULATE')
        sim = Simulator()
        sim.simulate(policy, 50)
        if policy.epsilon > 0.01:
            policy.epsilon -= 0.002
        print(sim)
        states, actions, rewards = sim.get_memory()
        print('TRAINING')
        for i in range(30):
            m.train(states, actions, rewards)
        m.save()
