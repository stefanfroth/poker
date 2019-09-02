import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from poker_presentation import Game
from agent import Agent
import time
start_time = time.time()

PLAYERS = 6
BLIND = 50
STACK = 1000
LIMIT = 100
GAMES = 10
DB_TABLE = 'pres'

agent1 = Agent()
agent1.load('v7_2019-08-19-23:30_20_epochs')

agent2 = Agent()
agent2.load('v7_2019-08-19-23:30_20_epochs')

agent3 = Agent()
agent3.load('v6_2019-08-19-09:30_20_epochs')

agent4 = Agent()
agent4.load('v6_2019-08-19-09:30_20_epochs')

# instantiate random agent
agent5 = Agent()

g = Game(PLAYERS, BLIND, STACK, agents=[agent1, agent2, agent3, agent4, agent5], db_table=DB_TABLE, limit=LIMIT)

for i in range(GAMES):
    start = input('Press s to start: ')
    g.play_one_complete_game()

print('''We have generated data points.
This took {} to run'''.format(time.time() - start_time))
