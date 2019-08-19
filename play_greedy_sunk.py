import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import datetime
from poker_greedy_sunk import Game
from agent_greedy_sunk import Agent

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 50000
AGENT = Agent()

version = 0
db_table = f'v{version}'

for i in range(20):

    g = Game(PLAYERS, BLIND, STACK, AGENT, db_table, LIMIT)

    for i in range(GAMES):
        g.play_one_complete_game()

    AGENT.save(db_table)

    AGENT.read_data(db_table)

    AGENT.create_embedding_input()

    AGENT.create_state_input()

    losses = []

    for i in range(20):
        losses.append(AGENT.train_model(AGENT.input_card_embedding, AGENT.input_state, np.array(AGENT.input['action']), np.array(AGENT.input['reward'])))

    # increase version
    version += 1
    db_table = f'v{version}'

    AGENT.save(db_table)
    loss = pd.DataFrame(losses)

    loss.to_csv(f'loss_{db_table}_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}.csv')

#print('''We have generated data points.
#This took {} to run'''.format(time.time() - start_time))