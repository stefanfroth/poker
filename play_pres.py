import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from poker_greedy_pres import Game
from agent import Agent
import time
start_time = time.time()

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

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


#print(f'The models weights before playing are: {AGENT.model.get_weights()}')


#AGENT.build_model()
# #print(AGENT.model.layers[1].get_weights())
#
g = Game(PLAYERS, BLIND, STACK, agents=[agent1, agent2, agent3, agent4, agent5], db_table=DB_TABLE, limit=LIMIT)
#
#

for i in range(GAMES):
    start = input('Press s to start: ')
    g.play_one_complete_game()
#
# print(f'{GAMES} games were played!')
#
#AGENT.save()

#AGENT.read_data('results')

#AGENT.create_embedding_input()
#AGENT.create_state_input()

#for i in range(10):
#    AGENT.train_model(AGENT.input_card_embedding, AGENT.input_state, np.array(AGENT.input['action']), np.array(AGENT.input['reward']))

#print(f'The models weights after playing are: {AGENT.model.get_weights()}')
# print('''The weights were saved! Now the model will be retrained on the generated
#         data''')
#time.sleep(60)
#AGENT.save()

print('''We have generated data points.
This took {} to run'''.format(time.time() - start_time))
