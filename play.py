import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from poker_greedy import Game
from agent import Agent
import time
start_time = time.time()

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 20000
DB_TABLE = 'v6_2019_08_18_12_43'
AGENT = Agent()
AGENT.load('v6_2019-08-19-09:30_20_epochs')
print(f'The models weights before playing are: {AGENT.model.get_weights()}')


#AGENT.build_model()
# #print(AGENT.model.layers[1].get_weights())
#
g = Game(PLAYERS, BLIND, STACK, AGENT, DB_TABLE, LIMIT)
#
#
for i in range(GAMES):
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
