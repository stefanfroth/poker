import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from poker import Game
from agent import Agent
import sqlalchemy as sqa
import time
start_time = time.time()

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 50000
AGENT = Agent()


#AGENT.build_model()
# #print(AGENT.model.layers[1].get_weights())
#
# g = Game(PLAYERS, BLIND, STACK, AGENT, LIMIT)
#
#
# for i in range(GAMES):
#     g.play_one_complete_game()
#
# print(f'{GAMES} games were played!')
#
# AGENT.save()
#
# print('''The weights were saved! Now the model will be retrained on the generated
#         data''')

AGENT.load('2019-08-14-00:24')
#print('Loaded successfully')
AGENT.read_data()
#print('Read the data successfully')
AGENT.create_embedding_input()
#print(f'Created embedding input {AGENT.input_card_embedding}')
AGENT.create_state_input()
#print(f'Created state input {AGENT.input_state}')
#print(f'The actions are {AGENT.input['action']}')
#print(f'The rewards are {AGENT.input['reward']}')
#print(f'The model operations are: {AGENT.model.get_operations()}')
AGENT.train_model(AGENT.input_card_embedding, AGENT.input_state, np.array(AGENT.input['action']), np.array(AGENT.input['reward']))

AGENT.save()

print('''The model has been retrained and the new weights were saved.
This took {} to run'''.format(time.time() - start_time))
