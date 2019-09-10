'''
The module play imports the game of poker and the agent playing and generates
data points by self play against other agents of the same version.
'''
import time
import warnings
import numpy as np
import sqlalchemy as sqa
from poker_game import Game
from agent import Agent

warnings.simplefilter(action='ignore', category=FutureWarning)

START_TIME = time.time()

DB = f'postgres://localhost/poker'
ENGINE = sqa.create_engine(DB)

# Rules of the game
PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 10
DB_TABLE = 'cleanup'

# Chose whether to show print statements while running the program or not.
VERBOSE = 1

# Chose whether to include the training process or not.
TRAINING = 0

# Chose the agent that is supposed to play.
AGENT = Agent()
AGENT.load('v7_2019-08-19-23:30_20_epochs')
if VERBOSE == 1:
    print(f'The models weights before playing are: {AGENT.model.get_weights()}')

AGENT.build_model()

# save the initial weights of the agent
#AGENT.save()

# instantiate the game and play the games
G = Game(PLAYERS, BLIND, STACK, AGENT, DB_TABLE, LIMIT)

for i in range(GAMES):
    G.play_one_complete_game()

if VERBOSE == 1:
    print(f'{GAMES} games were played!')
    print('''We have generated data points. This took {} to run'''.format(time.time() - START_TIME))


if TRAINING == 1:
    # read in the data created by playing the game
    AGENT.read_data(DB_TABLE)

    # Train the model
    AGENT.create_embedding_input()
    AGENT.create_state_input()

    for i in range(10):
        AGENT.train_model(AGENT.input_card_embedding, AGENT.input_state,
                          np.array(AGENT.input['action']), np.array(AGENT.input['reward']))

    if VERBOSE == 1:
        print(f'The models weights after playing are: {AGENT.model.get_weights()}')
        print('The weights were saved! Now the model will be retrained on the generated data')

    #time.sleep(60)

    # save the weights after training the model
    AGENT.save(DB_TABLE)
