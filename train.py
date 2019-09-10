'''
The module train trains the model based on the weights of a chosen version.
'''

import time
import warnings
import datetime
import numpy as np
import pandas as pd
from agent import Agent

warnings.simplefilter(action='ignore', category=FutureWarning)

# Chose whether print statements are to be shown while running the code or not.
VERBOSE = 1
# Chose the version that shall be trained.
WEIGHTS = '2019-08-18-16:51_greedy'
# Indicate on which data it is to be trained.
DB_TABLE = 'sunk'
VERSION = 'v5'

START_TIME = time.time()

# instantiate the Agent
AGENT = Agent()

# Load the weights of the version to be trained.
AGENT.load(WEIGHTS)
if VERBOSE == 1:
    print(f'The models weights after loading are: {AGENT.model.get_weights()}')

AGENT.read_data(DB_TABLE)

# Create model inputs
AGENT.create_embedding_input()
AGENT.create_state_input()

# Create a list to save the losses of the epochs.
LOSSES = []

# Train the model and save the loss
for i in range(20):
    LOSSES.append(AGENT.train_model(AGENT.input_card_embedding,
                                    AGENT.input_state,
                                    np.array(AGENT.input['action']),
                                    np.array(AGENT.input['reward'])))

if VERBOSE == 1:
    print(f'The models weights after training are: {AGENT.model.get_weights()}')

AGENT.save(VERSION)
LOSS = pd.DataFrame(LOSSES)
#print(f'The loss is {loss}')
LOSS.to_csv(f'loss_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}.csv')

if VERBOSE == 1:
    print('''We have retrained the network. This took {} to run'''.format(time.time() - START_TIME))
