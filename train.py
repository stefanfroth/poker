import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
#import tensorflow as tf
import datetime
from agent import Agent
import time

#log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

start_time = time.time()
AGENT = Agent()

#print(f'The models weights before loading are: {AGENT.model.get_weights()}')

AGENT.load('2019-08-15-15:29')
#print(f'The models weights after loading are: {AGENT.model.get_weights()}')
AGENT.read_data('v1')
#print('Read the data successfully')
AGENT.create_embedding_input()
#print(f'Created embedding input {AGENT.input_card_embedding}')
AGENT.create_state_input()
#print(f'Created state input {AGENT.input_state}')
#print(f'The actions are {AGENT.input['action']}')
#print(f'The rewards are {AGENT.input['reward']}')
#print(f'The model operations are: {AGENT.model.get_operations()}')
losses = []
#print(f'The optimizer is: {AGENT.adam}')
for i in range(20):
    losses.append(AGENT.train_model(AGENT.input_card_embedding, AGENT.input_state, np.array(AGENT.input['action']), np.array(AGENT.input['reward'])))
#print(f'The models weights after training are: {AGENT.model.get_weights()}')
#time.sleep(60)
AGENT.save()
loss = pd.DataFrame(losses)
print(f'The loss is {loss}')
loss.to_csv(f'./loss/loss_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}.csv')

print('''We have retrained the network.
This took {} to run'''.format(time.time() - start_time))

#tensorboard --logdir logs/fit
