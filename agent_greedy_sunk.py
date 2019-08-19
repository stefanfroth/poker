import itertools
import pandas as pd
import sqlalchemy as sqa
import numpy as np
from tensorflow.keras import models, layers, metrics, optimizers, utils
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import backend as K
import tensorflow as tf
import datetime

host = 'poker.crvv64d8bwgs.eu-central-1.rds.amazonaws.com'
port = '5432'
user = 'postgres'
database = 'poker'
password = 'forthewin'

DB = f'postgres://{user}:{password}@{host}:{port}/{database}'
ENGINE = sqa.create_engine(DB)

max_features = 53
max_len = 7
vector_size = 4


class Agent:
    '''
    The class Agent takes the following parameters as input:

    and uses them to decide on his action in a specified game of poker
    '''

    def __init__(self):
        self.max_features = 53
        self.vector_size = vector_size
        self.max_len = 7
        self.model = ""
        self.input_card_embedding = ''
        self.input_state = ''
        self.input = ''
        self.train_fn = ''

        self.build_model()


    def build_model(self):
        '''
        The function build_model sets up the basic structure for the model that is going to be trained.
        '''
        #K.clear_session()

        # Card embeddings
        card_input = Input(shape=(7,))
        card_output = layers.Embedding(self.max_features, self.vector_size, input_length=self.max_len)(card_input)
        #card_output = layers.BatchNormalization()(card_output)
        card_output = layers.Flatten()(card_output)

        # Other information
        state_input = Input(shape=(19,))

        #Merge and add dense layer
        merge_layer = layers.concatenate([state_input, card_output])
        x = layers.BatchNormalization()(merge_layer)
        x = layers.Dense(64, activation='relu')(x)
        #x = layers.Dense(64, activation='relu')(merge_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        main_output = layers.Dense(3, activation='softmax')(x)

        # Define model with two inputs
        self.model = models.Model(inputs=[card_input, state_input], outputs=[main_output])


        #    action_prob_ph = self.model.output
        action_prob_ph = self.model.output#K.sum(self.model.output)
        #print(f'The probabilities for the three actions are {action_prob_ph}')
        action_onehot_ph = K.placeholder(shape=(None, 3),
                                                 name="action_onehot")
        reward_ph = K.placeholder(shape=(None,),
                                                    name="reward")

        action_prob = K.sum(action_prob_ph * action_onehot_ph, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * reward_ph
        loss = K.mean(loss)

        self.adam = optimizers.Adam(clipvalue=0.5)

        updates = self.adam.get_updates(params=self.model.trainable_weights,
                                   #constraints=[],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input[0],
                                            self.model.input[1],
                                           action_onehot_ph,
                                           reward_ph],
                                   outputs=[loss, action_onehot_ph+1, action_prob_ph, action_prob, log_action_prob, reward_ph+1], updates=updates)

        self.model.compile(optimizer=self.adam, loss='categorical_crossentropy')

        # for i in tf.get_default_graph().get_operations():
        #     if i.name == 'action_onehot' or i.name == 'reward':
        #         print(i.name)

        return self.model


    def read_data(self, table_name):
        '''
        The method read_model reads the data saved into the postgres database.
        '''
        self.input = pd.read_csv(f'{table_name}.csv')
        #print(f'The data read in is {self.input}')

        # for i in tf.get_default_graph().get_operations():
        #     if i.name == 'action_onehot' or i.name == 'reward':
        #         print(i.name + 'read')


    def create_embedding_input(self):
        '''
        The function create_embedding_input creates the input for the embedding of the card vectors
        '''
        # I will have to refactor the code here because this code will create matrices for the action functions but I only want an array
        # However, for the model improvement I will want to have all of them
        self.input_card_embedding = self.input[['hand1', 'hand2', 'community1', 'community2', 'community3', 'community4', 'community5']].to_numpy()

#        self.input_card_embedding = np.squeeze(cards)
        #print('The card embedding input is {}'.format(self.input_card_embedding))

        # for i in tf.get_default_graph().get_operations():
        #     if i.name == 'action_onehot' or i.name == 'reward':
        #         print(i.name + 'embedding')


    def create_state_input(self):
        '''
        The function create_state_input creates the input for the neural network apart from the card embeddings
        '''
        self.input_state = self.input[['position', 'round', 'action_last_0', 'bet', \
        'action_last_0', 'action_last_1', 'action_last_2', 'action_last_3', 'action_last_4',  \
        'action_second_0', 'action_second_1', 'action_second_2', 'action_second_3', 'action_second_4', \
        'action_third_0', 'action_third_1', 'action_third_2', 'action_third_3', 'action_third_4'\
        ]].to_numpy()

#        self.input_state = np.squeeze(state)

        # for i in tf.get_default_graph().get_operations():
        #     if i.name == 'action_onehot' or i.name == 'reward':
        #         print(i.name + 'state')


    def train_model(self, cards, states, actions, rewards):
        '''
        The method train_model trains the model after every nth game.
        '''
        for i in tf.get_default_graph().get_operations():
            if i.name == 'action_onehot' or i.name == 'reward':
                print(i.name)

#        print(f'The operations are: {tf.get_default_graph().get_operations()}')
        #b = np.zeros((actions.shape[0], 3))
        #b[np.arange(actions.shape[0]), actions] = 1

        action_onehot = utils.to_categorical(actions-1, num_classes=3)
        rewards = np.float32(rewards)
        cards = np.float32(cards)
        states = np.float32(states)

        n = actions.shape[0]
        assert rewards.shape[0] == n
        assert cards.shape[0] == n
        assert states.shape[0] == n
        assert action_onehot.shape[1] == 3
        assert cards.shape[1] == 7
        assert states.shape[1] == 19
        print(f'''The states are {states}, the cards are {cards}, the actions are {action_onehot} and the rewards are {rewards}.
        #We are going to train the model using these inputs.''')
        loss = self.train_fn(inputs=[cards, states, action_onehot, rewards])
        #log_loss = np.log(loss[1])
        #log_loss
        return loss[0]


    def train(self, X, y, epochs):
        self.model.fit(X, y, epochs=epochs, batch_size=50)
        #score = self.model.evaluate(X, y, batch_size=50)
        print("The model was trained.")

    def save(self):
        date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        with open(f"weights_{date_string}.json", "w") as json_file:
        #with open('./weights/weights_test2.json', 'w') as json_file:
            json_file.write(self.model.to_json())
        #self.model.save_weights('./weights/weights_test2.h5')
        self.model.save_weights(f'weights_{date_string}.h5')

    def load(self, date_string):
        # for i in tf.get_default_graph().get_operations():
        #     if i.name == 'action_onehot' or i.name == 'reward':
        #         print(i.name + 'load_1')
        #K.clear_session()
        #self.build_model()
        with open(f'weights_{date_string}.json', 'r') as f:
            json = f.read()
        self.model = model_from_json(json)
        self.model.load_weights(f'weights_{date_string}.h5')
        # 'self.adam' als string nimmt allerdings die Standardeinstellungen von self.adam
        #self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        #self.build_model()

        #    action_prob_ph = self.model.output
        action_prob_ph = self.model.output#K.sum(self.model.output)
        #print(f'The probabilities for the three actions are {action_prob_ph}')
        action_onehot_ph = K.placeholder(shape=(None, 3),
                                                 name="action_onehot")
        reward_ph = K.placeholder(shape=(None,),
                                                    name="reward")

        action_prob = K.sum(action_prob_ph * action_onehot_ph, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * reward_ph
        loss = K.mean(loss)

        self.adam = optimizers.Adam(clipvalue=0.5)

        updates = self.adam.get_updates(params=self.model.trainable_weights,
                                   #constraints=[],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input[0],
                                            self.model.input[1],
                                           action_onehot_ph,
                                           reward_ph],
                                   outputs=[loss, action_onehot_ph+1, action_prob_ph, action_prob, log_action_prob, reward_ph+1], updates=updates)

        self.model.compile(optimizer=self.adam, loss='categorical_crossentropy')

        # for i in tf.get_default_graph().get_operations():
        #     if i.name == 'action_onehot' or i.name == 'reward':
        #         print(i.name + 'load_2')
