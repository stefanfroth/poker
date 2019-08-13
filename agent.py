import itertools
import pandas as pd
import sqlalchemy as sqa
import numpy as np
from tensorflow.keras import models, layers, metrics, optimizers
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import backend as K
import datetime

DB = f'postgres://localhost/{"poker"}'
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


    def build_model(self):
        '''
        The function build_model sets up the basic structure for the model that is going to be trained.
        '''
        # Card embeddings
        card_input = Input(shape=7)
        card_output = layers.Embedding(self.max_features, self.vector_size, input_length=self.max_len)(card_input)
        card_output = layers.Flatten()(card_output)

        # Other information
        state_input = Input(shape=13)

        #Merge and add dense layer
        merge_layer = layers.concatenate([state_input, card_output])
        x = layers.Dense(64, activation='relu')(merge_layer)
        x = layers.Dense(32, activation='relu')(x)
        main_output = layers.Dense(3, activation='softmax')(x)

        # Define model with two inputs
        self.model = models.Model(inputs=[card_input, state_input], outputs=[main_output])

        #    action_prob_ph = self.model.output
        action_prob = K.sum(self.model.output)
        action_onehot_ph = K.placeholder(shape=(None, 3),
                                                 name="action_onehot")
        discount_reward_ph = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        #action_prob = K.sum(action_prob_ph * action_onehot_ph, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_ph
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   #constraints=[],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_ph,
                                           discount_reward_ph],
                                   outputs=[action_prob], updates=updates)

        self.model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

        return self.model


    def read_model(self):
        '''
        The method read_model reads the data saved into the postgres database.
        '''
        self.input = pd.read_sql('results', con=ENGINE)


    def create_embedding_input(self):
        '''
        The function create_embedding_input creates the input for the embedding of the card vectors
        '''
        # I will have to refactor the code here because this code will create matrices for the action functions but I only want an array
        # However, for the model improvement I will want to have all of them
        if self.input.shape[0] > 1:
            cards = np.array(self.input[['hand1', 'hand2', 'community1', 'community2', 'community3', 'community4', 'community5']].loc[0])
            for i in range(1, self.input.shape[0]):
                cards = np.vstack((cards, np.array(self.input[['hand1', 'hand2', 'community1', 'community2', 'community3', 'community4', 'community5']].loc[i])))
        else:
            cards = np.array(self.input[['hand1', 'hand2', 'community1', 'community2', 'community3', 'community4', 'community5']])

        self.input_card_embedding = np.squeeze(cards)
        print('The card embedding input is {}'.format(self.input_card_embedding))


    def create_state_input(self):
        '''
        The function create_state_input creates the input for the neural network apart from the card embeddings
        '''
        if self.input.shape[0] > 1:
            state = np.array(self.input[['position', 'round', 'active_0', 'active_1', 'active_2', 'active_3', 'active_4', \
            'bet', 'bet_0', 'bet_1', 'bet_2', 'bet_3', 'bet_4']].loc[0])
            for i in range(1, self.input.shape[0]):
                state = np.vstack((state, np.array(self.input[['position', 'round', 'active_0', 'active_1', 'active_2', 'active_3', 'active_4', \
                'bet', 'bet_0', 'bet_1', 'bet_2', 'bet_3', 'bet_4']].loc[i])))
        else:
            state = np.array(self.input[['position', 'round', 'active_0', 'active_1', 'active_2', 'active_3', 'active_4', \
            'bet', 'bet_0', 'bet_1', 'bet_2', 'bet_3', 'bet_4']])

        self.input_state = np.squeeze(state)


    def train_model(self, states, actions, rewards):
        '''
        The method train_model trains the model after every nth game.
        '''
        train_fn(inputs=[states, actions, rewards])


    def train(self, X, y, epochs):
        self.model.fit(X, y, epochs=epochs, batch_size=50)
        #score = self.model.evaluate(X, y, batch_size=50)
        print("The model was trained.")

    def save(self):
        date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        with open(f"./weights/weights_{date_string}.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(f'./weights/weights_{date_string}.h5')

    def load(self, date_string):
        K.clear_session()
        with open(f'./weights/weights_{date_string}.json', 'r') as f:
            json = f.read()
        self.model = model_from_json(json)
        self.model.load_weights(f'./weights/weights_{date_string}.h5')
        self.model.compile(optimizer=adam, loss='categorical_crossentropy')
