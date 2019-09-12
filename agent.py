'''
The module agent defines the agent that plays the game of poker. She is an
Artificial Neural Network that is trained via Deep Reinforcement Learning.
'''

import datetime
import warnings
import pandas as pd
import sqlalchemy as sqa
import numpy as np
from tensorflow.keras import models, layers, optimizers, utils
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
# import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)

# Create database connection
DB = 'postgres://localhost/poker'
ENGINE = sqa.create_engine(DB)


class Agent:
    '''
    The class Agent defines the agent that takes the actions in poker_games
    and is trained via Deep Reinforcement Learning.
    '''

    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.max_features = 53
        self.vector_size = 4
        self.max_len = 7
        self.model = ''
        self.input_card_embedding = ''
        self.input_state = ''
        self.input = ''
        self.train_fn = ''

        # instantiate the model optimizer
        self.adam = optimizers.Adam(clipnorm=1.0, clipvalue=0.1)

        # instantiate the model
        self.build_model()

    def build_model(self):
        '''
        The function build_model sets up the basic structure for the model
        that is going to be trained.

        Replaces `model.fit(X, y)` because we use the output of model
        and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.

        https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
        '''

        # Card embeddings
        card_input = Input(shape=(7,))
        card_output = layers.Embedding(self.max_features, self.vector_size,
                                       input_length=self.max_len)(card_input)
        # Change: should we include batch normalization for the embedding layer?
        #card_output = layers.BatchNormalization()(card_output)
        card_output = layers.Flatten()(card_output)

        # Other input
        state_input = Input(shape=(18,))

        # Merge and add dense layer
        merge_layer = layers.concatenate([state_input, card_output])
        merge_layer = layers.BatchNormalization()(merge_layer)
        merge_layer = layers.Dense(64, activation='relu')(merge_layer)
        #merge_layer = layers.Dense(64, activation='relu')(merge_layer)
        merge_layer = layers.BatchNormalization()(merge_layer)
        merge_layer = layers.Dense(32, activation='relu')(merge_layer)
        merge_layer = layers.BatchNormalization()(merge_layer)
        main_output = layers.Dense(3, activation='softmax')(merge_layer)

        # Define model with two inputs
        self.model = models.Model(inputs=[card_input, state_input], outputs=[main_output])

        action_prob_ph = self.model.output
        action_onehot_ph = K.placeholder(shape=(None, 3), name="action_onehot")
        reward_ph = K.placeholder(shape=(None,), name="reward")

        action_prob = K.sum(action_prob_ph * action_onehot_ph, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * reward_ph
        loss = K.mean(loss)

        updates = self.adam.get_updates(params=self.model.trainable_weights,
                                        loss=loss)

        self.train_fn = K.function(inputs=[self.model.input[0],
                                           self.model.input[1],
                                           action_onehot_ph,
                                           reward_ph],
                                   outputs=[loss, action_onehot_ph+1,
                                            action_prob_ph, action_prob,
                                            log_action_prob, reward_ph+1],
                                   updates=updates)

        self.model.compile(optimizer=self.adam, loss='categorical_crossentropy')

        return self.model

    # Change: This should be changed to a database table again.

    def read_data(self, table_name):
        '''
        The method read_model reads the data saved into the postgres database.

        :param name of the document the training data is saved in.
        '''
        self.input = pd.read_sql(table_name, con=ENGINE)

    def create_embedding_input(self):
        '''
        The function create_embedding_input creates the input for the embedding
        of the card vectors
        '''
        # I will have to refactor the code here because this code will create matrices
        # for the action functions but I only want an array --> Why?
        # However, for the model improvement I will want to have all of them
        self.input_card_embedding = self.input[['hand1', 'hand2', 'community1',
                                                'community2', 'community3',
                                                'community4', 'community5']].to_numpy()

#        self.input_card_embedding = np.squeeze(cards)

    def create_state_input(self):
        '''
        The function create_state_input creates the input for the neural network
        apart from the card embeddings
        '''
        self.input_state = self.input[['position', 'round', 'bet',
                                       'action_last_0', 'action_last_1', 'action_last_2',
                                       'action_last_3', 'action_last_4',
                                       'action_second_0', 'action_second_1', 'action_second_2',
                                       'action_second_3', 'action_second_4',
                                       'action_third_0', 'action_third_1', 'action_third_2',
                                       'action_third_3', 'action_third_4'
                                       ]].to_numpy()

#        self.input_state = np.squeeze(state)

    def train_model(self, cards, states, actions, rewards):
        '''
        The method train_model trains the model after every nth game.

        :param cards: The card_embedding inputs generated by self-play.
        :param states: The rest of the variables describing the state.
        :param actions: The actions taken by the agents.
        :param rewards: The rewards received by the agents.
        '''
#        for i in tf.get_default_graph().get_operations():
#            if i.name == 'action_onehot' or i.name == 'reward':
#                print(i.name)

        action_onehot = utils.to_categorical(actions-1, num_classes=3)
        rewards = np.float32(rewards)
        cards = np.float32(cards)
        states = np.float32(states)

        # Test: formulate the following lines as tests.
        # num_of_actions = actions.shape[0]
        # assert rewards.shape[0] == num_of_actions
        # assert cards.shape[0] == num_of_actions
        # assert states.shape[0] == num_of_actions
        # assert action_onehot.shape[1] == 3
        # assert cards.shape[1] == 7
        # assert states.shape[1] == 18
        # print(f'''The states are {states}, the cards are {cards},
        # the actions are {action_onehot} and the rewards are {rewards}.
        # We are going to train the model using these inputs.''')
        loss = self.train_fn(inputs=[cards, states, action_onehot, rewards])

        return loss[0]

    def save(self, document_name):
        '''
        The method save saves the weights of the model as a json file and as
        a h5 file.

        :param document_name: Name of the document the weights are saved as.
        '''
        date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        with open(f"weights_{document_name}_{date_string}.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(f'weights_{document_name}_{date_string}.h5')

    def load(self, date_string):
        '''
        The method load loads weights into the model.

        :param date_string: Timestampt of when the weights were created.
        '''
        with open(f'weights_{date_string}.json', 'r') as file:
            json = file.read()
        self.model = model_from_json(json)
        self.model.load_weights(f'weights_{date_string}.h5')

        # Optimization instructions have to be given again.
        action_prob_ph = self.model.output
        action_onehot_ph = K.placeholder(shape=(None, 3),
                                         name="action_onehot")
        reward_ph = K.placeholder(shape=(None,),
                                  name="reward")

        action_prob = K.sum(action_prob_ph * action_onehot_ph, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * reward_ph
        loss = K.mean(loss)

        updates = self.adam.get_updates(params=self.model.trainable_weights,
                                        loss=loss)

        self.train_fn = K.function(inputs=[self.model.input[0],
                                           self.model.input[1],
                                           action_onehot_ph,
                                           reward_ph],
                                   outputs=[loss, action_onehot_ph+1, action_prob_ph,
                                            action_prob, log_action_prob, reward_ph+1],
                                   updates=updates)

        self.model.compile(optimizer=self.adam, loss='categorical_crossentropy')
