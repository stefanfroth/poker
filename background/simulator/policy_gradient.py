
from keras.models import model_from_json
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import metrics
from keras import backend as K
import numpy as np

from model import Model


class PolicyGradientModel(Model):

    def train(self, states, actions, rewards):
        self.train_fn(inputs=[states, actions, rewards])
        

    def build_train_fn(self):
        """Replaces `model.fit(X, y)` because we use the output of model
        and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.

        https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
        """
        action_prob_ph = self.model.output
        action_onehot_ph = K.placeholder(shape=(None, 4),
                                                  name="action_onehot")
        discount_reward_ph = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_ph * action_onehot_ph, axis=1)
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
                                   outputs=[], updates=updates)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
