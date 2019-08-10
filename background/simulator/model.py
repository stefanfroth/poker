from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras import metrics
from keras import backend as K
import numpy as np


class Model:

    def create(self):
        self.model = Sequential([
            #Conv2D(8, (3, 3), strides=(1, 1), padding='valid', input_shape=(16, 16, 1)),
            #Activation('relu'),
            #MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
            #Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
            #Activation('relu'),
            #MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'),
            #Flatten(),
            Dense(8, input_shape=(25,)),
            Activation('relu'),
            #Dropout(0.5),
            Dense(4),
            Activation('softmax'),
        ])
        print(self.model.summary())
        #self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def train(self, X, y, epochs):
        self.model.fit(X, y, epochs=epochs, batch_size=50)
        score = self.model.evaluate(X, y, batch_size=50)
        print(f"training: {score}")

    def save(self):
        with open("weights.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights('weights.h5')

    def load(self):
        K.clear_session()
        with open('weights.json', 'r') as f:
            json = f.read()
        self.model = model_from_json(json)
        self.model.load_weights("weights.h5")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def predict(self, state):
        #X = np.array([state]).reshape(1, 16, 16, 1)
        X = np.array([state]).reshape(1, 25)
        ypred = self.model.predict(X)[0]
        return ypred


if __name__ == '__main__':
    sim = Simulator()
    sim.simulate(AlwaysLeftPolicy(), 10)

    # train a simple action-based model
    m = Model()
    m.create()
    m.train(states, actions, 50)
    m.save()
