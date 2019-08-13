import numpy as np
from poker import Card, Evaluator, Player, Game
from agent import Agent
import sqlalchemy as sqa

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 2
AGENT = Agent()

AGENT.build_model()
print(AGENT.model.layers[1].get_weights())

g = Game(PLAYERS, BLIND, STACK, AGENT, LIMIT)


for i in range(GAMES):
    g.play_one_complete_game()

print(f'{GAMES} games were played!')

AGENT.save()

print('''The weights were saved! Now the model will be retrained on the generated
        data''')

AGENT.read_model()
AGENT.create_embedding_input()
AGENT.create_state_input()
AGENT.train([AGENT.input_card_embedding, AGENT.input_state], np.array(AGENT.input['action']), 10)

AGENT.save()

print('The model has been retrained and the new weights were saved as well.')
