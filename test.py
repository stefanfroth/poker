from poker import Game
from agent import Agent

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 1000

agent1 = Agent()
agent1.load('2019-08-14-00:24')

agent2 = Agent()
agent2.load('2019-08-14-17:59')

AGENTS = [agent1, agent2]

g = Game(PLAYERS, BLIND, STACK, agents=[agent1, agent2], limit=LIMIT)

for i in range(GAMES):
     g.play_one_complete_game()

print(f'Yeahy, agent1 and agent2 played {GAMES} games of poker against each other.')
