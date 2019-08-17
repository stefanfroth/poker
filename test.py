from poker_verbose import Game
from agent import Agent

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 1000
DB_TABLE = 'v1vv2_2019_08_17_11:01'

agent1 = Agent()
agent1.load('2019-08-15-15:29')

agent2 = Agent()
agent2.load('2019-08-17-11:01')

AGENTS = [agent1, agent2]

g = Game(PLAYERS, BLIND, STACK, agents=[agent1, agent2], db_table=DB_TABLE, limit=LIMIT)
print('Let the games begin!')

for i in range(GAMES):
     g.play_one_complete_game()
     print(f'Game {i+1} has been played!')

print(f'Yeahy, agent1 and agent2 played {GAMES} games of poker against each other.')
