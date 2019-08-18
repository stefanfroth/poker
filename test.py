from poker_verbose import Game
from agent import Agent

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 1000
DB_TABLE = 'v3vv4_2019_08_18_10:11'

agent1 = Agent()
agent1.load('v3_2019-08-17-19:11_20_epochs')

agent2 = Agent()
agent2.load('v4_2019-08-18-10:02_20_epochs')

AGENTS = [agent1, agent2]

g = Game(PLAYERS, BLIND, STACK, agents=[agent1, agent2], db_table=DB_TABLE, limit=LIMIT)
print('Let the games begin!')

for i in range(GAMES):
     g.play_one_complete_game()
     print(f'Game {i+1} has been played!')

print(f'Yeahy, agent1 and agent2 played {GAMES} games of poker against each other.')
