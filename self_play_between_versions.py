from poker_greedy import Game, Player, Evaluator, Card
from agent import Agent

#DB = f'postgres://localhost/{"poker"}'
#ENGINE = sqa.create_engine(DB)

PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 1000
DB_TABLE = 'v7vv8_2019_08_20_14:19'

agent1 = Agent()
agent1.load('v7_2019-08-19-23:30_20_epochs')

agent2 = Agent()
agent2.load('v8_2019-08-20-14:15_20_epochs')

AGENTS = [agent1, agent2]

g = Game(PLAYERS, BLIND, STACK, agents=[agent1, agent2], db_table=DB_TABLE, limit=LIMIT)
print('Let the games begin!')

for i in range(GAMES):
     g.play_one_complete_game()
     print(f'Game {i+1} has been played!')

print(f'Yeahy, agent1 and agent2 played {GAMES} games of poker against each other.')
