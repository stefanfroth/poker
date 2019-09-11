'''
The module self_play_between_versions lets two different versions of the poker
bot play against each other, creates a png displaying the rewards of each version
in the first 50 games and tests if the reward of the newer version is significantly
different from zero.
'''

import warnings
import sqlalchemy as sqa
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from poker_game import Game
from agent import Agent

warnings.simplefilter(action='ignore', category=FutureWarning)

# db connection
DB = f'postgres://localhost/poker'
ENGINE = sqa.create_engine(DB)

# Versions and db table
OLD_VERSION = 'v7'
NEW_VERSION = 'v8'
DB_TABLE = 'v7vv8_2019_08_20_14:19'

# Rules of the game
PLAYERS = 6
BLIND = 50
STACK = 10000
LIMIT = 100
GAMES = 1000

# load the agents
AGENT1 = Agent()
AGENT1.load('v7_2019-08-19-23:30_20_epochs')

AGENT2 = Agent()
AGENT2.load('v8_2019-08-20-14:15_20_epochs')

AGENTS = [AGENT1, AGENT2]

# instantiate and play the game
G = Game(PLAYERS, BLIND, STACK, agents=[AGENT1, AGENT2], db_table=DB_TABLE, limit=LIMIT)
print('Let the games begin!')

for i in range(GAMES):
    G.play_one_complete_game()
    print(f'Game {i+1} has been played!')

print(f'Yeahy, agent1 and agent2 played {GAMES} games of poker against each other.')


# analyse the result of the self play between the different versions.
# Red in the database table
DF = pd.read_sql(f'{DB_TABLE}', con=ENGINE)
DF['game'] = DF.game.astype(int)


for i in range(DF.shape[0]):
    if DF.at[i, 'player'] == 0.0 or DF.at[i, 'player'] == 2.0 or DF.at[i, 'player'] == 4.0:
        DF.at[i, 'agent'] = NEW_VERSION
    else:
        DF.at[i, 'agent'] = OLD_VERSION

# Create a png displaying the differences in rewards between the two versions of the bot
# for the first 50 games.
DFP = DF[['player', 'game', 'reward', 'agent']].\
groupby(['player', 'game', 'agent'], as_index=False).mean()
DFP[['agent', 'game', 'reward']].groupby(['agent', 'game']).mean()\
.unstack(0).iloc[:50, :].plot(kind='bar', figsize=(15, 10), fontsize=15)
plt.xlabel('Game', fontsize=30)
plt.ylabel('Reward', fontsize=30)
plt.title('Total reward per Agent in Games 1-50', fontsize=30)
plt.legend()
plt.savefig(f'{OLD_VERSION}v{NEW_VERSION}.png')

# Test for statistical difference in rewards
DF2 = DF[DF['agent'] == NEW_VERSION]
DF4 = DF2[['player', 'game', 'reward', 'agent']].groupby(['player', 'agent', 'game']).mean()

# One sided t-test testing whether the reward of the NEW_VERSION agent is less or
# equal than 0. This assumes that the distribution of rewards is normal.
STAT = stats.ttest_1samp(np.array(DF4['reward']), 0)
PVALUE = STAT.pvalue/2

if PVALUE > 0.05:
    print('The Null hypothesis of the reward of the NEW_Version agent being less or equal',
          ' to zero cannot be rejected. There is no empirical evidence that the NEW_VERSION',
          ' agent is playing better.')
else:
    print('The Null hypothesis of the reward of the NEW_Version agent being less or equal',
          ' to zero is rejected. This suggests that the NEW_VERSION agent is',
          ' better at playing poker. The result is statistically significant with'
          f'a p-value of {PVALUE}.')
