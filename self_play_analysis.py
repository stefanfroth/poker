'''
The module self_play_analysis creates a png showing the rewards of each version
of the poker bot playing against each other for the first 50 games. Furthermore,
it tests if the rewards of the newer version bot are statistically significant from
zero.
'''

import sqlalchemy as sqa
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


DB = 'postgres://localhost/poker'
ENGINE = sqa.create_engine(DB)

OLD_VERSION = 'v7'
NEW_VERSION = 'v8'
DB_TABLE = 'v7vv8_2019_08_20_14:19'

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

STAT = stats.ttest_1samp(np.array(DF4['reward']), 0)
PVALUE = STAT.pvalue

if PVALUE > 0.05:
    print('The difference in rewards between the two versions of the poker bot',
          'are not statistically significant at a 95% confidence level.')
else:
    print('The difference in rewards between the two versions of the poker bot',
          f'are statistically significant. The p-value is {PVALUE}.')
