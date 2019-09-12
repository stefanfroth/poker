'''
The module test_file defines the tests that are automatically run by Travis.
'''

import random
import warnings
from poker_game import Game
from agent import Agent

warnings.simplefilter(action='ignore', category=FutureWarning)

DB_TABLE = 'tests'
NR_OF_PLAYERS = random.randint(1, 10)

def test_number_of_players():
    '''
    The method test_number_of_players tests if the class Game instantiates the
    correct number of players.
    '''
    agent = Agent()
    GAME = Game(NR_OF_PLAYERS, 50, 10000, agent, DB_TABLE)
    assert len(GAME.players) == NR_OF_PLAYERS, f'The number of players should be {NR_OF_PLAYERS}'


def test_all_agents_in_game():
    '''
    The method test_all_agents_in_game tests whether all agents that are created
    are also assigned to a player in the game of poker.
    '''
    agents = []
    for agent in range(NR_OF_PLAYERS):
        agent = Agent()
        agents.append(agent)

    GAME = Game(NR_OF_PLAYERS, 50, 10000, agents, DB_TABLE)

    created_agents = [player.agent for player in GAME.players]
    assert len(set(created_agents)) == NR_OF_PLAYERS, f'The number of agents should equal {NR_OF_PLAYERS}'



def test_something():
    '''
    The method test_something is a dummy test to make the Travis build work.
    '''
    assert 1 + 1 == 2


if __name__ == '__main__':
    test_something()
    test_number_of_players()
    test_all_agents_in_game()
    print('Everything passed!')
