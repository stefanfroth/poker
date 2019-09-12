'''
The module test_file defines the tests that are automatically run by Travis.
'''

import warnings
import sqlalchemy
import pandas as pd
from poker_game import Game
from agent import Agent

warnings.simplefilter(action='ignore', category=FutureWarning)

NR_OF_PLAYERS = 6
BLIND = 50
STACK = 10_000
DB_TABLE = 'tests'
NR_OF_GAMES = 10

AGENT = Agent()
GAME = Game(NR_OF_PLAYERS, BLIND, STACK, AGENT, DB_TABLE)

DB = 'postgres://localhost/poker'
ENGINE = sqlalchemy.create_engine(DB)


# def test_len_deck():
#     '''
#     The function test_len_deck tests that the deck always holds 52
#     unique cards.
#     '''
#     assert len(GAME.deck) == 52, 'The deck has not the right amount of cards'
#     assert len(GAME.deck) == len(set(GAME.deck)),\
#         'There are duplicates among the cards.'
#
#
# test_len_deck()

for _ in range(NR_OF_GAMES):
    GAME.play_one_complete_game()

DF = pd.read_sql(DB_TABLE, ENGINE)


def test_number_of_players():
    '''
    The function test_number_of_players tests if the class Game instantiates
    the correct number of players.
    '''
    assert len(GAME.players) == NR_OF_PLAYERS,\
        f'The number of players should be {NR_OF_PLAYERS}'


def test_all_agents_in_game():
    '''
    The function test_all_agents_in_game tests whether all agents that are
    created are also assigned to a player in the game of poker.
    '''
    agents = []
    for agent in range(NR_OF_PLAYERS):
        agent = Agent()
        agents.append(agent)
    multiagent_game = Game(NR_OF_PLAYERS, BLIND, STACK, agents, DB_TABLE)

    created_agents = [player.agent for player in multiagent_game.players]
    assert len(set(created_agents)) == NR_OF_PLAYERS,\
        f'The number of agents should equal {NR_OF_PLAYERS}'


def test_chosen_action_exists():
    '''
    The function test_chosen_action_exists tests whether agents only chose
    actions that are defined for the agent
    '''
    # Change: Will be easier and make more sense to check for entries
    # in database.
    possible_actions = {1, 2, 3}
    assert possible_actions.issuperset(set(DF.action.unique())),\
        f'Unknown actions are chosen!'


def test_not_all_players_inactive():
    '''
    The function test_not_all_players_inactive tests that not all players are
    inactive at the same time.
    '''
    assert DF[(round(DF.action_last_0, 2) == 0.33) &
              (round(DF.action_last_1, 2) == 0.33) &
              (round(DF.action_last_2, 2) == 0.33) &
              (round(DF.action_last_3, 2) == 0.33) &
              (round(DF.action_last_4, 2) == 0.33)].shape[0] == 0,\
        'All players are inactive at the same time!'


def test_no_missing_data():
    '''
    The function test_no_missing_data tests that there is no data missing in
    the database.
    '''
    assert not DF.isnull().values.any(), 'There are missing values!'


def test_all_values_in_right_range():
    '''
    The function test_all_values_in_right_range tests if all values in the
    database are in the expected range.
    '''
    variables = ['action_last_0', 'action_last_1', 'action_last_2',
                 'action_last_3', 'action_last_4',
                 'action_second_0', 'action_second_1', 'action_second_2',
                 'action_second_3', 'action_second_4',
                 'action_third_0', 'action_third_1', 'action_third_2',
                 'action_third_3', 'action_third_4',
                 'bet', 'bet_0', 'bet_1', 'bet_2', 'bet_3', 'bet_4',
                 'community1', 'community2', 'hand1', 'hand2', 'position',
                 'round', 'reward_of_action'
                 ]
    for variable in variables:
        assert not (0 < DF[variable].values.any() > 1),\
            f'{variable} is out of bounds!'


def test_bet_less_or_equal_than_stack():
    '''
    The function test_bet_less_or_equal_than_stack tests for each player that
    he never bets more money than he has.
    '''
    assert not DF.bet.values.any() > 1

#  Database
# Test: royal flush always wins
# Test: worst hand always looses


if __name__ == '__main__':
    test_number_of_players()
    test_all_agents_in_game()
    test_chosen_action_exists()
    test_not_all_players_inactive()
    test_no_missing_data()
    test_all_values_in_right_range()
    test_bet_less_or_equal_than_stack()
    print('Everything passed!')
