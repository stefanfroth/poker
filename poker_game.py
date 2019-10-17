'''
The module poker contains all the classes, attributes and methods to play
a game of poker.
'''


# import the necessary packages
import itertools
import random
import time
import numpy as np
import sqlalchemy as sqa
import pandas as pd
from hand_evaluator import Evaluator

# pylint: disable=line-too-long

# establish a connection to the database
DB = 'postgres://localhost/poker'
ENGINE = sqa.create_engine(DB)


# Chose some global input variables
# VERBOSE indicates whether print statements are to be active
VERBOSE = 1

# PRESENTATION indicates whether we are going to use the fancy suits or letters
PRESENTATION = 0

# Allow for the possibility of human players.
if PRESENTATION == 1:
    HUMANS = 1
else:
    HUMANS = 0


class Card:
    '''
    The class Card takes a tuple(=card) as input and makes
    its face and suit accessible.
    '''

    # pylint: disable=too-few-public-methods

    def __init__(self, card):
        self.face, self.suit = card


class Player():
    '''
    The class Player represents a player in a poker game.
    The player has an initial stack of money and at the beginning of each game
    she is dealt two cards. During the game she takes actions.
    '''
    # pylint: disable=too-many-instance-attributes

    handrankorder = {'straight-flush': 1, 'four-of-a-kind': 2, 'full-house': 3,
                     'flush': 4, 'straight': 5, 'three-of-a-kind': 6,
                     'two-pair': 7, 'one-pair': 8, 'high-card': 9}
    cardrankorder = {'a': 1, 'k': 2, 'q': 3, 'j': 4, 't': 5, '9': 6,
                     '8': 7, '7': 8, '6': 9, '5': 10,
                     '4': 11, '3': 12, '2': 13}

    def __init__(self, stack, blind, name, agent):
        # stack_original will always save the original stack of the player
        self.stack_original = stack
        # stack will save the current stack of the player
        self.stack = stack
        # stack_old is a variable in order to calculate the reward at the end of each game.
        self.stack_old = stack
        self.blind = blind
        self.name = name
        self.agent = agent

        self.hand = []
        self.own_bet = 0
        self.active = 1
        self.last_actions = []
        # self.best_hand = []
        self.actions = {1: 'fold', 2: 'call', 3: 'raise'}

        # variables for the Deep Reinforcement Learning.
        self.input_card_embedding = np.zeros((1, 7))
        self.input_state = np.zeros((1, 18))

        self.reward = 0

        '''
        :param stack: The initial stack of the player.
        :param blind: The small blind of the game.
        :param name: The name of the player.
        :param agent: The agent behind the player (Neural Network).
        '''

    def create_embedding_input(self, input_vars):
        '''
        The method create_embedding_input creates the input for the embedding of the card vectors
        '''
        # I will have to refactor the code here because this code will create
        # matrices for the action functions but I only want an array
        cards = input_vars[['hand1', 'hand2', 'community1', 'community2',
                            'community3', 'community4', 'community5']].to_numpy()

        self.input_card_embedding = np.squeeze(cards)

    def create_state_input(self, input_vars):
        '''
        The method create_state_input creates the input for the neural network
        apart from the card embeddings
        '''
        state = input_vars[['position', 'round', 'bet',
                            'action_last_0', 'action_last_1', 'action_last_2',
                            'action_last_3', 'action_last_4',
                            'action_second_0', 'action_second_1', 'action_second_2',
                            'action_second_3', 'action_second_4',
                            'action_third_0', 'action_third_1', 'action_third_2',
                            'action_third_3', 'action_third_4']].to_numpy()

        self.input_state = np.squeeze(state)

    def chose_action(self, state):
        '''
        The method chose_action simulates the choice of the agent regarding his
        action.

        :param state: Observation the agent makes about the state of the game.
        '''
        action = \
            np.squeeze(np.random.choice(3, 1, p=np.squeeze(self.agent.model.predict(state)))+1)

        return action

    def fold(self):
        '''
        The player folds.
        '''
        self.last_actions.append(1)
        self.active = 0
        if PRESENTATION == 1:
            print(f'Player {self.name}: My cards are sh..! I fold!')

    def call(self, highest_bet):
        '''
        The player calls.
        '''
        self.last_actions.append(2)
        self.own_bet = highest_bet
        if PRESENTATION == 1:
            print(f'Player {self.name}: I think I can win this game. I am calling!')

    def raise_bet(self, highest_bet, limit):
        '''
        The player raises the highest_bet.
        '''
        self.last_actions.append(3)
        self.own_bet = highest_bet + limit
        # if PRESENTATION == 1:
        #    print(f'Player {self.name}: I am raising!')

    def take_action(self, highest_bet, limit, state, blind=None):
        '''
        The method take_action determines the action of the player when it is his turn
        to bet.
        '''

        # pylint: disable=too-many-branches

        if blind == 'small':
            self.own_bet = self.blind
            self.last_actions.append(3)
            if VERBOSE == 1:
                print(f'Player {self.name}: I have bet the small blind of ${self.blind}!')

        elif blind == 'big':
            self.own_bet = self.blind * 2
            self.last_actions.append(3)
            if VERBOSE == 1:
                print(f'Player {self.name}: I have bet the big blind of ${self.blind*2}!')

        # Action if the player is non-human:
        elif self.agent != 'human':
            if highest_bet == self.stack_original:
                self.call(highest_bet)
                print('I called because I do not have more money to spend!')
            else:
                # incorporate some randomness into the decision in order to
                # ensure exploration.
                eps = random.random()
                if eps > 0.2:
                    action = self.chose_action(state)
                else:
                    # Make sure the agent does not fold when his
                    # bet is the highest.
                    if self.own_bet < highest_bet:
                        action = np.squeeze(np.random.choice(3)+1)
                    else:
                        action = np.squeeze(np.random.choice(2)+2)

                if action == 1:
                    self.fold()
                elif action == 2:
                    self.call(highest_bet)
                else:
                    self.raise_bet(highest_bet, limit)
                print(f'Player {self.name}: I have',
                      f'{self.actions[self.last_actions[-1]]}ed',
                      f'and am betting ${self.own_bet}!')

        # Action if the player is human:
        else:
            ans = 0
            while not ans:
                try:
                    txt = int(input('''Please chose your action (fold (1)
                                    check or call(2), raise(3)): '''))
                    if txt not in (1, 2, 3):
                        raise ValueError
                    ans = 1
                except ValueError:
                    ans = 0
                    print("That's not an option! Please chose between:",
                          "1 for fold, 2 for call/check or 3 for raise.")

            action = int(txt)
            if action == 1:
                self.fold()
            elif action == 2:
                self.call(highest_bet)
            elif action == 3:
                self.raise_bet(highest_bet, limit)
                print(f'Player {self.name}: I {self.actions[self.last_actions[-1]]}',
                      f'ed and am betting ${self.own_bet}!')

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(1)

        return self.last_actions[-1]

    def evaluate_hand(self):
        '''
        The function evaluate_hand evaluates the highest ranked hand out of
        all the possible 5 card combinations of the player.
        '''
        possible_hands = list(itertools.combinations(self.hand, 5))
        hand_values_rank = []
        for i, hand in enumerate(possible_hands):
            # print([(hand[i].face, hand[i].suit) for i in range(len(hand))])
            evaluator = Evaluator(hand, PRESENTATION)
            rank, highest_card = evaluator.find_best_hand()
            hand_values_rank.append((i, rank, self.handrankorder[rank],
                                     [self.cardrankorder[highest_card[i]] for i in range(len(highest_card))], highest_card, self.name))

        ranked = sorted(hand_values_rank, key=lambda x: (x[2], x[3]), reverse=False)
        if VERBOSE == 1:
            print(f'The best hand of player {self.name} is a {ranked[0][1]}')
            print([(card.face, card.suit) for card in possible_hands[ranked[0][0]]])

        # self.best_hand = 0#possible_hands[np.argmax(hand_values)]
        # print(hand_values)

        return ranked[0]


class Game:
    '''
    The class Game is a representation of a Fixed-Limit Texas Hold'em game.
    '''

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self, nr_of_players, blind, stack, agents, db_table, limit=200):
        self.nr_of_players = nr_of_players
        self.blind = blind
        self.stack = stack
        self.agent = agents
        self.db_table = db_table
        self.limit = limit

        # initialize the players, their order and positions
        self.players = []

        # Create players:
        # Assign Neural Network to the players:
        for player_number in range(self.nr_of_players-HUMANS):
            if isinstance(self.agent, list):
                print(f'Player {player_number} will be of type agent',
                      f'{self.agent[player_number%len(self.agent)]}.')
                player = Player(self.stack, self.blind, player_number+1,
                                self.agent[player_number % len(self.agent)])
            else:
                player = Player(self.stack, self.blind, player_number+1, self.agent)
            self.players.append(player)
            if VERBOSE == 1:
                print(f'successfully created player {player_number+1}.',
                      f'He is a {self.players[player_number].agent}.')

        # Create human player:
        for player_number in range(HUMANS):
            pos = HUMANS-player_number-1
            self.players.append(Player(self.stack, self.blind, self.nr_of_players-pos, 'human'))

        self.player_names = list(range(nr_of_players))
        self.order = list(range(nr_of_players))
        self.position_small = 0
        self.active_players = nr_of_players

        # create a variable that stores who has called to determine when to end the round
        self.call_counter = 0

        # initialize counters and storage for the money amounts
        self.round = 1
        self.game_count = 1

        self.highest_bet = 0
        self.pot = 0
        self.winner = 0

        # Suits of the cards
        if PRESENTATION == 1:
            self.suit = '♥ ♦ ♣ ♠'.split()
        else:
            self.suit = ['h', 'd', 'c', 's']

        # Faces of the cards
        self.faces = '2 3 4 5 6 7 8 9 t j q k a'
        self.face = self.faces.split()

        # Create a deck, a dictionary ordering the cards and a placeholder for the community cards
        self.deck = list(itertools.product(self.face, self.suit))
        self.card_ordering = {self.deck[i]: i+1 for i in range(len(self.deck))}
        self.community_cards = []

        # Create a data frame to store the data that is saved
        self.output = pd.DataFrame()

        '''
        :param nr_of_players: The number of players that will participate in the game.
        :param blind: The size of the small blind.
        :param stack: The size of the initial stack of the game.
        :param agents: The weights of the model that is playing.
        :param db_table: The name of the database table the results are stored in.
        :param limit: The (fixed) limit of the bets pre-flop and flop.
        It will be doubled for the turn and the river.
        The default is 200.
        '''

    def determine_order(self):
        '''
        The method determine_order determines the order of players in each game.
        Additionally it determines the position of small and big blind
        '''
        if self.game_count != 1:
            self.order.append(self.order.pop(0))
            self.position_small = self.order[0]

    def create_deck(self):
        '''
        The method create_deck takes the faces and suits of the Game as input,
        combines and shuffles them to create a new deck for each game.
        '''
        self.deck = list(itertools.product(self.faces.split(), self.suit))
        np.random.shuffle(self.deck)

    def deal_cards(self):
        '''
        The method deal_cards takes the list of players (self.players) and
        deals a hand of two cards for each player.
        '''
        for player in self.players:
            player.hand = [Card(self.deck.pop()), Card(self.deck.pop())]

            if VERBOSE == 1:
                print(f"Player {player.name}'s hand is",
                      f"{[(card.face, card.suit) for card in player.hand]}")

    def collect_data(self, player, position):
        '''
        The method collect_data collects the data that is to be used as input
        for the deep reinforcement learning. Observe that the data is scaled.
        Min-max scaling is applied.

        :param player: The player which has to take an action.
        :param position: The position of the player in the game (relative to the dealer).
        '''

        # pylint: disable=too-many-statements

        # create dicitonary for appending the self.output data frame:
        data = {}

        # The player and the game are not inputs to the model but are needed to
        # keep track for evaluation purposes.
        data['player'] = player
        data['game'] = self.game_count
        data['agent'] = str(self.players[player].agent)

        # input variables for the model
        data['position'] = position/self.nr_of_players
        data['round'] = (self.round-1)/(4-1)
        data['bet'] = self.players[player].own_bet/self.players[player].stack
        data['hand1'] = self.card_ordering[(self.players[player].hand[0].face,
                                            self.players[player].hand[0].suit)]/52
        data['hand2'] = self.card_ordering[(self.players[player].hand[1].face,
                                            self.players[player].hand[1].suit)]/52
        if self.round == 1:
            data['community1'] = 0
            data['community2'] = 0
            data['community3'] = 0
            data['community4'] = 0
            data['community5'] = 0
        elif self.round == 2:
            data['community1'] = self.card_ordering[self.community_cards[0]]/52
            data['community2'] = self.card_ordering[self.community_cards[1]]/52
            data['community3'] = self.card_ordering[self.community_cards[2]]/52
            data['community4'] = 0
            data['community5'] = 0
        elif self.round == 3:
            data['community1'] = self.card_ordering[self.community_cards[0]]/52
            data['community2'] = self.card_ordering[self.community_cards[1]]/52
            data['community3'] = self.card_ordering[self.community_cards[2]]/52
            data['community4'] = self.card_ordering[self.community_cards[3]]/52
            data['community5'] = 0
        else:
            data['community1'] = self.card_ordering[self.community_cards[0]]/52
            data['community2'] = self.card_ordering[self.community_cards[1]]/52
            data['community3'] = self.card_ordering[self.community_cards[2]]/52
            data['community4'] = self.card_ordering[self.community_cards[3]]/52
            data['community5'] = self.card_ordering[self.community_cards[4]]/52

        count = 0
        for player_number in self.order:
            if player_number != player:
                # Take the bet and active columns for tests on the sql database table
                data[f'bet_{count}'] = self.players[player_number].own_bet\
                    / self.stack
                data[f'active_{count}'] = self.players[player_number].active

                # introduce (scaled) values for the last three actions of the player
                if not self.players[player_number].last_actions:
                    data[f'action_last_{count}'] = 0
                    data[f'action_second_{count}'] = 0
                    data[f'action_third_{count}'] = 0
                elif len(self.players[player_number].last_actions) == 1:
                    data[f'action_last_{count}'] = self.players[player_number].last_actions[-1]/3
                    data[f'action_second_{count}'] = 0
                    data[f'action_third_{count}'] = 0
                elif len(self.players[player_number].last_actions) == 2:
                    data[f'action_last_{count}'] = self.players[player_number].last_actions[-1]/3
                    data[f'action_second_{count}'] = self.players[player_number].last_actions[-2]/3
                    data[f'action_third_{count}'] = 0
                else:
                    data[f'action_last_{count}'] = self.players[player_number].last_actions[-1]/3
                    data[f'action_second_{count}'] = self.players[player_number].last_actions[-2]/3
                    data[f'action_third_{count}'] = self.players[player_number].last_actions[-3]/3

                count += 1

        # output variables for the model
        data['action'] = ''

        self.output = self.output.append(data, ignore_index=True)

    def save_action(self, player):
        '''
        The method save_action appends the action chosen by the player to the
        data frame and reduces the number of active players by one if the
        player folds.

        :param player: The player whose action is saved.
        :param action: The action he took.
        '''
        # save the action of the agent after he made his decision
        self.output.at[self.output.shape[0]-1, 'action'] = self.players[player].last_actions[-1]

    # pylint: disable=inconsistent-return-statements

    def eliminate_and_check_activity(self, action):
        '''
        The method eliminate_and_check_activity eliminates player that fold and
        checks whether a round is still active within the order.

        :param action: The action the player chose.
        '''

        # eliminate players from the game if they folded
        if action == 1:
            self.active_players -= 1

        # increase call_counter if someone called and reset it if someone raised
        elif action == 2:
            self.call_counter += 1
        else:
            self.call_counter = 0

            # pylint: disable=no-else-return

        if self.active_players == 1:
            return True

        elif self.call_counter == self.active_players - 1:
            # advance to the next round
            self.round += 1
            return True

    def iterative_play(self):
        '''
        The module iterative_play is called when the players enter in a phase
        of iterative play.
        '''

        check_activity = False

        # run until the round is exhaustively played
        while not check_activity:

            for position, player in enumerate(self.order):

                if self.players[player].own_bet <= self.highest_bet\
                        and self.players[player].active == 1:
                    # if VERBOSE == 1:
                    #    print(f'The highest bet is {self.highest_bet} and
                    # the stack is {self.stack}')

                    if self.highest_bet >= self.stack and\
                       self.players[player].own_bet >= self.stack:
                        action = self.players[player].\
                            take_action(self.highest_bet, self.limit,
                                        [[self.players[player].input_card_embedding],
                                         [self.players[player].input_state]])

                        check_activity = self.eliminate_and_check_activity(action)
                        if check_activity:
                            # if VERBOSE == 1:
                            #    print('The round is over because of calling out in iterative play.')
                            return True

                    else:
                        self.collect_data(player, position)

                        self.players[player].\
                            create_embedding_input(self.output.loc[[self.output.shape[0]-1]])

                        self.players[player].\
                            create_state_input(self.output.loc[[self.output.shape[0]-1]])

                        action = self.players[player].\
                            take_action(self.highest_bet, self.limit,
                                        [[self.players[player].input_card_embedding],
                                         [self.players[player].input_state]])

                        self.save_action(player)

                        if self.players[player].own_bet > self.highest_bet:
                            self.highest_bet = self.players[player].own_bet

                        check_activity = self.eliminate_and_check_activity(action)
                        if check_activity:
                            # if VERBOSE == 1:
                            #    print('The round is over because of calling out in iterative play.')
                            return True

    def action_first_round(self):
        '''
        The function action calls all agents sequentially to decide on their action
        '''
        # action of the small blind player
        self.players[self.order[0]].take_action(self.highest_bet,
                                                self.limit,
                                                [],
                                                blind='small')

        # action of the big blind player
        self.players[self.order[1]].take_action(self.highest_bet,
                                                self.limit,
                                                [],
                                                blind='big')
        self.highest_bet = self.players[self.order[1]].own_bet

        # let the non-blind players take their turn
        for position, player in enumerate(self.order[2:]):
            if self.highest_bet >= self.stack:
                action = self.players[player].\
                    take_action(self.highest_bet, self.limit,
                                [[self.players[player].input_card_embedding],
                                 [self.players[player].input_state]])

                self.eliminate_and_check_activity(action)
            # for jedem take_action muss die Information ausgelesen werden
            # self.write_data(player, position)
            else:
                self.collect_data(player, position+2)

                self.players[player].\
                    create_embedding_input(self.output.loc[[self.output.shape[0]-1]])

                self.players[player].\
                    create_state_input(self.output.loc[[self.output.shape[0]-1]])

                action = self.players[player].\
                    take_action(self.highest_bet, self.limit,
                                [[self.players[player].input_card_embedding],
                                 [self.players[player].input_state]])
                self.save_action(player)

                if self.players[player].own_bet > self.highest_bet:
                    self.highest_bet = self.players[player].own_bet

                self.eliminate_and_check_activity(action)

        self.iterative_play()

        if VERBOSE == 1:
            print(f'round {self.round-1} is over and {self.active_players}',
                  'players are still in the game!')

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(1)

    def deal_flop(self):
        '''
        The method deal_flop deals the flop.
        '''

        # pylint: disable=expression-not-assigned
        [self.community_cards.append(self.deck.pop()) for x in range(3)]
        [player.hand.append(Card(x)) for player in self.players for x in self.community_cards]

        if VERBOSE == 1:
            print(f'The flop is: {self.community_cards}')

            for player in self.players:
                if player.active == 1:
                    print('Player {} has the hand {}'
                          .format(player.name, [(card.face, card.suit) for card in player.hand]))

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(1)

    def action_post_flop(self):
        '''
        The method action_post_flop calls the players to execute their
        action after observing the additional community card(s).
        '''
        self.iterative_play()

        if VERBOSE == 1:
            print(f'Round {self.round} is over and {self.active_players}',
                  'players are still in the game')

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(1)

    def deal_turn(self):
        '''
        The method deal_turn deals out the fourth community card.
        '''
        self.community_cards.append(self.deck.pop())
        # pylint: disable=expression-not-assigned
        [player.hand.append(Card(self.community_cards[-1])) for player in self.players]

        if VERBOSE == 1:
            print(f'The community cards after the turn are: {self.community_cards}')

            for player in self.players:
                if player.active == 1:
                    print('Player {} has the hand {}'
                          .format(player.name, [(card.face, card.suit) for card in player.hand]))

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(1)

    def deal_river(self):
        '''
        The method deal_river deals out the fifth community card.
        '''
        self.community_cards.append(self.deck.pop())
        # pylint: disable=expression-not-assigned
        [player.hand.append(Card(self.community_cards[-1])) for player in self.players]

        if VERBOSE == 1:
            print(f'The community cards after the river are: {self.community_cards}')

            for player in self.players:
                if player.active == 1:
                    print('Player {} has the hand {}'
                          .format(player.name, [(card.face, card.suit) for card in player.hand]))

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(1)

    def determine_pot_size(self):
        '''
        The method determine_pot_size calculates how much money is in the pot.
        '''
        pot = 0
        for player_number in self.players:
            pot += player_number.own_bet

        self.pot = pot

    def pass_to_next_game(self):
        '''
        The method pass_to_next_game finishes of one game of poker and
        passes on to the new one.
        '''

        # determine who won the game
        if self.active_players > 1:
            best_hands = []

            for player in self.players:
                if player.active == 1:
                    best_hands.append(player.evaluate_hand())

            winning_hand = sorted(best_hands, key=lambda x: (x[2], x[3]),
                                  reverse=False)[0]

            if VERBOSE == 1:
                print(f'''Player {winning_hand[5]} wins with a {winning_hand[1]}.\
                His hand is {winning_hand[4]}''')

            self.winner = winning_hand[5]

        else:
            for player in self.players:
                if player.active == 1:
                    self.winner = player.name

                    if VERBOSE == 1:
                        print(f'''Player {self.winner} wins because everyone else dropped out.''')

        # distribute the pot
        self.determine_pot_size()
        self.players[self.winner-1].stack += self.pot

        # update the stack and reset the own_bet
        for player in self.players:
            player.stack -= player.own_bet
            if VERBOSE == 1:
                print(f"Player {player.name}'s stack after game",
                      f"{self.game_count} is {player.stack}.")
            player.reward = player.stack - player.stack_old
            # update old stack for the next round
            player.stack_old = player.stack
            player.own_bet = 0

        # set pot to zero again
        self.pot = 0

        # increase game count
        self.game_count += 1

        # reset limit and round
        if self.round > 2:
            self.limit = self.limit/2
        self.round = 1

        # reset all players to be active and delete their last actions
        for player in self.players:
            player.active = 1
            player.last_actions = []
        self.active_players = self.nr_of_players

        # reset highest_bet
        self.highest_bet = 0

        # reset the call_counter
        self.call_counter = 0

        # reset the community cards
        self.community_cards = []

    def add_reward(self):
        '''
        The method add_reward adds the reward of each player for each action
        to the output data frame that is written to the database.
        '''
        # print(self.output)
        self.output['reward'] = self.output['player']
        self.output['reward_of_action'] = ''
        for i in range(self.output.shape[0]):
            player = self.players[int(self.output.at[i, 'reward'])].name - 1
            reward = self.players[player].reward
            self.output.at[i, 'reward'] = reward
            # if VERBOSE == 1:
            #    print(f'Player {player+1} has made a reward in this game of {reward}!')

            bet_until_decision = self.output.at[i, 'bet'] * self.stack
            if VERBOSE == 1:
                print(f'Player bet until this round was {bet_until_decision}')
            if reward >= 0:
                # Change: once players with different stack sizes play,
                # this has to be updated because the scaling will not work
                self.output.at[i, 'reward_of_action'] = (reward-bet_until_decision+self.stack)\
                    / (self.stack*(self.nr_of_players-1)+self.stack)
            else:
                # Note: negative rewards are not scaled as much as positive rewards
                self.output.at[i, 'reward_of_action'] = (reward+bet_until_decision+self.stack)\
                    / (self.stack*(self.nr_of_players-1)+self.stack)
            if VERBOSE == 1:
                print(f"The reward of the last action was {self.output.at[i, 'reward_of_action']}")

    def write_data(self):
        '''
        The function write_data writes the collected data into the postgres
        database "poker"
        '''
        self.output.to_sql(self.db_table, con=ENGINE, if_exists='append')

        # reset the output_frame
        self.output = pd.DataFrame()

    def play_one_complete_game(self):
        '''
        This function simulates a complete game of poker without all the steps
        in between.
        '''
        self.determine_order()
        self.create_deck()
        self.deal_cards()
        self.action_first_round()
        self.call_counter = -1
        if self.active_players > 1:
            self.deal_flop()
            self.action_post_flop()
            self.call_counter = -1
            if self.active_players > 1:
                # increase the limit for the betting rounds following turn and river
                self.limit = self.limit * 2
                self.deal_turn()
                self.action_post_flop()
                self.call_counter = -1
                if self.active_players > 1:
                    self.deal_river()
                    self.action_post_flop()
        self.pass_to_next_game()
        self.add_reward()
        self.write_data()

        # introduce some latency for the human eye to follow
        if PRESENTATION == 1:
            time.sleep(5)

    def __repr__(self):
        return '''The game is in round {} and in game {}. The highest bet is {}.
        The position of the small blind is {}.
        '''.format(self.round, self.game_count, self.highest_bet, self.position_small+1)
