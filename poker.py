'''
The module poker contains all the classes, attributes and methods to play
a game of poker.
'''


# import the necessary packages
import itertools
import random
import numpy as np
#from collections import namedtuple
import sqlalchemy as sqa
import pandas as pd


# establish a connection to the database
DB = f'postgres://localhost/{"poker"}'
ENGINE = sqa.create_engine(DB)


class Card:
    '''
    The class Card takes a tuple(=card) as input and makes
    its face and suit accessible.
    '''
    def __init__(self, card):
        self.face, self.suit = card



class Evaluator:
    '''
    The class Evaluator evaluates hands in a game of poker.
    It takes the hand of the player, flop, turn and river as inputs
    and returns the best possible hand for the player
    '''

    def __init__(self, hand):
        self.hand = hand
        self.faces = '2 3 4 5 6 7 8 9 t j q k a'
        self.lowaces = 'a 2 3 4 5 6 7 8 9 t j q k'
        self.face = self.faces.split()
        self.lowace = self.lowaces.split()
        #self.suit = '♥ ♦ ♣ ♠'.split()
        self.suit = ['h', 'd', 'c', 's']


    def straightflush(self, hand):
        '''
        The function straightflush determines whether the hand of the player
        is a straight flush.
        '''
        f, fs = ((self.lowace, self.lowaces) if any(card.face == '2' for card in hand)
                 else (self.face, self.faces))
        ordered = sorted(hand, key=lambda card: (f.index(card.face), card.suit))
        first, rest = ordered[0], ordered[1:]
        if (all(card.suit == first.suit for card in rest) and
             ' '.join(card.face for card in ordered) in fs):
            return 'straight-flush', ordered[-1].face
        return False


    def fourofakind(self, hand):
        '''
        The function fourofakind determines whether the player
        has four of a kind.
        '''
        allfaces = [hand[l].face for l in range(len(hand))]
        allftypes = set(allfaces)
        if len(allftypes) != 2:
            return False
        for f in allftypes:
            if allfaces.count(f) == 4:
                allftypes.remove(f)
                return 'four-of-a-kind', [f, allftypes.pop()]
            else:
                return False


    def fullhouse(self, hand):
        '''
        The function fullhouse determines whether the hand of the player
        is a full house.
        '''
        allfaces = [hand[l].face for l in range(len(hand))]
        allftypes = set(allfaces)
        if len(allftypes) != 2:
            return False
        for f in allftypes:
            if allfaces.count(f) == 3:
                allftypes.remove(f)
                return 'full-house', [f, allftypes.pop()]
            else:
                return False


    def flush(self, hand):
        '''
        The function flush determines whether the hand of the player
        is a flush.
        '''
        allstypes = {hand[l].suit for l in range(len(hand))}
        if len(allstypes) == 1:
            allfaces = [hand[l].face for l in range(len(hand))]
            return 'flush', sorted(allfaces,
                                   key=lambda f: self.face.index(f),
                                   reverse=True)
        else:
            return False


    def straight(self, hand):
        '''
        The function straigh determines whether the hand of the player
        is a straight.
        '''
        f, fs = ((self.lowace, self.lowaces) if any(card.face == '2' for card in hand)
                 else (self.face, self.faces))
        ordered = sorted(hand, key=lambda card: (f.index(card.face), card.suit))
        first, rest = ordered[0], ordered[1:]
        if ' '.join(card.face for card in ordered) in fs:
            return 'straight', ordered[-1].face
        return False


    def threeofakind(self, hand):
        '''
        The function threeofakind determines whether the player
        has three of a kind.
        '''
        allfaces = [hand[l].face for l in range(len(hand))]
        allftypes = set(allfaces)
        if len(allftypes) <= 2:
            return False
        for f in allftypes:
            if allfaces.count(f) == 3:
                allftypes.remove(f)
                return ('three-of-a-kind', [f] +
                    sorted(allftypes,
                    key=lambda f: self.face.index(f),
                    reverse=True))
        else:
            return False


    def twopair(self, hand):
        '''
        The function twopair determines whether the player
        has two pairs and which pairs he has.
        '''
        allfaces = [hand[l].face for l in range(len(hand))]
        allftypes = set(allfaces)
        pairs = [f for f in allftypes if allfaces.count(f) == 2]
        if len(pairs) != 2:
            return False
        p0, p1 = pairs
        other = [(allftypes - set(pairs)).pop()]
        return 'two-pair', pairs + other\
        if self.face.index(p0) > self.face.index(p1)\
        else pairs[::-1] + other


    def onepair(self, hand):
        '''
        The function onepair determines whether the player
        has a pairs and which pair he has.
        '''
        allfaces = [hand[l].face for l in range(len(hand))]
        allftypes = set(allfaces)
        pairs = [f for f in allftypes if allfaces.count(f) == 2]
        if len(pairs) != 1:
            return False
        allftypes.remove(pairs[0])
        return 'one-pair', pairs + sorted(allftypes,
                                          key=lambda f: self.face.index(f),
                                          reverse=True)


    def highcard(self, hand):
        '''
        The function highcard determines the highest card of the player
        in case he has no better hand.
        '''
        allfaces = [hand[l].face for l in range(len(hand))]
        return 'high-card', sorted(allfaces,
                                   key=lambda f: self.face.index(f),
                                   reverse=True)


    def find_best_hand(self):
        '''
        The function find_best_hand finds the highest ranked outcome
        for the players hand.
        '''
        straightflush = self.straightflush(self.hand)
        if straightflush:
            return straightflush

        fourofakind = self.fourofakind(self.hand)
        if fourofakind:
            return fourofakind

        fullhouse = self.fullhouse(self.hand)
        if fullhouse:
            return fullhouse

        flush = self.flush(self.hand)
        if flush:
            return flush

        straight = self.straight(self.hand)
        if straight:
            return straight

        threeofakind = self.threeofakind(self.hand)
        if threeofakind:
            return threeofakind

        twopair = self.twopair(self.hand)
        if twopair:
            return twopair

        onepair = self.onepair(self.hand)
        if onepair:
            return onepair

        return self.highcard(self.hand)


#    handrankorder =  (straightflush, fourofakind, fullhouse,
#                      flush, straight, threeofakind,
#                      twopair, onepair, highcard)

#    def rank(self, cards):
#        hand = handy(cards)
#        for ranker in handrankorder:
#            rank = ranker(hand)
#            if rank:
#                break
#        assert rank, "Invalid: Failed to rank cards: %r" % cards
#        return rank

#    def handy(self, cards='2♥ 2♦ 2♣ k♣ q♦'):
#        hand = []
#        for card in cards.split():
#            f, s = card[:-1], card[-1]
#            assert f in face, "Invalid: Don't understand card face %r" % f
#            assert s in suit, "Invalid: Don't understand card suit %r" % s
#            hand.append(Card(f, s))
#        assert len(hand) == 5, "Invalid: Must be 5 cards in a hand, not %i" % len(hand)
#        assert len(set(hand)) == 5, "Invalid: All cards in the hand must be unique %r" % cards
#        return hand



class Player():
    '''
    The class Player represents a player in a poker game.
    The player has an initial stack of money and at the beginning of each game
    she is dealt two cards. She decides on her actions given her cards.
    '''

    def __init__(self, stack, blind, name):
        self.hand = []
        self.stack = stack
        self.stack_old = stack
        self.reward = 0
        self.blind = blind
        self.name = name
        self.own_bid = 0
        self.active = 1
        self.last_actions = []
        self.best_hand = []
        self.handrankorder = {'straightflush': 1, 'four-of-a-kind': 2, 'full-house': 3,
                              'flush': 4, 'straight': 5, 'three-of-a-kind': 6,
                              'two-pair': 7, 'one-pair': 8, 'high-card': 9}
        self.cardrankorder = {'a': 1, 'k': 2, 'q': 3, 'j': 4, 't': 5, '9': 6,
                              '8': 7, '7': 8, '6': 9, '5': 10,
                              '4': 11, '3': 12, '2': 13}
        self.actions = {'fold': 0, 'small blind': 1, 'big blind': 2,
                        'check': 3, 'call': 4, 'raise': 5}


    def fold(self):
        '''
        The player folds.
        '''
        self.last_actions.append(0)
        self.active = 0
        print(f'Player {self.name}: My cards are sh..! I fold!')


    def check(self):
        '''
        The player checks.
        '''
        self.last_actions.append(3)
        print(f'Player {self.name}: I am curious what is going to happen.\
            I am checking!')


    def call(self, highest_bid):
        '''
        The player calls.
        '''
        self.last_actions.append(4)
        self.own_bid = highest_bid
        print(f'Player {self.name}: I think I can win this game. I am calling!')


    def raise_bet(self, highest_bid, limit):
        '''
        The player raises the highest_bid.
        '''
        self.last_actions.append(5)
        self.own_bid = highest_bid + limit


    def do(self, highest_bid, limit, blind=None):
        '''
        The function do determines the action of the player when it is his turn
        to bet.
        '''
        if blind == 'small': # was, wenn er nicht mehr genug Geld für den small blind hat?
            self.own_bid = self.blind
            self.last_actions.append(1)
            print(f'Player {self.name}: I have bet the small blind of ${self.blind}!')
        elif blind == 'big': # was, wenn er nicht mehr genug Geld für den big blind hat?
            self.own_bid = self.blind * 2
            self.last_actions.append(2)
            print(f'Player {self.name}: I have bet the big blind of ${self.blind*2}!')
        else:
            if self.own_bid < highest_bid:
                action = random.choice(['fold', 'call', 'raise'])
            else:
                action = random.choice(['check', 'raise'])
            #print(f'Player {self.name} chose action {action}.')
            if action == 'fold':
                self.fold()
            elif action == 'check':
                self.check()
            elif action == 'call':
                self.call(highest_bid)
            else:
                self.raise_bet(highest_bid, limit)
            print(f'''Player {self.name}: I have {self.last_actions[-1]}
            and am betting ${self.own_bid}!''')


    def evaluate_hand(self):
        '''
        The function evaluate_hand evaluates the highest ranked hand out of
        all the possible 5 card combinations of the player.
        '''
        possible_hands = list(itertools.combinations(self.hand, 5))
#        print([(possible_hands[i][j].face, possible_hands[i][j].suit)\
#               for i in range(len(possible_hands))\
#               for j in range(len(possible_hands[i])) ])
        hand_values_rank = []
        for i, hand in enumerate(possible_hands):
            #print([(hand[i].face, hand[i].suit) for i in range(len(hand))])
            evaluator = Evaluator(hand)
            rank, highest_card = evaluator.find_best_hand()
            hand_values_rank.append((i, rank, self.handrankorder[rank]\
                , [self.cardrankorder[highest_card[i]] for i in range(len(highest_card))]\
                , highest_card, self.name))

        ranked = sorted(hand_values_rank, key=lambda x: (x[2], x[3]), reverse=False)
        print([(card.face, card.suit) for card in possible_hands[ranked[0][0]]])
        print(f'The best hand of player {self.name} is a {ranked[0][1]}')

        #if ranked[0][3][0] in [ranked[i][3][0] for i in range(len(hand_values_rank))]:
        #    print('This outcomes exists more than once')


        self.best_hand = 0#possible_hands[np.argmax(hand_values)]
        #print(hand_values)

        return ranked[0]



class Game:
    '''
    This class takes as input:

    nr_of_players: the number of players that will participate in the game
    limit: the limit of the bets in each game
    blind: the size of the big blind
    stack: the size of the initial stack of the game

    and creates a pokergame where you can deal the cards, players can fold, call or raise
    and the game thus unfolds.
    '''

    def __init__(self, nr_of_players, blind, stack, limit=200):
        self.nr_of_players = nr_of_players
        self.limit = limit
        self.blind = blind
        self.stack = stack
        self.players = []
        self.player_names = list(range(nr_of_players))
        self.order = list(range(nr_of_players))
        self.position_small = 0
        self.highest_bid = 0
        self.round = 1
        self.game_count = 1
        self.active_players = nr_of_players
        self.flop = []
        self.pot = 0
        self.winner = 0
        #self.suit = '♥ ♦ ♣ ♠'.split()
        self.suit = ['h', 'd', 'c', 's']
        # ordered strings of faces
        self.faces = '2 3 4 5 6 7 8 9 t j q k a'
        self.deck = []
        self.d = {}
        # faces as lists
        self.face = self.faces.split()

        for p in range(self.nr_of_players):
            player = Player(self.stack, self.blind, p+1)
            self.players.append(player)

        self.output = pd.DataFrame()


    def create_deck(self):
        '''
        The function create_deck takes the faces and suits of the Game as input,
        combines and shuffles them.
        '''
        # die nächsten zwei Zeilen kann man auch auslagern
        self.deck = list(itertools.product(self.faces.split(), self.suit))
        self.d = {self.deck[i]: i for i in range(len(self.deck))}
        np.random.shuffle(self.deck)


    def determine_order(self):
        '''
        The function determine_order() determines the order of bidding in each game
        '''
        if self.game_count != 1:
            order = []

            # create list of order in which players play
            print(f'The position of the small blind is {self.position_small+1}')

            if (self.position_small) < (self.nr_of_players-1):
                order = [self.position_small, self.position_small+1]
            else:# (self.position_small) == (self.nr_of_players-1):
                order = [self.position_small, 0]

            #print(f'The order is {order}')
            add = list(range(self.nr_of_players))
            #print(order)
            #print(add)
            [add.remove(x) for x in order]
            #print(add)
            lower_help = []
            upper_help = []

            for p in add:
                if p < self.position_small:
                    lower_help.append(p)
                else:
                    upper_help.append(p)
            #print(upper_help, lower_help)
            order = order + upper_help + lower_help
            #print(order)

            self.order = order
        #print(f'The order is {self.order}')


    def check_activity_round(self):
        '''
        The function check_activity_round checks whether the betting in the
        current round is still active.
        '''
        highest = 0
        self.active_players = self.nr_of_players
        for position in self.order:
            if self.players[position].own_bid == self.highest_bid:
                highest += 1
            if self.players[position].active == 0:
                self.active_players -= 1
            #print(f'highest is {highest} and active players are {self.active_players}')

        if highest == self.active_players and self.active_players >= 1:

            # increase the round by one
            self.round += 1

            return True

        else:

            return False


    def check_end_of_game(self):
        '''
        The function check_end_of_game checks if the game has ended.
        '''
        active = 0
        ap = 0
        for player in self.players:
            if player.active == 1:
                active += 1
                ap = player
        if active == 1:
            print('The game has ended')
            self.winner = ap
            self.active_players = 1
            return 0
        if active == 0:
            print('Something went wrong')
            return 1
        else:
#            print('The game has not ended')
            return 2


    def eliminate_players(self):
        '''
        The function eliminate_players eliminates, after each game,
        the players that did not call or raise.
        '''
        eliminated_players = 0
        for player in self.players:
            if player.active == 0:
                eliminated_players += 1

        active_players = self.active_players - eliminated_players

        return active_players


    def determine_pot_size(self):
        '''
        The function determine_pot_size calculates at each point in time how
        much money is in the pot.
        '''
        pot = 0
        for p in self.players:
            pot += p.own_bid

        return pot


    def deal_cards(self):
        '''
        The function deal_cards takes the list of players (self.players) and
        deals a hand of two cards for each player
        '''
        for player in self.players:
            card1 = Card(self.deck.pop())
            #print(card1, card1.face, card1.suit)
            card2 = Card(self.deck.pop())
            #print(card2, card2.face, card2.suit)
            player.hand = [card1, card2]
            print(f"{player.name}'s hand is {[(card.face, card.suit) for card in player.hand]} ")
            #assert len(player.hand) == 2


    def iterative_play(self):
        '''
        The function iterative_play is called when the players enter in a phase
        of iterative play.
        '''

        check_activity_round = False

        # run until the round is exhaustively played
        while not check_activity_round:
            #print('entered while loop')
#            print(f'We have to go on playing with {self.active_players}')
            for position, player in enumerate(self.order):
                #print('entered for loop')
                if self.players[player].own_bid <= self.highest_bid\
                and self.players[player].active == 1:
                    print(f'The highest bid is {self.highest_bid}')
                    #self.write_data(player, position)
                    self.players[player].do(self.highest_bid, self.limit)
                    self.collect_data(player, position)
                    if self.players[player].own_bid > self.highest_bid:
                        self.highest_bid = self.players[player].own_bid
                    # need to include some way to stop the game here
                    if self.check_end_of_game() == 0:
                        return 0
                    #print(f'player {position+1} plays {self.players[position].own_bid}')
            check_activity_round = self.check_activity_round()
#            print(check_activity_round)
#            print(not check_activity_round)


    def collect_data(self, player, position):
        '''
        The function collect data collects the data that is to be used as input
        for the deep reinforcement learning.
        '''
        # Set up a dataframe that stores the following information:
        # create dicitonary for appending the data frame:
        d = {}
        # agent_id of agent that is making the observation
        d['player'] = player
        d['position'] = position
        d['round'] = self.round
        d['game'] = self.game_count
        d['action'] = self.players[player].last_actions[-1]
        d['bid'] = self.players[player].own_bid
        d['hand1'] = self.d[(self.players[player].hand[0].face, self.players[player].hand[0].suit)]
        d['hand2'] = self.d[(self.players[player].hand[1].face, self.players[player].hand[1].suit)]
        if self.round == 1:
            d['community1'] = 999
            d['community2'] = 999
            d['community3'] = 999
            d['community4'] = 999
            d['community5'] = 999
        elif self.round == 2:
            d['community1'] = self.d[self.flop[0]]
            d['community2'] = self.d[self.flop[1]]
            d['community3'] = self.d[self.flop[2]]
            d['community4'] = 999
            d['community5'] = 999
        elif self.round == 3:
            d['community1'] = self.d[self.flop[0]]
            d['community2'] = self.d[self.flop[1]]
            d['community3'] = self.d[self.flop[2]]
            d['community4'] = self.d[self.flop[3]]
            d['community5'] = 999
        else:
            d['community1'] = self.d[self.flop[0]]
            d['community2'] = self.d[self.flop[1]]
            d['community3'] = self.d[self.flop[2]]
            d['community4'] = self.d[self.flop[3]]
            d['community5'] = self.d[self.flop[4]]
        #[self.flop[i] for i in range(len(self.flop))]
        count = 0
        for p in self.order:
            if p != player:
                d[f'bid_{count}'] = self.players[p].own_bid
                d[f'active_{count}'] = self.players[p].active
                #d[f'action_{count}'] = self.players[p].last_actions
                count += 1

        #print(f'The row {d} will be added to the output frame')
        self.output = self.output.append(d, ignore_index=True)
        # agent_position
        # round
        # for each of the agents store their current bet
        # for each of the agents store whether they are active
        # for each of the agents store their past actions
        # store the own hand
        # store the community cards
        # store the complete list of playable cards
        # at the end of each game write the payoff into each row of the agent


    def add_reward(self):
        '''
        The function add_reward adds the reward of each player to the output
        data frame that is written to the database.
        '''
        self.output['reward'] = self.output['player']
        for i in range(self.output.shape[0]):
            self.output.at[i, 'reward'] =\
            self.players[int(self.output.at[i, 'reward'])].reward


    def write_data(self):
        '''
        The function write_data writes the collected data into the postgres
        database "poker"
        '''
        self.output.to_sql('results', con=ENGINE, if_exists='append')

        #print('Wrote the data into the database!')

        # reset the output_frame
        self.output = pd.DataFrame()


    def action_first_round(self):
        '''
        The function action calls all agents sequentially to decide on their action
        '''
        # action of the small blind player
        self.players[self.order[0]].do(self.highest_bid, self.limit, blind='small')

        # action of the big blind player
        self.players[self.order[1]].do(self.highest_bid, self.limit, blind='big')
        self.highest_bid = self.players[self.order[1]].own_bid
        #print(f'The highest bid after the big blind is {self.highest_bid}')

        # let the non-blind players take their turn
        for position, player in enumerate(self.order[2:]):
            # for jedem do muss die Information ausgelesen werden
            #self.write_data(player, position)
            self.players[player].do(self.highest_bid, self.limit)
            self.collect_data(player, position+2)
            #print(f'player {position+1} bet {self.players[position].own_bid}')
            if self.players[player].own_bid > self.highest_bid:
                self.highest_bid = self.players[player].own_bid

        self.active_players = self.eliminate_players()

        # check if the round is still active
        check_activity_round = self.check_activity_round()
        if check_activity_round:
            return print(f'''round {self.round} is over and {self.active_players}
            players are still in the game!''')

        # play iterative_play
        self.iterative_play()

        return print(f'''round {self.round} is over and {self.active_players} players
        are still in the game''')


    def deal_flop(self):
        '''
        The function deal_flop deals out the first three cards of the flop
        '''
        [self.flop.append(self.deck.pop()) for x in range(3)]
        [player.hand.append(Card(x)) for player in self.players for x in self.flop]
        print(f'The flop is: {self.flop}')
        for player in self.players:
            if player.active == 1:
                print('Player {} has the hand {}'\
                .format(player.name, [(card.face, card.suit) for card in player.hand]))


    def action_post_flop(self):
        '''
        The function action_post_flop calls the players to execute their
        action after observing the additional community card(s)
        '''
        self.iterative_play()

        return print(f'Round {self.round} is over and {self.active_players} \
        players are still in the game')


    def deal_turn(self):
        '''
        The function deal_flop_2 deals out the fourth card of the flop
        '''
        self.flop.append(self.deck.pop())
        [player.hand.append(Card(self.flop[-1])) for player in self.players]
        print(f'The cards after the turn are: {self.flop}')


    def deal_river(self):
        '''
        The function deal_flop_3 deals out the fifth card of the flop
        '''
        self.flop.append(self.deck.pop())
        [player.hand.append(Card(self.flop[-1])) for player in self.players]
        print(f'The cards after the river are: {self.flop}')


    def pass_to_next_game(self):
        '''
        The function pass_to_next_game finishes of one game of poker and
        starts the new one.
        '''

        # determine who won the game
        if self.active_players > 1:
            best_hands = []
            for player in self.players:
                if player.active == 1:
                    best_hands.append(player.evaluate_hand())
            winning_hand = sorted(best_hands, key=lambda x: (x[2], x[3]),\
                reverse=False)[0]
            print(f'Player {winning_hand[5]} wins with a {winning_hand[1]}. \
            His hand is {winning_hand[4]}')
            self.winner = winning_hand[5]
        else:
            for player in self.players:
                if player.active == 1:
                    self.winner = player.name
                    print(f'Player {self.winner} wins because everyone else dropped\
                    out.')

        # distribute the pot
        self.pot = self.determine_pot_size()
        #print(f'The winner is {self.winner}')
        self.players[self.winner-1].stack += self.pot

        for player in self.players:
            player.stack -= player.own_bid
            print(f"Player {player.name}'s stack after game {self.game_count} \
            is {player.stack}.")
            player.reward = player.stack - player.stack_old
            # update old stack for the next round
            player.stack_old = player.stack

        # set pot to zero again
        self.pot = 0

        # increase game count
        self.game_count += 1

        # reset rounds
        self.round = 1

        # reset all players to be active and delete their last actions
        for player in self.players:
            player.active = 1
            player.last_actions = []

        # increase the small blind position
        # this is currently incorrect because we have to go in a circle
        if self.position_small == self.nr_of_players-1:
            self.position_small = 0
        else:
            self.position_small += 1


        # reset highest_bid
        self.highest_bid = 0

        # reset all players to be active
        self.active_players = self.nr_of_players

        # reset the flop
        self.flop = []

        # create a new deck and shuffle it
        # currently done through the create_deck() function at the beginning of the play_one_complete_game() function.


    def play_one_complete_game(self):
        '''
        This function simulates a complete game of poker without all the steps
        in between.
        '''
        self.determine_order()
        self.create_deck()
        self.deal_cards()
        self.action_first_round()
        if self.active_players > 1:
            self.deal_flop()
            self.action_post_flop()
            if self.active_players > 1:
                self.deal_turn()
                self.action_post_flop()
                if self.active_players > 1:
                    self.deal_river()
                    self.action_post_flop()
        self.pass_to_next_game()
        self.add_reward()
        self.write_data()

    def __repr__(self):
        return '''The game is in round {} and in game {}. The highest bid is {}.\
        The position of the small blind is {}. \
        The total money in the pot will yet have to be created.\
        \n The players who are still in the game will have to be created.
        '''.format(self.round, self.game_count, self.highest_bid, self.position_small+1)
