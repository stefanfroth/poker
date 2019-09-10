'''
The module hand_evaluator contains solely the class Evaluator, which evaluates
the strength of hands in Texas Hold'em poker.

The class Evaluator is a modified version of the "Poker hand analyser" from
rosettacode.org. It is published under the GNU Free Documentation License 1.2
and therefore is published by me under the same GNU Free Documentation License 1.2
as agreed upon by using the published code.
'''



class Evaluator:
    '''
    The class Evaluator evaluates hands in a game of poker.
    It takes the hand of the player, flop, turn and river as inputs
    and returns the best possible hand for the player
    '''

    def __init__(self, hand, presentation):
        self.hand = hand
        self.presentation = presentation
        self.faces = '2 3 4 5 6 7 8 9 t j q k a'
        self.lowaces = 'a 2 3 4 5 6 7 8 9 t j q k'
        self.face = self.faces.split()
        self.lowace = self.lowaces.split()
        if self.presentation == 1:
            self.suit = '♥ ♦ ♣ ♠'.split()
        else:
            self.suit = ['h', 'd', 'c', 's']

    '''
    :param hand: The hand of cards that is to be evaluated.
    :param ation: Binary variable indicating whether the fancy suits should
    be displayed (1) or the letters (0).
    '''


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
