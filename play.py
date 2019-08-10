from poker import Card, Evaluator, Player, Game

PLAYERS = 6
BLIND = 100
STACK = 10000
GAMES = 10

g = Game(PLAYERS, BLIND, STACK)

for i in range(GAMES):
    g.play_one_complete_game()

print(f'{GAMES} games were played!')
