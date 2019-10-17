from flask import Flask, render_template, request
import csv
import play
from poker_game import Game
from agent import Agent

# Configure App
app = Flask(__name__)


def play_game():
    # Rules of the game
    PLAYERS = 6
    BLIND = 50
    STACK = 10000
    LIMIT = 100
    GAMES = 10
    DB_TABLE = 'online'

    # Chose the agent that is supposed to play.
    AGENT = Agent()
    AGENT.load('v7_2019-08-19-23:30_20_epochs')

    AGENT.build_model()

    # instantiate the game and play the games
    G = Game(PLAYERS, BLIND, STACK, AGENT, DB_TABLE, LIMIT)

    for i in range(GAMES):
        G.play_one_complete_game()


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/start', methods=['Post'])
def start():
    name = request.form.get('name')
    if not name:
        return 'Please enter a name!'
    player = name
    file = open('players.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(request.form.get('name'))
    play.play_game()
    file.close

    return render_template('game.html', player=player)


if __name__ == '__main__':
    app.run(debug=True)
