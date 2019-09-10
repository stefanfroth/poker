# Poker

[![Build Status](https://travis-ci.com/stefanfroth/poker.svg?token=s3wrUz4phpxxGfMxPpPT&branch=epsilon)](https://travis-ci.com/stefanfroth/poker)

The aim of the project Poker is to train a poker bot to beat human players!
The project is inspired by the advances of Facebook and their pokerbot Pluribus
in this field. See: https://www.forbes.com/sites/samshead/2019/07/11/facebooks-new-poker-ai-can-beat-the-worlds-top-players/#63333ed6fefa

The project is structured as follows:
The poker bot plays 6 player Fixed-Limit Texas Hold'em version of the game with
blinds of 50/100 and limits of 100/200.

Given the rules the first step consisted in defining the structure of the program.
The whole program is written in Python. All of this code can be found in the module
poker_game.

1) The backbone of the program are the classes Game, Player, Card (in the module poker_game)
and Evaluator (in the module hand_evaluator).
2) At the beginning of the game all players have to be created, followed by the deck of cards.
3) In the next step two cards are distributed among each player. Then the betting rounds follow:
5) First betting round
  6) The flop is dealt
  7) Second betting round
    8) The turn is dealt
    9) Third betting round (with big limit)
      10) The river is dealt
      11) Fourth betting round
12) The game is over, the winner is determined, the pot is distributed, the dealer button/small blind position
is passed on and everything starts over.

The second step is to create an agent that is taking the decisions given his
observation of the state of the game. The code for this step can be found in the
module agent. For this an artificial neural network is created.
It takes its own cards and the community cards as input creating an embedding.
On top of that more ovserved variables, the last actions of the other players, the
own bet and the position in the game are taken as further input for the agent.

In the third step data is generated. The code for the data generation can be found
in the module play. Herefore, in the beginning the model is randomly initialized.

In the fourth step the model is trained and the weights are adjusted.
