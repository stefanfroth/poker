# Poker

[![Build Status](https://travis-ci.com/stefanfroth/poker.svg?token=s3wrUz4phpxxGfMxPpPT&branch=epsilon)](https://travis-ci.com/stefanfroth/poker)


## Goal

The aim of the project Poker is to train a poker bot to beat human players!
The project is inspired by the advances of Facebook and their pokerbot Pluribus
in this field. See: https://www.forbes.com/sites/samshead/2019/07/11/facebooks-new-poker-ai-can-beat-the-worlds-top-players/#63333ed6fefa


## Rules

The poker bot plays 6 player Fixed-Limit Texas Hold'em version of the game with
blinds of 50/100 and limits of 100/200.


## Structure

Given the rules the first step consisted in defining the structure of the program.


### The Game

The game can be best explained as a chronological flow of the game of poker. The backbone of the structure of the game are the classes Game, Player, Card (in the module poker_game) and Evaluator (in the module hand_evaluator).

1. At the beginning of the game all players are automatically created when the game is instantiated.
2. Following the generation of the players, the deck has to be created and shuffled.
3. In the next step two cards are distributed among each player. Then the betting rounds follow:
5. First betting round
  6. The flop is dealt
  7. Second betting round
    8. The turn is dealt
    9. Third betting round (the limit is increased)
      10. The river is dealt
      11. Fourth betting round
12. The game is over, the winner is determined, the pot is distributed, the dealer button/small blind position
is passed on and everything starts over at step 2.

The code defining the game can be found mainly in the modules *poker_game* and *hand_evaluator*.


### The Agent

The second step is to create an agent that is taking the decisions given his observation of the state of the game. For this an artificial neural network is created. It takes its own cards and the community cards as input creating an embedding. On top of that the last actions of the other players, the own bet and the position in the game are taken as further input for the agent. These inputs are concatenated with the embedding layers of the cards and fed into a fully connected Artificial Neural Network with two hidden layers. During each betting rounds the agent decides which action to take.

The code defining the agent is contained in the module *agent*.


### Data Generation

In the third step data is generated. In the beginning the Artificial Neural Network is randomly initialized. It then plays against itself over a chosen amount of rounds (100.000 rounds in this case) to generate data on which the model can be trained. All input variables, the outcome of the game (the reward and the reward following the current action), the action and some additional observations for analysis purposes are saved in a data frame and written into an PostGres database.

The code for the data generation, the self play of an agent, is written in the module *play*.


### Training the model

In the fourth step the model is trained and the weights are adjusted. Having generated the data by self play of the agent, the model is trained by minimizing the loss of the agents.

The code for the training of the model is written in the modules *train* and *agent*.


## Data Pipeline

The original data pipeline consisted of running the two scripts *play.py* and *train.py* sequentially over and over again on my local machine. The weights contained in this repository and the .png file displaying rewards of agents over 50 games of poker were created in that matter.


## Results

The preliminary result is that the poker bot in the first step learned to beat the random playing agent by not folding. As a random playing agent will at some point drop out of the game regardless of hand, staying in the game yields better results in expectation.
This however lead to the fact that the poker bot disregarded the possibility of folding altogether because it did not explore situations in which folding might lead to the better outcome. I therefore introduced some randomness into the decisions of the agent, a so called epsilon-greedy policy, in order to force the agent to make observations about folding as well.
The last observed version of the poker bot leaned slightly towards always calling.

The total amount of data generated for training purposes was ~8 GB. The result of version 7 playing against version 8 can be seen in the image *v7vv8.png* where we see that v8 is  during the first 50 games on average reaching higher rewards. This intuition created by eyeballing the graph is corroborated by testing for the hypothesis that agent v8 is making rewards different from 0 for which we find empirical evidence.

The code for creating the plot and the statistical test is written in the module *self_play_between_versions*.


## Tech stack

- Python (TensorFlow, pandas, numpy, scipy, itertools, random, sqlalchemy)
- PostGreSQL
- AWS (EC2, RDS)
- Travis CI
- Pylint
