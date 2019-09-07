# Poker

[![Build Status](https://travis-ci.com/stefanfroth/poker.svg?token=s3wrUz4phpxxGfMxPpPT&branch=epsilon)](https://travis-ci.com/stefanfroth/poker)

This readme is an introduction into my Poker project.

Aim: Train a poker bot to beat (human) players!
Inspiration: Pluribus!
Structure: Throughout the project I was considering a 6 player Fixed-Limit Texas Hold'em version of the game.

First step: The first step consisted in building  the structure of the poker game:

1) Consists of classes Game, Player, Card and Evaluator
2) At the beginning of the game all the (6) players have to be created.
3) Next: Deck is created
4) Next: cards are distributed
5) First betting round
  6) Deal Flop
  7) Second betting round
    8) Deal turn
    9) Third betting round (with big limit)
      10) Deal river
      11) Fourth betting round
12) End the game


) Fix the reward system
