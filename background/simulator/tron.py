"""
A simple game: you need to paint the screen 
but must not hit already painted areas.
"""
import time
import pygame
from pygame import Rect
import random
import numpy as np

from tilegamelib import TiledMap
from tilegamelib.game import Game
from tilegamelib.sprites import Sprite
from tilegamelib.config import config
from tilegamelib.vector import UP, DOWN, LEFT, RIGHT


FIELD = """################
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
################"""

config.RESOLUTION = (550, 550)
config.FRAME = Rect(10, 10, 550, 550)

FRUIT = 'a'
WALL_TILE = '#'


class Tron:

    def __init__(self):
        self.game = Game()
        self.tm = TiledMap(self.game)
        startx = random.randint(1,14)
        starty = random.randint(1,14)
        self.player = Sprite(self.game, 'b.pac_right', (startx, starty), speed=8)
        self.tm.set_map(FIELD)
        self.place_fruit()
        self.draw()
        self.events = None
        self.score = 0
        self.dead = False

    def place_fruit(self):
        for i in range(5):
            x = random.randint(1,14)
            y = random.randint(1,14)
            self.tm.set_tile((x,y), 'a')

    def draw(self):
        self.player.move()
        self.tm.draw()
        self.player.draw()

    def move(self, direction):
        if self.player.finished:
            nearpos = self.player.pos + direction
            near = self.tm.at(nearpos)
            if near == FRUIT:
                self.score += 50
            elif near == WALL_TILE:
                self.dead = True
                if hasattr(self.game, 'events'):
                    self.game.exit()
            self.player.add_move(direction)
            self.score += 1
            self.tm.set_tile(self.player.pos, '#')


    def run(self):
        self.game.event_loop(figure_moves=self.move, draw_func=self.draw)


if __name__ == '__main__':
    """Play the game"""
    tron = Tron()
    tron.run()
    print('final score:', tron.score)
