
from tron import Tron, UP, DOWN, LEFT, RIGHT
import numpy as np
import pygame
import time


class Environment:
    """
    Interface helping programs to play Tron.
    """
    actions = UP, DOWN, LEFT, RIGHT

    def __init__(self):
        self.finished = False
        self.tron = Tron()
        self.xsize = 16
        self.ysize = 16

    def get_window(self, state):
        padded = np.pad(state, (1, 1), 'constant')
        #print(padded)
        ix, iy = np.where(padded == 2.0)
        ix, iy = ix[0], iy[0]
        #print(ix, iy)
        five = padded[ix-2:ix+3, iy-2:iy+3]
        #print(five)
        assert five.shape == (5, 5)
        return five

    def get_state(self):
        """Returns a 2D float array."""
        tiles = {
            '#' : 1.0,
            '.' : 0.0,
            'a' : 0.1
            # player: 1.0
            }
        state = [[tiles[self.tron.tm.map[y][x]] for x in range(16)] for y in range(16)]
        pos = self.tron.player.pos
        state[pos.y][pos.x] = 2.0
        state = np.array(state, dtype=np.float32)
        return self.get_window(state)

    def get_score(self):
        """Returns number of painted squares or zero if dead"""
        if not self.finished:
            return self.tron.score
        else:
            return -10

    def perform_action(self, action):
        """action is a one-hot encoded vector of length 4"""
        action = action.argmax()
        action = self.actions[action]
        self.tron.move(action)
        while not self.tron.player.finished:
            self.tron.draw()
            pygame.display.update()
        self.finished = self.tron.dead

if __name__ == '__main__':
    env = Environment()
    for i in range(5):
        left = np.array([0, 0, 1, 0])
        env.perform_action(left)
    
    print(env.get_state())
    print(env.get_score())
