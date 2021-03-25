import numpy
import sys
import random
import math

class Terrians:
    types = ['flat', 'hilly', 'forested', 'caves']
    terrian_type = None
    contains_target = False
    probability = 0.0
    cell_location = ()

    def __init__(self, i, j):
        self.cell_location = (i, j)
        self.terrian_type = random.choice(self.types)
        self.probability = 0.25

class Map_Search:
    dim = 0
    map_board = numpy.zeros((dim, dim), dtype=object)

    def __init__(self, dim):
        self.dim = dim
        self.map_board = numpy.zeros((self.dim, self.dim), dtype=object)
        self.generate_board()
    
    def generate_board(self):
        # generate a random spot for the target to be located
        indicies = list(range(0, self.dim))
        target_location = (random.choice(indicies), random.choice(indicies))

        # generate all the locations
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                # if the target location == i and j then occupy i, j with the target
                if target_location[0] == i and target_location[1] == j:
                    self.map_board[i][j] = Terrians(i, j)
                    self.map_board[i][j].contains_target = True
                else:
                    # otherwise occupy the rest of the map with the random terrians
                    self.map_board[i][j] = Terrians(i, j)
    
    def start(self):
        print(self.map_board)

board = Map_Search(20)
board.start()