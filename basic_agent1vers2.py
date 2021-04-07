import numpy
import sys
import random
import math
from Terrain import Terrians
from Target import Target


class Basic_Agent_1:
    dim = 0
    map_board = numpy.zeros((dim, dim), dtype=object)
    target_info = None
    belief_state = numpy.zeros((dim, dim), dtype=object)
    # stores all the information about the previous cells here in the form (x, y)
    previous_cells = []
    target_found = False

    def __init__(self, dim):
        self.dim = dim
        self.map_board = numpy.zeros((self.dim, self.dim), dtype=object)
        self.belief_state = numpy.zeros((self.dim, self.dim), dtype=object)
        self.generate_board()

    # generate the board for the agent that is going to traverse
    def generate_board(self):
        # generate all the locations
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                # if the target location == i and j then occupy i, j with the target
                self.map_board[i][j] = Terrians(i, j)
                self.belief_state[i][j] = 1/(self.dim*self.dim)

        # generate a random spot for the target to be located
        indicies = list(range(0, self.dim))
        target_location = (random.choice(indicies), random.choice(indicies))
        terrian_type = self.map_board[target_location[0]
            ][target_location[1]].terrian_type
        self.target_info = Target(
            target_location[0], target_location[1], terrian_type)

    # update the belief state based on bayesian updating
    def bayesian_update(self, x, y):

        '''
        For Bayes updating we want to update the belief state with P(Target in Cell i | Observations at t and Failure in Cell j)
        '''

        '''
        Step 1: Update the current cell then update the rest of the cells according to the cell
        '''
        # update the probability of failure for the previous cell
        curr_prev = self.previous_cells.pop(0)
        # obtain the probability of the previous cell failing and the target is in the current cell
        fnr = self.map_board[x][y].false_neg  # FNR of current cell
        # obtain the probability of the target being in the location based on the observation
        curr_cell_belief = self.belief_state[x][y]
        prob_failure = (fnr * curr_cell_belief) + (1 - curr_cell_belief)
        # update the probability of the current cell
        # self.belief_state[x][y] = curr_cell_belief/prob_failure

        '''
        Step 2: Update the remaining probabilities so everything is equal to 1
        '''
        # go to each cell and make sure the total probability equals 1
        sum_equal_1 = False
        for i in range(self.dim):
            for j in range(self.dim):
                # if the false negative rates are same we have same terrain cell as the one search failed on in last step
                if fnr == self.map_board[i][j].false_neg:
                    self.belief_state[i][j] = (fnr * self.belief_state[i][j]) / prob_failure
                else:
                    # if the other case isn't true that means we are at every other cell
                    self.belief_state[i][j] = self.belief_state[i][j] / prob_failure

    def calculate_neighbors(self, x, y):
        neighbors = []
        if x+1 < 50 and y < 50:
            neighbors.append((x+1, y))
        if x < 50 and y+1 < 50:
            neighbors.append((x, y+1))
        if x-1 < 50 and y < 50:
            neighbors.append((x-1, y))
        if x < 50 and y-1 < 50:
            neighbors.append((x, y-1))
        return neighbors

    def clear_ties(self, board, x, y):
        min_distance = 0.0
        ties = []
        j = 0
        for i in board:
            x_cord = i[0]
            y_cord = i[1]
            distance = abs(x_cord-x) + abs(y_cord-y)
            if j == 0:
                min_distance = distance
                ties.append((x_cord, y_cord))
                j = 1
            elif distance < min_distance:
                min_distance = distance
                ties.clear()
                ties.append((x_cord, y_cord))
            elif min_distance == distance:
                ties.append((x_cord, y_cord))
        return ties[0]

    def calculate_location(self, board):
        location = ()
        ties = []
        max_val = 0.0
        for i in range(0, 50):
            for j in range(0, 50):
                if board[i][j] > max_val:
                    max_val = board[i][j]
                    location = (i, j)
                    ties.clear()
                    ties.append(location)
                elif board[i][j] == max_val:
                    ties.append((i, j))
        return ties

    def basic_agent(self,x,y):
        x_cord=x
        y_cord=y
        moves_counted=0
        distance=0
        while self.target_found==False:
            if (x_cord,y_cord)==self.target_info.location:
                fnr=self.map_board[x_cord][y_cord].false_neg
                rand=random.random()
                if rand>fnr:
                    moves_counted+=1
                    self.target_found=True
            else:
                self.previous_cells.append((x_cord,y_cord))
                moves_counted+=1
                self.bayesian_update(x_cord,y_cord)
                locations=self.calculate_location(self.belief_state)
                print(locations)
                if len(locations)>1:
                    location=self.clear_ties(locations,x_cord,y_cord)
                    locations.clear()
                    locations.append(location)
                coords=locations[0]
                distance+=abs(x_cord-coords[0])+abs(y_cord-coords[1])
                x_cord=coords[0]
                y_cord=coords[1]

        return moves_counted+distance        

x=random.randint(0,50)
y=random.randint(0,50)
info=Basic_Agent_1(50)
print(info.basic_agent(x,y))           
# self.start()
