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

        print(self.belief_state.sum())

        # generate a random spot for the target to be located
        indicies = list(range(0, self.dim))
        target_location = (random.choice(indicies), random.choice(indicies))
        terrain_type = self.map_board[target_location[0]][target_location[1]].terrain_type
        self.target_info = Target(target_location[0], target_location[1], terrain_type)


    # update the belief state based on bayesian updating
    def bayesian_update(self, x, y):

        '''
        For Bayes updating we want to update the belief state with P(Target in Cell i | Observations at t and Failure in Cell j)
        '''

        '''
        Step 1: Update the current cell then update the rest of the cells according to the cell
        '''
        # obtain the probability of the previous cell failing and the target is in the current cell
        fnr = self.map_board[x][y].false_neg  # FNR of current cell
        # obtain the probability of the target being in the location based on the observation
        curr_cell_belief = self.belief_state[x][y]
        prob_failure = (fnr * curr_cell_belief) + (1 - curr_cell_belief)
        # update the probability of the current cell
        self.belief_state[x][y] = (fnr*curr_cell_belief)/prob_failure
        print("point:", x, y)
        print("belief[x][y]:", self.belief_state[x][y])

        '''
        Step 2: Update the remaining probabilities so everything is equal to 1
        '''
        # go to each cell and make sure the total probability equals 1

        for i in range(self.dim):
            for j in range(self.dim):
                if (i, j) != (x, y):
                    self.belief_state[i][j] /= prob_failure

        print("sum", self.belief_state.sum())

    
    #breaks any ties that exist between cells with equal probabilities or the case where cells have equal probabilities and same shortest distance and returns one cell
    def clear_ties(self, board, x, y):
        min_distance = 0.0 #initializes minimum distance
        ties = [] #represents the list of cells to return
        j = 0
        #iterates through board
        for i in board:
            x_cord = i[0]
            y_cord = i[1]
            distance = abs(x_cord-x) + abs(y_cord-y) #manhatten distance between current element of list and current cell the agent is in
            if j == 0:
                min_distance = distance #sets minimum distance to the first cell in the list
                ties.append((x_cord, y_cord)) #adds current element in list to list
                j = 1
            elif distance < min_distance:
                min_distance = distance #sets a new minimum distance
                ties.clear() #in the case that there is a new minimum distance value, removes cells with the previous minimum value distance
                ties.append((x_cord, y_cord)) #adds location of minimum distance value in the board to the list
            #in the case that there is a tie (current cells distance is equal to the minimum distance value in the board, adds current cell to list
            elif min_distance == distance:
                ties.append((x_cord, y_cord))
        if len(ties)==1:
            return ties[0] #in the case that there is one cell whose distance is equivalent to the shortest distance to the current cell
        else:
            return random.choice(ties) #in the case that there are multiple cells whose distance are equivalent to the shortest distance to the current cell
        
    #calculates possible next cells to go to after a bayesian update has been done and returns that list of cells
    def calculate_location(self, board):
        location = () #possible location to be added
        ties = [] #list of cells to return 
        max_val = 0.0 #intial probability value to compare
        #iterates through whole board
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if board[i][j] > max_val:
                    max_val = board[i][j] #largest probability value in the board
                    location = (i, j) #location of largest probability value in the board
                    ties.clear() #in the case that there is a new largest probability value, removes cells with the previous largest value probability
                    ties.append(location) #adds location of largest probability value in the board to the list
                #in the case that there is a tie (current cells probability is equal to the largest probability value in the board, adds current cell to list   
                elif board[i][j] == max_val:
                    ties.append((i, j)) 
        return ties #returns list of possible next cells to go to

    def basic_agent(self,x,y):
        x_cord=x #current cell x coordinate
        y_cord=y #current cell y coordinate
        moves_counted=0 #represents amount of moves done so far
        distance=0  #represents distance covered so far
        while self.target_found==False: #while the target has not been found
            if (x_cord,y_cord)==self.target_info.location: #if current cell coordinates are equivalent to the target location
                fnr=self.map_board[x_cord][y_cord].false_neg #false negative rate of current cell
                rand=random.random() #generates random number between 0 and 1
                if rand>fnr:
                    moves_counted+=1 #increments moves counted
                    self.target_found=True #target has been found
                #false negative situation
                else:
                    self.previous_cells.append((x_cord,y_cord))
                    moves_counted+=1 #increments moves counted
                    self.bayesian_update(x_cord,y_cord) #perform bayesian update on belief state
                    locations=self.calculate_location(self.belief_state) #calculates list of possible next cells to go to
                    #print(locations)
                    if len(locations)>1: #ties exist between cells in the list
                        location=self.clear_ties(locations,x_cord,y_cord) #clears tie to return one possible next cell
                        locations.clear() #clears list so that u can add the singular location to list
                        locations.append(location) #add the next possible cell to go to list
                    coords=locations[0] #gets coordinates for next possible cell to go to
                    #adds its manhatten distance between current cell and next possible cell to go to to the cumulative distance
                    distance+=abs(x_cord-coords[0])+abs(y_cord-coords[1]) 
                    x_cord=coords[0] #new x coordinate to go to
                    y_cord=coords[1] #new y coordinate to go to
            #current cell corrdinates not equivalent to target location
            else:
                self.previous_cells.append((x_cord, y_cord))
                moves_counted += 1
                self.bayesian_update(x_cord, y_cord)
                locations = self.calculate_location(self.belief_state)
                # print(locations)
                if len(locations) > 1:
                    location = self.clear_ties(locations, x_cord, y_cord)
                    locations.clear()
                    locations.append(location)
                coords = locations[0]
                distance += abs(x_cord - coords[0]) + abs(y_cord - coords[1])
                x_cord = coords[0]
                y_cord = coords[1]
        #returns final result of moves and distance
        return moves_counted+distance

def start_agent():
    x = random.randint(0, 49) #random x coordinate to start agent1
    y = random.randint(0, 49) #random y coordinate to start agent1
    info = Basic_Agent_1(50) #creating object named info for basic agent class with a dimension of 50
    return info.basic_agent(x, y) #starts agent1

