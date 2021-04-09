import numpy
import random
from Terrain import Terrians
from Target import Target


class AgentClass:
    # information to store about the generated map
    dim = 0  # default size 0 to start with
    map_board = numpy.zeros((dim, dim),
                            dtype=object)  # in the start create the map_board with the default dimension of 0

    # target info will hold all the information related to the target
    target_info = None  # set to None as default because the map wasn't generated yet

    # information to store for the basic agent to use
    belief_state = numpy.zeros((dim, dim),
                               dtype=object)  # tracks the belief of target being in cell_i given the observations
    confidence_state = numpy.zeros((dim, dim),
                                   dtype=object)  # tracks the confidence of the target actually being in cell_i given the belief state and current observations
    previous_cells = []  # stores all the information about the previous cells here in the form (x, y)
    distance_traveled = 0  # stores the distanced traveled from the current cell you start with
    num_searches = 0  # stores the number of searches made by the agent
    target_found = False

    # initialize information related to the the agent and the map itself
    def __init__(self, dim):
        self.dim = dim
        self.map_board = numpy.zeros((self.dim, self.dim), dtype=object)
        self.belief_state = numpy.zeros((self.dim, self.dim), dtype=object)
        self.confidence_state = numpy.zeros((self.dim, self.dim), dtype=object)
        self.generate_board()

    # generate the board for the agent that is going to traverse
    def generate_board(self):
        # generate all the locations
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                # create every i,j instance with the Terrain or belief at t=0 respectively
                self.map_board[i][j] = Terrians(i, j)  # generated map default values
                self.belief_state[i][j] = 1 / (self.dim * self.dim)  # generated belief state default values
                # print(self.map_board[i][j].false_neg)
                # self.confidence_state[i][j] = self.belief_state[i][j] * (1 - self.map_board[i][i].false_neg)

        # generate a random spot for the target to be located
        indices = list(range(0, self.dim))  # create a list of numbers to choose from
        target_location = (random.choice(indices), random.choice(indices))  # get a random target location
        print(self.map_board[target_location[0]][target_location[1]].__dict__)
        terrain_type = self.map_board[target_location[0]][
            target_location[1]].terrain_type  # the terrain of the target will be whatever map_board[x][y] is
        self.target_info = Target(target_location[0], target_location[1],
                                  terrain_type)  # store the information in the target_info variable

    # update the belief state based on bayesian updating (for basic agent 2)
    def bayesian_update(self, x, y):

        """
        For Bayes updating we want to update the belief state with P(Target in Cell i | Observations at t and Failure in Cell j)
        """

        '''
        Step 1: Update the current cell then update the rest of the cells according to the cell
        '''
        # obtain the probability of the previous cell failing and the target is in the current cell
        fnr = self.map_board[x][y].false_neg  # FNR of current cell
        # obtain the probability of the target being in the location based on the observation
        curr_cell_belief = self.belief_state[x][y]
        prob_failure = (fnr * curr_cell_belief) + (1 - curr_cell_belief)
        # update the probability of the current cell
        self.belief_state[x][y] = (fnr * curr_cell_belief) / prob_failure
        self.confidence_state[x][y] = curr_cell_belief * (1 - fnr)
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
                    self.confidence_state[i][j] = self.belief_state[i][j] * (1 - self.map_board[i][j].false_neg)

        print("sum", self.belief_state.sum())
