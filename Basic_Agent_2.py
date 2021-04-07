import numpy, sys, random, math
from Terrain import Terrians
from Target import Target


class Basic_Agent_2:
    dim = 0
    map_board = numpy.zeros((dim, dim), dtype=object)
    target_info = None
    belief_state = numpy.zeros((dim, dim), dtype=object)
    previous_cells = []  # stores all the information about the previous cells here in the form (x, y)

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
        terrian_type = self.map_board[target_location[0]][target_location[1]].terrian_type
        self.target_info = Target(target_location[0], target_location[1], terrian_type)

    # update the belief state based on bayesian updating (for basic agent 1)
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
        Step 2: Update the remaining probabilities
        '''
        # go to each cell and make sure the total probability equals 1
        for i in range(self.dim):
            for j in range(self.dim):
                # if the false negative rates are same we have same terrain cell as the one search failed on in last step
                if fnr == self.map_board[i][j].false_neg:
                    self.belief_state[i][j] = (fnr * self.belief_state[i][j]) / prob_failure
                else:
                    # if the other case isn't true that means we are at every other cell
                    self.belief_state[i][j] = self.belief_state[i][j] / prob_failure
    
    def bayesian_update_v1(self, x, y):

        '''
        For Bayes updating we want to update the belief state with P(target found in cell_i | Observations_t)
        '''

        '''
        Step 1: 
        '''
    
    # start basic agent 2 here
    def basic_agent_2(self):
        print("Work on this tomorrow")

    
