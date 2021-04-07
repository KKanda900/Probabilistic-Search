import numpy, sys, random, math
from Terrain import Terrians
from Target import Target


class Basic_Agent_2:
    
    # information to store about the generated map
    dim = 0 # default size 0 to start with 
    map_board = numpy.zeros((dim, dim), dtype=object) # in the start create the map_board with the default dimension of 0
    
    # target info will hold all the information related to the target
    target_info = None # set to None as default because the map wasn't generated yet

    # information to store for the basic agent to use
    belief_state = numpy.zeros((dim, dim), dtype=object) # tracks the belief of target being in cell_i given the observations
    confidence_state = numpy.zeros((dim, dim), dtype=object) # tracks the confidence of the target actually being in cell_i given the belief state and current observations
    previous_cells = [] # stores all the information about the previous cells here in the form (x, y)
    distance_traveled = 0 # stores the distanced traveled from the current cell you start with 
    num_searches = 0 # stores the number of searches made by the agent

    # initialize information related to the the agent and the map itself
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
                # create every i,j instance with the Terrian or belief at t=0 respectively
                self.map_board[i][j] = Terrians(i, j) # generated map default values
                self.belief_state[i][j] = 1/(self.dim*self.dim) # generated belief state default values

        # generate a random spot for the target to be located
        indicies = list(range(0, self.dim)) # create a list of numbers to choose from
        target_location = (random.choice(indicies), random.choice(indicies)) # get a random target location
        terrian_type = self.map_board[target_location[0]][target_location[1]].terrian_type # the terrain of the target will be whatever map_board[x][y] is
        self.target_info = Target(target_location[0], target_location[1], terrian_type) # store the information in the target_info variable
    
    # update the belief state based on bayesian updating (for basic agent 2)
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

        '''
        Step 2: Update the remaining probabilities and Step 3: Update the confidence states for the current cell
        '''
        # go to each cell and make sure the total probability equals 1
        for i in range(self.dim):
            for j in range(self.dim):
                # if the false negative rates are same we have same terrain cell as the one search failed on in last step
                if fnr == self.map_board[i][j].false_neg:
                    self.belief_state[i][j] = (fnr * self.belief_state[i][j]) / prob_failure
                    
                    # obtain the probability of the Target being in Cell_i given the Observations at time t
                    curr_belief = self.belief_state[i][j]
                    # obtain the probability of Success in Cell_i given the Obseravations at time t
                    current_success = 1-self.map_board[i][j].false_neg
                    # update the current confidence state
                    self.confidence_state[i][j] = curr_belief*current_success
                else:
                    # if the other case isn't true that means we are at every other cell
                    self.belief_state[i][j] = self.belief_state[i][j] / prob_failure

                    # obtain the probability of the Target being in Cell_i given the Observations at time t
                    curr_belief = self.belief_state[i][j]
                    # obtain the probability of Success in Cell_i given the Obseravations at time t
                    current_success = 1-self.map_board[i][j].false_neg
                    # update the current confidence state
                    self.confidence_state[i][j] = curr_belief*current_success

    # start basic agent 2 here
    def start_agent(self, x, y):

        # termination variable to track if the target is found or not
        target_found = False

        # variables to keep track of for final result
        tracking_distance = 0
        searches = 0

        # iterate through every cell over and over until the target is found
        while target_found == False:

            # if the target is found immediately exit out of the loop and return the results
            if (x, y) == self.target_info.target_location:
                rand = random.randint(0, 1)
                if rand <= self.map_board[x][y].false_neg:
                    # lets add the cell to the previous failure list
                    self.previous_cells.append((x, y))

                    # update the belief state based on equation 5 from the pdf write-up
                    self.bayesian_update(x, y)

                    # now iterate through all the cells that when i != x and j != y to find the new highest probabilities
                    max_coord = (x, y, self.belief_state[x][y])
                    for i in range(len(self.belief_state)):
                        for j in range(len(self.belief_state)):
                            if i != x and j != y:
                                # if this condition returns true than we found the new highest probability out of all the coordinates
                                if max_coord[2] < self.confidence_state[i][j]:
                                    max_coord[0] = i
                                    max_coord[1] = j
                                    max_coord[2] = self.confidence_state[i][j]

                                # if the condition returns true there are certain factors we have to take care of
                                if max_coord[2] == self.confidence_state[i][j]:
                                    # returns the coordinates with the lowest manhattan distance
                                    smallest_coord = self.find_manhattan_distance((x, y), (max_coord[0], max_coord[1]), (i, j))
                                    # checks if the distance and probability are the same if the condition is met below
                                    if smallest_coord == (max_coord[0], max_coord[1]) and smallest_coord == (i, j):
                                        arr_choices = [0, 1] # store 0 and 1 to pick from randomly to use to distinguish if probability and distance are equal
                                        # if the random choice is 0 then choose max_coord 
                                        if random.choice(arr_choices) == 0:
                                            x = max_coord[0]
                                            y = max_coord[1]
                                        else:
                                            # otherwise choose i, j if the above condition isn't satisified
                                            x = i
                                            y = j
                                    else:
                                        # if the condition above isn't true that means we can use smallest_coord effectively
                                        x = smallest_coord[0]
                                        y = smallest_coord[1]

                else:
                    # once we reach here we should exit out the loop and return the results
                    target_found = True
            else:
                # lets add the cell to the previous failure list
                self.previous_cells.append((x, y))

                # update the belief state based on equation 5 from the pdf write-up
                self.bayesian_update(x, y)

                # now iterate through all the cells that when i != x and j != y to find the new highest probabilities
                max_coord = (x, y, self.confidence_state[x][y])
                for i in range(len(self.confidence_state)):
                     for j in range(len(self.confidence_state)):
                        if i != x and j != y:
                            # if this condition returns true than we found the new highest probability out of all the coordinates
                            if max_coord[2] < self.confidence_state[i][j]:
                                max_coord[0] = i
                                max_coord[1] = j
                                max_coord[2] = self.confidence_state[i][j]

                            # if the condition returns true there are certain factors we have to take care of
                            if max_coord[2] == self.confidence_state[i][j]:
                                # returns the coordinates with the lowest manhattan distance
                                smallest_coord = self.find_manhattan_distance((x, y), (max_coord[0], max_coord[1]), (i, j))
                                # checks if the distance and probability are the same if the condition is met below
                                if smallest_coord == (max_coord[0], max_coord[1]) and smallest_coord == (i, j):
                                    # store 0 and 1 to pick from randomly to use to distinguish if probability and distance are equal
                                    arr_choices = [0, 1]
                                    # if the random choice is 0 then choose max_coord
                                    if random.choice(arr_choices) == 0:
                                        x = max_coord[0]
                                        y = max_coord[1]
                                    else:
                                        # otherwise choose i, j if the above condition isn't satisified
                                        x = i
                                        y = j
                                else:
                                    # if the condition above isn't true that means we can use smallest_coord effectively
                                    x = smallest_coord[0]
                                    y = smallest_coord[1]
