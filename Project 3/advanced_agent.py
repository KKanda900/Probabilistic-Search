import numpy
import random
from Terrain import Terrians
from Target import Target

'''
Advanced Agent Class

Description: Contains all the logic and the map generation logic/functions related to the advanced agent.
'''
class AgentClass:

    # information to store about the generated map
    dim = 0  # default size 0 to start with
    map_board = numpy.zeros((dim, dim),
                            dtype=object)  # in the start create the map_board with the default dimension of 0

    # target info will hold all the information related to the target
    target_info = None  # set to None as default because the map wasn't generated yet

    # information to store for the advanced agent to use
    belief_state = numpy.zeros((dim, dim),
                               dtype=object)  # tracks the belief of target being in cell_i given the observations
    confidence_state = numpy.zeros((dim, dim),
                                   dtype=object)  # tracks the confidence of the target actually being in cell_i given the belief state and current observations
    previous_cells = []  # stores all the information about the previous cells here in the form (x, y)
    distance_traveled = 0  # stores the distanced traveled from the current cell you start with
    num_searches = 0  # stores the number of searches made by the agent
    target_found = False # stores the information related to if the target is found
    searches = 0 # stores the searches 

    # initialize information related to the the agent and the map itself
    def __init__(self, dim):
        self.dim = dim # stores the dimension of the agent instance
        self.map_board = numpy.zeros((self.dim, self.dim), dtype=object) # stores the map board for the agent instance 
        self.belief_state = numpy.zeros((self.dim, self.dim), dtype=object) # stores the belief state for the agent instance
        self.confidence_state = numpy.zeros((self.dim, self.dim), dtype=object) # stores the confidence state for the agent instance
        self.generate_board() # after stores all the agent related fields, the map is generated

    # generate the board for the agent that is going to traverse
    def generate_board(self):
        # generate all the locations
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                # create every i,j instance with the Terrain or belief at t=0 respectively
                self.map_board[i][j] = Terrians(i, j)  # generated map default values
                self.belief_state[i][j] = 1 / (self.dim * self.dim)  # generated belief state default values


        # generate a random spot for the target to be located
        indices = list(range(0, self.dim))  # create a list of numbers to choose from
        target_location = (random.choice(indices), random.choice(indices))  # get a random target location
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

        '''
        Step 2: Update the remaining probabilities so everything is equal to 1 and update the confidence levels
        '''
        for i in range(self.dim):
            for j in range(self.dim):
                if (i, j) != (x, y):
                    self.belief_state[i][j] /= prob_failure
                    self.confidence_state[i][j] = self.belief_state[i][j] * (1 - self.map_board[i][j].false_neg)

    # check if what you are checking is within the constraints of the board
    def check_constraints(self, ind1, ind2):
        if ind1 >= 0 and ind1 <= self.dim - 1 and ind2 >= 0 and ind2 <= self.dim - 1:
            return True

        return False

    # finds the next cell for the agent to go to
    def next_cell(self, x, y):

        max_total = 0 # stores the max confidence probability
        max_point = (0, 0)  # the cell that has the best neighboring cells we can search next

        alpha = (self.dim * self.dim) / (self.searches + (2 * self.dim * self.dim))  # as we do more searches the importance of confidence decreases
        
        # iterate through the full board to find the next cell to go to
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if (x, y) != (i, j): # skip any cell that we already went too
                    addition = 0  # represents the total of confidence and beliefs of it's neighbors
                    failure_prob = (self.map_board[i][j].false_neg * self.belief_state[i][j]) + (1 - self.map_board[i][j].false_neg) # total failure rate

                    # iterate through all neighbors to find the next cell to go to while storing the belief to utilize later
                    if self.check_constraints(i + 1, j):
                        belief = self.belief_state[i + 1][j] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha  # add confidence
                    if self.check_constraints(i - 1, j):
                        belief = self.belief_state[i - 1][j] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence
                    if self.check_constraints(i + 1, j + 1):
                        belief = self.belief_state[i + 1][j + 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence
                    if self.check_constraints(i + 1, j - 1):
                        belief = self.belief_state[i + 1][j - 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence
                    if self.check_constraints(i - 1, j - 1):
                        belief = self.belief_state[i - 1][j - 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence
                    if self.check_constraints(i - 1, j + 1):
                        belief = self.belief_state[i - 1][j + 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence
                    if self.check_constraints(i, j + 1):
                        belief = self.belief_state[i][j + 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence
                    if self.check_constraints(i, j - 1):
                        belief = self.belief_state[i][j - 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg) * alpha # add confidence

                    if addition > max_total:  # if the current addition is greater then we replace the max_point
                        max_total = addition
                        max_point = (i, j)
                    elif addition == max_total:  # if the additions are equal we choose the one with smaller distance
                        if (abs(x - i) + abs(y - j)) < (abs(x - max_point[0]) + abs(y - max_point[0])):
                            max_total = addition
                            max_point = (i, j)

        return max_point

    # returns a list of neighbors of the current cell that are valid
    def neighbors_list(self, x, y):  

        # store the neighbors of cell x, y to compare to find the valid list
        neighbors = []

        # iterate through all the neighbors of cell 
        if self.check_constraints(x + 1, y):
            neighbors.append((x + 1, y)) # append the neighbor
        if self.check_constraints(x - 1, y):
            neighbors.append((x - 1, y))  # append the neighbor
        if self.check_constraints(x + 1, y + 1):
            neighbors.append((x + 1, y + 1))  # append the neighbor
        if self.check_constraints(x + 1, y - 1):
            neighbors.append((x + 1, y - 1))  # append the neighbor
        if self.check_constraints(x - 1, y - 1):
            neighbors.append((x - 1, y - 1))  # append the neighbor
        if self.check_constraints(x - 1, y + 1):
            neighbors.append((x - 1, y + 1))  # append the neighbor
        if self.check_constraints(x, y + 1):
            neighbors.append((x, y + 1))  # append the neighbor
        if self.check_constraints(x, y - 1):
            neighbors.append((x, y - 1))  # append the neighbor

        return neighbors

    def test_moves(self, start_x, start_y, end_x, end_y):  # looks for cells that we can search on the way from current to next cell

        # we only use the 2 manhattan paths to find the cells that we can search
        x = start_x
        y = start_y + 1
        list1 = []  # list of cells that we should search in path 1
        list2 = []  # list of cells that we should search in path 2
        count1 = 0 
        count2 = 0
        list3 = []
        while y <= end_y:
            if self.belief_state[x][y] > 0.85*self.belief_state[end_x][end_y]:  # only add the cell to list if it's bigger than .85 * belief of end cell
                count1 += 1
            list1.append((x, y))
            y += 1
        y -= 1
        while x < end_x:
            if self.belief_state[x][y] > 0.85*self.belief_state[end_x][end_y]:
                count1 += 1
            list1.append((x, y))
            x += 1
        x = start_x+1
        y = start_y
        while x <= end_x:
            if self.belief_state[x][y] > 0.85*self.belief_state[end_x][end_y]:
                count2 += 1
            list2.append((x, y))
            x += 1
        x -= 1
        while y < end_y:
            if self.belief_state[x][y] > 0.85*self.belief_state[end_x][end_y]:
                count2 += 1
            list1.append((x, y))
            y += 1
        list3.append(list1)
        list3.append(list2)
        if count2 < count1:  # we return the list with more cells in the list
            return list1
        elif count1 < count2:
            return list2
        elif count1 == count2:  # if there are equal cells in the list we choose randomly
            return random.choice(list3)

    # this is the acronym we decided to use for the advanced agent: Thanks Aravind for Helping Us
    def tahu(self, x, y):
        x_cord = x  # current cell x coordinate
        y_cord = y  # current cell y coordinate
        moves_counted = 0  # represents amount of moves done so far
        distance = 0  # represents distance covered so far

        # we keep iterating until the target is found
        while self.target_found is False:  

            self.searches = moves_counted # searches == moves we counted so far
            moves_counted += 1 # increment moves
            neighbors = self.neighbors_list(x_cord, y_cord)  # list of valid neighbors of the current cell

            for z in range(0, len(neighbors)):  # we search all the neighbors
                check_cell = neighbors.pop()
                moves_counted += 2  # moving + searching
                fnr = self.map_board[check_cell[0]][check_cell[1]].false_neg
                rand = random.random()  # random value from 0 to 1
                self.bayesian_update(check_cell[0], check_cell[1])  # we update the probabilities
                if check_cell == self.target_info.location and rand > fnr:
                    self.target_found = True
                    break  # if the target is found we break out of the loop
                if self.map_board[check_cell[0]][check_cell[1]].terrain_type == "flat":  # if the cell is a flat terrain we check twice
                    rand = random.random()
                    self.bayesian_update(check_cell[0], check_cell[1])
                    if check_cell == self.target_info.location and rand > fnr:
                        moves_counted += 1
                        self.target_found = True
                        break

            moves_counted += 1  # to return to current cell

            if (x_cord, y_cord) == self.target_info.location:  # if the current cell is the target cell
                fnr = self.map_board[x_cord][y_cord].false_neg
                rand = random.random()
                if rand > fnr:
                    self.target_found = True

                rand = random.random()
                if self.map_board[x_cord][y_cord].terrain_type == "flat" and rand > fnr:  # if the cell is a flat terrain we check twice
                    moves_counted += 1
                    self.target_found = True

                else:
                    self.bayesian_update(x_cord, y_cord)  # update the probabilities as the search has failed
                    next_cell = self.next_cell(x_cord, y_cord)  # find the best next cell to search
                    path_cells = self.test_moves(x_cord, y_cord, next_cell[0], next_cell[1])  # list of cells that we can search on the way from current to next cell

                    if len(path_cells) > 1:  # if there are cells in path that have beliefs of higher than 85% of the next cell we are going to search, we search the cell

                        for z in range(0, len(path_cells)):  # search all the path_cells

                            check_cell = path_cells.pop()
                            moves_counted += 1
                            fnr = self.map_board[check_cell[0]][check_cell[1]].false_neg
                            rand = random.random()
                            self.bayesian_update(check_cell[0], check_cell[1])
                            if check_cell == self.target_info.location and rand > fnr:
                                self.target_found = True
                                break
                            if self.map_board[check_cell[0]][check_cell[1]].terrain_type == "flat":  # if the cell is a flat terrain we check twice
                                rand = random.random()
                                self.bayesian_update(check_cell[0], check_cell[1])
                                if check_cell == self.target_info.location and rand > fnr:
                                    moves_counted += 1
                                    self.target_found = True
                                    break

                    distance += abs(x_cord - next_cell[0]) + abs(y_cord - next_cell[1])  # add the distance travelled from current to next cell
                    x_cord = next_cell[0]  # replace to x coordinate of next cell to search
                    y_cord = next_cell[1]  # replace to y coordinate of next cell to search

            else:
                self.bayesian_update(x_cord, y_cord)
                next_cell = self.next_cell(x_cord, y_cord)
                path_cells = self.test_moves(x_cord, y_cord, next_cell[0], next_cell[1])

                if len(path_cells) > 1:  # if there are cells in path that have beliefs of higher than 85% of the next cell we are going to search, we search the cell

                    for z in range(0, len(path_cells)):

                        check_cell = path_cells.pop()
                        moves_counted += 1
                        fnr = self.map_board[check_cell[0]][check_cell[1]].false_neg
                        rand = random.random()
                        if check_cell == self.target_info.location and rand > fnr:
                            self.target_found = True
                            break
                        if self.map_board[check_cell[0]][check_cell[1]].terrain_type == "flat":  # if the cell is a flat terrain we check twice
                            rand = random.random()
                            if check_cell == self.target_info.location and rand > fnr:
                                moves_counted += 1
                                self.target_found = True
                                break

                distance += abs(x_cord - next_cell[0]) + abs(y_cord - next_cell[1])  # add the distance travelled from current to next cell
                x_cord = next_cell[0]  # replace to x coordinate of next cell to search
                y_cord = next_cell[1]  # replace to y coordinate of next cell to search

        return moves_counted + distance

# start the advanced agent 
def start_agent():
    print("Running Advanced Agent...")
    agent = AgentClass(50) # make an instance of a 50
    indicies = list(range(0, agent.dim)) # create a list of numbers to choose from
    score = agent.tahu(random.choice(indicies), random.choice(indicies)) # keep track of the agent's score
    results = {"agent": "Advanced Agent", "target location": agent.target_info.terrain_location, "score": score}
    print(results)
