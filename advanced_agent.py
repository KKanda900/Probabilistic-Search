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

        '''
        Step 2: Update the remaining probabilities so everything is equal to 1
        '''
        # go to each cell and make sure the total probability equals 1

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

    def next_cell(self, x, y):

        max_total = 0
        max_point = (0, 0)

        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if (x, y) != (i, j):
                    addition = 0  # represents the total of confidence and beliefs of it's neighbors
                    failure_prob = (self.map_board[i][j].false_neg * self.belief_state[i][j]) + (1 - self.map_board[i][j].false_neg)

                    if self.check_constraints(i + 1, j):
                        belief = self.belief_state[i + 1][j] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i - 1, j):
                        belief = self.belief_state[i - 1][j] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i + 1, j + 1):
                        belief = self.belief_state[i + 1][j + 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i + 1, j - 1):
                        belief = self.belief_state[i + 1][j - 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i - 1, j - 1):
                        belief = self.belief_state[i - 1][j - 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i - 1, j + 1):
                        belief = self.belief_state[i - 1][j + 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i, j + 1):
                        belief = self.belief_state[i][j + 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)
                    if self.check_constraints(i, j - 1):
                        belief = self.belief_state[i][j - 1] / failure_prob
                        addition += belief
                        addition += belief * (1 - self.map_board[i][j].false_neg)

                    if addition > max_total:
                        max_total = addition
                        max_point = (i, j)
                    elif addition == max_total:
                        if (abs(x - i) + abs(y - j)) < (abs(x - max_point[0]) + abs(y - max_point[0])):
                            max_total = addition
                            max_point = (i, j)

        return max_point

    def test_moves(self, start_x, start_y, end_x, end_y):
        x = start_x
        y = start_y + 1
        list1 = []
        list2 = []
        count1 = 0
        count2 = 0
        list3 = []
        while y <= end_y:
            if self.belief_state[x][y] > 0.85*self.belief_state[end_x][end_y]:
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
        if count2 < count1:
            return count1
        elif count1 < count2:
            return count2
        elif count1 == count2:
            return random.choice(list3)

    def advanced_agent(self, x, y):
        x_cord = x
        y_cord = y
        moves_counted = 0
        distance = 0

        while self.target_found is False:

            moves_counted += 1
            if (x_cord, y_cord) == self.target_info.location:
                fnr = self.map_board[x_cord][y_cord].false_neg
                rand = random.random()
                if rand > fnr:
                    self.target_found = True

                rand = random.random()
                if self.map_board[x_cord][y_cord].terrain_type == "flat" and rand > fnr:  # if the cell is a flat terrain we check twice
                    moves_counted += 1
                    self.target_found = True

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

                    distance += abs(x_cord - next_cell[0]) + abs(y_cord - next_cell[1])
                    x_cord = next_cell[0]
                    y_cord = next_cell[1]

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

                distance += abs(x_cord - next_cell[0]) + abs(y_cord - next_cell[1])
                x_cord = next_cell[0]
                y_cord = next_cell[1]

        return moves_counted + distance


def start_agent():
    agent = AgentClass(50)
    # create a list of numbers to choose from
    indicies = list(range(0, agent.dim))
    print(agent.advanced_agent(random.choice(indicies), random.choice(indicies)))


start_agent()
