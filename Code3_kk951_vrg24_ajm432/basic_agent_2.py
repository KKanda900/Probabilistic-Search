import numpy, random
from Terrain import Terrians
from Target import Target

'''
Basic_Agent_2 Class

Description: Contains all the logic pertaining to the second basic agent and utilizing the specific rule.
'''
class Basic_Agent_2:
    
    # information to store about the generated map
    dim = 0 # default size 0 to start with 
    map_board = numpy.zeros((dim, dim), dtype=object)  # in the start create the map_board with the default dimension of 0
    
    # target info will hold all the information related to the target
    target_info = None  # set to None as default because the map wasn't generated yet

    # information to store for the basic agent to use
    belief_state = numpy.zeros((dim, dim), dtype=object) # tracks the belief of target being in cell_i given the observations
    confidence_state = numpy.zeros((dim, dim), dtype=object) # tracks the confidence of the target actually being in cell_i given the belief state and current observations
    previous_cells = [] # stores all the information about the previous cells here in the form (x, y)
    distance_traveled = 0 # stores the distanced traveled from the current cell you start with 
    num_searches = 0 # stores the number of searches made by the agent
    target_found = False # stores if the target is found or not for the basic agent

    # initialize information related to the the agent and the map itself
    def __init__(self, dim):
        self.dim = dim  # stores the dimension of the agent instance
        self.map_board = numpy.zeros((self.dim, self.dim), dtype=object) # stores the map board for the agent instance 
        self.belief_state = numpy.zeros((self.dim, self.dim), dtype=object) # stores the belief state for the agent instance
        self.confidence_state = numpy.zeros((self.dim, self.dim), dtype=object) # stores the confidence state for the agent instance
        self.generate_board() # after stores all the agent related fields, the map is generated
    
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
        terrian_type = self.map_board[target_location[0]][target_location[1]].terrain_type # the terrain of the target will be whatever map_board[x][y] is
        self.target_info = Target(target_location[0], target_location[1], terrian_type) # store the information in the target_info variable

    # update the belief state based on bayesian updating (for advanced agent)
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
        self.confidence_state[x][y] = curr_cell_belief * (1 - fnr)

        '''
        Step 2: Update the remaining probabilities so everything is equal to 1
        '''
        for i in range(self.dim):
            for j in range(self.dim):
                if (i, j) != (x, y):
                    self.belief_state[i][j] /= prob_failure
                    self.confidence_state[i][j] = self.belief_state[i][j] * (1 - self.map_board[i][j].false_neg)

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
                ties.append((x_cord, y_cord)) #adds current element in board to list
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

    # start basic agent 2 here
    def start_agent(self, x, y):
        x_cord = x #current cell x coordinate
        y_cord=y  #current cell y coordinate
        moves_counted=0 #represents amount of moves done so far
        distance=0 #represents distance covered so far
        while self.target_found==False: #while the target has not been found
            moves_counted += 1 #increments moves counted
            if (x_cord,y_cord)==self.target_info.location: #if current cell coordinates are equivalent to the target location
                fnr=self.map_board[x_cord][y_cord].false_neg #false negative rate of current cell
                rand=random.random() #generates random number between 0 and 1
                # first check if we hit a false negative if we didn't return that we found the target and the results at the end
                if rand>fnr:
                    self.target_found=True #target has been found
                else:
                    self.previous_cells.append((x_cord,y_cord))
                    self.bayesian_update(x_cord,y_cord) #perform bayesian update on belief state and confidence
                    locations=self.calculate_location(self.confidence_state) #calculates list of possible next cells to go to

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
                self.bayesian_update(x_cord, y_cord) 
                locations = self.calculate_location(self.confidence_state) 
                if len(locations) > 1: #ties exist between cells in the list
                    location = self.clear_ties(locations, x_cord, y_cord) 
                    locations.clear()
                    locations.append(location)
                coords = locations[0] 
                distance += abs(x_cord - coords[0]) + abs(y_cord - coords[1])
                x_cord = coords[0] 
                y_cord = coords[1] 
        #returns final result of moves and distance
        return moves_counted+distance

# start the basic agent 2 process here
def start_agent():
    print("Running Basic Agent 2...")
    agent = Basic_Agent_2(50) # create the instance of the agent
    indicies = list(range(0, agent.dim)) # create a list of numbers to choose from
    score = agent.start_agent(random.choice(indicies), random.choice(indicies)) # track the agents score
    results = {"agent": "Basic Agent 2", "target location": agent.target_info.terrain_location, "score": score}
    print(results)
    