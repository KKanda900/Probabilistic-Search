import numpy, sys, random, math
from Terrain import Terrians
from Target import Target

class Basic_Agent_1:
    dim = 0
    map_board = numpy.zeros((dim, dim), dtype=object)
    target_info = None
    belief_state = numpy.zeros((dim, dim), dtype=object)
    previous_cells = [] # stores all the information about the previous cells here in the form (x, y)
    target_found=False
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
        prob_failure = 1 - (self.belief_state[curr_prev[0]][curr_prev[1]])
        # obtain the false negative probability for the cell
        fnr = self.map_board[x][y].false_neg
        # obtain the probability of the target being in the location based on the observation
        curr_belief = self.belief_state[x][y]
        # update the probability of the current cell
        self.belief_state[x][y] = (curr_belief*fnr)/(prob_failure)

        '''
        Step 2: Update the remaining probabilities so everything is equal to 1
        '''
        # go to each cell and make sure the total probability equals 1
        for i in range(self.dim):
            sum_equal_1 = False
            for j in range(self.dim):
                # if the i == x and j == y that means we are at the coordinate we just updated and we can just skip that
                if i == x and j == y:
                    pass
                else:
                    # if the other case isn't true that means we are at every other cell
                    if self.belief_state[x][y] == 0:
                        self.belief_state[x][y]+=.01
                    elif self.belief_state[x][y] > 0: # otherwise keep lowering the probability until we get back to the sum being 1
                        self.belief_state[x][y]-=.01
                        print(sum(self.belief_state).any())
                        if sum(self.belief_state).any() == 1:
                            sum_equal_1 = True
                            break
                        else:
                            pass
            
            if sum_equal_1 == True:
                break

    def calculate_neighbors(self,x,y):
        neighbors=[]
        if x+1<50 and y<50:
            neighbors.append((x+1,y))
        if x<50 and y+1<50:
            neighbors.append((x,y+1))
        if x-1<50 and y<50:
            neighbors.append((x-1,y))
        if x<50 and y-1<50:
            neighbors.append((x,y-1))
        return neighbors

    def clear_ties(self,board,x,y):
        min_distance=0.0
        ties=[]
        j=0
        for i in board:
            x_cord=i[0]
            y_cord=i[1]
            distance = abs(x_cord-x) + abs(y_cord-y)
            if j==0:
                min_distance=distance
                ties.append((x_cord,y_cord))
                j=1
            elif distance<min_distance:
                min_distance=distance
                ties.clear()
                ties.append((x_cord,y_cord))
            elif min_distance==distance:
                ties.append((x_cord,y_cord))
        return ties[0]

    def calculate_location(self,board):
        location=()
        ties=[]
        max_val=0.0
        for i in range(0,50):
            for j in range(0,50):
                if board[i][j]>max_val:
                    max_val=board[i][j]
                    location=(i,j)
                    ties.clear()
                    ties.append(location)
                elif board[i][j]==max_val:
                    ties.append((i,j))
        return ties

    def basic_agent(self,x,y):
        x_cord=x
        y_cord=y
        moves_counted=0
        distance=0
        while self.target_found==False:
             if (x_cord,y_cord)==self.target_info.location and self.belief_state[x_cord][y_cord]==1-self.map_board[x_cord][y_cord].terrian_type.false_neg:
                moves_counted+=1
                self.target_found=True
            else:
                self.previous_cells.append((x_cord,y_cord))
                moves_counted+=1
                self.bayesian_update(x_cord,y_cord)
                locations=self.calculate_location(self.belief_state)
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
#self.start()
