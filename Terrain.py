import random

class Terrians:
    types = ['flat', 'hilly', 'forested', 'caves']
    terrain_type = None
    # the probability of the target not being a cell given that we know it's in that particular cell
    false_neg = 0.0 # by default its set to 0.0 but we can set it based on the terrian type
    # the probability of the target being in a cell given that we know its not in a particular cell
    false_pos = 0.0  # by default its 0.0 but we can increase this later
    cell_location = () # store the location of the cell

    def __init__(self, i, j):
        self.cell_location = (i, j)
        self.terrain_type = random.choice(self.types)
        if self.terrain_type == 'flat':
            self.false_neg = 0.1
        elif self.terrain_type == 'hilly':
            self.false_neg = 0.3
        elif self.terrain_type == 'forested':
            self.false_neg = 0.7
        else:
            self.false_neg = 0.9
