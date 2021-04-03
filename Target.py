class Target:
    location = ()
    terrian_location = None

    def __init__(self, i, j, terrian_location):
        self.location = (i, j)
        self.terrian_location = terrian_location
