class Target:
    location = ()
    terrain_location = None

    def __init__(self, i, j, terrain_location):
        self.location = (i, j)
        self.terrain_location = terrain_location
