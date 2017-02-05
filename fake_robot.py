class Robot():

    def __init__(self):
        self.t = 0
        self.lastAction = None
        self.readingAfterAction = [0, 0, 40]

    def take_action(self, action):
        self.t += 1
        self.lastAction = None
        self.readingAfterAction = [0, 0, 35] if (self.t//1) % 2 == 0 else [0, 0, 40]

    def get_readings(self):
        return self.readingAfterAction

    def get_sensorspace_corners(self):
        low_corner = (0, 0, -10)
        high_corner = (1024, 2028, 90)

        return [low_corner, high_corner]


