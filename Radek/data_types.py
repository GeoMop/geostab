

class ElectrodeGroup:
    def __init__(self, gallery="", wall="", height=""):
        self.gallery = gallery
        self.wall = wall
        self.height = height
        self.electrodes = []


class Electrode:
    def __init__(self, id=0, offset=0.0, x=0.0, y=0.0, z=0.0):
        self.id = id
        self.offset = offset
        self.x = x
        self.y = y
        self.z = z


class Measurement:
    def __init__(self, number="", date=""):
        self.number = number
        """Measurement number, this is key"""
        self.date = date
