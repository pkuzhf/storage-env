from gym import spaces


class ShapedTuple(spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces
    Example usage:
    observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        self.shape = (len(spaces),) + spaces[0].shape
        super(ShapedTuple, self).__init__(spaces)
