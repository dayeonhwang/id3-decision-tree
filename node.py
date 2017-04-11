class Node:
    def __init__(self):
        self.label = None # output(Class value); None if there is a decision attribute
        self.split_attr = None # name of splitting attribute
        self.split_attr_index = None #index of splitting attribute
        self.split_attr_val = None # value of splitting attribute for a child node
        self.children = {} # {0: val, 1: val, ...,}
        self.depth = None