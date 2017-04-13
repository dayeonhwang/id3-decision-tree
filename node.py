class Node:
    def __init__(self):
        self.label = None # (int) output(Class value); None if there is a decision attribute
        self.split_attr = None # (str) name of splitting attribute; None if leaf node
        self.split_attr_index = None # (int) index of splitting attribute ; None if leaf node
        self.split_attr_val = None # (int) value of splitting attribute; None if leaf node
        self.mode = None # (int) mode output among examples sorted to the node
        self.children = {} # (dictionary) {val1: node, val2: node, ...} -> sub_examples_i : node
        self.unclearMode = 0 # (bool) if it is clear to have a mode or not