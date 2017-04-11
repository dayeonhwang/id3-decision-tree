from node import Node
import math
import random 
from collections import Counter

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  # target value: attribute with the key "Class"
  # label a leaf as 'Class'=0
  # default : node -> default.lable = attr, default.children = {}

  attributes = [key for key in examples.keys()] # a list of attributes
  # corner cases
  # 1. example set is empty
  if len(examples) == 0:
    return default 
  # 2. all examples have same class value
  else if check_same_class(examples):
    top_class = examples[0][-1]
    tree = Node(top_class, {})
    return tree
  # 3. non-trivial split of examples is possible (all examples have same attribute values) -> mode class value
  else if:
    return target_attr_mode(examples, attributes, target_attr)
  # general case:
  else:
    best_attr = choose_attr(examples, attributes)
    # create a root node
    root = Node(best_attr, {})

def prune_iter(node, examples):
  old_acc = test(node, examples)
  old_node = node
  freq = {}
  if node.children != None:
    for c in node.children.iteritems():
      if c.label in freq: 
        freq[c.label] += 1.0
      else:
        freq[c.label] = 1.0
    node.label = max(freq, key=freq.get)]]
    node.children = {}
    new_acc = test(node, examples)
    if new_acc >= old_acc:
      return node
  return old_node

def breadth_first_search_complete(root):
  q = []
  output = {}
  depthList = []
  level = 0
  q.append(root)
  q.append(None)
  while len(q) != 0:
      n = q.pop(0)
      if n == None:
        level += 1
        q.append(None)
        if (q[0] == None):
          break
        else:
          continue
      if level in output:
        output[level].append(n)
      else:
        output[level] = []
        output[level].append(n)
        
      #output.append(n)
      n.depth = level
      children = n.children
      if children != None:
        for c in children.itervalues():
          q.append(c)
  return output

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  
  pruning strategy - removing subtree rooted at node, making it a leaf node with the most common classification of the training examples affiliated with that node
                   - node removed only if pruned tree performs no worse than the original over the validation set
                   - pruning continues until further pruning is harmful'''

  dictLevel = breadth_first_search_complete(node)
  maxLevel = max(dictLevel, key=int)

  while maxLevel > 0:
    dictLevel = breadth_first_search_complete(node)
    maxLevel = max(dictLevel, key=int)
    thatLevel = dictLevel[maxLevel]
    for n in thatLevel:
      prune_iter(n, examples)
  return node


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

def entropy(examples, target_attr):
  '''
  Calculate entropy of a target attribute
  '''
  freq = {} # dictionary : [value, frequency of value]
  entropy = 0.0

  # calculate frequency of values in target attr
  for e in examples:
    if e[target_attr] in freq: # if e[target_attr] key is in freq object
      freq[e[target_attr]] += 1.0
    else:
      freq[e[target_attr]] = 1.0

  for f in freq.values():
    entropy += (-f/len(examples)) * math.log(f/len(examples), 2)
  return entropy

def info_gain(examples, target_attr):
  '''
  Calculate information gain of splitting examples on the chosen attribute
  Info_gain(x_i) = H_prior - Sum_v (P(x_i = v) * H(y|x_i=v)) (x_i: attr)
  '''
  prior_entropy = entropy(examples, target_attr)
  sub_entropy = 0.0
  freq = {} # dictionary : [value, frequency of value]

  # calculate frequency of values in target attr
  for e in examples:
    if e[target_attr] in freq:
      freq[e[target_attr]] += 1.0
    else:
      freq[e[target_attr]] = 1.0

  # calculate sum of entropy for each subset * prob
  for val, f_val in freq.iteritems():
    prob = f_val / sum(freq.values())
    sub_examples = [e for e in examples if e[target_attr] == val]
    sub_entropy += prob * entropy(sub_examples, target_attr)

  # subtract entropy of target_attr from entropy of entire data w.r.t target_attr
  return (prior_entropy - sub_entropy)

def choose_attr(examples, attributes):
  '''
  Choose which attribute to split on
  '''
  max_info_gain = 0
  curr_info_gain = 0
  for a in attributes:
    curr_info_gain = info_gain(examples, a)
    if curr_info_gain > max_info_gain:
      max_info_gain = curr_info_gain
      best_attr = a
  return best_attr

def check_same_class(examples):
  '''
  Check if all examples have the same class value
  '''
  top_class = examples[0][-1]
  for e in examples:
    if e[-1] != top_class:
      return False
  return True

def check_same_attr_val(examples, attributes):
  '''
  Check if all examples have same attribute values
  '''

def target_attr_mode(examples, attributes, target_attr):
  '''
  Get mode of target attribute's value
  '''
  attr_index = attributes.index(target_attr) # index of target_attr from first row of examples (= a list of attributes)
  target_examples = [e[attr_index] for e in examples]
  return Counter(target_examples).most_common()[0][0]

def fill_missing_attr(examples):
  '''
  Fill any missing attributes (denoted with a value of "?") with mode of attribute's value or 0.5
  '''
  for e in examples:
    for attr, attr_val in e.iteritems():
      if attr_val == '?':
        attr_val = target_attr_mode(examples, attr)
  return examples
