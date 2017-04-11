from node import Node
import math
import random 
from collections import Counter
from collections import defaultdict

#global variable
attributes = [key for key in examples.keys()] # a list of attributes

def ID3(examples, default):
  global attributes
  
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  fill_missing_attr(examples)
  tree = Node()
  same_class_value = check_same_class(examples)
  best_attr = choose_best_attr(examples, attributes)

  # corner cases
  # 1. example set is empty
  if len(examples) == 0:
    return default
  # 2. all examples have same class value -> tree is a leaf node
  else if check_same_class(examples):
    return check_same_class(examples)
  # 3. non-trivial split of examples is possible (all examples have same attribute values) -> mode class value
  else if len(attributes)==0:
    return mode(examples, {})
  # general case:
  else:
    best_attr = choose_best_attr(examples, attributes)
    tree.split_attr = best_attr
    count=0
    sub_examples = split_examples(examples, best_attr) #{val: {sub_examples}, val2: {sub_examples}, ...}
    for val, sub_e in sub_examples.iteritems():
      examples_i=sub_e
      attributes.remove(best_attr)

      subtree = ID3(sub_e, mode({best_attr:val}))
      ...
      count += 1
  return tree



def breadth_first_search_complete(root):
  '''
  Takes a root node and returns a dictionary of {level, [node @ the level]}
  '''
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

def prune_iter(node, examples):
  '''
  Takes a node and compares accuracy of tree w/ or w/o the node.
  Keeps pruning by removing the node if accuracy of tree w/o the node is bigger than that of w/ the node
  and replacing the node with its most popular class(mode)
  '''
  old_acc = test(node, examples)
  old_node = node
  freq = {}
  if node.children != None: # how about case of leaf??
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

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  
  *pruning strategy - removing subtree rooted at node, making it a leaf node with the most common classification of the training examples affiliated with that node
                    - node removed only if pruned tree performs no worse than the original over the validation set
                    - pruning continues until further pruning is harmful (reduced error pruning algorithm)'''

  dictLevel = breadth_first_search_complete(node) # dictionary : {level, node @ the level}
  maxLevel = max(dictLevel, key=int)

  while maxLevel > 0:
    dictLevel = breadth_first_search_complete(node)
    maxLevel = max(dictLevel, key=int)
    nodesAtLevel = dictLevel[maxLevel]
    for n in nodesAtLevel: # leaves
      prune_iter(n, examples)
  return node

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).

  *arguments "examples": validation set
  '''
  total_ex = len(examples)
  accurate_ex = 0

  for e in examples:
    actual_class_val = e['Class']
    computed_class_val = evaluate(node, e)
    if computed_class_val == actual_class_val:
      accurate_ex += 1

  accuracy = float(accurate_ex / total_ex)
  return accuracy

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

def choose_best_attr(examples, attributes):
  '''
  Choose which attribute to split on (attr w/ max info_gain)
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
  Return None(if don't have same class value) or value of the attribute
  '''
  first_ex_value = examples[0][0]
  for e in examples[1:]:
    if e[0] != first_ex_value:
      return None
  return first_ex_value

def target_attr_mode(examples, target_attr):
  '''
  Get mode of target attribute's value
  '''
  attributes = [key for key in examples.keys()]
  attr_index = attributes.index(target_attr)
  target_examples = [e[attr_index] for e in examples]
  return Counter(target_examples).most_common()[0][0] #value

def mode(examples, target_dict):
  '''
  Return mode of Class value from remaining set of examples
  Input: examples, target_dict (a dictionary of {attr1: val1, attr2: val2,...}) that nodes have
  Output: mode ('democrat' or 'republic')
  '''
  # initialize dictionary
  count = {}
  for k in keys:
    count[k] = 0
  # select examples that have matching attribute & value from target_dict
  for attr, val in target_dict.iteritems():
    subset = [e for e in examples if e[attr] == val]
  # count mode Class value
  for s in subset:
    attr_val = s["Class"] #label
    count[attr_val] += 1
  # return a key with the biggest value
  return max(count, key=count.get)

def fill_missing_attr(examples):
  '''
  Fill any missing attributes (denoted with a value of "?") with mode of attribute's value
  '''
  for e in examples:
    for attr, attr_val in e.iteritems():
      if attr_val == '?':
        attr_val = target_attr_mode(examples, attr)
  return examples

def split_examples(examples, target_attr):
  '''
  Split examples into a subset that has the target_attr as a key
  Return a dictionary of {val: {sub_examples}, val2: {sub_examples}, ...}
  '''
  subset = {}
  target_val = [e[target_attr] for e in examples] # list of all values of target_attribute
  for val in target_val:
    subset[val] = [e for e in examples if e[target_attr] == val]
  return subset