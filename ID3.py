from node import Node
import math
import random 
from collections import Counter
from collections import defaultdict

#global variable
attributes = [] # a list of attributes
flag = 0

def ID3(examples, default):
  global attributes
  global flag
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  fill_missing_attr(examples)
  tree = Node()
  tree.mode = mode(examples)
  
  if flag == 0:
    attributes = [key for key in examples[0]]
    if 'Class' in attributes:
      attributes.remove('Class')
    flag = 1

  # corner cases
  # 1. example set is empty
  if len(examples) == 0:
    tree.label = default

  # 2. all examples have same class value -> tree is a leaf node
  elif check_same_class(examples):
    tree.label = check_same_class(examples)

  # 3. non-trivial split of examples is possible (all examples have same attribute values) -> mode class value
  elif len(attributes) == 0:
    tree.label = mode(examples)
  # general case:
  else:
    best_attr = choose_best_attr(examples, attributes)
    if (best_attr == 'Class'):
      best_attr = None
    print 'chose best attr = ' + str(best_attr)
    #check if best_attr of all examples are identical
    if (best_attr == None):
      tree.label = mode(examples)
    elif check_same_value(examples, best_attr) == True:
      tree.label = mode(examples)
    #otherwise, continue
    else:
      tree.split_attr = best_attr
      tree.split_attr_index = attributes.index(best_attr)
      children = {}
      sub_examples = split_examples(examples, best_attr) #{val1: {sub_examples_1}, val2: {sub_examples_2}, ...}

      for val, sub_e in sub_examples.iteritems():
        if best_attr in attributes:
          attributes.remove(best_attr)
        children[val] = ID3(examples, mode(sub_e))
      tree.children = children
  return tree

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  
  *pruning strategy
  - start from node -> node's children (top to bottom)
  - compare accuracy before/after deleting each node -> delete if accuracy is higher
  - greedily grab each node @ each level and repeat recursively'''

  numDeletedNode = 0 # keep track of the number of nodes deleted during pruning
  q = []  # queue of nodes @ each level
  output = {}
  q.append(node)
  while (len(q) != 0):
    n = q.pop(0) #first node
    #if (n != None and n.label != None): # if we have nodes & haven't reached a leaf node yet
      
    # save the current node's state before deleting it
    old_acc = test(n, examples)
    old_node = n
    old_children = n.children
      
    # proceed to delete it
    n.label = n.mode
    n.children = {}
    new_acc = test(n, examples)
      
    # if decide to delete it
    if new_acc >= old_acc:
      numDeletedNode += 1
      continue

    # if decide to keep it
    else:
      #add children
      if old_children != None:
        for c in old_children.itervalues():
          q.append(c)

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

  accuracy = accurate_ex / float(total_ex)
  print total_ex
  print accurate_ex
  print accuracy
  return accuracy

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  nodeVisited = 0
  while len(node.children)!=0: # while you havent reached a leaf
    best_attr = node.split_attr # find which attribute you split on
    ex_split_val = example[best_attr] # find what value the example had for this attribute
    node = node.children[ex_split_val] # go to the child node whose value matches
    nodeVisited += 1

  print "number of node visited:", nodeVisited
  return node.label

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
  best_attr = None
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
  first_ex_value = examples[0]['Class']
  for e in examples:
    if e['Class'] != first_ex_value:
      return None
  return first_ex_value

def check_all_attr_same_value(examples, attributes):
  '''
  Check if all examples have the same attribute value for each attribute
  Return False or True
  '''
  for a in attributes:
    if check_same_value(examples, a) != True:
      return False
  return True

def check_same_value(examples, attr):
  '''
  Check if all examples have the same attribute value for each attribute
  Return False or True
  '''
  checking_value = examples[0][attr]
  for e in examples:
    if e[attr] != checking_value:
      return False
  return True

def target_attr_mode(examples, target_attr):
  '''
  Get mode of target attribute's value
  '''
  attr_index = attributes.index(target_attr)
  target_examples = [e[attr_index] for e in examples]
  return Counter(target_examples).most_common()[0][0] #value

def mode(examples):
  '''
  Return mode of Class value from remaining set of examples
  Input: examples
  Output: mode ('democrat' or 'republic')
  '''
  # initialize dictionary
  count = {}
  for e in examples:
    if e['Class'] in count:
      count[e['Class']] += 1.0
    else:
      count[e['Class']] = 1.0

  # return key with the biggest value
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
