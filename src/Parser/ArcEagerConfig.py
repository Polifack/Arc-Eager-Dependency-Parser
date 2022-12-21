from .constants import *

def get_rightmost_child(n, edge_list):
  '''
  Given an node n and a list of edges returns 
  the rightmost child of the n (i.e. the bigger node
  that has n as head)
  '''
  rightmost_child = None
  edge_list = sorted(edge_list)
  for e in edge_list:
    if e.dependant <  n:
      continue
    if e.head == n:
      rightmost_child = e.dependant
  return rightmost_child

def get_leftmost_child(n, edge_list):
  '''
  Given an node n and a list of edges returns
  the leftmost child of n (i.e. the smaller node
  that has n as a head)
  '''
  leftmost_child = None
  edge_list = sorted(edge_list)
  for e in edge_list:
    if e.dependant > n:
      break
    if e.head == n:
      leftmost_child = e.dependant
      break
  return leftmost_child
    


class ArcEagerFeatures:
  '''
  Class defining the elements to be used in the trining of the parser.
  '''
  def __init__(self, stack_words, stack_postags, buffer_words, buffer_postags, left_child_words, right_child_words, relation, action):
    self.stack_words = stack_words
    self.stack_postags = stack_postags
    self.buffer_words = buffer_words
    self.buffer_postags = buffer_postags
    self.left_child_words = left_child_words
    self.right_child_words = right_child_words
    self.relation = relation
    self.action = action
    
class ArcEagerConfig:
  '''
  Class defining the configuration of a given oracle. Represents the
  state of the parser at any given time. The information relevant for the
  training of the parser is:
        edit
        => stack: list of words in the stack at a given time
        => buffer: list of words in the buffer at a given time
        => action: action taken at a given time
        => arc   : dependency relationship arc shaped as ((d,h),r) where
                   d is the word being the dependant of the relation, h is the
                   head of the relation and r is the relationship type.
  '''
  def __init__(self, action, arc, stack, buffer):
    self.action     = action
    self.arc        = arc
    self.stack      = stack
    self.buffer     = buffer

  def get_train(self, words, postags, built_arcs, n_stack=2, n_buffer=2, n_lchild=2, n_rchild=2):
    # take the n top words from the stack
    stack = self.stack[-n_stack:] if len(self.stack) >= n_stack else self.stack
    
    # take the n top words from the buffer
    buffer = self.buffer[:n_buffer] if len(self.buffer) >= n_buffer else self.buffer

    # get the words and postags of the stack and buffer
    stack_words = [words[i] for i in stack]
    stack_postags = [postags[i] for i in stack]
    buffer_words = [words[i] for i in buffer]
    buffer_postags = [postags[i] for i in buffer]

    r  = self.arc.relation if self.arc is not None else "-"
    a  = self.action

    # get the leftmost or rightmost childs of the n top words in the stack
    # esto esta feo
    left_childs = []
    right_childs = []
    
    for i in range(n_lchild):
      current_word = stack[-i] if len(stack) >= i else None
      if current_word is not None:
        leftmost_child = get_leftmost_child(current_word, built_arcs)
        if leftmost_child is not None and leftmost_child<len(words):
          leftmost_child = get_leftmost_child(leftmost_child, built_arcs)
          if leftmost_child is not None and leftmost_child<len(words):
            left_childs.append(words[leftmost_child])
          else:
            left_childs.append("-NULL-")
        else:
          left_childs.append("-NULL-")
      else:
        left_childs.append("-NULL-")

    for i in range(n_rchild):
      current_word = stack[-i] if len(stack) >= i else None
      if current_word is not None:
        rightmost_child = get_rightmost_child(current_word, built_arcs)
        if rightmost_child is not None and rightmost_child<len(words):
          rightmost_child = get_rightmost_child(rightmost_child, built_arcs)
          if rightmost_child is not None and rightmost_child<len(words):
            right_childs.append(words[rightmost_child])
          else:
            right_childs.append("-NULL-")
        else:
          right_childs.append("-NULL-")
      else:
        right_childs.append("-NULL-")     
        

    config = ArcEagerFeatures(stack_words, stack_postags, buffer_words, buffer_postags, left_childs, right_childs, r, a)
    return config

  def __repr__(self):
    if self.action is not None:
      action = action_dict[self.action]
    else:
      action = "-"
    
    return str(self.stack)+" "+str(self.buffer)+"  ACTION: "+action+"  ARC: "+str(self.arc)
