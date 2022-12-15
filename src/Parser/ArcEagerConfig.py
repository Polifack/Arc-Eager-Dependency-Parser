from .constants import *

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
    self.action = action
    self.arc    = arc
    self.stack  = stack
    self.buffer = buffer

  def get_train(self, words, postags, n_stack=2, n_buffer=2):
    # take the n top words from the stack
    stack = self.stack[-n_stack:] if len(self.stack) >= n_stack else self.stack
    
    # take the n top words from the buffer
    buffer = self.buffer[:n_buffer] if len(self.buffer) >= n_buffer else self.buffer

    # get the words and postags of the stack and buffer
    stack_words = [words[i] for i in stack]
    stack_postags = [postags[i] for i in stack]
    buffer_words = [words[i] for i in buffer]
    buffer_postags = [postags[i] for i in buffer]

    r  = self.arc[1] if self.arc is not None else "-"
    a  = self.action

    return {"sw":stack_words, "bw":buffer_words, "sp":stack_postags, "bp":buffer_postags, "r":[r], "a":[a]}


  def __repr__(self):
    if self.action is not None:
      action = action_dict[self.action]
    else:
      action = "-"
    
    return str(self.stack)+" "+str(self.buffer)+"  ACTION: "+action+"  ARC: "+str(self.arc)
