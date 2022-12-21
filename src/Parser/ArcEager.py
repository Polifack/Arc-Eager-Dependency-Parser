from .ArcEagerConfig import ArcEagerConfig
from ..ConllTree.ConllTree import ConllTree
from ..ConllTree.ConllNode import DependencyEdge
from .constants import *

from keras.utils import pad_sequences

import copy
import numpy as np
from datetime import datetime



class ArcEagerParser:
    '''
    Class defining the arc-eager parser state machine. The parser
    is initialized with a sequence length and a ConllTree object
    containing the training data. The parser is then able to
    generate the configurations for the training of the oracle
    and to predict the next action given a configuration.
    
    The parser is initialized with a sequence length. This is the
    number of words and postags to consider in the training of the
    oracle. The parser is then initialized with a ConllTree object
    containing the training data. The parser is then able to generate
    the configurations for the training of the oracle and to predict
    the next action given a configuration.
    '''
  
    def __init__(self, seq_l_s=2, seq_l_b=2, seq_l_lc=2, seq_l_rc=2):    
        # training sequence length (i.e. the number of words/postags to consider)    
        self.seq_l_s        = seq_l_s
        self.seq_l_b        = seq_l_b
        self.seq_l_lc       = seq_l_lc
        self.seq_l_rc       = seq_l_rc
        
        # store words and postags to obtain a configuration to train the oracle
        self.words          = None
        self.postags        = None

        # arc-eager parser state machine fields
        self.conll_tree     = None
        self.target_edges   = None
        self.current_edges  = []
        self.buffer         = []
        self.stack          = []
        self.configuration  = []

    #############
    # INITIALIZE
    #############
    
    def reset(self):
        self.words          = []
        self.postags        = []
        
        self.conll_tree     = None
        self.target_edges   = None
        self.configuration  = []
        self.stack          = []
        self.buffer         = []

    def init_sentence_raw(self, words, postags):
        '''
        Given a raw sentence shaped as a string initializes
        the parser in order to predict it
        '''

        # reset state machine
        self.reset()

        # init sentence data
        self.words          = words
        self.postags        = postags
        self.current_edges  = []

        # init stack and buffer
        self.stack  = [0]
        self.buffer = [i+1 for i,w in enumerate(words[1:])]
        
        self.configuration = [ArcEagerConfig(None, None, self.stack, self.buffer)]
  
    def init_tree(self, conll_tree):
        '''
        Given a ConllTree object initializes the parser with
        its edges and resets the configuration
        '''

        # reset state machine
        self.reset()

        # init conll_tree data
        self.conll_tree    = conll_tree
        self.current_edges = []
        self.target_edges  = conll_tree.get_edges()      # [((0,0),-norel-), ((1,4),root), ((2,4),mark), ...]
        self.words         = conll_tree.get_words()      # [u'ROOT', u'El', u'gobierno', u'central', u'ha', ...]
        self.postags       = conll_tree.get_postags()    # [u'ROOT', u'DT', u'NN', u'JJ', u'VBZ', ...]
        
        # init stack and buffer
        self.stack         = [0]  
        self.buffer        = conll_tree.get_indexes()[1:]

    #############
    # ACTIONS
    #############

    def left_arc(self):
        self.stack.pop()

    def right_arc(self):
        self.stack.append(self.buffer[0])
        self.buffer.remove(self.buffer[0])

    def reduce(self):
        self.stack.pop()

    def shift(self):
        self.stack.append(self.buffer[0])
        self.buffer.remove(self.buffer[0])
  
    #############
    # GET PARSER CONFIG
    #############

    def get_actions_mask(self):
        '''
        Returns a boolean mask of valid actions given the current
        state of the parser
        '''

        valid_actions = [True, True, True, True]

        # without stack we cant reduce
        if len(self.stack) <= 0: valid_actions[A_REDUCE] = False
        
        # avoid dependant already having a head in left arcs
        # avoid setting ROOT as dependant in left arcs
        valid_actions[A_LEFT_ARC] = valid_actions[A_LEFT_ARC] and (not self.check_head_assigned(self.stack[-1]))
        valid_actions[A_LEFT_ARC] = valid_actions[A_LEFT_ARC] and (self.stack[-1] != 0)
  
        # avoid dependant already having a head in right arcs
        valid_actions[A_RIGHT_ARC] = valid_actions[A_RIGHT_ARC] and (not self.check_head_assigned(self.buffer[0]))

        # avoid reduce if top of the stack does not have a head
        valid_actions[A_REDUCE] = valid_actions[A_REDUCE] and self.check_head_assigned(self.stack[-1])

        return valid_actions

    def get_next_action(self):
        '''
        Returns an array with the valid actions to take at any given moment
        '''

        valid_actions = self.get_actions_mask()
        arc = None

        if valid_actions[A_LEFT_ARC]:
            # check right arc is a target arc; if it is, retrieve the arc
            l_arc = (self.stack[-1], self.buffer[0])
            if (l_arc) not in self.conll_tree.get_arcs(): 
                valid_actions[A_LEFT_ARC] = False 
            else:
                arc = self.get_target_arc(l_arc)
        
        if valid_actions[A_RIGHT_ARC]:
            # check left arc is a target arc; if it is, retireve the arc
            r_arc = (self.buffer[0], self.stack[-1])
            if (r_arc) not in self.conll_tree.get_arcs(): 
                valid_actions[A_RIGHT_ARC] = False 
            else:
                arc = self.get_target_arc(r_arc)

        if valid_actions[A_REDUCE]:
          # avoid reducing if top of the stack is still the head for any remaining target arc
          valid_actions[A_REDUCE] = valid_actions[A_REDUCE] and (not self.check_remaining_head(self.stack[-1]))
        
        return valid_actions, arc

    def get_parser_config_list(self, dependency_trees):
        '''
        Gets the arc-eager configuration for a list of
        dependency trees
        '''

        configs = []
        for tree in dependency_trees:
            tree_config = self.get_parser_config(tree)
            for c in tree_config:
                configs.append(c)
        return configs

    def get_parser_config(self, dependency_tree):
        '''
        Gets the arc-eager configuration for a single 
        dependency tree
        '''

        self.init_tree(dependency_tree)
        train_configs = []
        while self.buffer != []:
            # get action
            valid_actions, arc = self.get_next_action()
            action = np.argmax(valid_actions)
            
            # if no more actions can be executed, break
            if np.sum(valid_actions)==0: 
              config = ArcEagerConfig(A_REDUCE, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
              self.configuration.append(config)
              break
            
            # update configuration
            if arc: self.current_edges.append(arc)
            config = ArcEagerConfig(action, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
            self.configuration.append(config)
            train_configs.append(config.get_train(self.words, 
                                                  self.postags, 
                                                  self.current_edges, 
                                                  n_stack=self.seq_l_s, 
                                                  n_buffer=self.seq_l_b,
                                                  n_lchild=self.seq_l_lc,
                                                  n_rchild=self.seq_l_rc))

            # run action
            if action == A_LEFT_ARC: 
                self.left_arc()
            
            elif action == A_RIGHT_ARC: 
                self.right_arc()
            
            elif action == A_REDUCE: 
                self.reduce()
            
            elif action == A_SHIFT: 
                self.shift()
        
        return train_configs

    #############
    # PREDICTION
    #############

    def predict_next_action(self, oracle):
        '''
        Returns an array with the valid actions to take at any given moment
        '''

        current_config = self.configuration[-1].get_train(self.words, 
                                                          self.postags, 
                                                          self.current_edges, 
                                                          n_stack=self.seq_l_s, 
                                                          n_buffer=self.seq_l_b,
                                                          n_lchild=self.seq_l_lc,
                                                          n_rchild=self.seq_l_rc)
        valid_actions, relation  = oracle.predict([current_config])

        # filter the predicted actions with the valid actions
        valid_actions = valid_actions * self.get_actions_mask()
                    
        return valid_actions, relation


    def predict_dependency_tree_list(self, w_list, p_list, model):
        '''
        Predicts the dependency tree for a list of sentences
        '''

        dependency_trees = []
        t1 = datetime.now()
        n_sents = 0
        n_token = 0
        for w, p in zip(w_list, p_list):
            dependency_trees.append(self.predict_dependency_tree(w, p, model))
            n_token += len(w)
            n_sents += 1
        delta = datetime.now() - t1
        
        print("Time: ", delta)
        print("Total tokens: ", n_token)
        print("Total sentences: ", n_sents)
        print("Tokens per second: ", n_token/delta.total_seconds())
        print("Sentences per second: ", n_sents/delta.total_seconds())


        return dependency_trees

    def predict_dependency_tree(self, words, postags, model):
        '''
        Predicts the dependency tree for a single sentence
        '''

        # initialize sentence
        self.init_sentence_raw(words, postags)
        
        while self.buffer!=[]:
            # predict actions list and (if arc made) relation
            valid_actions, rel = self.predict_next_action(model)
            
            # the action will be the one that is non-zero item most on the left of the array (highest priority)
            action = np.argmax(valid_actions)
            arc = None

            # run action
            if action == A_LEFT_ARC:
              arc = DependencyEdge(self.stack[-1], self.buffer[0], rel)
              self.current_edges.append(arc)
              self.left_arc()
            
            elif action == A_RIGHT_ARC:
              arc = DependencyEdge(self.buffer[0], self.stack[-1], rel)
              self.current_edges.append(arc)
              self.right_arc()
            
            elif action == A_REDUCE: 
              self.reduce()
            
            elif action == A_SHIFT: 
              self.shift()
            
            # update configuration
            config = ArcEagerConfig(action, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
            self.configuration.append(config)
        
        return self.build_conll_tree()

    def build_conll_tree(self):
        '''
        Builds a conll tree from the configuration
        '''
        self.conll_tree = ConllTree.build_tree(self.words[1:], self.postags[1:], sorted(self.current_edges))
        return self.conll_tree

    #############
    # EVALUATION
    #############

    def assert_sentence(self, sentence):
        '''
        Given a sentence in ConllU format we assert that
        all its arcs are in the parser configuration
        '''
        sentence_arcs = sorted(sentence.get_edges()[1:])
        config_arcs = sorted([c.arc for c in self.configuration if c.arc is not None])
        
        return (sentence_arcs==config_arcs)
    
    def get_matching_tokens(self, dependency_tree):
        sentence_arcs = sorted(dependency_tree.get_arcs()[1:])
        config_arcs = sorted([c.arc[0] for c in self.configuration if c.arc is not None])
        matching = len([key for key, val in enumerate(sentence_arcs) if val in set(config_arcs)])

        return matching, len(sentence_arcs)
    
    #############
    # AUXILIARS
    #############

    def get_target_arc(self, test_arc):
        '''
        Returns the relationship type of a given arc
        in the target_edges list. After that, removes the
        arc from the list
        ''' 
        for edge in self.target_edges:
            if test_arc==edge.get_arc():
                self.target_edges.remove(edge)
                return edge
        return None
    
    def check_head_assigned(self, n):
        '''
        A node has its head assigned if the head
        is pressent on the 'current edges' list as
        the dependent of some relation.
        
        return any(arc.is_dependent(n) for arc in self.current_edges)
        '''
        for edge in self.current_edges:
            if edge.is_dependant(n):
                return True
        return False

    def check_remaining_head(self, w):
        '''
        A node has remaining heaads if the node 
        is still present on the 'target edges' list
        as the head of some relation.

        return any(arc.is_head(n) for arc in self.target_edges)
        '''
        for edge in self.target_edges:
            if edge.is_head(w):
                return True
        return False