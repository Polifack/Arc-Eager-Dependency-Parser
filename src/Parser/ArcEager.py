from .ArcEagerConfig import ArcEagerConfig
from ..ConllTree.ConllTree import ConllTree
from .constants import *
import keras
import pickle

from keras.utils import pad_sequences

import copy
import numpy as np


class ArcEagerParser:
    '''
    Class that given a target ConllTree returns the
    actions necesary to build it from the word sentence
    using the ArcEager Transition algorithm
    '''
  
    def __init__(self, seq_l=2):        
        self.n_words          = None
        self.n_postags        = None
        self.n_relations      = None
        self.model            = None
        self.history          = None
        self.word_tokenizer   = None
        self.postag_tokenizer = None
        self.rel_tokenizer    = None
        self.seq_l            = seq_l
        self.conll_tree     = None
        self.target_edges   = None
        self.words          = None
        self.postags        = None
        self.configuration  = []
        self.buffer         = []
        self.stack          = []

    #############
    # INITIALIZE
    #############
    
    def reset(self):
        self.configuration = []
        self.stack = []
        self.buffer = []
        self.words = []
        self.postags = []
        self.conll_tree = None
        self.target_edges = None


    def init_sentence_raw(self, words, postags):
        '''
        Given a raw sentence shaped as a string initializes
        the parser in order to predict it
        '''
        # reset state machine
        self.reset()

        # init sentence data
        self.words  = words
        self.postags = postags

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
        self.target_edges  = conll_tree.get_edges()      # [((0,0),-norel-), ((1,4),root), ((2,4),mark), ...]
        self.words         = conll_tree.get_words()      # [u'ROOT', u'El', u'gobierno', u'central', u'ha', ...]
        self.postags       = conll_tree.get_postags()    # [u'ROOT', u'DT', u'NN', u'JJ', u'VBZ', ...]
        
        # init stack and buffer
        self.stack         = [0]  
        self.buffer        = conll_tree.get_indexes()[1:]

    #############
    # EXECUTIONS
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
    # ORACLE
    #############

    def get_actions_mask(self):
        valid_actions = [True, True, True, True]
        
        # with no buffer we cant shift, r_arc or l_arc
        if len(self.buffer) <= 0:
            valid_actions[A_SHIFT] = False
            valid_actions[A_RIGHT_ARC] = False
            valid_actions[A_LEFT_ARC] = False

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

    def train_on_list(self, dependency_trees):
        '''
        Trains the parser on a list of dependency trees
        '''
        configs = []
        for tree in dependency_trees:
            tree_config = self.train(tree)
            for c in tree_config:
                configs.append(c)
        return configs

    def train(self, dependency_tree):
        # initialize dep tree
        self.init_tree(dependency_tree)
        
        while True:
            # get action
            valid_actions, arc = self.get_next_action()
            action = np.argmax(valid_actions)
            
            # if no more actions can be executed, break
            if np.sum(valid_actions)==0: 
              config = ArcEagerConfig(A_REDUCE, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
              self.configuration.append(config)
              break
            
            # update configuration
            config = ArcEagerConfig(action, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
            self.configuration.append(config)
            
            # run action
            if action == A_LEFT_ARC: 
                self.left_arc()
            
            elif action == A_RIGHT_ARC: 
                self.right_arc()
            
            elif action == A_REDUCE: 
                self.reduce()
            
            elif action == A_SHIFT: 
                self.shift()
        
        return [c.get_train(self.words, self.postags, n=self.seq_l) for c in  self.configuration]


    def predict_next_action(self):
        '''
        Returns an array with the valid actions to take at any given moment
        '''
        if self.model is None:
            if self.model is None:
                print("[*] Error: Model has not been yet created")
            return
        
        ws,wb,ps,pb,_,_ = self.configuration[-1].get_train(self.words, self.postags, n=self.seq_l)

        wsp = pad_sequences(self.word_tokenizer.texts_to_sequences([ws]), maxlen=self.seq_l, padding='pre', truncating='post', value=0)
        wbp = pad_sequences(self.word_tokenizer.texts_to_sequences([wb]), maxlen=self.seq_l, padding='post', truncating='post', value=0)
        
        psp = pad_sequences(self.postag_tokenizer.texts_to_sequences([ps]), maxlen=self.seq_l, padding='pre', truncating='post', value = 0)
        pbp = pad_sequences(self.postag_tokenizer.texts_to_sequences([pb]), maxlen=self.seq_l, padding='post', truncating='post', value = 0)

        x = [wsp, wbp, psp, pbp]
        valid_actions, relation = self.model.predict(x, verbose=False)

        # filter the predicted actions with the valid actions
        valid_actions = valid_actions * self.get_actions_mask()

        # decode relation
        relation = self.rel_tokenizer.sequences_to_texts([[np.argmax(relation)]])
                    
        return valid_actions, relation

    def predict(self, words, postags):
        # initialize sentence
        self.init_sentence_raw(words, postags)
        
        while len(self.stack)>0 or len(self.buffer)>0:
            # predict actions list and (if arc made) relation
            valid_actions, rel = self.predict_next_action()
            
            # the action will be the one that is non-zero item most on the left of the array (highest priority)
            action = np.argmax(valid_actions)
            arc = None

            # if no more actions can be executed, break
            if np.sum(valid_actions)==0: 
              config = ArcEagerConfig(A_REDUCE, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
              self.configuration.append(config)
              break

            # run action
            if action == A_LEFT_ARC:
              arc = ((self.stack[-1], self.buffer[0]), rel)
              self.left_arc()
            
            elif action == A_RIGHT_ARC:
              arc = ((self.buffer[0], self.stack[-1]), rel)
              self.right_arc()
            
            elif action == A_REDUCE: 
              self.reduce()
            
            elif action == A_SHIFT: 
              self.shift()
            
            # update configuration
            config = ArcEagerConfig(action, arc, copy.deepcopy(self.stack), copy.deepcopy(self.buffer))
            self.configuration.append(config)
        
        self.build_conll_tree(self)
        return self.conll_tree

    def predict_file(self, file_path_in, file_path_out):
        '''
        Given a conll-u file with a list of dependency trees
        it predicts the dependency tree for each one of them and
        saves them to file_path
        '''
        trees = ConllTree.read_conllu_file(file_path_in)
        for tree in trees:
            self.predict(tree.get_words(), tree.get_postags())
            self.conll_tree.write_conllu_file(file_path_out)
            self.reset()

    def build_conll_tree(self):
        '''
        Builds a conll tree from the configuration
        '''
        # get the arcs from the configuration
        arcs = []
        for c in self.configuration:
            if c.arc is not None:
                arcs.append(c.arc)

        # build the tree
        self.conll_tree = ConllTree.build_tree(self.words, self.postags, arcs)

    #############
    # EVALUATION
    #############
  
    def get_train_config(self):
        '''
        Returns the configuration formated to train a neural network.
        '''
        # extract required info
        dataset       = []
        
        for c in self.configuration:
            dataset.append(c.get_train(self.words, self.postags, self.seq_l))
        return dataset

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
        for arc in self.target_edges:
            (d,h), r = arc
            if (d, h) == test_arc:
                self.target_edges.remove(arc)
                return arc
        return None
    
    def check_head_assigned(self, w):
        '''
        Checks if a given word has its head already assigned
        in the list of configurations.
        '''
        for config in self.configuration:
            if config.arc is None:
                continue
            
            (d,h),r = config.arc
            if (w==d):
                return True
        return False

    def check_remaining_head(self, w):
        '''
        Checks if a given word is still head to any word
        in the target arcs.
        '''
        for arc in self.target_edges:
            if self.target_edges is None:
                continue
            
            (d,h),r = arc
            if (w==h):
                return True
        return False