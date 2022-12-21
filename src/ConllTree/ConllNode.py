from .ConllTreeConstants import *

class DependencyEdge:
    '''
    Class that represents a dependency edge between two nodes in a 
    dependency tree and the relation type that relates them. We will
    call a DependencyArc the tuple (dependant, head) and a DependencyEdge
    the tuple (dependant, head, relation).
    
    We say that a dependency edge is left if the
    dependant is to the left of the head, and right otherwise.
    '''
    def __init__(self, dependant, head, relation):
        self.dependant = dependant
        self.head = head
        self.relation = relation

    def is_left_arc(self):
        return self.dependant < self.head
    
    def is_right_arc(self):
        return self.dependant > self.head

    def get_arc(self):
        return (self.dependant, self.head)

    def is_head(self, node):
        return self.head == node
    
    def is_dependant(self, node):
        return self.dependant == node

    def is_root_arc(self):
        return self.head == 0

    def __repr__(self):
        return f'{self.dependant}({self.relation})->{self.head}'

    def __eq__(self, other):
        return self.dependant == other.dependant and self.head == other.head and self.relation == other.relation
    def __gt__(self, other):
        return self.dependant > other.dependant
    def __ge__(self, other):
        return self.dependant >= other.dependant
    def __lt__(self, other):
        return self.dependant < other.dependant
    def __le__(self, other):
        return self.dependant < other.dependant


class ConllNode:
    def __init__(self, wid, form, lemma=None, upos=None, xpos=None, feats=None, head=None, deprel=None, deps=None, misc=None):
        self.id = wid                           # word id
        
        self.form = form if form else "_"       # word 
        self.lemma = lemma if lemma else "_"    # word lemma/stem
        self.upos = upos if upos else "_"       # universal postag
        self.xpos = xpos if xpos else "_"       # language_specific postag
        self.feats = feats if feats else "_"    # morphological features
        
        self.head = head                        # id of the word that depends on
        self.relation = deprel                  # type of relation with head

        self.deps = deps if deps else "_"       # enhanced dependency graph
        self.misc = misc if misc else "_"       # miscelaneous data
    
    def __repr__(self):
        return '\t'.join(str(e) for e in list(self.__dict__.values()))+'\n'

    def get_edge(self):
        return DependencyEdge(self.id, self.head, self.relation)

    @staticmethod
    def from_string(conll_str):
        wid,form,lemma,upos,xpos,feats,head,deprel,deps,misc = conll_str.split('\t')
        return ConllNode(int(wid), form, lemma, upos, xpos, feats, int(head), deprel, deps, misc)

    @staticmethod
    def dummy_root():
        return ConllNode(0, D_POSROOT, None, D_POSROOT, None, None, 0, D_EMPTYREL, None, None)
    
    @staticmethod
    def empty_node():
        return ConllNode(0, None, None, None, None, None, 0, None, None, None)