from .ConllNode import ConllNode
from .ConllTreeConstants import *

class ConllTree:
    def __init__(self, nodes):
        self.nodes = nodes
    
    def get_node(self, id):
        return self.nodes[id-1]

    def get_edges(self):
        '''
        Return sentence dependency edges as a tuple 
        shaped as ((d,h),r) where d is the dependant of the relation,
        h the head of the relation and r the relationship type
        '''
        return list(map((lambda x :((x.id, x.head), x.relation)), self.nodes))
    
    def get_arcs(self):
        '''
        Return sentence dependency edges as a tuple 
        shaped as (d,h) where d is the dependant of the relation,
        and h the head of the relation.
        '''
        return list(map((lambda x :(x.id, x.head)), self.nodes))

    def get_relations(self):
        '''
        Return a list of relationships betwee nodes
        '''
        return list(map((lambda x :x.relation), self.nodes))
    
    def get_sentence(self):
        '''
        Return the sentence as a string
        '''
        return " ".join(list(map((lambda x :x.form), self.nodes)))

    def get_words(self):
        '''
        Returns the words of the sentence as a list
        '''
        return list(map((lambda x :x.form), self.nodes))

    def get_indexes(self):
        '''
        Returns a list of integers representing the words of the 
        dependency tree
        '''
        return list(map((lambda x :x.id), self.nodes))

    def get_postags(self):
        '''
        Returns the part of speech tags of the tree
        '''
        return list(map((lambda x :x.upos), self.nodes))

    def get_lemmas(self):
        '''
        Returns the lemmas of the tree
        '''
        return list(map((lambda x :x.lemma), self.nodes))

    def get_heads(self):
        '''
        Returns the heads of the tree
        '''
        return list(map((lambda x :x.head), self.nodes))

    
    def append_node(self, node):
        '''
        Append a node to the tree and sorts the nodes by id
        '''
        self.nodes.append(node)
        self.nodes.sort(key=lambda x: x.id)


    def update_head(self, node_id, head_value):
        '''
        Update the head of a node indicated by its id
        '''
        for node in self.nodes:
            if node.id == node_id:
                node.head = head_value
                break
    
    def update_relation(self, node_id, relation_value):
        '''
        Update the relation of a node indicated by its id
        '''
        for node in self.nodes:
            if node.id == node_id:
                node.relation = relation_value
                break
    

    def is_projective(self):
        '''
        Returns a boolean indicating if the dependency tree
        is projective (i.e. no edges are crossing)
        '''
        arcs = self.get_arcs()
        for (i,j) in arcs:
            for (k,l) in arcs:
                if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
                    return False
        return True
    

    def postprocess_tree(self, root_uniqueness=True, search_root_strat=D_ROOT_HEAD):
        '''
        Postprocess the tree by finding the root according to the selected 
        strategy and fixing cycles and out of bounds heads
        '''
        # 1) Find the root
        root = self.root_search(search_root_strat)
        # 2) Fix oob heads
        self.fix_oob_heads()
        # 3) Fix cycles
        self.fix_cycles(root)
        
        # 4) Set all null heads to root
        for node in self.nodes:
            if node.id == root:
                continue
            if node.head == D_NULLHEAD:
                node.head = root
            if root_uniqueness and node.head == 0:
                node.head = root

    def root_search(self, search_root_strat):
        '''
        Search for the root of the tree using the method indicated
        '''
        root = 1 # Default root
        for node in self.nodes:    
            if search_root_strat == D_ROOT_HEAD and node.head == 0:
                root = node.id
                return root
            
            elif search_root_strat == D_ROOT_REL and (node.rel == 'root' or node.rel == 'ROOT'):
                root = node.id
                return root
        
        # ensure root
        self.nodes[root].head = 0
        self.nodes[root].relation = 'root'

        return root

    def fix_oob_heads(self):
        '''
        Fixes heads of the tree (if they dont exist, if they are out of bounds, etc)
        If a head is out of bounds set it to nullhead
        '''
        for node in self.nodes:
            if node.head==D_NULLHEAD:
                continue
            if int(node.head) < 0:
                node.head = D_NULLHEAD
            elif int(node.head) > len(self.nodes):
                node.head = D_NULLHEAD
    
    def fix_cycles(self, root):
        '''
        Breaks cycles in the tree by setting the head of the node to root_id
        '''
        for node in self.nodes:
            visisted = []
            current = node
            while current.head != D_NULLHEAD and (current.id != root):
                if current.head in visisted:
                    current.head = D_NULLHEAD
                    break
                visisted.append(current.head)
                current = self.get_node(current.head)

        
    
    def __repr__(self):
        return "".join(str(e) for e in self.nodes)+"\n"
    
    @staticmethod
    def from_string(conll_str, dummy_root=True, clean_contractions=True, clean_omisions=True):
        '''
        Create a ConllTree from a dependency tree conll-u string.
        '''
        data = conll_str.split('\n')
        dependency_tree_start_index = 0
        for line in data:
            if line[0]!="#":
                break
            dependency_tree_start_index+=1
        data = data[dependency_tree_start_index:]
        nodes = []
        if dummy_root:
            nodes.append(ConllNode.dummy_root())
        
        for line in data:
            # check if not valid line (empty or not enough fields)
            if (len(line)<=1) or len(line.split('\t'))<10:
                continue 
            
            wid = line.split('\t')[0]

            # check if node is a comment (comments are marked with #)
            if "#" in wid:
                continue
            
            # check if node is a contraction (multiexp lines are marked with .)
            if clean_contractions and "-" in wid:    
                continue
            
            # check if node is an omited word (empty nodes are marked with .)
            if clean_omisions and "." in wid:
                continue

            conll_node = ConllNode.from_string(line)
            nodes.append(conll_node)
        
        return ConllTree(nodes)
    
    @staticmethod
    def build_tree(words, postags, relations):
        '''
        Build a ConllTree from a set of words, part of speech tags
        and dependency relations shaped as r = ((d,h),r)
        '''
        nodes = []
        nodes.append(ConllNode.dummy_root())
        for i in range(len(words)):

            # get the relation for the current word
            # if no relation is found, set the head to 0
            r  = list(filter(lambda x: x[0][0]==i+1, relations))
            if len(r)==0:
                h = 0
                dr = 'root'
            else:
                h  = r[0][0][1]
                dr = r[0][1]

            nodes.append(ConllNode(wid=i+1, 
                                    form=words[i], 
                                    upos=postags[i], 
                                    head=h, 
                                    deprel=dr))
        return ConllTree(nodes[1:])

    @staticmethod
    def read_conllu_file(file_path, filter_projective = True, dummy_root = True):
        '''
        Read a conllu file and return a list of ConllTree objects.
        '''
        with open(file_path, 'r') as f:
            data = f.read()
        data = data.split('\n\n')[:-1]
        
        trees = []
        for x in data:
            t = ConllTree.from_string(x, dummy_root=dummy_root)
            if not filter_projective or t.is_projective():
                trees.append(t)
        return trees    

    @staticmethod
    def write_conllu_file(file_path, trees):
        '''
        Write a list of ConllTree objects to a conllu file.
        '''
        with open(file_path, 'w') as f:
            f.write("".join(str(e) for e in trees))