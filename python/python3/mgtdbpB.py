"""
file: mgtdbpB.py
      minimalist grammar top-down beam parser, refactored from E. Stabler's original.

   This is part of effort to make an output-equivalent mgtdbp that can be used as a Kataja plugin.
   The code aims for readability and not efficiency, so most of the parameter passings with complex
   lists are turned into objects where necessary parameters can be get by their name and not by their index in ad-hoc lists or tuples.  

   Refactoring by Jukka Purma                                      || 11/21/16 
   mgtdbp-dev Modified for py3.1 compatibility by: Erik V Arrieta. || Last modified: 9/15/12
   mgtdbp-dev by Edward P. Stabler

   mgtdbpA -- Turned most of the complex list parameters for functions to class instances 
   mgtdbpB -- More informative variable names and neater conversion to output trees

   
Comments welcome: jukka.purma--at--gmail.com
"""
import sys
from nltktreeport import (TreeView, Tree, CanvasWidget, TextWidget,
                AbstractContainerWidget, BoxWidget, OvalWidget, ParenWidget,
                ScrollWatcherWidget)    

import heapq_mod 
import time
import io
import pprint
from collections import OrderedDict

"""
   We represent trees with lists, where the first element is the root,
   and the rest is the list of the subtrees.

   First, a pretty printer for these list trees:
"""
def pptreeN(out, n, t): # pretty print t indented n spaces
    if isinstance(t,list) and len(t)>0:
        out.write('\n'+' '*n+'[')
        out.write(str(t[0])), # print root and return
        if len(t[1:])>0:
            out.write(',') # comma if more coming
        for i,subtree in enumerate(t[1:]):  # then print subtrees indented by 4
            pptreeN(out, n+4,subtree)
            if i<len(t[1:])-1:
                out.write(',') # comma if more coming
        out.write(']')
    else:
        out.write('\n'+' '*n)
        out.write(str(t))

def pptree(out, t):
    if len(t)==0: # catch special case of empty tree, lacking even a root
        out.write('\n[]\n')
    else:
        pptreeN(out, 0,t)
        out.write('\n')
"""
example: 
pptree([1, 2, [3, 4], [5, 6]])
pptree(['TP', ['DP', ['John']], ['VP', ['V',['praises']], ['DP', ['Mary']]]])

I have intensionally written this prettyprinter so that the
prettyprinted form is a well-formed term.
"""

"""
   We can convert a list tree to an NLTK tree with the following:           #NLTK
"""
def list_tree_to_nltk_tree(listtree):
    if isinstance(listtree,tuple): # special case for MG lexical items at leaves
        return (' '.join(listtree[0]) + ' :: ' + ' '.join(listtree[1]))
    elif isinstance(listtree,str): # special case for strings at leaves
        return listtree
    elif isinstance(listtree,list) and listtree==[]:
        return []
    elif isinstance(listtree,list):
        subtrees=[list_tree_to_nltk_tree(e) for e in listtree[1:]]
        if subtrees == []:
            return listtree[0]
        else:
            return Tree(listtree[0],subtrees)
    else:
        raise RuntimeError('list_tree_to_nltk_tree')

"""
With an NLTK tree, we can use NLTK tree display:

list_tree_to_nltk_tree(t).draw()
TreeView(t)
"""

class Feature:

    def __init__(self, ftype, value):
        self.ftype = ftype
        self.value = value

    def __str__(self):
        if self.ftype == 'cat':
            return self.value
        elif self.ftype == 'sel':
            return '=' + self.value
        elif self.ftype == 'neg':
            return '-' + self.value
        elif self.ftype == 'pos':
            return '+' + self.value

    #def __repr__(self):
    #    return 'F(%s, %s)' % (self.ftype, self.value)

    def __repr__(self):
        if self.ftype == 'cat':
            return self.value
        elif self.ftype == 'sel':
            return '=' + self.value
        elif self.ftype == 'neg':
            return '-' + self.value
        elif self.ftype == 'pos':
            return '+' + self.value

    def __eq__(self, other):
        return self.ftype == other.ftype and self.value == other.value

class LexItem:
    """ These are more typical LIs. It takes in list of words instead of one string for compability with mgtdbp output. These are not used in parsing at all, they are only to print out the grammar. """

    def __init__(self, words, features):
        self.words = words
        self.features = features

    def __str__(self):
        return ' '.join(self.words) + '::' + ' '.join([str(f) for f in self.features])

class LexTreeNode:    
    def __init__(self, feature):
        self.feature = feature  
        self.nonterminals = [] # LexTreeNodes
        self.terminals = [] 

    def nonterminal_with_feature(self, value):
        for nonterminal in self.nonterminals:
            if nonterminal.feature and nonterminal.feature.value == value:
                return nonterminal

    def __repr__(self):
        return 'LN(key=%r, children=%r, roots=%r)' % (self.feature, self.nonterminals, self.terminals)

    def __str__(self):
        if self.nonterminals:
            return '[%s, [%s]]' % (self.feature, ', '.join([str(x) for x in self.nonterminals]))
        else:
            return '[%s, %s]' % (self.feature, self.terminals)

class Prediction:
    def __init__(self, head, movers=None, head_path=None, mover_paths=None, dt=None):
        self.head = head
        self.movers = movers or {} 
        self.head_path = head_path or []
        self.mover_paths = mover_paths or {}
        self.dt = dt
        self._min_index = None
        self.update_ordering()

    def update_ordering(self):
        """ ICs can be stored in queue if they can be ordered. Original used tuples to provide ordering, we can have an innate property _min_index for the task. It could be calculated for each comparison, but for efficiency's sake do it manually before pushing to queue. """ 
        mini = self.head_path
        for x in self.mover_paths.values():
            if x and x < mini:
                mini = x
        self._min_index = mini

    def copy(self):
        return Prediction(self.head, self.movers.copy(), self.head_path[:], self.mover_paths.copy(), self.dt.copy())

    def __repr__(self):
        return 'Prediction(head=%r, m=%r, head_path=%r, mover_paths=%r, dt=%r)' % (self.head, self.movers, self.head_path, self.mover_paths, self.dt)

    def __getitem__(self, key):
        return self._min_index

    def __lt__(self, other):
        return self._min_index < other._min_index

    def row_dump(self):
        print('h: ', self.head)
        print('m: ', self.movers)
        print('hx: ', self.head_path)
        print('mover_paths: ', self.mover_paths)
        print('dt: ', self.dt)


class DerivationTree:
    def __init__(self, features, path=None, moving_features=None):
        self.features = features
        self.path = path or []
        self.moving_features = moving_features or {}

    def copy(self):
        return DerivationTree(self.features[:], self.path[:], self.moving_features.copy())

    def __repr__(self):
        return 'DerivationTree(features=%r, path=%r, moving_features=%r)' % (self.features, self.path, self.moving_features)


class Expansion:
    def __init__(self, prediction=None, prediction1=None, words=None):
        self.prediction = prediction
        self.prediction1 = prediction1
        self.words = words or []

    def __repr__(self):
        return 'Expansion(prediction=%r, prediction1=%r, words=%r)' % (self.prediction, self.prediction1, self.words)


class Derivation:
    def __init__(self, p, inpt, iqs, dnodes):
        self.probability = p
        self.input = inpt
        self.prediction_queue = iqs
        self.dnodes = dnodes

    def __getitem__(self, key):
        return (self.probability, self.input, self.prediction_queue, self.dnodes)[key]

    def __lt__(self, other):
        return (self.probability, self.input, self.prediction_queue) < (other.probability, other.input, other.prediction_queue)

    def __repr__(self):
        return repr([self.probability, self.input, self.prediction_queue, self.dnodes])

class DerivationNode:
    """ DerivationNodes are constituent nodes that are represented in a queer way: 
        instead of having relations to other nodes, they have 'path', a list of 1:s and 0:s that tells their place in a binary tree. Once there is a list of DerivationNodes, a common tree can be composed from it. """  
    def __init__(self, path, label=None, features=None):
        self.path = path
        self.label = label
        self.features = features

    def __repr__(self):
        return 'DerivationNode(path=%r, label=%r, features=%r)' % (self.path, self.label, self.features)

    def __str__(self):
        if self.label or self.features:
            return '%s, (%s, %s)' % (str(self.path), self.label, self.features)
        else:
            return self.path

    def __lt__(self, other):
        return self.path < other.path

class Parser:

    def __init__(self, lex_tuples, start, min_p, sentence):
        print('****** Starting Parser *******')
        self.d = []
        self.min_p = min_p
        self.lex = OrderedDict()
        self.results = {}
        # Read LIs and features from grammar. 
        feature_values = set()
        base = LexTreeNode(None)
        for words, feature_tuples in lex_tuples:
            features = []
            for ftype, value in feature_tuples:
                feat = Feature(ftype, value)
                feature_values.add(value)
                features.append(feat)
            self.d.append(LexItem(words, features))
            # Build LexTrees
            node = base
            for f in reversed(features):
                found = False
                for nonterminal in node.nonterminals:
                    if nonterminal.feature == f:
                        found = True
                        node = nonterminal
                        break                        
                if not found:
                    new_node = LexTreeNode(f)
                    node.nonterminals.append(new_node)
                    node = new_node
            node.terminals.append(words) 

        for node in base.nonterminals: 
            self.lex[node.feature.value] = node # dict for quick access to starting categories

        success, dnodes = self.parse(start, sentence)
        if success:
            self.print_results(dnodes)
        else:
            print('parse failed, what we have is:')
            print(dnodes)

    def show(self, out):
        for item in self.d:
            out.write(str(item))

    def __str__(self):
        return str(self.d)

    def parse(self, start, sentence):
        # Prepare prediction queue. We have a prediction that the derivation will finish 
        # in a certain kind of category, e.g. 'C'  
        final_features = [Feature('cat', start)] 
        topmost_head = self.lex[start]
        prediction = Prediction(topmost_head, dt=DerivationTree(final_features)) 
        pred_queue = [prediction]
        heapq_mod.heapify(pred_queue)  

        inpt = sentence.split()
        print('inpt =' + str(inpt))  

        # Prepare derivation queue. It gets expanded by derive.  
        dq = [Derivation(-1.0, inpt, pred_queue, [DerivationNode([])])]
        heapq_mod.heapify(dq)   

        # The work is done by derive.
        t0 = time.time()
        success, dnodes, remaining_dq = self.derive(dq) 
        t1 = time.time()
        if success:
            print('parse found') 
        else:
            print('no parse found')
        print(str(t1 - t0) + "seconds") 
        return success, dnodes

    def print_results(self, dnodes):
        dt = DTree.dnodes_to_dtree(dnodes)
        results = {}
        # d -- derivation tree
        results['d'] = list_tree_to_nltk_tree(dt.as_list_tree())
        # pd -- pretty-printed derivation tree
        output = io.StringIO()
        pptree(output, dt.as_list_tree())
        results['pd'] = output.getvalue()
        output.close()
        # s -- state tree
        results['s'] = list_tree_to_nltk_tree(StateTree(dt).as_list_tree())
        # ps -- pretty-printed state tree
        output = io.StringIO()
        pptree(output, StateTree(dt).as_list_tree())
        results['ps'] = output.getvalue()
        output.close()
        # b -- bare tree
        results['b'] = list_tree_to_nltk_tree(BareTree(dt).as_list_tree())
        # pb -- pretty-printed bare tree
        output = io.StringIO()
        pptree(output, BareTree(dt).as_list_tree())
        results['pb'] = output.getvalue()
        output.close()
        # x -- xbar tree
        results['x'] = list_tree_to_nltk_tree(XBarTree(dt).as_list_tree())
        # px -- pretty-printed xbar tree
        output = io.StringIO()
        pptree(output, XBarTree(dt).as_list_tree())
        results['px'] = output.getvalue()
        output.close()
        # pg -- print grammar as items
        output = io.StringIO()
        self.show(output)
        results['pg'] = output.getvalue()
        output.close()
        # l -- grammar as tree
        results['l'] = list_tree_to_nltk_tree(['.'] + self.lex_array_as_list())
        # pl -- pretty-printed grammar as tree
        output = io.StringIO()
        pptree(output, ['.'] + self.lex_array_as_list())   #changed EA
        results['pl'] = output.getvalue()
        output.close()

        self.results = results

    def lex_array_as_list(self):
        def as_list(node):
            if isinstance(node, LexTreeNode):
                return [str(node.feature)] + [as_list(x) for x in node.nonterminals] + node.terminals
            else:
                return node
        return [as_list(y) for y in self.lex.values()]

    def derive(self, derivation_queue): 
        p = 1.0
        while derivation_queue:
            d = heapq_mod.heappop(derivation_queue) 
            print('# of parses in beam=%s, p(best parse)=%s' % (len(derivation_queue) + 1, -d.probability))  
            if not (d.prediction_queue or d.input):
                return True, d.dnodes, derivation_queue # success 
            elif d.prediction_queue:
                prediction = heapq_mod.heappop(d.prediction_queue)
                self.new_parses = []
                self.create_expansions_from_head(prediction, d.input)
                if self.new_parses:
                    new_p = d.probability / len(self.new_parses)
                    if new_p < self.min_p:
                        self.insert_new_parses(d, new_p, derivation_queue)
                    else:
                        print('improbable parses discarded')  
        return False, d.dnodes, derivation_queue # fail

    def create_expansions_from_head(self, prediction, inpt): 
        """ Expand possibilities. If we assume current {prediction}, what are the operations that
         could have lead into it? Prediction has features we know about, and those fix its place in LexTree. The next generation of nodes, {nonterminals}, in LexTree are those that have these and additional features and for each we make a prediction where the nonterminal node got there because of merge or move with something else. 
         All predictions get written into self.new_parses
         """   

        #e.g. if current head is C, nodes are =V, -wh 
        for node in prediction.head.nonterminals:       
            if node.feature.ftype == 'sel':
                if node.terminals: 
                    # merge a (non-moving) complement
                    self.merge1(node, prediction)
                    # merge a (moving) complement
                    self.merge3(node, prediction)
                elif node.nonterminals: 
                    # merge a (non-moving) specifier
                    self.merge2(node, prediction)
                    # merge a (moving) specifier
                    self.merge4(node, prediction)
            elif node.feature.ftype == 'pos': 
                self.move1(node, prediction)
                self.move2(node, prediction)
            else:
                raise RuntimeError('exps')
        for terminal in prediction.head.terminals:
            #the next node is a string node
            self.scan(terminal, inpt, prediction)

    # These operations reverse familiar minimalist operations: external merges, moves and select. 
    # They create predictions of possible child nodes that could have resulted in current head.
    # Predictions are packed into Expansions, and after all Expansions for this head are created, 
    # the good ones are inserted to parse queue by insert_new_parses.  
    # merge a (non-moving) complement
    def merge1(self, node, prediction):    
        """ This reverses a situation when a new element (pr0) is external-merged to existing head (pr1) as a complement. 
        given node is child of current prediction and it is the last feature before the terminals.
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        print('doing merge1')
        #print(node)
        category = node.feature.value
        pr0 = prediction.copy() # no movers to lexical head
        pr0.head = node # one part of the puzzle is given, the other part is deduced from this 
        pr0.head_path.append(0) 
        pr0.movers = {}
        pr0.mover_paths = {}
        pr0.dt.features.append(Feature('sel', category)) 
        pr0.dt.path.append(0) 
        pr0.dt.moving_features = {}

        pr1 = prediction.copy() # movers to complement only
        pr1.head = self.lex[category] # head can be any LI in this category 
        pr1.head_path.append(1) 
        pr1.dt.features = [Feature('cat', category)]
        pr1.dt.path.append(1) 
        self.new_parses.append(Expansion(pr0, pr1))

    # merge a (non-moving) specifier
    def merge2(self, node, prediction):
        """ This reverses a situation when a non-terminal element (pr0) is external-merged to existing head (pr1) as a specifier. 
        node is non-terminal
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        print('doing merge2')
        #print(node)
        cat = node.feature.value
        pr0 = prediction.copy() # movers to head
        pr0.head = node
        pr0.head_path.append(1) 
        pr0.dt.features.append(Feature('sel', cat))
        pr0.dt.path.append(0) 

        pr1 = prediction.copy()
        pr1.head = self.lex[cat]
        pr1.movers = {}
        pr1.head_path.append(0)
        pr1.mover_paths = {}
        pr1.dt.features = [Feature('cat', cat)]
        pr1.dt.path.append(1) 
        pr1.dt.moving_features = {}
        self.new_parses.append(Expansion(pr0, pr1))

    # merge a (moving) complement
    def merge3(self, node, prediction):      
        """ This reverses a situation when a terminal moving element (pr0) is merged to existing head (pr1) as a complement. 
        what we know about {node} is that it is terminal
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        cat = node.feature.value
        for mover_cat, mover in prediction.movers.items(): # look into movers
            matching_tree = mover.nonterminal_with_feature(cat) # matching tree is a child of mover 
            if matching_tree:
                print('doing merge3')
                #print(node)
                # pr0 is prediction about {node}. It is a simple terminal that selects for 
                pr0 = prediction.copy()
                pr0.head = node
                pr0.movers = {}
                pr0.mover_paths = {}      
                pr0.dt.features.append(Feature('sel', cat)) 
                pr0.dt.path.append(0)
                pr0.dt.moving_features = {}

                # pr1 doesn't have certain movers that higher prediction has
                pr1 = prediction.copy() # movers passed to complement
                pr1.head = matching_tree
                del pr1.movers[mover_cat] # we used the licensee, so now empty
                pr1.head_path = pr1.mover_paths[mover_cat]
                del pr1.mover_paths[mover_cat]
                pr1.dt.features = pr1.dt.moving_features[mover_cat][:] # movers to complement
                pr1.dt.features.append(Feature('cat', cat)) 
                pr1.dt.path.append(1)
                del pr1.dt.moving_features[mover_cat]
                self.new_parses.append(Expansion(pr0, pr1))

    # merge a (moving) specifier
    def merge4(self, node, prediction):          
        """ This reverses a situation when a non-terminal element (pr0) is merged to existing head (pr1) as a specifier. 
        node is non-terminal
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        cat = node.feature.value
        for nxt, m_nxt in prediction.movers.items():
            matching_tree = m_nxt.nonterminal_with_feature(cat)
            if matching_tree:
                print('doing merge4')
                #print(node)
                # pr0 doesn't have certain movers that higher prediction has
                pr0 = prediction.copy()
                pr0.head = node
                del pr0.movers[nxt] # we used the "next" licensee, so now empty
                del pr0.mover_paths[nxt]
                pr0.dt.features.append(Feature('sel', cat)) 
                pr0.dt.path.append(0) 
                del pr0.dt.moving_features[nxt]

                pr1 = prediction.copy() # movers passed to complement
                pr1.head = matching_tree
                pr1.movers = {}
                pr1.head_path = pr1.mover_paths[nxt]
                pr1.mover_paths = {} 
                pr1.dt.features = pr1.dt.moving_features[nxt][:] # copy
                pr1.dt.features.append(Feature('cat', cat))
                pr1.dt.path.append(1) 
                pr1.dt.moving_features = {}                
                self.new_parses.append(Expansion(pr0, pr1))

    def move1(self, node, prediction):    
        cat = node.feature.value
        if cat not in prediction.movers:  # SMC
            #print('doing move1')
            print('doing move1')
            #print(node)
            
            pr0 = prediction.copy() 
            pr0.head = node #node is remainder of head branch
            pr0.movers[cat] = self.lex[cat]
            pr0.head_path.append(1)
            pr0.mover_paths[cat] = prediction.head_path[:]
            pr0.mover_paths[cat].append(0)
            pr0.dt.features.append(Feature('pos', cat)) 
            pr0.dt.path.append(0) 
            pr0.dt.moving_features[cat] = [Feature('neg', cat)] # begin new mover with (neg cat)
            self.new_parses.append(Expansion(pr0))

    def move2(self, node, prediction):  
        cat = node.feature.value
        for mover_cat, mover in prediction.movers.items(): # <-- look into movers
            matching_tree = mover.nonterminal_with_feature(cat) # ... for category shared with prediction
            if matching_tree:
                root_f = matching_tree.feature.value # value of rootLabel
                assert(root_f == cat)
                print(root_f, mover_cat)
                if root_f == mover_cat or not prediction.movers.get(root_f, []): # SMC
                    print('doing move2')
                    print(node)
                    #print('doing move2')
                    mts = matching_tree #matchingTree[1:][:]
                    pr0 = prediction.copy()
                    pr0.head = node
                    del pr0.movers[mover_cat] # we used the "next" licensee, so now empty
                    pr0.movers[root_f] = mts
                    pr0.mover_paths[root_f] = pr0.mover_paths[mover_cat][:]
                    del pr0.mover_paths[mover_cat]
                    pr0.dt.moving_features[root_f] = pr0.dt.moving_features[mover_cat][:]
                    pr0.dt.moving_features[root_f].append(Feature('neg', cat)) # extend prev features of mover with (neg cat)
                    del pr0.dt.moving_features[mover_cat]
                    pr0.dt.features.append(Feature('pos', cat)) 
                    pr0.dt.path.append(0)
                    self.new_parses.append(Expansion(pr0))

    def scan(self, words, inpt, prediction):
        # this actually checks to see if word is present in given sentence
        if not any(self.new_parses) and inpt[:len(words)] == words:             
            #print('ok scan')
            new_prediction = prediction.copy()
            new_prediction.head = []
            new_prediction.head_path = []
            self.new_parses.append(Expansion(new_prediction, words=words))

    def insert_new_parses(self, d, new_p, dq):
        for exp in self.new_parses:
            dnodes = d.dnodes[:]
            # scan is a special case, identifiable by empty head
            if not exp.prediction.head:
                label = exp.words
                features = exp.prediction.dt.features
                features.reverse()
                path = exp.prediction.dt.path
                dnode = DerivationNode(path, label, features)
                dnodes.append(dnode)
                if d.input[:len(label)] == label:
                    remainder_input = d.input[len(label):]
                else:
                    remainder_input = d.input[:]
                new_parse = Derivation(d.probability, remainder_input, d.prediction_queue, dnodes)
            else: # put indexed categories ics onto iq with new_p
                pred_queue = d.prediction_queue[:]
                dnodes.append(DerivationNode(exp.prediction.dt.path))
                exp.prediction.update_ordering()
                heapq_mod.heappush(pred_queue, exp.prediction) 
                if exp.prediction1:
                    dnodes.append(DerivationNode(exp.prediction1.dt.path))
                    exp.prediction1.update_ordering()
                    heapq_mod.heappush(pred_queue, exp.prediction1) 
                new_parse = Derivation(new_p, d.input, pred_queue, dnodes)
            heapq_mod.heappush(dq, new_parse) 

#### Output trees ########

class DTree:
    """ Basic constituent tree, base for other kinds of trees. """
    def __init__(self, label='', features=None, parts=None):
        self.label = label or []
        self.features = features or []
        self.parts = parts or []

    def __repr__(self):
        return '[%r, %r]' % (self.label, self.features or self.parts)

    def build_from_dnodes(self, parent_path, dnodes, terminals):
        if terminals and terminals[0].path == parent_path:
            leaf = terminals.pop(0)
            self.label = ' '.join(leaf.label)
            self.features = leaf.features
        elif dnodes and parent_path == dnodes[0].path[0:len(parent_path)]:
            root = dnodes.pop(0)
            child0 = DTree() 
            child0.build_from_dnodes(root.path, dnodes, terminals)
            self.parts.append(child0)
            if dnodes and parent_path == dnodes[0].path[0:len(parent_path)]:
                self.label = '*'  
                root1 = dnodes.pop(0)
                child1 = DTree()
                child1.build_from_dnodes(root1.path, dnodes, terminals)
                self.parts.append(child1)
            else:
                self.label = 'o' 
        else:
            raise RuntimeError('build_from_dnodes: error')

    def as_list_tree(self):
        if len(self.parts) == 2:            
            return [self.label, self.parts[0].as_list_tree(), self.parts[1].as_list_tree()]
        elif len(self.parts) == 1:
            return [self.label, self.parts[0].as_list_tree()]            
        elif self.features:
            if self.label:
                label = [self.label]
            else:
                label = []
            return (label, [str(f) for f in self.features])

    @staticmethod
    def dnodes_to_dtree(dnodes):
        nonterms = []
        terms = []
        for dn in dnodes:
            if dn.label or dn.features:  
                terms.append(dn)
            else:
                nonterms.append(dn)
        if len(nonterms) == 0:
            raise RuntimeError('buildIDtreeFromDnodes: error')
        else:
            terms.sort()
            nonterms.sort()
            root = nonterms.pop(0)
            dtree = DTree()
            dtree.build_from_dnodes(root.path, nonterms, terms)
            if terms or nonterms:  
                print('dnodes_to_dtree error: unused derivation steps')
                print('terms=' + str(terms))  
                print('nonterms='+ str(nonterms))  
            return dtree



class StateTree:
    """
    convert derivation tree to state tree
    """
    def __init__(self, dtree):        
        self.features = []
        self.movers = []
        self.part0 = None
        self.part1 = None
        if dtree.features:
            self.features = dtree.features
        elif dtree.label == '*':
            self.part0 = StateTree(dtree.parts[0])
            self.part1 = StateTree(dtree.parts[1])
            self.merge_check()
        elif dtree.label == 'o':
            self.part0 = StateTree(dtree.parts[0])
            self.move_check()

    def merge_check(self):
        headf0, *remainders0 = self.part0.features
        headf1, *remainders1 = self.part1.features
        if headf0.ftype == 'sel' and headf1.ftype == 'cat' and headf0.value == headf1.value:
            self.features = remainders0 
            if remainders1:
                self.movers = [remainders1]
            self.movers += self.part0.movers
            self.movers += self.part1.movers
        else:
            raise RuntimeError('merge_check error')

    def move_check(self):
        mover_match, *remaining = self.part0.features 
        self.features = remaining
        found = False
        mover = []
        self.movers = []
        for mover_f_list in self.part0.movers:
            if mover_f_list[0].value == mover_match.value:
                if found:
                    raise RuntimeError('SMC violation in move_check')
                mover = mover_f_list[1:]
                found = True
            else:
                self.movers.append(mover_f_list)
        assert(found)
        if mover:
            self.movers.append(mover)

    def as_list_tree(self):
        fss = []
        if self.features:
            fss.append(self.features)
        fss += self.movers
        sfs = ','.join([' '.join([str(f) for f in fs]) for fs in fss])
        if self.part0 and self.part1: # merge
            return [sfs, self.part0.as_list_tree(), self.part1.as_list_tree()]
        elif self.part0: # move
            return [sfs,self.part0.as_list_tree()]
        else: # leaf
            return [sfs]


class BareTree:
    """
    convert derivation tree to bare tree
    """
    def __init__(self, dtree):        
        self.features = []
        self.movers = []
        self.moving = []
        self.label = ''
        self.part0 = None
        self.part1 = None
        if dtree:
            self.label = dtree.label
            if dtree.features:
                self.features = dtree.features
            elif dtree.label == '*':            
                self.part0 = BareTree(dtree.parts[0])
                self.part1 = BareTree(dtree.parts[1])            
                self.moving = [] + self.part0.moving + self.part1.moving
                self.merge_check()
            elif dtree.label == 'o':
                self.part1 = BareTree(dtree.parts[0])
                self.move_check()

    def merge_check(self):
        headf0, *remainders0 = self.part0.features
        headf1, *remainders1 = self.part1.features
        if headf0.ftype == 'sel' and headf1.ftype == 'cat' and headf0.value == headf1.value:
            self.features = remainders0 
            self.movers = self.part0.movers + self.part1.movers
            if remainders1:
                self.movers.append(remainders1)
                self.moving.append((remainders1, self.part1))
                self.part1 = BareTree(None) # trace
        else:
            raise RuntimeError('merge_check error')
        if not (self.part0.part0 or self.part0.part1): # is it leaf?
            self.label = '<'
        else:
            self.label = '>' # switch order to part1, part0 
            temp = self.part0
            self.part0 = self.part1
            self.part1 = temp

    def move_check(self):
        mover_match, *remaining = self.part1.features 
        self.features = remaining
        found = False
        mover = []
        self.movers = []
        for mover_f_list in self.part1.movers:
            if mover_f_list[0].value == mover_match.value:
                if found:
                    raise RuntimeError('SMC violation in move_check')
                mover = mover_f_list[1:]
                found = True
            else:
                self.movers.append(mover_f_list)
        assert(found)
        self.moving = []
        for (fs, moving_tree) in self.part1.moving:
            if fs[0].value == mover_match.value:
                self.part0 = moving_tree
            else:
                self.moving.append((fs, moving_tree))
        if mover:
            self.movers.append(mover)
            self.moving.append((mover, self.part0))
        self.label = '>'
        assert(self.part0)

    def as_list_tree(self):
        if not (self.part0 or self.part1):
            if isinstance(self.label, list):
                w = ' '.join(self.label)
            else:
                w = self.label
            return '%s::%s' % (w, ' '.join([str(f) for f in self.features]))
        elif self.part0 and self.part1: # merge
            return [self.label, self.part0.as_list_tree(), self.part1.as_list_tree()]
        else:
            raise RuntimeError('BareTree.as_list_tree')


class XBarTree:
    """
    convert derivation tree to X-bar tree -
      similar to the bare tree conversion
    """

    def __init__(self, dtree, cntr=0, top=True):        
        self.features = []
        self.movers = []
        self.label = ''
        self.part0 = None
        self.part1 = None
        self.moving = []
        self.category = ''
        self.cntr = cntr
        self.lexical = False
        if dtree:
            self.label = dtree.label
            if dtree.features:
                self.features = dtree.features
                self.lexical = True
                for f in dtree.features:
                    if f.ftype == 'cat':
                        self.category = f.value
                        break
                assert(self.category)

            elif dtree.label == '*':            
                self.part0 = XBarTree(dtree.parts[0], self.cntr, top=False)
                self.part1 = XBarTree(dtree.parts[1], self.part0.cntr, top=False)            
                self.moving = [] + self.part0.moving + self.part1.moving
                self.merge_check()
            elif dtree.label == 'o':
                self.part1 = XBarTree(dtree.parts[0], self.cntr, top=False)
                self.cntr = self.part1.cntr
                self.move_check()
        if top:
            self.label = self.category + 'P'

    def merge_check(self):
        headf0, *remainders0 = self.part0.features
        headf1, *remainders1 = self.part1.features
        if headf0.ftype == 'sel' and headf1.ftype == 'cat' and headf0.value == headf1.value:
            self.features = remainders0 # copy remaining head1 features
            self.movers = self.part0.movers + self.part1.movers # add movers1 and 2
            self.cntr = self.part1.cntr
            if remainders1:
                self.movers.append(remainders1)
                new_label = '%sP(%s)' % (self.part1.category, self.part1.cntr)
                trace = XBarTree(None, top=False) 
                trace.category = self.part1.category
                trace.label = new_label
                self.part1.label = new_label
                self.cntr += 1
                self.moving.append((remainders1, self.part1))
                self.part1 = trace
            elif self.part1.lexical:
                self.part1.category += 'P'
            else:
                self.part1.label = self.part1.category + 'P'
        else:
            raise RuntimeError('merge_check error')
        self.category = self.part0.category
        self.label = self.category + "'"
        if not self.part0.lexical:            
            temp = self.part0
            self.part0 = self.part1
            self.part1 = temp

    def move_check(self):
        mover_match, *remaining = self.part1.features 
        self.features = remaining
        found = False
        mover = []
        self.movers = []
        for mover_f_list in self.part1.movers:
            if mover_f_list[0].value == mover_match.value:
                if found:
                    raise RuntimeError('SMC violation in move_check')
                mover = mover_f_list[1:]
                found = True
            else:
                self.movers.append(mover_f_list)
        assert(found)
        self.moving = []
        for (fs, moving_tree) in self.part1.moving:
            if fs[0].value == mover_match.value:
                self.part0 = moving_tree
            else:
                self.moving.append((fs, moving_tree))
        if mover:
            self.movers.append(mover)
            self.moving.append((mover, self.part0))
        self.category = self.part1.category
        self.label = self.category + "'"
        assert(self.part0)

    def as_list_tree(self):
        if not (self.part0 or self.part1):
            if self.lexical:
                if self.label and isinstance(self.label, str):
                    return [self.category, [self.label]]
                else:
                    return [self.category, [self.label]]                   
            else:
                return ([self.label], [])
        elif self.part0 and self.part1: # merge
            return [self.label, self.part0.as_list_tree(), self.part1.as_list_tree()]
        else:
            raise RuntimeError('XBarTree.as_list_tree')


############################################################################################

if __name__ == '__main__':
    import mg0 as grammar
    sentences = ["the king prefers the beer", 
        "which king says which queen knows which king says which wine the queen prefers",
        "which queen says the king knows which wine the queen prefers",
        "which wine the queen prefers",
    ]
    sentences = ["which wine the queen prefers"]
    t = time.time()
    for s in sentences:
        gr = Parser(grammar.g, 'C', -0.0001, sentence=s)
        results = gr.results
        if False:
            for key in sorted(list(results.keys())):
                print(key)
                print(results[key])
    print(time.time()-t)