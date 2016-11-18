"""
file: mgtdbp-dev.py
      minimalist grammar top-down beam parser, development version, port for
      python 3.1.

   (JP: This is part of effort to make a functionally equivalent but easier to read version of mgtdbp as a Kataja plugin. 
   )

   This is a working, development version, with print routines and examples.

   Using the included packages derived of NLTK (which officially is supported only
   for Python 2.7) in the file nltktreeport.py, and the heapq_mod.py file as well,
   you can start the read-parse loop by typing in the terminal window
   (assumming your system defaults on 3.1):
   
            python mgtdbp-dev.py [grammar] [startCategory] [minimumProbability]
            
   For example:
            python mgtdbp-dev.py mg0 C 0.0001
            
   (To get line mode editing, I start the loop with: rlwrap python mgtdbp-dev.py)
   The loop is started by the command at the bottom of this file.
   Python will prompt you for a sentence to parse with the grammar specified in that line.
   If the grammar is mg0, for example, you could type one of:
            the king prefers the beer
            which queen says the king knows which wine the queen prefers
   Then you will be given a prompt to which you can type
            h
   to get a list of the options.

   This file extends mgtdb-dev.py to a parser, by keeping a tree in each partial analysis.
   Note that, although this is a TD parser, derivation nodes are not expanded left-to-right,
     so we record their positions with indices (similar to indexing of predicted cats).
     To each indexed category (iCat) we add its own dtree node index,
        and we also add a list of the features checked in its own projection so far.
     To each derivation (der) we add its list of indexed category dtree node indices.
     In each step of the derivation, we extend the parents node index,
        putting the results into the derivation list, and in the respective children.

   So each indexed category ic = (i,c,dt) where dt is a "dtuple", that is:
       (Fs checked so far, index of current node, array of Fs moving elements).
   Here, dtuples are never checked during the parse, but they could be used to influence
   probability assignments at each step.

   For the moment, we compute just the most probable parse,
        using a uniform distribution at each choice point,
        returning the derivation (as a "dnode list", or else error,
        instead of just true or false as the recognizer does.
   TODO: implement more sophisticated pruning rule (cf Roark) and more sophisticated
        determination of which trees should be returned.

   * For cats that lack subtrees in lex, tA is not set. This does not matter,
     but we could probably get rid of tA altogether.
   * sA sets all features, but lA only needs non-empty lexTrees.
   * We might speed things up by numbering all subtrees of the lexArray,
     and using int list arrays to encode the whole lexTree.

Comments welcome: stabler@ucla.edu
Modified for py3.1 compatibility by: Erik V Arrieta. || Last modified: 9/15/12
"""
import sys
from nltktreeport import (TreeView, Tree, CanvasWidget, TextWidget,
                AbstractContainerWidget, BoxWidget, OvalWidget, ParenWidget,
                ScrollWatcherWidget)    #added EA

import heapq_mod    #added EA
import heapq
import time
import io
import pprint

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

"""
OK, we now begin implementing the beam parser. 
We will number the categories so that we have only integer comparisons
at runtime, and we can use those integers as position indices.

The human readable form of the grammar:
"""

"""
The following grammars will be included in the package as seperate files, and if run
as is currently configured, it relies on an external file mg0.py as a grammar input
-EA

mg0 = [ ([],[('sel','V'),('cat','C')]),
        ([],[('sel','V'),('pos','wh'),('cat','C')]),
        (['the'],[('sel','N'),('cat','D')]), 
        (['which'],[('sel','N'),('cat','D'),('neg','wh')]), 
        (['king'],[('cat','N')]),
        (['queen'],[('cat','N')]),
        (['wine'],[('cat','N')]),
        (['beer'],[('cat','N')]),
        (['drinks'],[('sel','D'),('sel','D'),('cat','V')]),
        (['prefers'],[('sel','D'),('sel','D'),('cat','V')]),
        (['knows'],[('sel','C'),('sel','D'),('cat','V')]),
        (['says'],[('sel','C'),('sel','D'),('cat','V')])
        ]

mg1 = [ ([],[('sel','V'),('cat','C')]), # = mg0 but without wh features, so no move
        (['the'],[('sel','N'),('cat','D')]), 
        (['king'],[('cat','N')]),
        (['queen'],[('cat','N')]),
        (['wine'],[('cat','N')]),
        (['beer'],[('cat','N')]),
        (['drinks'],[('sel','D'),('sel','D'),('cat','V')]),
        (['prefers'],[('sel','D'),('sel','D'),('cat','V')]),
        (['knows'],[('sel','C'),('sel','D'),('cat','V')]),
        (['says'],[('sel','C'),('sel','D'),('cat','V')])
        ]

mg2 = [ ([],[('sel','V'),('cat','C')]), # = mg1 but without specs, so no merge2
        (['the'],[('sel','N'),('cat','D')]), 
        (['king'],[('cat','N')]),
        (['queen'],[('cat','N')]),
        (['wine'],[('cat','N')]),
        (['beer'],[('cat','N')]),
        (['drinks'],[('sel','D'),('cat','V')]),
        (['prefers'],[('sel','D'),('cat','V')]),
        (['knows'],[('sel','C'),('cat','V')]),
        (['says'],[('sel','C'),('cat','V')])
        ]

# mgxx defines the copy language {ww| w\in{a,b}*}
# this grammar has lots of local ambiguity, and does lots of movement
#  (more than in any human language, I think)
#  so it gives the parser a good workout.
mgxx= [ ([],[('cat', 'T'),('neg','r'),('neg','l')]), 
        ([],[('sel','T'),('pos','r'),('pos','l'),('cat','T')]),
        (['a'],[('sel','T'),('pos','r'),('cat', 'A'),('neg','r')]), 
        (['b'],[('sel','T'),('pos','r'),('cat', 'B'),('neg','r')]), 
        (['a'],[('sel','A'),('pos','l'),('cat', 'T'),('neg','l')]),
        (['b'],[('sel','B'),('pos','l'),('cat', 'T'),('neg','l')]) 
        ] 

"""
ftype_map = {'cat':'', 'sel':'=', 'neg': '-', 'pos': '+'}
ftypes = ['cat', 'sel', 'neg', 'pos']


class Feature:

    def __init__(self, ftype, value):
        self.ftype = ftype
        self.value = value

    def __str__(self):
        return ftype_map[self.ftype] + self.value

    #def __repr__(self):
    #    return 'F(%s, %s)' % (self.ftype, self.value)

    def __repr__(self):
        return ftype_map[self.ftype] + self.value

    def __eq__(self, other):
        return self.ftype == other.ftype and self.value == other.value

# it remains to be seen if there should be both LexItems and LexTreeNodes or if LexTreeNodes are all we need.  

class LexItem:

    def __init__(self, words, features):
        self.words = words
        self.features = features
        self.rev_features = reversed(features)

    def __str__(self):
        return ' '.join(self.words) + '::' + ' '.join([str(f) for f in self.features])

class LexTreeNode:
    def __init__(self, key):
        self.key = key # these are Features 
        self.children = []
        self.roots = []

    def feature_in_children(self, value):
        for child in self.children:
            if child.key and child.key.value == value:
                return child

    def __repr__(self):
        return 'LN(key=%r, children=%r, roots=%r)' % (self.key, self.children, self.roots)

class IndexedCategory:
    def __init__(self, head, movers, head_path, mover_paths, dt):
        self.head = head
        self.movers = movers
        self.head_path = head_path
        self.mover_paths = mover_paths
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
        return IndexedCategory(self.head, self.movers.copy(), self.head_path[:], self.mover_paths.copy(), self.dt.copy())

    def __repr__(self):
        return 'IndexedCategory(head=%r, m=%r, head_path=%r, mover_paths=%r, dt=%r)' % (self.head, self.movers, self.head_path, self.mover_paths, self.dt)

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
    def __init__(self, features, path, moving_features):
        self.features = features
        self.path = path
        self.moving_features = moving_features

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
        return (self.probability, self.input, self.prediction_queue) < (self.probability, self.input, self.prediction_queue)

class DerivationNode:
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

class Grammar:

    def __init__(self, lex_tuples, start, min_p, sentence):
        print('****** Starting Grammar *******')
        self.d = []
        self.min_p = min_p
        self.lex = {}
        # Read LIs and features from grammar. 
        # Also prepare integer representations of features and give each LexItem a reversed 
        # feature list
        feature_values = set()
        for words, feature_tuples in lex_tuples:
            features = []
            for ftype, value in feature_tuples:
                feat = Feature(ftype, value)
                feature_values.add(value)
                features.append(feat)
            self.d.append(LexItem(words, features))
        self.build_lex_trees()

        # Preparing iq-list
        tree_size = len(feature_values)
        head = self.lex[start]
        m = {}
        mover_paths = {}
        features = [Feature('cat', start)] # for derivation tree
        path = [] # for derivation tree
        moving_features = {} # for derivation tree
        dt = DerivationTree(features,path,moving_features)      # for derivation tree
        prediction = IndexedCategory(head, m, [], mover_paths, dt) # dt = dtuple for derivation tree
        iq = [prediction]
        heapq_mod.heapify(iq)   #modifed EA
        return self.auto_runner(sentence, iq)

    def show(self, out):
        for item in self.d:
            out.write(str(item))

    def __str__(self):
        return str(self.d)

    def build_lex_trees(self):
        base = LexTreeNode(None)
        for lexitem in self.d:
            node = base
            for f in lexitem.rev_features:
                found = False
                for child in node.children:
                    if child.key == f:
                        found = True
                        node = child
                        break                        
                if not found:
                    new_node = LexTreeNode(f)
                    node.children.append(new_node)
                    node = new_node
            node.roots.append(lexitem.words) 
        self.lex_trees = base.children
        for node in self.lex_trees:
            self.lex[node.key.value] = node

    def auto_runner(self, sentence, iq):
        #gA = (sA, lA, tA) = self.feature_values, self.lex_array, self.type_array
        new_iq = iq[:]
        inpt = sentence.split()
        print('inpt =' + str(inpt))  #changed EA
        dq = [Derivation(-1.0, inpt, new_iq, [DerivationNode([])])]
        heapq_mod.heapify(dq)   #changed EA
        t0 = time.time()
        (dnodes, remaining_dq) = self.derive(dq)  #now returns dq 
        #(dnodes, remaining_dq) = derive(gA,minP,dq)  #now returns dq 
        t1 = time.time()
        print(str(t1 - t0) + "seconds") #changed EA
        dt = DTree.dnodes_to_dtree(dnodes)
        print(dt)
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
                return [str(node.key)] + [as_list(x) for x in node.children] + node.roots
            else:
                return node
        return [as_list(y) for y in self.lex_trees]

    def derive(self, dq): # modify this to return dq, so alternative parses can be found (CHECK! :) )
        p = 1.0
        while dq:
            d = heapq_mod.heappop(dq) 
            print('# of parses in beam=%s, p(best parse)=%s' % (len(dq)+1, -1 * d.probability))  #changed EA
            if not (d.prediction_queue or d.input):
                print('parse found')               #changed EA  -- END OF PARSE
                return (d.dnodes, dq)  # success!
            elif d.prediction_queue:
                prediction = heapq_mod.heappop(d.prediction_queue)
                #print(prediction)
                self.new_parses = []
                self.create_expansions_from_head(prediction, d.input)
                if self.new_parses:
                    new_p = d.probability / float(len(self.new_parses))
                    if new_p < self.min_p:
                        self.insert_new_parses(d, new_p, dq)
                    else:
                        print('improbable parses discarded')     #changed EA
        print('no parse found')   #changed EA     #changed EA
        return ([[],([],(['no parse'],[]))], dq) # failure! #return dq now as well (an empty list now) EA

    def create_expansions_from_head(self, prediction, inpt): 
        #ic.row_dump()
        for child in prediction.head.children:       #"for sub-branch in category branch"
            cat = child.key.value
            if child.key.ftype == 'sel':
                if child.roots: 
                    # merge a (non-moving) complement
                    self.merge1(child, cat, prediction)
                    # merge a (moving) complement
                    self.merge3(child, cat, prediction)
                elif child.children: 
                    # merge a (non-moving) specifier
                    self.merge2(child, cat, prediction)
                    # merge a (moving) specifier
                    self.merge4(child, cat, prediction)
            elif child.key.ftype == 'pos': 
                self.move1(child, cat, prediction)
                self.move2(child, cat, prediction)
            else:
                raise RuntimeError('exps')
        for root in prediction.head.roots:
            #the next node is a string node
            self.scan(root, inpt, prediction)

#ftypes = ['cat', 'sel', 'neg', 'pos']

    # merge a (non-moving) complement
    def merge1(self, node, cat, prediction):    
        #print('doing merge1')
        pr0 = prediction.copy() # no movers to lexical head
        pr0.head = node
        pr0.head_path.append(0) 
        pr0.movers = {}
        pr0.mover_paths = {}
        pr0.dt.features.append(Feature('sel', cat)) 
        pr0.dt.path.append(0) 
        pr0.dt.moving_features = {}

        pr1 = prediction.copy() # movers to complement only
        pr1.head = self.lex[cat] 
        pr1.head_path.append(1) 
        pr1.dt.features = [Feature('cat', cat)]
        pr1.dt.path.append(1) 
        self.new_parses.append(Expansion(pr0, pr1))

    # merge a (non-moving) specifier
    def merge2(self, node, cat, prediction):
        #print('doing merge2')
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
    def merge3(self, node, cat, prediction):      
        #print('doing merge3')
        for nxt, m_nxt in prediction.movers.items():
            matching_tree = m_nxt and m_nxt.feature_in_children(cat) #check to see if term is a mover plain and simple
            if matching_tree:
                pr0 = prediction.copy()
                pr0.head = node
                pr0.movers = {}
                pr0.mover_paths = {}      
                pr0.dt.features.append(Feature('sel', cat)) 
                pr0.dt.path.append(0)
                pr0.dt.moving_features = {}

                pr1 = prediction.copy() # movers passed to complement
                pr1.head = matching_tree
                del pr1.movers[nxt] # we used the "next" licensee, so now empty
                pr1.head_path = pr1.mover_paths[nxt]
                del pr1.mover_paths[nxt]
                pr1.dt.features = pr1.dt.moving_features[nxt][:] # movers to complement
                pr1.dt.features.append(Feature('cat', cat)) 
                pr1.dt.path.append(1)
                del pr1.dt.moving_features[nxt]
                self.new_parses.append(Expansion(pr0, pr1))

    # merge a (moving) specifier
    def merge4(self, node, cat, prediction):          
        #print('doing merge4')
        for nxt, m_nxt in prediction.movers.items():
            matching_tree = m_nxt and m_nxt.feature_in_children(cat)
            if matching_tree:
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

    def move1(self, node, cat, prediction):    
        if cat not in prediction.movers:  # SMC
            #print('doing move1')
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

    def move2(self, node, cat, prediction):  
        for mover_cat, mover in prediction.movers.items():
            matching_tree = mover.feature_in_children(cat)
            if matching_tree:
                root_f = matching_tree.key.value # value of rootLabel
                print(root_f, mover_cat)
                if root_f == mover_cat or not prediction.movers.get(root_f, []): # SMC
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
            #print(exp.ics)
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
        self.head_features = []
        self.movers = []
        self.part0 = None
        self.part1 = None
        if dtree.features:
            self.head_features = dtree.features
        elif dtree.label == '*':
            self.part0 = StateTree(dtree.parts[0])
            self.part1 = StateTree(dtree.parts[1])
            self.merge_check()
        elif dtree.label == 'o':
            self.part0 = StateTree(dtree.parts[0])
            self.move_check()

    def merge_check(self):
        headf0, *remainders0 = self.part0.head_features
        headf1, *remainders1 = self.part1.head_features
        if headf0.ftype == 'sel' and headf1.ftype == 'cat' and headf0.value == headf1.value:
            self.head_features = remainders0 
            if remainders1:
                self.movers = [remainders1]
            self.movers += self.part0.movers
            self.movers += self.part1.movers
        else:
            raise RuntimeError('merge_check error')

    def move_check(self):
        mover_match, *remaining = self.part0.head_features 
        self.head_features = remaining
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
        if self.head_features:
            fss.append(self.head_features)
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
        self.head_features = []
        self.movers = []
        self.moving = []
        self.label = ''
        self.part0 = None
        self.part1 = None
        if dtree:
            self.label = dtree.label
            if dtree.features:
                self.head_features = dtree.features
            elif dtree.label == '*':            
                self.part0 = BareTree(dtree.parts[0])
                self.part1 = BareTree(dtree.parts[1])            
                self.moving = [] + self.part0.moving + self.part1.moving
                self.merge_check()
            elif dtree.label == 'o':
                self.part1 = BareTree(dtree.parts[0])
                self.move_check()

    def merge_check(self):
        headf0, *remainders0 = self.part0.head_features
        headf1, *remainders1 = self.part1.head_features
        if headf0.ftype == 'sel' and headf1.ftype == 'cat' and headf0.value == headf1.value:
            self.head_features = remainders0 
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
        mover_match, *remaining = self.part1.head_features 
        self.head_features = remaining
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
            return '%s::%s' % (w, ' '.join([str(f) for f in self.head_features]))
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
        self.head_features = []
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
                self.head_features = dtree.features
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
        headf0, *remainders0 = self.part0.head_features
        headf1, *remainders1 = self.part1.head_features
        if headf0.ftype == 'sel' and headf1.ftype == 'cat' and headf0.value == headf1.value:
            self.head_features = remainders0 # copy remaining head1 features
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
        mover_match, *remaining = self.part1.head_features 
        self.head_features = remaining
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
                    return [self.category, self.label]                   
            else:
                return ([self.label], [])
        elif self.part0 and self.part1: # merge
            return [self.label, self.part0.as_list_tree(), self.part1.as_list_tree()]
        else:
            raise RuntimeError('XBarTree.as_list_tree')


############################################################################################

if __name__ == '__main__':
    import mg0 as grammar
    sentence = "the king prefers the beer"
    sentence = "which king says which queen knows which king says which wine the queen prefers"
    sentence = "which queen says the king knows which wine the queen prefers"
    #sentence = "the king says the queen prefers wine"
    gr = Grammar(grammar.g, 'C', -0.0001, sentence=sentence)
    results = gr.results
    if True:
        for key in sorted(list(results.keys())):
            print(key)
            print(results[key])
