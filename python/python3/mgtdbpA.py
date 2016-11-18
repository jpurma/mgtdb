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
def list2nltktree(listtree):
    if isinstance(listtree,tuple): # special case for MG lexical items at leaves
        return (' '.join(listtree[0]) + ' :: ' + ' '.join(listtree[1]))
    elif isinstance(listtree,str): # special case for strings at leaves
        return listtree
    elif isinstance(listtree,list) and listtree==[]:
        return []
    elif isinstance(listtree,list):
        subtrees=[list2nltktree(e) for e in listtree[1:]]
        if subtrees == []:
            return listtree[0]
        else:
            return Tree(listtree[0],subtrees)
    else:
        raise RuntimeError('list2nltktree')

"""
With an NLTK tree, we can use NLTK tree display:

list2nltktree(t).draw()
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

    def __init__(self, ftype, value, value_list=None):
        self.ftype = ftype
        self.value = value
        if value_list:
            self.int_ftype = ftypes.index(ftype)
            self.int_value = value_list.index(value)
        else:
            self.int_ftype = 0
            self.int_value = 0

    def __str__(self):
        return ftype_map[self.ftype] + self.value

    def __repr__(self):
        return 'F(%s, %s)' % (self.int_ftype, self.int_value)

    def __eq__(self, other):
        return self.int_ftype == other.int_ftype and self.int_value == other.int_value

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

    def feature_in_children(self, i):
        for child in self.children:
            if child.key and child.key.int_value == i:
                return child

    def __repr__(self):
        return 'LN(key=%r, children=%r, roots=%r)' % (self.key, self.children, self.roots)

class IC:
    def __init__(self, h, m, hx, mx, dt):
        self.h = h
        self.m = m
        self.hx = hx
        self.mx = mx
        self.dt = dt

    def min_index(self):
        mini = self.hx
        for x in self.mx:
            if x != [] and x < mini:
                mini = x
        return mini

    def copy(self):
        return IC(self.h, self.m[:], self.hx[:], self.mx[:], self.dt.copy())

    def __repr__(self):
        return 'IC(h=%r, m=%r, hx=%r, mx=%r, dt=%r)' % (self.h, self.m, self.hx, self.mx, self.dt)

    def row_dump(self):
        print('h: ', self.h)
        print('m: ', self.m)
        print('hx: ', self.hx)
        print('mx: ', self.mx)
        print('dt: ', self.dt)


class DT:
    def __init__(self, ifs, dx, mifs):
        self.ifs = ifs
        self.dx = dx
        self.mifs = mifs

    def copy(self):
        return DT(self.ifs[:], self.dx[:], self.mifs[:])

    def __repr__(self):
        return 'DT(ifs=%r, dx=%r, mifs=%r)' % (self.ifs, self.dx, self.mifs)


class Exp:
    def __init__(self, words=None, ics=None):
        self.words = words or []
        self.ics = ics or []

    def __repr__(self):
        return 'Exp(words=%r, ics=%r)' % (self.words, self.ics)


class Grammar:
    ftypes = ['cat', 'sel', 'neg', 'pos']

    def __init__(self, lex_tuples, start, min_p, sentence):
        print('****** Starting Grammar *******')
        self.d = []
        self.min_p = min_p
        self.feature_values = [] # sA
        self.lex_array = []
        self.type_array = []
        # Read LIs and features from grammar. 
        # Also prepare integer representations of features and give each LexItem a reversed 
        # feature list
        for words, feature_tuples in lex_tuples:
            features = []
            for ftype, value in feature_tuples:
                if value not in self.feature_values:
                    self.feature_values.append(value)
                feat = Feature(ftype, value, value_list=self.feature_values)
                features.append(feat)
            self.d.append(LexItem(words, features))
        self.build_lex_trees()
        self.start_int = self.feature_values.index(start)

        # Preparing iq-list
        tree_size = len(self.feature_values)
        h = self.lex_array[self.start_int]
        m = [[]] * tree_size
        mx = [[]] * tree_size
        ifs = [self.create_feature('cat', self.start_int)]    # for derivation tree
        dx = []                 # for derivation tree
        mifs = [[]] * tree_size     # for derivation tree
        dt = DT(ifs,dx,mifs)      # for derivation tree
        ic = IC(h, m, [], mx, dt) # dt = dtuple for derivation tree
        iq = [([],ic)]
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
        self.lex_array = [[]] * len(self.feature_values)
        self.type_array = [0] * len(self.feature_values)
        for node in base.children:
            index = node.key.int_value
            self.lex_array[index] = node
            self.type_array[index] = node.key.int_ftype

    def auto_runner(self, sentence, iq):
        #gA = (sA, lA, tA) = self.feature_values, self.lex_array, self.type_array
        new_iq = iq[:]
        inpt = sentence.split()
        print('inpt =' + str(inpt))  #changed EA
        dq = [(-1.0,inpt,new_iq,[[]])]
        heapq_mod.heapify(dq)   #changed EA
        t0 = time.time()
        (dns, remaining_dq) = self.derive(dq)  #now returns dq 
        #(dns, remaining_dq) = derive(gA,minP,dq)  #now returns dq 
        t1 = time.time()
        print(str(t1 - t0) + "seconds") #changed EA
        dt = dnodes_to_dtree(dns)
        results = {}
        # d -- derivation tree
        results['d'] = list2nltktree(dt.as_tree())
        # pd -- pretty-printed derivation tree
        output = io.StringIO()
        pptree(output, dt.as_tree())
        results['pd'] = output.getvalue()
        output.close()
        # s -- state tree
        results['s'] = list2nltktree(StateTree(dt).as_tree())
        # ps -- pretty-printed state tree
        output = io.StringIO()
        pptree(output, StateTree(dt).as_tree())
        results['ps'] = output.getvalue()
        output.close()
        # b -- bare tree
        results['b'] = list2nltktree(BareTree(dt).as_tree())
        # pb -- pretty-printed bare tree
        output = io.StringIO()
        pptree(output, BareTree(dt).as_tree())
        results['pb'] = output.getvalue()
        output.close()
        # x -- xbar tree
        results['x'] = list2nltktree(XBarTree(dt).as_tree())
        # px -- pretty-printed xbar tree
        output = io.StringIO()
        pptree(output, XBarTree(dt).as_tree())
        results['px'] = output.getvalue()
        output.close()
        # pg -- print grammar as items
        output = io.StringIO()
        self.show(output)
        results['pg'] = output.getvalue()
        output.close()
        # l -- grammar as tree
        results['l'] = list2nltktree(['.'] + self.lex_array_as_list())
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
        return [as_list(y) for y in self.lex_array]

    def create_feature(self, ftype, int_value):
        return Feature(ftype, self.feature_values[int_value], value_list=self.feature_values)

    def derive(self, dq): # modify this to return dq, so alternative parses can be found (CHECK! :) )
        p = 1.0
        while len(dq) > 0:
            (p,inpt,iq,dns) = heapq_mod.heappop(dq)
            print('# of parses in beam=' +str(len(dq)+1)+', p(best parse)=' + str((-1 * p)))  #changed EA
            if len(iq)==0 and len(inpt)==0:
                print('parse found')               #changed EA  -- END OF PARSE
                return (dns, dq)  # success!
            elif len(iq) > 0:
                prediction = heapq_mod.heappop(iq)
                ic = prediction[1]
                self.sofar = []
                self.exps(inpt, ic)
                if len(self.sofar) > 0:
                    new_p = p / float(len(self.sofar))
                    if new_p < self.min_p:
                        self.insert_new_parses(inpt, p, new_p, iq, dq, dns)
                    else:
                        print('improbable parses discarded')     #changed EA
        print('no parse found')   #changed EA     #changed EA
        return ([[],([],(['no parse'],[]))], dq) # failure! #return dq now as well (an empty list now) EA

    def exps(self, inpt, ic):  #last def input to be changed....
        #ic.row_dump()
        if ic.h == [[]]:
            return

        for child in ic.h.children:       #"for sub-branch in category branch"
            feature_i = child.key.int_value # set i to feature value
            if child.key.ftype == 'sel': # feature type 1 is 'sel'
                if child.roots: 
                    # merge a (non-moving) complement
                    self.merge1(child, feature_i, ic)
                    # merge a (moving) complement
                    self.merge3(child, feature_i, ic)
                elif child.children: 
                    # merge a (non-moving) specifier
                    self.merge2(child, feature_i, ic)
                    # merge a (moving) specifier
                    self.merge4(child, feature_i, ic)
            elif child.key.ftype == 'pos': # feature type 3 is 'pos'
                # again, either children or roots of child are of interest below
                self.move1(child, feature_i, ic)
                self.move2(child, feature_i, ic)
            else:
                raise RuntimeError('exps')
        for root in ic.h.roots:
            #the next node is a string node
            self.scan(root, inpt, ic)

#ftypes = ['cat', 'sel', 'neg', 'pos']

    # merge a (non-moving) complement
    def merge1(self, node, i, ic):       # dt=(ifs,dx,mifs)
        #print('doing merge1')
        ic1 = ic.copy() # no movers to lexical head
        ic1.h = node
        ic1.hx.append(0) # new head index
        ic1.m = [[]] * len(ic.m)
        ic1.mx = [[]] * len(ic.mx)
        ic1.dt.ifs.append(self.create_feature('sel', i)) # extend ifs
        ic1.dt.dx.append(0) # extend dx
        ic1.dt.mifs = [[]]*len(ic.m)

        ic2 = ic.copy() # movers to complement only
        ic2.h = self.lex_array[i] 
        ic2.hx.append(1) # new comp index
        ic2.dt.ifs = [self.create_feature('cat', i)]
        ic2.dt.dx.append(1) # extend new_dx
        exp = Exp(ics=[ic1, ic2])
        self.sofar.append(exp)

    # merge a (non-moving) specifier
    def merge2(self, node, i, ic):    # dt=(ifs,dx,mifs) EA
        #print('doing merge2')
        ic1 = ic.copy() # movers to head
        ic1.h = node
        ic1.hx.append(1) # new head index
        ic1.dt.ifs.append(self.create_feature('sel', i)) # extend ifs
        ic1.dt.dx.append(0) # extend dx

        ic2 = ic.copy()
        ic2.h = self.lex_array[i]
        ic2.m = [[]] * len(ic.m)
        ic2.hx.append(0)
        ic2.mx = [[]] * len(ic.mx)
        ic2.dt.ifs = [self.create_feature('cat', i)]
        ic2.dt.dx.append(1) # extend new_dx
        ic2.dt.mifs = [[]] * len(ic.m)
        exp = Exp(ics=[ic1,ic2])
        self.sofar.append(exp)

    # merge a (moving) complement
    def merge3(self, node, i, ic):      
        #print('doing merge3')
        for nxt, m_nxt in enumerate(ic.m):
            matching_tree = m_nxt and m_nxt.feature_in_children(i) #check to see if term is a mover plain and simple
            if matching_tree:
                ic1 = ic.copy()
                ic1.h = node
                ic1.m = [[]] * len(ic.m)
                ic1.mx = [[]] * len(ic.mx)      
                ic1.dt.ifs.append(self.create_feature('sel', i)) # extend ifs with (sel i)
                ic1.dt.dx.append(0) # extend dx
                ic1.dt.mifs = [[]] * len(ic.m)

                ic2 = ic.copy() # movers passed to complement
                ic2.h = matching_tree
                ic2.m[nxt] = [] # we used the "next" licensee, so now empty
                ic2.hx = ic2.mx[nxt]
                ic2.mx[nxt] = []
                ic2.dt.ifs = ic2.dt.mifs[nxt][:] # movers to complement
                ic2.dt.ifs.append(self.create_feature('cat', i)) # add (cat i) feature
                ic2.dt.dx.append(1) # extend new_dx
                ic2.dt.mifs[nxt] = []
                exp = Exp(ics=[ic1,ic2])
                self.sofar.append(exp)

    # merge a (moving) specifier
    def merge4(self, node, i, ic):          
        #print('doing merge4')
        for nxt, m_nxt in enumerate(ic.m):
            matching_tree = m_nxt and m_nxt.feature_in_children(i)
            if matching_tree:
                ic1 = ic.copy()
                ic1.h = node
                ic1.m[nxt] = [] # we used the "next" licensee, so now empty
                ic1.mx[nxt] = []
                ic1.dt.ifs.append(self.create_feature('sel', i)) # extend ifs
                ic1.dt.dx.append(0) # extend dx
                ic1.dt.mifs[nxt] = []

                ic2 = ic.copy() # movers passed to complement
                ic2.h = matching_tree
                ic2.m = [[]] * len(ic.m)
                ic2.hx = ic2.mx[nxt]
                ic2.mx = [[]] * len(ic.mx) 
                ic2.dt.ifs = ic2.dt.mifs[nxt][:] # copy
                ic2.dt.ifs.append(self.create_feature('cat', i))
                ic2.dt.dx.append(1) # extend new_dx
                ic2.dt.mifs = [[]] * len(ic.m)                
                exp = Exp(ics=[ic1, ic2])
                self.sofar.append(exp)

    def move1(self, node, i, ic):    
        if ic.m[i] == []:  # SMC
            #print('doing move1')
            ic1 = ic.copy() 
            ic1.h = node #node is remainder of head branch
            ic1.m[i] = self.lex_array[i]
            ic1.hx.append(1)
            ic1.mx[i] = ic.hx[:]
            ic1.mx[i].append(0)
            ic1.dt.ifs.append(self.create_feature('pos', i)) # extend ifs with (pos i)
            ic1.dt.dx.append(0) # extend dx
            ic1.dt.mifs[i] = [self.create_feature('neg', i)] # begin new mover with (neg i)
            exp = Exp(ics=[ic1])
            self.sofar.append(exp)

    def move2(self, node, i, ic):  
        for nxt, m_nxt in enumerate(ic.m):
            matching_tree = m_nxt and m_nxt.feature_in_children(i)
            if matching_tree:
                root_f = matching_tree.key.int_value # value of rootLabel
                if root_f == nxt or ic.m[root_f] == []: # SMC
                    #print('doing move2')
                    mts = matching_tree #matchingTree[1:][:]
                    ic1 = ic.copy()
                    ic1.h = node
                    ic1.m[nxt] = [] # we used the "next" licensee, so now empty
                    ic1.m[root_f] = mts
                    ic1.mx[root_f] = ic1.mx[nxt][:]
                    ic1.mx[nxt] = []
                    ic1.dt.mifs[root_f] = ic1.dt.mifs[nxt][:]
                    ic1.dt.mifs[root_f].append(self.create_feature('neg', i)) # extend prev ifs of mover with (neg i)
                    ic1.dt.mifs[nxt] = []
                    ic1.dt.ifs.append(self.create_feature('pos', i)) # extend ifs with (pos i)
                    ic1.dt.dx.append(0)
                    exp = Exp(ics=[ic1])
                    self.sofar.append(exp)

    def scan(self, words, inpt, ic):
        # this actually checks to see if word is present in given sentence
        if not any(self.sofar) and inpt[:len(words)] == words:             
            #print('ok scan')
            new_ic = ic.copy()
            new_ic.h = []
            new_ic.hx = []
            exp = Exp(words=words, ics=[new_ic])  # unlike recognizer, we return w(ord) here
            self.sofar.append(exp)

    def insert_new_parses(self, inpt, p, new_p, q, dq, dns0):
        for exp in self.sofar:
            # scan is a special case, identifiable by empty head
            # (w,[(([],m),([],mx),(ifs,dx,mifs))]) <-- we check for that empty head
            #if exp[1][0][0][0]==[]:

            if not exp.ics[0].h:
                dns = dns0[:]
                #w = exp[0]
                words = exp.words
                #ifs = exp[1][0][2][0][:] # copy
                ifs = exp.ics[0].dt.ifs
                ifs.reverse()
                dx = exp.ics[0].dt.dx
                #dx = exp[1][0][2][1]
                dns.append((dx,(words,ifs)))
                if inpt[:len(words)] == words:
                    remainder_input = inpt[len(words):]
                else:
                    remainder_input = inpt[:]
                new_parse = (p, remainder_input, q, dns)
                heapq_mod.heappush(dq, new_parse)  #modified EA
            else: # put indexed categories ics onto iq with new_p
                safe_q = q[:]
                dns = dns0[:]
                for ic in exp.ics: # ic = ((h,m),(hx,mx),(ifs,dx,mifs))
                    dx = ic.dt.dx
                    dns.append(dx)
                    new_index = ic.min_index()
                    heapq_mod.heappush(safe_q, (new_index, ic))  #modified EA
                new_parse = (new_p, inpt, safe_q, dns)
                heapq_mod.heappush(dq, new_parse) #modified EA


class DTree:
    def __init__(self, label='', features=None, parts=None):
        self.label = label or []
        self.features = features or []
        self.parts = parts or []

    def __repr__(self):
        return '[%r, %r]' % (self.label, self.features or self.parts)

    def as_tree(self):
        if len(self.parts) == 2:            
            return [self.label, self.parts[0].as_tree(), self.parts[1].as_tree()]
        elif len(self.parts) == 1:
            return [self.label, self.parts[0].as_tree()]            
        elif self.features:
            if self.label:
                label = [self.label]
            else:
                label = []
            return (label, [str(f) for f in self.features])


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

    def as_tree(self):
        fss = []
        if self.head_features:
            fss.append(self.head_features)
        fss += self.movers
        sfs = ','.join([' '.join([str(f) for f in fs]) for fs in fss])
        if self.part0 and self.part1: # merge
            return [sfs, self.part0.as_tree(), self.part1.as_tree()]
        elif self.part0: # move
            return [sfs,self.part0.as_tree()]
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

    def as_tree(self):
        if not (self.part0 or self.part1):
            if isinstance(self.label, list):
                w = ' '.join(self.label)
            else:
                w = self.label
            return '%s::%s' % (w, ' '.join([str(f) for f in self.head_features]))
        elif self.part0 and self.part1: # merge
            return [self.label, self.part0.as_tree(), self.part1.as_tree()]
        else:
            raise RuntimeError('BareTree.as_tree')


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
                trace = XBarTree(None, top=False) # trace
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

    def as_tree(self):
        if not (self.part0 or self.part1):
            if self.lexical:
                if self.label and isinstance(self.label, str):
                    return [self.category, [self.label]]
                else:
                    return [self.category, self.label]                   
            else:
                return ([self.label], [])
        elif self.part0 and self.part1: # merge
            return [self.label, self.part0.as_tree(), self.part1.as_tree()]
        else:
            raise RuntimeError('XBarTree.as_tree')

#### Tree conversions and printing

def dnodes_to_dtree(dns):
    nonterms = []
    terms = []
    for dn in dns:
        if isinstance(dn,tuple):  # a terminal is a pair (node,terminal string list)
            terms.append(dn)
        else:
            nonterms.append(dn) # node is an integer list
    if len(nonterms) == 0:
        raise RuntimeError('buildIDtreeFromDnodes: error')
    else:
        terms.sort()
        nonterms.sort()
        root = nonterms.pop(0)
        n = build_dtree_from_dnodes(root, nonterms, terms, DTree())
        if len(terms)!=0 or len(nonterms)!=0:   #changed EA(<> is now only !=)
            print('dNodes2idtree error: unused derivation steps') #changed EA
            print('terms=' + str(terms))   #changed EA
            print('nonterms='+ str(nonterms))  #changed EA
        return n

def build_dtree_from_dnodes(parent, nodes, terminals, dtree):
    def child(n1,n2): # boolean: is n1 a prefix of n2? If so: n2 is a child of n1
        return n1 == n2[0:len(n1)]

    if terminals and terminals[0][0] == parent:
        leaf = terminals.pop(0)[1]
        if leaf[0]:
            label = leaf[0][0]
        else:
            label = ''
        print(leaf)
        features = leaf[1]
        dtree.parts.append(DTree(label=label, features=features))
        return dtree
    elif nodes and child(parent, nodes[0]):
        root = nodes.pop(0)
        new_node = DTree() 
        dtree.parts.append(new_node)
        child0 = build_dtree_from_dnodes(root, nodes, terminals, new_node)
        if nodes and child(parent, nodes[0]):
            new_node.label = '*'  
            root1 = nodes.pop(0)
            child1 = build_dtree_from_dnodes(root1, nodes, terminals, new_node)
        else:
            new_node.label = 'o' 
        return new_node
    else:
        raise RuntimeError('build_dtree_from_dnodes: error')

############################################################################################

if __name__ == '__main__':
    import mg0 as grammar
    sentence = "the king prefers the beer"
    sentence = "which king says which queen knows which king says which wine the queen prefers"
    gr = Grammar(grammar.g, 'C', -0.0001, sentence=sentence)
    results = gr.results
    if False:
        for key in sorted(list(results.keys())):
            print(key)
            print(results[key])

