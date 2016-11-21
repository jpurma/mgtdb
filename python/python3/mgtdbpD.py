"""
file: mgtdbpD.py
      minimalist grammar top-down beam parser, refactored from E. Stabler's original.

   This is part of effort to make an output-equivalent mgtdbp that can be used as a Kataja plugin.
   The code aims for readability and not efficiency, so most of the parameter passings with complex
   lists are turned into objects where necessary parameters can be get by their name and not by
   their index in ad-hoc lists or tuples.

   mgtdbpA -- Turned most of the complex list parameters for functions to class instances 
   mgtdbpB -- More informative variable names and neater conversion to output trees
   mgtdbpC -- Removed heapq_mod -- now the whole thing is faster, as there is less implicit
   sorting and new parses are inserted close to their final resting place.
   mgtdbpD -- Combined DerivationTree and DerivationNode. As names tell, they were quite similar.
   Sortable indices are strings instead of lists. Easier for output and almost as fast.

   Refactoring by Jukka Purma                                      || 11/21/16 
   mgtdbp-dev Modified for py3.1 compatibility by: Erik V Arrieta. || Last modified: 9/15/12
   mgtdbp-dev by Edward P. Stabler
   
Comments welcome: jukka.purma--at--gmail.com
"""
import io
import time
from collections import OrderedDict
from nltktreeport import (Tree)

"""
   We represent trees with lists, where the first element is the root,
   and the rest is the list of the subtrees.

   First, a pretty printer for these list trees:
"""
def pptreeN(out, n, t):  # pretty print t indented n spaces
    if isinstance(t, list) and len(t) > 0:
        out.write('\n' + ' ' * n + '[')
        out.write(str(t[0])),  # print root and return
        if len(t[1:]) > 0:
            out.write(',')  # comma if more coming
        for i, subtree in enumerate(t[1:]):  # then print subtrees indented by 4
            pptreeN(out, n + 4, subtree)
            if i < len(t[1:]) - 1:
                out.write(',')  # comma if more coming
        out.write(']')
    else:
        out.write('\n' + ' ' * n)
        out.write(str(t))


def pptree(out, t):
    if len(t) == 0:  # catch special case of empty tree, lacking even a root
        out.write('\n[]\n')
    else:
        pptreeN(out, 0, t)
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
    if isinstance(listtree, tuple):  # special case for MG lexical items at leaves
        return ' '.join(listtree[0]) + ' :: ' + ' '.join(listtree[1])
    elif isinstance(listtree, str):  # special case for strings at leaves
        return listtree
    elif isinstance(listtree, list) and listtree == []:
        return []
    elif isinstance(listtree, list):
        subtrees = [list_tree_to_nltk_tree(e) for e in listtree[1:]]
        if not subtrees:
            return listtree[0]
        else:
            return Tree(listtree[0], subtrees)
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
    """ These are more typical LIs. It takes in list of words instead of one string for
    compability with mgtdbp output. These are not used in parsing at all, they are only to print
    out the grammar. """

    def __init__(self, words, features):
        self.words = words
        self.features = features

    def __str__(self):
        return ' '.join(self.words) + '::' + ' '.join([str(f) for f in self.features])


class LexTreeNode:
    def __init__(self, feature):
        self.feature = feature
        self.subtrees = []  # LexTreeNodes
        self.terminals = []

    def subtree_with_feature(self, value):
        for subtree in self.subtrees:
            if subtree.feature and subtree.feature.value == value:
                return subtree

    def __repr__(self):
        return 'LN(key=%r, children=%r, roots=%r)' % (self.feature, self.subtrees, self.terminals)

    def __str__(self):
        if self.subtrees:
            return '[%s, [%s]]' % (self.feature, ', '.join([str(x) for x in self.subtrees]))
        else:
            return '[%s, %s]' % (self.feature, self.terminals)


class Prediction:
    def __init__(self, head, movers=None, head_path=None, mover_paths=None, tree=None):
        self.head = head
        self.movers = movers or {}
        self.head_path = head_path or ''
        self.mover_paths = mover_paths or {}
        self.tree = tree
        self._min_index = None
        self.update_ordering()

    def update_ordering(self):
        """ ICs can be stored in queue if they can be ordered. Original used tuples to provide
        ordering, we can have an innate property _min_index for the task. It could be calculated
        for each comparison, but for efficiency's sake do it manually before pushing to queue. """
        self._min_index = min([self.head_path] + [x for x in self.mover_paths.values()])

    def copy(self):
        return Prediction(self.head, self.movers.copy(), self.head_path[:], self.mover_paths.copy(),
                          self.tree.copy())

    def __repr__(self):
        return 'Prediction(head=%r, movers=%r, head_path=%r, mover_paths=%r, tree=%r)' % (
        self.head, self.movers, self.head_path, self.mover_paths, self.tree)

    def __getitem__(self, key):
        return self._min_index

    def __lt__(self, other):
        return self._min_index < other._min_index


class Derivation:
    def __init__(self, p, inpt, iqs, results):
        self.probability = p
        self.input = inpt
        self.predictions = iqs
        self.results = results

    def __getitem__(self, key):
        return (self.probability, self.input, self.predictions, self.results)[key]

    def __str__(self):
        return '(%s, %s, %s, [%s])' % \
               (self.probability, ' '.join(self.input), str(self.predictions),
                ', '.join([str(x) for x in self.results]))

    def copy(self):
        return Derivation(self.probability, self.input[:], self.predictions[:], self.results[:])

    def __lt__(self, other):
        return self.probability < other.probability


class DerivationNode:
    """ DerivationNodes are constituent nodes that are represented in a queer way: 
        instead of having relations to other nodes, they have 'path', a list of 1:s and 0:s that
        tells their place in a binary tree. Once there is a list of DerivationNodes, a common
        tree can be composed from it. """

    def __init__(self, path, label=None, features=None, moving_features=None, terminal=False):
        self.path = path or ''
        self.label = label or []
        self.features = features or []
        self.moving_features = moving_features or {}
        self.terminal = terminal

    def copy(self):
        return DerivationNode(self.path[:], self.label[:], self.features[:],
                              self.moving_features.copy(), self.terminal)

    def __repr__(self):
        return 'DerivationNode(path=%r, label=%r, features=%r)' % \
               (self.path, self.label, self.features)

    def __str__(self):
        if self.label or self.features:
            return '%s, (%s, %s)' % (self.path, self.label, self.features)
        else:
            return self.path

    def __lt__(self, other):
        return self.path < other.path


class Parser:
    def __init__(self, lex_tuples, min_p, start=None, sentence=None):
        print('****** Starting Parser *******')
        self.d = []
        self.min_p = min_p
        self.lex = OrderedDict()
        self.derivation_stack = []
        self.new_parses = []
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
                for subtree in node.subtrees:
                    if subtree.feature == f:
                        found = True
                        node = subtree
                        break
                if not found:
                    new_node = LexTreeNode(f)
                    node.subtrees.append(new_node)
                    node = new_node
            node.terminals.append(words)
        for node in base.subtrees:
            self.lex[node.feature.value] = node  # dict for quick access to starting categories

        if start and sentence:
            success, dnodes = self.parse(start, sentence)
            if success:
                self.print_results(dnodes)
            else:
                print('parse failed, what we have is:')
                print(dnodes)

    def __str__(self):
        return str(self.d)

    def parse(self, start, sentence):
        # Prepare prediction queue. We have a prediction that the derivation will finish 
        # in a certain kind of category, e.g. 'C'  
        final_features = [Feature('cat', start)]
        topmost_head = self.lex[start]
        prediction = Prediction(topmost_head, tree=DerivationNode('', features=final_features))

        inpt = sentence.split()
        print('inpt =' + str(inpt))

        # Prepare derivation queue. It gets expanded by derive.  
        self.derivation_stack = [Derivation(-1.0, inpt, [prediction], [DerivationNode('')])]

        # The work is done by derive.
        t0 = time.time()
        success, dnodes = self.derive()
        t1 = time.time()
        if success:
            print('parse found')
        else:
            print('no parse found')
        print(str(t1 - t0) + "seconds")
        return success, dnodes

    def derive(self):
        d = None
        while self.derivation_stack:
            d = self.derivation_stack.pop()
            print('# of parses in beam=%s, p(best parse)=%s' %
                  (len(self.derivation_stack) + 1, -1 * d.probability))
            if not (d.predictions or d.input):
                return True, d.results  # success
            elif d.predictions:
                prediction = d.predictions.pop(0)
                self.new_parses = []
                self.create_expansions(prediction)
                if self.new_parses:
                    new_p = d.probability / len(self.new_parses)
                    if new_p < self.min_p:
                        self.insert_new_parses(d, new_p)
                    else:
                        print('improbable parses discarded')
                else:
                    self.scan_and_insert_terminals(prediction, d)
        return False, d.results  # fail

    def create_expansions(self, prediction):
        """ Expand possibilities. If we assume current {prediction}, what are the operations that
         could have lead into it? Prediction has features we know about, and those fix its place
         in LexTree. The next generation of nodes, {subtrees}, in LexTree are those that have
         these and additional features and for each we make a prediction where the subtree node
         got there because of merge or move with something else.
         All predictions get written into self.new_parses
         """

        # e.g. if current head is C, nodes are =V, -wh
        for node in prediction.head.subtrees:
            if node.feature.ftype == 'sel':
                if node.terminals:
                    self.merge1(node, prediction)  # merge a (non-moving) complement
                    self.merge3(node, prediction)  # merge a (moving) complement
                elif node.subtrees:
                    self.merge2(node, prediction)  # merge a (non-moving) specifier
                    self.merge4(node, prediction)  # merge a (moving) specifier
            elif node.feature.ftype == 'pos':
                self.move1(node, prediction)
                self.move2(node, prediction)
            else:
                raise RuntimeError('exps')

    def scan_and_insert_terminals(self, prediction, derivation):
        for words in prediction.head.terminals:
            # scan -operation
            if derivation.input[:len(words)] == words:
                # print('doing scan:', terminal)
                new_pred = prediction.copy()
                new_pred.head = []
                new_pred.head_path = []
                new_parse = derivation.copy()
                new_pred.tree.label = words
                new_pred.tree.terminal = True
                new_parse.results.append(new_pred.tree)
                if new_parse.input[:len(words)] == words:
                    new_parse.input = new_parse.input[len(words):]
                self.derivation_stack.append(new_parse)
                self.derivation_stack.sort(reverse=True)
                break  # there is only one match for word+features in lexicon

    # These operations reverse familiar minimalist operations: external merges, moves and select. 
    # They create predictions of possible child nodes that could have resulted in current head.
    # Predictions are packed into Expansions, and after all Expansions for this head are created, 
    # the good ones are inserted to parse queue by insert_new_parses.  
    # merge a (non-moving) complement
    def merge1(self, node, prediction):
        """ This reverses a situation when a new element (pr0) is external-merged to existing
        head (pr1) as a complement.
        given node is child of current prediction and it is the last feature before the terminals.
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        # print('doing merge1')
        # print(node)
        category = node.feature.value
        pr0 = prediction.copy()  # no movers to lexical head
        pr0.head = node  # one part of the puzzle is given, the other part is deduced from this
        pr0.head_path += '0'
        pr0.movers = {}
        pr0.mover_paths = {}
        pr0.tree.features.append(Feature('sel', category))
        pr0.tree.path += '0'
        pr0.tree.moving_features = {}

        pr1 = prediction.copy()  # movers to complement only
        pr1.head = self.lex[category]  # head can be any LI in this category
        pr1.head_path += '1'
        pr1.tree.features = [Feature('cat', category)]
        pr1.tree.path += '1'
        self.new_parses.append((pr0, pr1))

    # merge a (non-moving) specifier
    def merge2(self, node, prediction):
        """ This reverses a situation when a non-terminal element (pr0) is external-merged to
        existing head (pr1) as a specifier.
        node is non-terminal
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        # print('doing merge2')
        # print(node)
        cat = node.feature.value
        pr0 = prediction.copy()  # movers to head
        pr0.head = node
        pr0.head_path += '1'
        pr0.tree.features.append(Feature('sel', cat))
        pr0.tree.path += '0'

        pr1 = prediction.copy()
        pr1.head = self.lex[cat]
        pr1.movers = {}
        pr1.head_path += '0'
        pr1.mover_paths = {}
        pr1.tree.features = [Feature('cat', cat)]
        pr1.tree.path += '1'
        pr1.tree.moving_features = {}
        self.new_parses.append((pr0, pr1))

    # merge a (moving) complement
    def merge3(self, node, prediction):
        """ This reverses a situation when a terminal moving element (pr0) is merged to existing
        head (pr1) as a complement.
        what we know about {node} is that it is terminal
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        cat = node.feature.value
        for mover_cat, mover in prediction.movers.items():  # look into movers
            matching_tree = mover.subtree_with_feature(cat)  # matching tree is a child of mover
            if matching_tree:
                # print('doing merge3')
                # print(node)
                # pr0 is prediction about {node}. It is a simple terminal that selects for 
                pr0 = prediction.copy()
                pr0.head = node
                pr0.movers = {}
                pr0.mover_paths = {}
                pr0.tree.features.append(Feature('sel', cat))
                pr0.tree.path += '0'
                pr0.tree.moving_features = {}

                # pr1 doesn't have certain movers that higher prediction has
                pr1 = prediction.copy()  # movers passed to complement
                pr1.head = matching_tree
                del pr1.movers[mover_cat]  # we used the licensee, so now empty
                pr1.head_path = pr1.mover_paths[mover_cat]
                del pr1.mover_paths[mover_cat]
                pr1.tree.features = pr1.tree.moving_features[mover_cat][:]  # movers to complement
                pr1.tree.features.append(Feature('cat', cat))
                pr1.tree.path += '1'
                del pr1.tree.moving_features[mover_cat]
                self.new_parses.append((pr0, pr1))

    # merge a (moving) specifier
    def merge4(self, node, prediction):
        """ This reverses a situation when a non-terminal element (pr0) is merged to existing
        head (pr1) as a specifier.
        node is non-terminal
        Predictions are copies of the given parent prediction and modify it slightly, so that the
        parent prediction is what would result if the two predictions are merged.
        :param node: hypothetical node that could lead to given {prediction}  
        :param prediction: the known result of hypothetical merge
        """
        cat = node.feature.value
        for nxt, m_nxt in prediction.movers.items():
            matching_tree = m_nxt.subtree_with_feature(cat)
            if matching_tree:
                # print('doing merge4')
                # print(node)
                # pr0 doesn't have certain movers that higher prediction has
                pr0 = prediction.copy()
                pr0.head = node
                del pr0.movers[nxt]  # we used the "next" licensee, so now empty
                del pr0.mover_paths[nxt]
                pr0.tree.features.append(Feature('sel', cat))
                pr0.tree.path += '0'
                del pr0.tree.moving_features[nxt]

                pr1 = prediction.copy()  # movers passed to complement
                pr1.head = matching_tree
                pr1.movers = {}
                pr1.head_path = pr1.mover_paths[nxt]
                pr1.mover_paths = {}
                pr1.tree.features = pr1.tree.moving_features[nxt][:]  # copy
                pr1.tree.features.append(Feature('cat', cat))
                pr1.tree.path += '1'
                pr1.tree.moving_features = {}
                self.new_parses.append((pr0, pr1))

    def move1(self, node, prediction):
        cat = node.feature.value
        if cat not in prediction.movers:  # SMC
            # print('doing move1')
            # print(node)
            pr0 = prediction.copy()
            pr0.head = node  # node is remainder of head branch
            pr0.movers[cat] = self.lex[cat]
            pr0.head_path += '1'
            pr0.mover_paths[cat] = prediction.head_path[:]
            pr0.mover_paths[cat] += '0'
            pr0.tree.features.append(Feature('pos', cat))
            pr0.tree.path += '0'
            pr0.tree.moving_features[cat] = [Feature('neg', cat)]  # begin new mover with (neg cat)
            self.new_parses.append((pr0, None))

    def move2(self, node, prediction):
        cat = node.feature.value
        for mover_cat, mover in prediction.movers.items():  # <-- look into movers
            matching_tree = mover.subtree_with_feature(
                cat)  # ... for category shared with prediction
            if matching_tree:
                root_f = matching_tree.feature.value  # value of rootLabel
                assert (root_f == cat)
                print(root_f, mover_cat)
                if root_f == mover_cat or not prediction.movers.get(root_f, []):  # SMC
                    print('doing move2')
                    print(node)
                    # print('doing move2')
                    mts = matching_tree  # matchingTree[1:][:]
                    pr0 = prediction.copy()
                    pr0.head = node
                    del pr0.movers[mover_cat]  # we used the "next" licensee, so now empty
                    pr0.movers[root_f] = mts
                    pr0.mover_paths[root_f] = pr0.mover_paths[mover_cat][:]
                    del pr0.mover_paths[mover_cat]
                    pr0.tree.moving_features[root_f] = pr0.tree.moving_features[mover_cat][:]
                    pr0.tree.moving_features[root_f].append(
                        Feature('neg', cat))  # extend prev features of mover with (neg cat)
                    del pr0.tree.moving_features[mover_cat]
                    pr0.tree.features.append(Feature('pos', cat))
                    pr0.tree.path += '0'
                    self.new_parses.append((pr0, None))

    def insert_new_parses(self, derivation, new_p):
        for pred0, pred1 in self.new_parses:
            new_parse = derivation.copy()
            new_parse.results.append(pred0.tree)
            pred0.update_ordering()
            new_parse.predictions.append(pred0)
            if pred1:
                new_parse.results.append(pred1.tree)
                pred1.update_ordering()
                new_parse.predictions.append(pred1)
            new_parse.predictions.sort()
            new_parse.probability = new_p
            self.derivation_stack.append(new_parse)
        self.derivation_stack.sort(reverse=True)


# Output trees ########

class DTree:
    """ Basic constituent tree, base for other kinds of trees. """

    def __init__(self, label='', features=None, parts=None):
        self.label = label or []
        self.features = features or []
        self.parts = parts or []
        # self.coords = ''

    def __repr__(self):
        return '[%r, %r]' % (self.label, self.features or self.parts)

    def build_from_dnodes(self, parent_path, dnodes, terminals):
        if terminals and terminals[0].path == parent_path:
            leaf = terminals.pop(0)
            self.label = ' '.join(leaf.label)
            self.features = leaf.features
            self.features.reverse()
            # if dnodes:
            #     if leaf.path and leaf.path[-1] == '1':
            #         self.coords = str(len(parent_path)) + 'b' 
            #     else:
            #         self.coords = str(len(parent_path)) + 'a' 
            #     print('-', self.coords, leaf.path)            

        elif dnodes and parent_path == dnodes[0].path[0:len(parent_path)]:
            root = dnodes.pop(0)
            # if root.path and root.path[-1] == '1':
            #     self.coords = str(len(parent_path)) + 'b' 
            # else:
            #     self.coords = str(len(parent_path)) + 'a' 
            # print('*', self.coords, root.path)            
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
            return label, [str(f) for f in self.features]

    @staticmethod
    def dnodes_to_dtree(dnodes):
        nonterms = []
        terms = []
        for dn in dnodes:
            if dn.terminal:
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
                print('nonterms=' + str(nonterms))
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
        assert found
        if mover:
            self.movers.append(mover)

    def as_list_tree(self):
        fss = []
        if self.features:
            fss.append(self.features)
        fss += self.movers
        sfs = ','.join([' '.join([str(f) for f in fs]) for fs in fss])
        if self.part0 and self.part1:  # merge
            return [sfs, self.part0.as_list_tree(), self.part1.as_list_tree()]
        elif self.part0:  # move
            return [sfs, self.part0.as_list_tree()]
        else:  # leaf
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
                self.part1 = BareTree(None)  # trace
        else:
            raise RuntimeError('merge_check error')
        if not (self.part0.part0 or self.part0.part1):  # is it leaf?
            self.label = '<'
        else:
            self.label = '>'  # switch order to part1, part0
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
        assert found
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
        assert self.part0

    def as_list_tree(self):
        if not (self.part0 or self.part1):
            if isinstance(self.label, list):
                w = ' '.join(self.label)
            else:
                w = self.label
            return '%s::%s' % (w, ' '.join([str(f) for f in self.features]))
        elif self.part0 and self.part1:  # merge
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
                assert self.category

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
            self.features = remainders0  # copy remaining head1 features
            self.movers = self.part0.movers + self.part1.movers  # add movers1 and 2
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
        assert found
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
        assert self.part0

    def as_list_tree(self):
        if not (self.part0 or self.part1):
            if self.lexical:
                if self.label and isinstance(self.label, str):
                    return [self.category, [self.label]]
                else:
                    return [self.category, [self.label]]
            else:
                return [self.label], []
        elif self.part0 and self.part1:  # merge
            return [self.label, self.part0.as_list_tree(), self.part1.as_list_tree()]
        else:
            raise RuntimeError('XBarTree.as_list_tree')

############################################################################################

def print_results(dnodes, lexicon=None):
        dt = DTree.dnodes_to_dtree(dnodes)
        res = {}
        # d -- derivation tree
        res['d'] = list_tree_to_nltk_tree(dt.as_list_tree())
        # pd -- pretty-printed derivation tree
        output = io.StringIO()
        pptree(output, dt.as_list_tree())
        res['pd'] = output.getvalue()
        output.close()
        # s -- state tree
        res['s'] = list_tree_to_nltk_tree(StateTree(dt).as_list_tree())
        # ps -- pretty-printed state tree
        output = io.StringIO()
        pptree(output, StateTree(dt).as_list_tree())
        res['ps'] = output.getvalue()
        output.close()
        # b -- bare tree
        res['b'] = list_tree_to_nltk_tree(BareTree(dt).as_list_tree())
        # pb -- pretty-printed bare tree
        output = io.StringIO()
        pptree(output, BareTree(dt).as_list_tree())
        res['pb'] = output.getvalue()
        output.close()
        # x -- xbar tree
        res['x'] = list_tree_to_nltk_tree(XBarTree(dt).as_list_tree())
        # px -- pretty-printed xbar tree
        output = io.StringIO()
        pptree(output, XBarTree(dt).as_list_tree())
        res['px'] = output.getvalue()
        output.close()
        # pg -- print grammar as items
        output = io.StringIO()
        res['pg'] = output.getvalue()
        output.close()
        if lexicon:
            # l -- grammar as tree
            res['l'] = list_tree_to_nltk_tree(['.'] + lex_array_as_list(lexicon))
            # pl -- pretty-printed grammar as tree
            output = io.StringIO()
            pptree(output, ['.'] + lex_array_as_list(lexicon))  # changed EA
            res['pl'] = output.getvalue()
            output.close()
        return res

def show(out):
    for item in self.d:
        out.write(str(item))

def lex_array_as_list(lexicon):
    def as_list(node):
        if isinstance(node, LexTreeNode):
            return [str(node.feature)] + [as_list(x) for x in node.subtrees] + node.terminals
        else:
            return node

    return [as_list(y) for y in lexicon.values()]

############################################################################################

if __name__ == '__main__':
    import mg0 as grammar

    sentences = ["the king prefers the beer",
                 "which king says which queen knows which king says which wine the queen prefers",
                 "which queen says the king knows which wine the queen prefers",
                 "which wine the queen prefers",
                 "which king says which queen knows which king says which wine the queen prefers"]
    sentences = ["which king says which queen knows which king says which wine the queen prefers"]
    # sentences = ["which wine the queen prefers"]
    t = time.time()
    for s in sentences:
        pr = Parser(grammar.g, -0.0001)
        success, dnodes = pr.parse(sentence=s, start='C')
        results = print_results(dnodes, pr.lex)
        if True:
            for key in sorted(list(results.keys())):
                print(key)
                print(results[key])
    print(time.time() - t)
