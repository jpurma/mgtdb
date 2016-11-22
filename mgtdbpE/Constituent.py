try:
    from syntax.BaseConstituent import BaseConstituent as MyBaseClass
    from kataja.SavedField import SavedField
    in_kataja = True
except ImportError:
    MyBaseClass = object
    in_kataja = False


class Constituent(MyBaseClass):
    """ Basic constituent tree, base for other kinds of trees. """
    replaces = "ConfigurableConstituent"

    def __init__(self, label='', features=None, parts=None):
        if in_kataja:
            if features is not None:
                features = list(features)
            else:
                features = []
            if parts:
                super().__init__(label=label, parts=parts, features=features)
            else:
                super().__init__(label=label, features=features)

        self.label = label or []
        self.features = features or []
        self.parts = parts or []

    def __repr__(self):
        return '[%r:%r, %r]' % (self.label, self.features, self.parts)

    def build_from_dnodes(self, parent_path, dnodes, terminals, all_features=False):
        if terminals and terminals[0].path == parent_path:
            leaf = terminals.pop(0)
            self.label = ' '.join(leaf.label)
            self.features = list(leaf.features)
            self.features.reverse()
            # s = ''
            # for char in leaf.path:
            #     if char == '0':
            #         s += 'L'
            #     else:
            #         s += 'R'
            # s += ':' + self.label
            # print(s)
        elif dnodes and dnodes[0].path.startswith(parent_path):
            root = dnodes.pop(0)
            if all_features:
                self.features = list(root.features)
                self.features.reverse()

            child0 = Constituent()
            child0.build_from_dnodes(root.path, dnodes, terminals, all_features=all_features)
            self.parts.append(child0)
            if dnodes and dnodes[0].path.startswith(parent_path):
                self.label = '*'
                root1 = dnodes.pop(0)
                child1 = Constituent()
                child1.build_from_dnodes(root1.path, dnodes, terminals, all_features=all_features)
                self.parts.append(child1)
            else:
                self.label = 'o'
                # s = ''
                # for char in parent_path:
                #     if char == '0':
                #         s += 'L'
                #     else:
                #         s += 'R'
                # s += ':' + self.label
                # print(s)

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
    def dnodes_to_dtree(dnodes, all_features=False):
        nonterms = []
        terms = []
        for dn in dnodes:
            if dn.terminal:
                terms.append(dn)
            else:
                nonterms.append(dn)
        terms.sort()
        nonterms.sort()
        root = nonterms.pop(0)
        dtree = Constituent()
        dtree.build_from_dnodes(root.path, nonterms, terms, all_features=all_features)
        if terms or nonterms:
            print('dnodes_to_dtree error: unused derivation steps')
            print('terms=' + str(terms))
            print('nonterms=' + str(nonterms))
        return dtree

    ################## Old constituentB starts here

