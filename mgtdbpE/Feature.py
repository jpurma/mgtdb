from kataja.SavedObject import SavedObject
from kataja.SavedField import SavedField


class Feature(SavedObject):
    syntactic_object = True

    editable = {}
    addable = {}

    def __init__(self, ftype=None, value=None):
        super().__init__()
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

    def __hash__(self):
        return hash(str(self))


    # ############## #
    #                #
    #  Save support  #
    #                #
    # ############## #

    name = SavedField("name")
    value = SavedField("value")
    #assigned = SavedField("assigned")
    #family = SavedField("family")
