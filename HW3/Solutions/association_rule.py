from dataclasses import dataclass


@dataclass
class AssociationRule:
    rule_left: set = None
    rule_right: set = None
    confidence: int = None
    support: int = None

    def __init__(self, item_set, rule_left=None, rule_right=None, support=None, confidence=None):
        if type(item_set) is set:
            self.item_set = item_set
        else:
            self.item_set = {item_set}
        self.rule_left = rule_left
        self.rule_right = rule_right
        self.support = support
        self.confidence = confidence

    def __eq__(self, o: object) -> bool:
        if not type(o) is AssociationRule:
            return False
        if o.rule_left is None:
            if self.rule_left is None:
                return o.item_set == self.item_set
            else:
                return False
        elif self.rule_left is None:
            return False
        else:
            return o.rule_left == self.rule_left and o.rule_right == self.rule_left and o.item_set == self.item_set

    def __str__(self) -> str:
        if self.rule_left is None:
            return str(self.item_set) + ', s = {}'.format(self.support)
        else:
            return str(self.rule_left) + ' -> ' \
                   + str(self.rule_right) + ', s = {}, c = {}'.format(self.support, self.confidence)

