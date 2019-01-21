from implementation.association_rule import AssociationRule


class APriori:

    def __init__(self, transactions):
        self.transactions = transactions
        self._extract_vocabulary(transactions)

    def get_frequent_item_sets(self, min_support, size=int(1e10)):
        return self._apriori(min_support, max_size=size)

    def get_extracted_rules(self, min_confidence, min_support=0, frequent_item_set=None):
        rules = []
        if frequent_item_set is None:
            frequent_item_set = self._apriori(min_support)
        max_k = max(frequent_item_set.keys())
        if max_k < 2:
            raise ArithmeticError('No valid rule available!')
        for k in range(2, max_k+1):
            for association_rule in frequent_item_set[k]:
                rules += self._extract_item_set_rules(association_rule, min_confidence)
        return rules

    def _extract_vocabulary(self, transactions):
        self.vocabulary = sorted(list(set([item for item_set in transactions for item in item_set])))

    def _get_item_set_count(self, item_set):
        return sum([item_set.issubset(transaction) for transaction in self.transactions])

    def _check_support_frequency(self, candidates, min_support):
        frequents = []
        infrequents = []
        for association_rule in candidates:
            count = self._get_item_set_count(association_rule.item_set)
            association_rule.support = count / len(self.transactions)
            if association_rule.support >= min_support:
                frequents.append(association_rule)
            else:
                infrequents.append(association_rule.item_set)
        return frequents, infrequents

    def _check_rule_confidence(self, candidates, min_confidence):
        confident_rules = []
        unconfident_lefts = []
        for rule in candidates:
            if self._get_item_set_count(rule.item_set) == 0:
                continue
            else:
                rule.confidence = self._get_item_set_count(rule.item_set) / self._get_item_set_count(rule.rule_left)
            if rule.confidence >= min_confidence:
                confident_rules.append(rule)
            else:
                unconfident_lefts.append(rule.rule_left)
        return confident_rules, unconfident_lefts

    def _apriori(self, min_support, max_size=int(1e10)):
        max_size = min(len(self.vocabulary), max_size)
        unit_candidates = [AssociationRule(item) for item in self.vocabulary]
        unit_frequents, infrequents = self._check_support_frequency(unit_candidates, min_support)
        last_frequents_size = 0
        frequents = {1: unit_frequents}
        k = 1
        while True:
            if len(frequents) == last_frequents_size:
                break
            elif k == max_size:
                break
            else:
                last_frequents_size = len(frequents)
            k_frequents = frequents[k]
            new_candidates = []
            for item in unit_frequents:
                for item_set in [rule.item_set for rule in k_frequents]:
                    candidate = AssociationRule(item_set.union(item.item_set))
                    if candidate not in new_candidates and len(candidate.item_set) > k:
                        has_infrequent = False
                        for infrequent in infrequents:
                            if infrequent.issubset(candidate.item_set):
                                has_infrequent = True
                                break
                        if not has_infrequent:
                            new_candidates.append(candidate)
            # after generating candidates
            new_frequents, infrequents = self._check_support_frequency(new_candidates, min_support)
            if len(new_frequents) > 0:
                frequents[k + 1] = new_frequents
            k += 1
        return frequents

    def _extract_item_set_rules(self, association_rule, min_confidence):
        confident_rules = {}
        unconfident_lefts = []
        dummy_rule = AssociationRule(association_rule.item_set, rule_left=association_rule.item_set,
                                     rule_right=set(), support=association_rule.support)
        confident_rules[0] = [dummy_rule]
        for k in range(0, len(association_rule.item_set)-1):
            parent_confident_rules = confident_rules[k]
            candidate_rules = []
            for parent_rule in parent_confident_rules:
                for item in parent_rule.rule_left:
                    right_side = parent_rule.rule_right.union({item})
                    left_side = parent_rule.rule_left.difference({item})
                    if not any([left_side.issubset(unconfident_left) for unconfident_left in unconfident_lefts]):
                        new_rule = AssociationRule(parent_rule.item_set, left_side, right_side, parent_rule.support)
                        if new_rule not in candidate_rules:
                            candidate_rules.append(new_rule)
            new_confident_rules, new_unconfident_lefts = self._check_rule_confidence(candidate_rules, min_confidence)
            confident_rules[k + 1] = new_confident_rules
            unconfident_lefts += new_unconfident_lefts
        del confident_rules[0]
        return [rule for sized_rules in confident_rules.values() for rule in sized_rules]
