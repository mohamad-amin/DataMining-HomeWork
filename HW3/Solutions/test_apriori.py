from apriori import APriori

transactions = [
    {'A', 'B', 'D', 'G'},
    {'B', 'D', 'E'},
    {'A', 'B', 'C', 'E', 'F'},
    {'B', 'D', 'E', 'G'},
    {'A', 'B', 'C', 'E', 'F'},
    {'B', 'E', 'G'},
    {'A', 'C', 'D', 'E'},
    {'B', 'E'},
    {'A', 'B', 'E', 'F'},
    {'A', 'C', 'D', 'E'},
]

solver = APriori(transactions)
frequents = solver.get_frequent_item_sets(.4)
rules = solver.get_extracted_rules(.1, frequent_item_set=frequents)
print('\n'.join(map(str, rules)))
