import json
d = json.load(open('cas_singleton_results_1000.json'))
failures = d['failures']
out = [{'derivative': f['derivative'], 'our_k': f['our_k'], 'our_expr': f['our_expr'],
        'sympy': 'failed', 'mathematica': 'not_tested', 'any_solved': False}
       for f in failures]
with open('singleton_failures_1000_for_fricas.json', 'w') as fout:
    json.dump(out, fout, indent=2)
print(f'Extracted {len(out)} SymPy failures for FriCAS')
