"""Summarise cas_audit JSONL outputs: status counts per engine and example problem cases."""
import collections
import json
import sys

for fn in sys.argv[1:]:
    try:
        rows = [json.loads(l) for l in open(fn)]
    except FileNotFoundError:
        print('==', fn, 'missing')
        continue
    c = collections.Counter((r['engine'], r['status']) for r in rows)
    print('==', fn, len(rows), 'tasks')
    for e in sorted({k[0] for k in c}):
        print('  ', e, {s: n for (ee, s), n in sorted(c.items()) if ee == e})
    seen = set()
    for r in rows:
        if r['status'] in ('harness_error', 'noresult', 'unverified', 'wrong') and (r['engine'], r['status']) not in seen:
            seen.add((r['engine'], r['status']))
            print('   EX', r['engine'], r['status'], r['fid'], r['result'][:80], '|', r['tail'][-250:].replace('\n', ' '))
