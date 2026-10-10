"""Analyse the strict CAS audit (Job B and stage 2).

usage: python cas_audit_analyse.py forms.json results.jsonl [results2.jsonl ...] [--write-survivors out.json]

An item counts as solved by an engine only if some strategy returned a verified antiderivative (status
'solved'). 'branch', 'wrong', 'unverified' etc. count as not solved, so the survivor list is conservative.
"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument('forms')
ap.add_argument('results', nargs='+')
ap.add_argument('--write-survivors')
ap.add_argument('--show', type=int, default=0)
args = ap.parse_args()

forms = {f['fid']: f for f in json.load(open(args.forms))}
solved = collections.defaultdict(set)
status = collections.defaultdict(lambda: collections.defaultdict(set))
tested = collections.defaultdict(set)
for fn in args.results:
    for line in open(fn):
        r = json.loads(line)
        if r['fid'] not in forms:
            continue
        tested[r['fid']].add(r['engine'])
        status[r['fid']][r['engine']].add(r['status'])
        if r['status'] == 'solved':
            solved[r['fid']].add(r['engine'])

engines = sorted({e for v in tested.values() for e in v})
sets = collections.OrderedDict()
for fid, f in forms.items():
    sets.setdefault(f['set'], []).append(fid)

hdr = f'{"set":20s} {"N":>5s} ' + ' '.join(f'{e:>7s}' for e in engines) + f' {"any":>6s} {"none":>6s}'
print(hdr)
tot = collections.Counter()
surv = []
for s, fids in sets.items():
    row = collections.Counter()
    for fid in fids:
        for e in solved[fid]:
            row[e] += 1
        if solved[fid]:
            row['any'] += 1
        else:
            row['none'] += 1
            surv.append(fid)
    tot.update(row)
    tot['N'] += len(fids)
    print(f'{s:20s} {len(fids):5d} ' + ' '.join(f'{row[e]:7d}' for e in engines) +
          f' {row["any"]:6d} {row["none"]:6d}')
print(f'{"TOTAL":20s} {tot["N"]:5d} ' + ' '.join(f'{tot[e]:7d}' for e in engines) +
      f' {tot["any"]:6d} {tot["none"]:6d}')

only = collections.Counter()
for fid in forms:
    if len(solved[fid]) == 1:
        only[next(iter(solved[fid]))] += 1
print('solved by exactly one engine:', dict(only))

if args.show:
    for fid in surv[:args.show]:
        f = forms[fid]
        print(fid, f['set'], f['F'][:90], {e: sorted(v) for e, v in status[fid].items()})

if args.write_survivors:
    json.dump([forms[fid] for fid in surv], open(args.write_survivors, 'w'), indent=0)
    print('wrote', len(surv), 'survivors to', args.write_survivors)
