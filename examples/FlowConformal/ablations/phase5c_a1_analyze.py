"""Analyze A1 probe output: decompose q-variance into flow/cal/scenario.

For each instance, compute:
  Var_F = empirical variance of q over trial set F
  Var_C = empirical variance of q over trial set C
  Var_S = empirical variance of q over trial set S

Aggregate across instances. Print a decision rule output.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).parent / 'outputs' / 'phase5c_a1_variance_probe.csv'

def main():
    if not CSV.exists():
        raise FileNotFoundError(f'A1 probe CSV not found: {CSV}')
    rows = list(csv.DictReader(open(CSV)))
    by = defaultdict(lambda: defaultdict(list))  # instance -> trial_set -> [q]
    skipped = []
    for r in rows:
        if r.get('verdict') in ('ERROR', 'SAT') or not r.get('q'):
            skipped.append((r.get('instance'), r.get('trial_set'),
                            r.get('trial_idx'), r.get('verdict'),
                            r.get('error', '')))
            continue
        try:
            q_val = float(r['q'])
        except ValueError:
            continue
        by[r['instance']][r['trial_set']].append(q_val)

    print(f'A1 variance decomposition  (read {len(rows)} rows from {CSV.name})')
    if skipped:
        print(f'\n{len(skipped)} rows skipped (ERROR/SAT/empty-q):')
        for s in skipped:
            print(f'  {s[0]}  set={s[1]} trial={s[2]}  verdict={s[3]}  err={s[4]}')

    print(f'\n{"instance":<55} {"Var_F":>10} {"Var_C":>10} {"Var_S":>10} {"dominant":<10}')
    overall = {'F': [], 'C': [], 'S': []}
    for inst in sorted(by):
        v = {ts: statistics.pvariance(by[inst][ts]) if len(by[inst][ts]) > 1 else 0.0
             for ts in ('F', 'C', 'S')}
        for ts in ('F', 'C', 'S'):
            overall[ts].append(v[ts])
        dom = max(v, key=v.get)
        print(f'{inst:<55} {v["F"]:>10.4f} {v["C"]:>10.4f} {v["S"]:>10.4f} {dom:<10}')

    print('\nOverall (mean across instances):')
    means = {ts: statistics.mean(overall[ts]) if overall[ts] else 0.0
             for ts in ('F', 'C', 'S')}
    print(f'  Var_F mean = {means["F"]:.4f}')
    print(f'  Var_C mean = {means["C"]:.4f}')
    print(f'  Var_S mean = {means["S"]:.4f}')

    threshold = 0.5
    big = {ts: means[ts] > threshold for ts in ('F', 'C', 'S')}
    n_big = sum(big.values())
    if n_big == 0:
        decision = 'A3 SKIP — all variances small (B-side knobs do all work)'
    elif n_big >= 2:
        decision = 'A3 ENSEMBLE — multiple variance sources contribute'
    elif big['F']:
        decision = 'A3 BUMP — increase n_train/flow_epochs'
    elif big['C']:
        decision = 'A3 SKIP standalone — merged into B2 (calibration tightening)'
    elif big['S']:
        decision = 'A3 SKIP standalone — merged into B1 (scenario tightening)'
    else:
        decision = 'A3 (unreachable — review dispatch logic)'
    print(f'\nDecision: {decision}')

if __name__ == '__main__':
    main()
