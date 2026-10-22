# -*- coding: utf-8 -*-
"""Sections 4.7 and 5.5.1, Table 5: clause inventories on the 1 182 treebank sentences.

A = rule V0 on system tags; B = gold clause boundaries (narrow definition) projected onto the same system tags;
C = gold boundaries and gold tags. Grouping B2. Exponents on a common grid (100, 250, 500, 1000, 2000, N_c;
N_c = the smallest of the three unit counts), 20 shuffles (seed 1000+s), mean and sample SD.
B - A: 95 % half-sample interval = point estimate +/- 1.96 x SD of 1 000 half-sample differences
(591 sentences without replacement, seed 700000+r). Also: the same comparison per treebank, without grouping,
and for the conservative finite-only rule set (patterns stored in data/ud_system_output.json).

Section 4.7: number of gold units and of discontinuous gold units (data/ud_system_output.json).

Output: results/ud_gold.json
"""
import math
import random
import statistics as S
from collections import Counter

import common as C

GRID = (100, 250, 500, 1000, 2000)
SHUFFLES, SHUFFLE_BASE = 20, 1000
HALF_REPS, HALF_BASE = 1000, 700000
Z = 1.96


def grid(n_c):
    return [k for k in GRID if k < n_c] + [n_c]


def ols(points):
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(v) for _, v in points]
    xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)


def betas(patterns, n_c):
    g = set(grid(n_c))
    out = []
    for s in range(SHUFFLES):
        order = list(range(len(patterns)))
        random.Random(SHUFFLE_BASE + s).shuffle(order)
        seen, pts = set(), []
        for i in range(n_c):
            seen.add(patterns[order[i]])
            if i + 1 in g:
                pts.append((i + 1, len(seen)))
        out.append(ols(pts))
    return out


def pats(G, key, idx=None):
    idx = range(len(G)) if idx is None else idx
    return [tuple(p) for i in idx for p in G[i][key] if p]


def coverage(p, tops=(20, 100)):
    c = Counter(p)
    cum, run = [], 0
    for _, k in c.most_common():
        run += k
        cum.append(run)
    out = {f'top{k}': 100.0 * (cum[k - 1] if k <= len(cum) else cum[-1]) / len(p) for k in tops}
    out['types'] = len(c)
    return out


def pair_delta(G, idx, ka, kb):
    P = {k: pats(G, k, idx) for k in {'A', 'B', 'C', ka, kb}}
    n_c = min(len(v) for v in P.values())
    return S.mean(betas(P[kb], n_c)) - S.mean(betas(P[ka], n_c))


def half_interval(G, ka, kb):
    n = len(G)
    d = pair_delta(G, range(n), ka, kb)
    half = [pair_delta(G, random.Random(HALF_BASE + r).sample(range(n), n // 2), ka, kb) for r in range(HALF_REPS)]
    s = S.stdev(half)
    return {'delta': d, 'interval': [d - Z * s, d + Z * s], 'sd': s}


def unit_codes(codes, word_unit, n_units):
    out = [[] for _ in range(n_units)]
    for c, k in zip(codes, word_unit):
        out[k].append(c)
    return out


def main():
    Y = C.load_json('ud_system_output.json')['sentences']
    G = []
    for y in Y:
        A = [[y['codes'][i] for i in idx] for idx in y['v0_units']]
        B = unit_codes(y['codes'], y['gold_unit'], y['gold_n'])
        G.append({'tb': y['tb'], 'A': [C.to_blocks(k, 'B2') for k in A], 'B': [C.to_blocks(k, 'B2') for k in B],
                  'C': [C.to_blocks(k, 'B2') for k in y['gold_tags_units']], 'A_w': A, 'B_w': B})
    P = {k: pats(G, k) for k in ('A', 'B', 'C')}
    n_c = min(len(v) for v in P.values())
    out = {'sentences': len(G), 'N_c': n_c, 'table5': {}}
    for k in ('A', 'B', 'C'):
        b = betas(P[k], n_c)
        out['table5'][k] = {'units': len(P[k]), 'units_per_sentence': len(P[k]) / len(G),
                            'beta': S.mean(b), 'sd': S.stdev(b), **coverage(P[k])}
    out['gold_units'] = sum(y['gold_n'] for y in Y)
    out['gold_units_discontinuous'] = sum(sum(y['gold_unit_discontinuous']) for y in Y)
    out['B_minus_A'] = half_interval(G, 'A', 'B')
    Pw = {k: pats(G, k) for k in ('A_w', 'B_w')}
    ncw = min(n_c, len(Pw['A_w']), len(Pw['B_w']))
    out['ungrouped'] = {k: S.mean(betas(Pw[k], ncw)) for k in Pw}
    out['per_treebank'] = {}
    for tb in ('UT', 'UzUDT'):
        idx = [i for i, g in enumerate(G) if g['tb'] == tb]
        Pt = {k: pats(G, k, idx) for k in ('A', 'B', 'C')}
        nt = min(len(v) for v in Pt.values())
        bt = {k: S.mean(betas(Pt[k], nt)) for k in Pt}
        out['per_treebank'][tb] = {'N_c': nt, **bt, 'B_minus_A': bt['B'] - bt['A']}
    F = [{'tb': y['tb'], **{k: [C.to_blocks(list(p), 'B2') for p in y['split_clauses_patterns'][k]]
                           for k in ('A', 'B', 'C')}} for y in Y]
    out['conservative_rule'] = half_interval(F, 'A', 'B')
    C.write_result('ud_gold', out)
    t = out['table5']
    for k in t:
        print(k, round(t[k]['units_per_sentence'], 2), round(t[k]['beta'], 3), round(t[k]['sd'], 3), t[k]['types'],
              round(t[k]['top20'], 1), round(t[k]['top100'], 1))
    print('B-A', out['B_minus_A'], 'per tb', {tb: round(v['B_minus_A'], 3) for tb, v in out['per_treebank'].items()})
    print('ungrouped', out['ungrouped'], 'conservative', out['conservative_rule'])


if __name__ == '__main__':
    main()
