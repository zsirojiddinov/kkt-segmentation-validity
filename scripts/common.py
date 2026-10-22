# -*- coding: utf-8 -*-
"""Shared helpers for all scripts in this package. Standard library only (Python 3.9+).

`to_blocks` and `heaps_beta` are verbatim copies of the estimator used for every exponent in the paper.
The V0 clause cut (section 4.2) is re-implemented here from the part-of-speech codes alone and is checked
against the stored clauses of the coded corpora every time `load_coded` is called.
"""
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

CORPORA = ('K1', 'K2')

# ----------------------------------------------------------------------------- estimator (verbatim)
NOMINAL_CODES = {"C", "P", "F", "M", "?", "X4", "X5", "X6"}
BLOCK_NOMINAL_B2 = {"C", "P", "F", "M"}
VERB_CODES = {"G"}
SEEDS = list(range(20))
N_POINTS = 30
N_MIN = 50


def to_blocks(codes, mode="B2"):
    """Collapse NP runs to a single C block and VP runs to a single G block.

    B2 (primary): '?' (out of vocabulary) is normalised to C first — an unknown
        word defaults to nominal — then runs of {C,P,F,M} collapse to one C and
        runs of G collapse to one G. N (adverb), Y (conjunction), G1
        (postposition) and D stay as their own blocks.
    B1 (sensitivity variant): '?' is kept distinct, and a nominal run is
        labelled by its LAST token's code (head-final principle).
    """
    if mode == "B2":
        codes = ["C" if c == "?" else c for c in codes]
        nominal, head_label = BLOCK_NOMINAL_B2, False
    else:
        nominal, head_label = NOMINAL_CODES, True

    out, i, n = [], 0, len(codes)
    while i < n:
        c = codes[i]
        if c in nominal:
            j = i
            while j + 1 < n and codes[j + 1] in nominal:
                j += 1
            out.append(codes[j] if head_label else "C")
            i = j + 1
        elif c in VERB_CODES:
            j = i
            while j + 1 < n and codes[j + 1] in VERB_CODES:
                j += 1
            out.append("G")
            i = j + 1
        else:
            out.append(c)
            i += 1
    return out


def heaps_beta(items, seeds=SEEDS, n_min=N_MIN, n_points=N_POINTS):
    """beta by log-log OLS regression, averaged over shuffles.

    NOTE: `items` order matters — see the module docstring.
    """
    total = len(items)
    if total < n_min * 2:
        n_min = max(10, total // 10)
    if total < 20:
        return {"beta": float("nan"), "sd": float("nan"),
                "n": total, "v": len(set(items))}

    grid = sorted({int(round(math.exp(x))) for x in [
        math.log(n_min) + i * (math.log(total) - math.log(n_min)) / (n_points - 1)
        for i in range(n_points)]})

    betas = []
    for seed in seeds:
        rnd = random.Random(seed)
        arr = items[:]
        rnd.shuffle(arr)
        seen, pts, gi = set(), [], 0
        for k, it in enumerate(arr, start=1):
            seen.add(it)
            if gi < len(grid) and k == grid[gi]:
                pts.append((k, len(seen)))
                gi += 1
        if not pts or pts[-1][0] != total:
            pts.append((total, len(seen)))
        xs = [math.log(a) for a, _ in pts]
        ys = [math.log(b) for _, b in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        betas.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
                     if den else float("nan"))

    mean = sum(betas) / len(betas)
    sd = (math.sqrt(sum((b - mean) ** 2 for b in betas) / len(betas))
          if len(betas) > 1 else 0.0)
    return {"beta": mean, "sd": sd, "n": total, "v": len(set(items))}


# ----------------------------------------------------------------------------- segmentation (section 4.2)
def v0_clauses(codes):
    """Cut after every maximal run of G codes. Returns a list of code lists."""
    out, cur = [], []
    for i, c in enumerate(codes):
        cur.append(c)
        if c == 'G' and (i + 1 == len(codes) or codes[i + 1] != 'G'):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def v0_index_clauses(codes):
    """Same cut as `v0_clauses`, returning token indices."""
    out, cur = [], []
    for i, c in enumerate(codes):
        cur.append(i)
        if c == 'G' and (i + 1 == len(codes) or codes[i + 1] != 'G'):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def b2(codes):
    return ' '.join(to_blocks(list(codes), 'B2'))


# ----------------------------------------------------------------------------- data
def load_json(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def load_coded(k):
    """Sentences of coded corpus K1 or K2: [{'codes': [...], 'clauses': [[...], ...]}]."""
    s = load_json(f'coded_corpus_{k}.json')['sentences']
    bad = sum(1 for x in s if v0_clauses(x['codes']) != x['clauses'])
    if bad:
        raise SystemExit(f'{k}: stored clauses differ from the V0 cut in {bad} sentences')
    return s


def unknown_codes(k, sents):
    """Codes with the tagger's unclassified tokens restored as '?' (they are stored as C)."""
    pos = load_json('unknown_positions.json')['positions'][k]
    kod = [list(s['codes']) for s in sents]
    for i, idx in pos:
        for j in idx:
            if kod[i][j] != 'C':
                raise SystemExit(f'{k}#{i}.{j}: unknown position is not C')
            kod[i][j] = '?'
    return kod


def heldout(k):
    return set(load_json('heldout_sentences.json')['sentences'][k])


def ranking(counter):
    """[(schema, count)] by count descending, ties by schema string."""
    return sorted(counter.items(), key=lambda t: (-t[1], t[0]))


def topk_pct(counter, k):
    tot = sum(counter.values())
    return 100 * sum(v for _, v in counter.most_common(k)) / tot


def write_result(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def Counter_of(items):
    return Counter(items)
