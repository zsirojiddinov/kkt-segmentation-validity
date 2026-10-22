# -*- coding: utf-8 -*-
"""Section 4.2: precision and recall of the unit boundaries of rule V0 against UD gold clause boundaries.

Gold boundaries are derived from the UD trees (`ud_tools.gap_etaloni`): wide definition 'asosiy' (acl and xcomp
are clauses) and narrow 'xcomp_aclsiz'. System boundaries: the V0 cuts, as UD token ids (data/ud_system_output.json).
A boundary counts only if at least one non-punctuation token lies on each side. Exact match.
Reported on the sentences whose units could be aligned to the gold tokens, and on all sentences.
Gold units per sentence (one unit per clause head) are also given for both definitions.
Missed gold boundaries are classified by their clause head (finite / non-finite / nominal / undetermined);
false boundaries by the gold category of the last non-punctuation token before them.

Output: results/ud_segmentation.json
"""
from collections import Counter, defaultdict

import common as C
import ud_tools as U


def main():
    gaplar = []
    for tb in U.TREEBANK:
        seen = set()
        for g in U.load_treebank(tb):
            key = U.squash(g['text'])
            if key in seen:
                continue
            seen.add(key)
            gaplar.append(g)
    feats_all = defaultdict(lambda: defaultdict(dict))
    for fl in {g['fayl'] for g in gaplar}:
        for (sid, i), f in U.feats_xarita((U.UD_DIR / fl).read_text(encoding='utf-8')).items():
            feats_all[fl][sid][i] = f
    sysout = {(y['tb'], y['sent_id']): y for y in C.load_json('ud_system_output.json')['sentences']}
    aligned = set()
    for g in gaplar:
        y = sysout[(g['tb'], g['sent_id'])]
        ch, _ = U.tizim_chegara(g['text'], g['tokenlar'], [{'text': x} for x in y['split_clauses_text']])
        if ch is not None:
            aligned.add((g['tb'], g['sent_id']))
    pairs = [(g, U.v0_tizim(sysout[(g['tb'], g['sent_id'])]['v0_cuts_ud'], g['tokenlar'])) for g in gaplar]
    out = {'sentences': len(gaplar), 'n_aligned': len(aligned), 'n_not_aligned': len(gaplar) - len(aligned)}
    out['gold_units'] = {gold: sum(len(U.klauza_boshlari(g, gold)) for g in gaplar) for gold in ('asosiy', 'xcomp_aclsiz')}
    out['gold_units_per_sentence'] = {gold: n / len(gaplar) for gold, n in out['gold_units'].items()}
    for name, sel in (('aligned', [p for p in pairs if (p[0]['tb'], p[0]['sent_id']) in aligned]), ('all', pairs)):
        r = U.baholash(sel, feats_all)
        res = {}
        for gold in ('asosiy', 'xcomp_aclsiz'):
            j = r[gold]['jami']
            fp = j['soxta_chegara']
            res[gold] = {'p': j['qatiy']['p'], 'r': j['qatiy']['r'], 'f1': j['qatiy']['f1'],
                         'missed_nonfinite_pct': j['sinf']['NOFINIT']['otkazilganda_ulush'],
                         'false_total': j['soxta_jami'],
                         'false_after_nonverb_pct': 100 * sum(v for k, v in fp.items() if k.startswith("fe'l emas")) / j['soxta_jami'],
                         'false_after_participle_pct': 100 * fp.get('VERB Part', 0) / j['soxta_jami']}
        out[name] = res
    C.write_result('ud_segmentation', out)
    for name in ('aligned', 'all'):
        print(name, {g: (round(v['p'], 3), round(v['r'], 3), v['missed_nonfinite_pct'],
                         round(v['false_after_nonverb_pct'], 1), round(v['false_after_participle_pct'], 1))
                     for g, v in out[name].items()})
    print('aligned', out['n_aligned'], 'not aligned', out['n_not_aligned'])


if __name__ == '__main__':
    main()
