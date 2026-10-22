# -*- coding: utf-8 -*-
"""Sections 2.4 and 3.2 (first measurement): UD treebank sizes and tagging accuracy of the corpus tagger
against the manual UPOS annotation of UD_Uzbek-UT and UD_Uzbek-UzUDT (release 2.18).

Sentences: all UD sentences except internal duplicates (a sentence whose text, reduced to letters and digits,
already occurred earlier in the same treebank; files read train before test). System words are aligned to UD tokens (`ud_tools.hizala`); accuracy is computed over 1:1
units whose UD token has a mapped UPOS (data/ud_kkt_mapping.tsv); PUNCT is excluded, INTJ/SYM/X are counted
separately, many-to-one units are not scored.

Input: data/ud/*, data/ud_system_output.json.  Output: results/ud_tagging.json
"""
from collections import Counter

import common as C
import ud_tools as U


def count_conllu(path):
    sents = toks = 0
    for g in U.conllu_gaplar(path.read_text(encoding='utf-8')):
        sents += 1
        toks += len(g['tokenlar'])
    return sents, toks


def main():
    out = {'sizes': {}}
    for name, rel in (('UT', ['UD_Uzbek-UT/uz_ut-ud-test.conllu']),
                      ('UzUDT', ['UD_Uzbek-UzUDT/uz_uzudt-ud-train.conllu', 'UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu']),
                      ('UT_r2.15', ['UD_Uzbek-UT_r2.15/uz_ut-ud-test.conllu'])):
        s = [count_conllu(U.UD_DIR / r) for r in rel]
        out['sizes'][name] = {'sentences': sum(x[0] for x in s), 'tokens': sum(x[1] for x in s)}
    ud, dup = {}, {}
    for tb in U.TREEBANK:
        seen, dup[tb] = set(), []
        for g in U.load_treebank(tb):
            key = U.squash(g['text'])          # letters and digits only
            if key in seen:
                dup[tb].append(g['sent_id'])
                continue
            seen.add(key)
            if (tb, g['sent_id']) in ud:
                raise SystemExit(f'sent_id not unique: {tb} {g["sent_id"]}')
            ud[(tb, g['sent_id'])] = g
    out['duplicates_removed'] = dup
    out['sentences_used'] = len(ud)
    sysout = C.load_json('ud_system_output.json')['sentences']
    if {(y['tb'], y['sent_id']) for y in sysout} != set(ud):
        raise SystemExit('system output and UD sentence sets differ')
    mat = {'UT': Counter(), 'UzUDT': Counter()}
    buckets, unaligned = Counter(), 0
    for y in sysout:
        g = ud[(y['tb'], y['sent_id'])]
        units, _ = U.hizala(g['text'], g['tokenlar'], [{'word': w, 'code': c} for w, c in zip(y['words'], y['codes'])])
        if units is None:
            unaligned += 1
            continue
        for u in units:
            buckets[u['savat']] += 1
            if u['savat'] == 'taqqoslanadi':
                mat[y['tb']][(U.MOSLASH[u['ud'][0]['upos']], u['tizim'])] += 1

    def score(m):
        n = sum(m.values())
        ok = sum(v for (o, t), v in m.items() if o == t)
        p = U.prf(m, U.KKT_TEGLAR)
        return {'n': n, 'correct': ok, 'accuracy_pct': 100 * ok / n,
                'macro_f1': sum(p[c]['f1'] for c in U.KKT_TEGLAR) / len(U.KKT_TEGLAR)}

    tot = mat['UT'] + mat['UzUDT']
    out['accuracy'] = {'UT': score(mat['UT']), 'UzUDT': score(mat['UzUDT']), 'all': score(tot)}
    out['top_confusions'] = [[o, t, v] for (o, t), v in sorted(tot.items(), key=lambda x: -x[1]) if o != t][:10]
    outside = Counter()
    for (o, t), v in tot.items():
        if t not in U.KKT_TEGLAR:
            outside[t] += v
    out['system_codes_outside_tagset'] = dict(outside)
    out['system_codes_outside_tagset_total'] = sum(v for (o, t), v in tot.items() if t not in U.KKT_TEGLAR)
    upos = Counter(t['upos'] for g in ud.values() for t in g['tokenlar'])
    out['ud_unmappable_tokens'] = {u: upos[u] for u in ('INTJ', 'X', 'SYM')}
    out['ud_nonpunct_tokens'] = sum(v for u, v in upos.items() if u != 'PUNCT')
    out['alignment_buckets'] = dict(buckets)
    out['unaligned_sentences'] = unaligned
    C.write_result('ud_tagging', out)
    a = out['accuracy']
    print(f"sizes {out['sizes']} · used {len(ud)} · dup {dup}")
    print(f"accuracy {a['all']['correct']}/{a['all']['n']} = {a['all']['accuracy_pct']:.2f} % macro-F1 "
          f"{a['all']['macro_f1']:.3f} · UT {a['UT']['accuracy_pct']:.2f} · UzUDT {a['UzUDT']['accuracy_pct']:.2f}")
    print('confusions', out['top_confusions'][:4], 'outside', out['system_codes_outside_tagset'],
          'unmappable', out['ud_unmappable_tokens'])


if __name__ == '__main__':
    main()
