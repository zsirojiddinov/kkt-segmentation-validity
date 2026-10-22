# -*- coding: utf-8 -*-
"""UD evaluation functions used for sections 3.2, 4.2 and 5.5.1 (standard library only).

The function bodies are the evaluation code used for the paper, copied mechanically (docstrings and comments
removed, names unchanged; most identifiers are Uzbek). Glossary: gap = sentence, tokenlar = tokens,
oltin = gold, tizim = system, chegara = boundary, klauza = clause, sinf = class, savat = alignment bucket,
hizala = align, moslash = match, yaroqli = valid, qatiy = strict (exact), yumshoq = +/-1 tolerance,
soxta = false, otkazildi = missed, FINIT/NOFINIT/NOMINAL/ANIQLANMAGAN = finite / non-finite / nominal /
undetermined clause head. Gold clauses: heads with deprel root, ccomp, xcomp, advcl, acl, csubj, parataxis,
or conj between two predicates; variant 'asosiy' = wide, 'xcomp_aclsiz' = narrow (acl and xcomp not clauses).
The UPOS mapping is read from data/ud_kkt_mapping.tsv.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'
UD_DIR = DATA / 'ud'
TREEBANK = {
    'UT': ('UD_Uzbek-UT/uz_ut-ud-test.conllu',),
    'UzUDT': ('UD_Uzbek-UzUDT/uz_uzudt-ud-train.conllu', 'UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu'),
}
with open(DATA / 'ud_kkt_mapping.tsv', encoding='utf-8') as _f:
    _MAP = {r['UPOS']: r for r in csv.DictReader(_f, delimiter='\t')}
MOSLASH = {u: r['tag'] for u, r in _MAP.items() if r['treatment'] == 'mapped'}
KKT_TEGLAR = ('C', 'P', 'G', 'N', 'M', 'F', 'Y', 'G1')


def kkt_teg(upos):
    r = _MAP.get(upos)
    if r is None:
        return "noma'lum", None
    return {'mapped': ('teg', r['tag']), 'excluded': ('chiqariladi', None), 'unmappable': ('moslanmagan', None)}[r['treatment']]


def load_treebank(key, excluded=()):
    out = []
    for rel in TREEBANK[key]:
        for g in conllu_gaplar((UD_DIR / rel).read_text(encoding='utf-8')):
            if g['sent_id'] in excluded:
                continue
            g['fayl'] = rel
            g['tb'] = key
            out.append(g)
    return out


# ---- etalon


KLAUZA_ASOS = ('root', 'ccomp', 'xcomp', 'advcl', 'acl', 'csubj', 'parataxis')


XCOMP_ACL = ('xcomp', 'acl')


PREDIKAT_UPOS = ('VERB', 'AUX')


def conllu_gaplar(matn):
    (gaplar, sid, text, toks) = ([], None, None, [])
    for satr in matn.splitlines() + ['']:
        s = satr.rstrip('\n')
        if not s.strip():
            if toks or text is not None:
                gaplar.append({'sent_id': sid, 'text': text, 'tokenlar': toks})
            (sid, text, toks) = (None, None, [])
            continue
        if s.startswith('#'):
            if s.startswith('# sent_id'):
                sid = s.split('=', 1)[1].strip()
            elif s.startswith('# text'):
                text = s.split('=', 1)[1].strip()
            continue
        p = s.split('\t')
        if '-' in p[0] or '.' in p[0]:
            continue
        toks.append({'id': int(p[0]), 'form': p[1], 'upos': p[3], 'head': int(p[6]) if p[6] != '_' else 0, 'deprel': p[7]})
    return gaplar


def _bolalar(toks):
    b = {t['id']: [] for t in toks}
    b[0] = []
    for t in toks:
        b.setdefault(t['head'], []).append(t['id'])
    return b


def _subtree(kok, bolalar):
    (yigin, stek) = (set(), [kok])
    while stek:
        v = stek.pop()
        if v in yigin:
            continue
        yigin.add(v)
        stek.extend(bolalar.get(v, ()))
    return yigin


def predikatmi(t, bolalar, xarita):
    if t['upos'] in PREDIKAT_UPOS:
        return True
    return any((xarita[c]['deprel'].split(':')[0] == 'cop' for c in bolalar.get(t['id'], ())))


def klauza_boshlari(gap, variant):
    toks = gap['tokenlar']
    xarita = {t['id']: t for t in toks}
    bolalar = _bolalar(toks)
    asos = tuple((d for d in KLAUZA_ASOS if not (variant == 'xcomp_aclsiz' and d in XCOMP_ACL)))
    chiqish = []
    for t in toks:
        d = t['deprel'].split(':')[0]
        if d in asos:
            chiqish.append({'id': t['id'], 'deprel': t['deprel'], 'sabab': d})
        elif d == 'conj':
            bosh = xarita.get(t['head'])
            if variant == 'conj_shartsiz':
                chiqish.append({'id': t['id'], 'deprel': t['deprel'], 'sabab': 'conj (shartsiz)'})
            elif bosh is not None and predikatmi(t, bolalar, xarita) and predikatmi(bosh, bolalar, xarita):
                chiqish.append({'id': t['id'], 'deprel': t['deprel'], 'sabab': 'conj (predikat)'})
    return chiqish


def gap_etaloni(gap, variant):
    toks = gap['tokenlar']
    n = len(toks)
    bolalar = _bolalar(toks)
    (klauza, chegara) = ([], set())
    for b in klauza_boshlari(gap, variant):
        st = _subtree(b['id'], bolalar)
        (i, j) = (min(st), max(st))
        klauza.append({**b, 'span': [i, j], 'token_soni': len(st), 'uzluksiz': j - i + 1 == len(st)})
        for p in (i, j + 1):
            if 1 < p <= n:
                chegara.add(p)
    return {'n': n, 'klauza': sorted(klauza, key=lambda k: (k['span'][0], k['id'])), 'chegara': sorted(chegara)}


# ---- teg_baho


APOSTROF = "'’ʻʼ`‘"


MAX_CHUNK = 4


ATLASH_OYNASI = 8


def squash(s):
    t = (s or '').lower()
    for ch in APOSTROF:
        t = t.replace(ch, '')
    return ''.join((ch for ch in t if ch.isalnum()))


def chunk_spanlari(text):
    (spanlar, i, n) = ([], 0, len(text))
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and (not text[j].isspace()):
            j += 1
        spanlar.append((i, j))
        i = j
    return spanlar


def ud_chunk_xarita(text, ud_toklar):
    spanlar = chunk_spanlari(text)
    (xarita, kursor) = ([], 0)
    for t in ud_toklar:
        idx = text.find(t['form'], kursor)
        if idx < 0:
            return None
        kursor = idx + len(t['form'])
        ci = next((k for (k, (a, b)) in enumerate(spanlar) if a <= idx < b), None)
        if ci is None:
            return None
        xarita.append(ci)
    return xarita


def tizim_chunk_xarita(text, sys_toklar):
    chunklar = text.split()
    (xarita, ci) = ([], 0)
    for st in sys_toklar:
        nishon = squash(st.get('word'))
        topildi = None
        for skip in range(0, ATLASH_OYNASI + 1):
            bosh = ci + skip
            if bosh >= len(chunklar):
                break
            for k in range(1, MAX_CHUNK + 1):
                if bosh + k > len(chunklar):
                    break
                cand = ' '.join(chunklar[bosh:bosh + k])
                sc = squash(cand)
                if sc == nishon and (sc or cand == (st.get('word') or '')):
                    topildi = (bosh, bosh + k)
                    break
            if topildi:
                break
        if not topildi:
            return None
        xarita.append(topildi)
        ci = topildi[1]
    return xarita


def hizala(text, ud_toklar, sys_toklar):
    uc = ud_chunk_xarita(text, ud_toklar)
    if uc is None:
        return (None, 'ud_matn_mos_emas')
    sc = tizim_chunk_xarita(text, sys_toklar)
    if sc is None:
        return (None, 'tizim_greedy_yiqildi')
    chunk_ud = defaultdict(list)
    for (t, ci) in zip(ud_toklar, uc):
        chunk_ud[ci].append(t)
    (birliklar, korilgan) = ([], set())
    for (st, (a, b)) in zip(sys_toklar, sc):
        korilgan.update(range(a, b))
        ud = [t for c in range(a, b) for t in chunk_ud.get(c, ())]
        nonpunct = [t for t in ud if kkt_teg(t['upos'])[0] != 'chiqariladi']
        if not nonpunct:
            savat = 'faqat_tizim'
        elif len(nonpunct) > 1:
            savat = 'birlashgan'
        elif kkt_teg(nonpunct[0]['upos'])[0] == 'moslanmagan':
            savat = 'moslanmagan'
        else:
            savat = 'taqqoslanadi'
        birliklar.append({'savat': savat, 'chunk': [a, b], 'tizim': st.get('code'), 'tizim_word': st.get('word'), 'ud': nonpunct})
    for (ci, toklar) in chunk_ud.items():
        if ci in korilgan:
            continue
        nonpunct = [t for t in toklar if kkt_teg(t['upos'])[0] != 'chiqariladi']
        if nonpunct:
            birliklar.append({'savat': 'faqat_ud', 'chunk': [ci, ci + 1], 'tizim': None, 'tizim_word': None, 'ud': nonpunct})
    return (birliklar, None)


def prf(matritsa, sinflar):
    chiqish = {}
    for s in sinflar:
        tp = matritsa.get((s, s), 0)
        oltin = sum((v for ((o, _t), v) in matritsa.items() if o == s))
        tizim = sum((v for ((_o, t), v) in matritsa.items() if t == s))
        p = tp / tizim if tizim else 0.0
        r = tp / oltin if oltin else 0.0
        chiqish[s] = {'p': p, 'r': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0, 'oltin': oltin, 'tizim': tizim, 'togri': tp}
    return chiqish


# ---- seg_baho


def yaroqli_kesimlar(ud_toklar):
    nonpunct = [t['id'] for t in ud_toklar if kkt_teg(t['upos'])[0] != 'chiqariladi']
    if not nonpunct:
        return set()
    n = len(ud_toklar)
    return {p for p in range(2, n + 1) if nonpunct[0] < p <= nonpunct[-1]}


def _subseq(chunklar, sozlar, boshlash):
    (i, birinchi, oxirgi) = (boshlash, None, None)
    for w in sozlar:
        while i < len(chunklar) and chunklar[i] != w:
            i += 1
        if i >= len(chunklar):
            return None
        if birinchi is None:
            birinchi = i
        oxirgi = i
        i += 1
    return None if birinchi is None else (birinchi, oxirgi)


def klauza_spanlari(text, klauzalar):
    chunklar = [squash(w) for w in text.split()]
    (spanlar, kursor) = ([], 0)
    for c in klauzalar:
        sozlar = [w for w in (squash(x) for x in (c.get('text') or '').split()) if w]
        if not sozlar:
            continue
        span = _subseq(chunklar, sozlar, kursor)
        if span is None:
            span = _subseq(chunklar, sozlar, 0)
        if span is None:
            return None
        spanlar.append(span)
        if span[0] >= kursor:
            kursor = span[1] + 1
    return spanlar


def tizim_chegara(text, ud_toklar, klauzalar):
    uc = ud_chunk_xarita(text, ud_toklar)
    if uc is None:
        return (None, None)
    spanlar = klauza_spanlari(text, klauzalar)
    if spanlar is None:
        return (None, None)
    n = len(ud_toklar)
    yaroqli = yaroqli_kesimlar(ud_toklar)
    birinchi_token = {}
    for (t, ci) in zip(ud_toklar, uc):
        birinchi_token.setdefault(ci, t['id'])

    def _p(c):
        q = birinchi_token.get(c)
        return q if q is not None and 1 < q <= n and (q in yaroqli) else None
    chegara = set()
    for (a, b) in spanlar:
        for c in (a, b + 1):
            q = _p(c)
            if q is not None:
                chegara.add(q)
    tartib = sorted(spanlar)
    mintaqa = []
    for k in range(len(tartib) - 1):
        (oxir, keyingi) = (tartib[k][1], tartib[k + 1][0])
        qq = {x for x in (_p(c) for c in range(oxir + 1, keyingi + 1)) if x is not None}
        if qq:
            mintaqa.append(sorted(qq))
    yolgiz = set(chegara) - {x for m in mintaqa for x in m}
    mintaqalar = [[x] for x in sorted(yolgiz)] + mintaqa
    return (chegara, {'spanlar': spanlar, 'mintaqalar': mintaqalar})


def oltin_provenans(gap, variant):
    e = gap_etaloni(gap, variant)
    n = e['n']
    yaroqli = yaroqli_kesimlar(gap['tokenlar'])
    beruvchi = {}
    for k in e['klauza']:
        (i, j) = k['span']
        for p in (i, j + 1):
            if 1 < p <= n and p in yaroqli:
                beruvchi.setdefault(p, []).append(k['uzluksiz'])
    chegara = set(beruvchi)
    faqat = {p for (p, uzl) in beruvchi.items() if not any(uzl)}
    return (chegara, faqat)


def moslash(oltin, tizim, tolerans=0):
    bosh = sorted(tizim)
    ishlatilgan = set()
    mos = set()
    for p in sorted(oltin):
        nomzod = [q for q in bosh if q not in ishlatilgan and abs(q - p) <= tolerans]
        if nomzod:
            ishlatilgan.add(nomzod[0])
            mos.add(p)
    return (len(mos), mos)


def prf1(tp, n_tizim, n_oltin):
    p = tp / n_tizim if n_tizim else 0.0
    r = tp / n_oltin if n_oltin else 0.0
    return {'p': p, 'r': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0, 'tp': tp, 'tizim': n_tizim, 'oltin': n_oltin}


# ---- tarif


OLTINLAR = ('asosiy', 'xcomp_aclsiz')


SINFLAR = ('FINIT', 'ANIQLANMAGAN', 'NOMINAL', 'NOFINIT')


NOFINIT_VF = ('Conv', 'Part', 'Vnoun', 'Inf', 'Ger')


def feats_xarita(matn):
    (out, sid) = ({}, None)
    for s in matn.splitlines():
        if s.startswith('# sent_id'):
            sid = s.split('=', 1)[1].strip()
        elif s and (not s.startswith('#')):
            p = s.split('\t')
            if '-' in p[0] or '.' in p[0]:
                continue
            out[sid, int(p[0])] = dict((x.split('=', 1) for x in p[5].split('|') if '=' in x))
    return out


def kesim_sinfi(bosh, bolalar, feats):
    kesim = [bosh] + [b for b in bolalar if b['deprel'].split(':')[0] in ('aux', 'cop')]
    for t in kesim:
        f = feats.get(t['id'], {})
        if f.get('VerbForm') == 'Fin':
            return 'FINIT'
        if 'VerbForm' not in f and ('Person' in f or 'Evident' in f or ('Tense' in f and 'Case' not in f)):
            return 'FINIT'
    fb = feats.get(bosh['id'], {})
    if fb.get('VerbForm') in NOFINIT_VF:
        return 'NOFINIT'
    if bosh['upos'] == 'VERB' and 'VerbForm' not in fb and ('Case' in fb):
        return 'NOFINIT'
    if bosh['upos'] not in ('VERB', 'AUX'):
        return 'NOMINAL'
    return 'ANIQLANMAGAN'


def chegara_sinflari(gap, oltin, feats):
    e = gap_etaloni(gap, oltin)
    n = e['n']
    yaroqli = yaroqli_kesimlar(gap['tokenlar'])
    x = {t['id']: t for t in gap['tokenlar']}
    bol = defaultdict(list)
    for t in gap['tokenlar']:
        bol[t['head']].append(t)
    egalar = defaultdict(list)
    for k in e['klauza']:
        (i, j) = k['span']
        s = kesim_sinfi(x[k['id']], bol[k['id']], feats)
        for p in (i, j + 1):
            if 1 < p <= n and p in yaroqli:
                egalar[p].append((k['deprel'].split(':')[0], feats.get(k['id'], {}).get('VerbForm', '—'), s))
    return {p: (next((c for c in SINFLAR if any((s == c for (_, _, s) in eg)))), eg) for (p, eg) in egalar.items()}


def fin_chegaralari(m_F, n_F, u_m, U):
    return {'R_F': m_F / n_F if n_F else 0.0, 'R_past': m_F / (n_F + U - u_m) if n_F + U - u_m else 0.0, 'R_yuqori': (m_F + u_m) / (n_F + u_m) if n_F + u_m else 0.0}


def fp_sinfi(gap, p, feats):
    t = [t for t in gap['tokenlar'] if t['id'] < p and kkt_teg(t['upos'])[0] != 'chiqariladi'][-1]
    vf = feats.get(t['id'], {}).get('VerbForm')
    if t['upos'] in ('VERB', 'AUX'):
        return {'Fin': 'VERB/AUX Fin', 'Part': 'VERB Part', 'Conv': 'VERB Conv', 'Vnoun': 'VERB Vnoun/Inf', 'Inf': 'VERB Vnoun/Inf'}.get(vf, 'VERB/AUX belgisiz')
    return f"fe'l emas ({t['upos']})"


def v0_tizim(v0_p, tokenlar):
    yaroqli = yaroqli_kesimlar(tokenlar)
    n = len(tokenlar)
    return {p for p in v0_p if 1 < p <= n and p in yaroqli}


def baholash(juftlar, feats_all):
    out = {}
    for oltin in OLTINLAR:
        blok = {}
        for tb in ('UT', 'UzUDT', 'jami'):
            nq = [(g, sy) for (g, sy) in juftlar if tb == 'jami' or g['tb'] == tb]
            ol_n = sy_n = tp_q = tp_y = 0
            sinf = {c: Counter() for c in SINFLAR}
            nofin = Counter()
            fp = Counter()
            for (g, sy) in nq:
                feats = feats_all[g['fayl']][g['sent_id']]
                cs = chegara_sinflari(g, oltin, feats)
                ol = set(cs)
                assert ol == oltin_provenans(g, oltin)[0]
                (q, mos_q) = moslash(ol, sy, 0)
                (y, mos_y) = moslash(ol, sy, 1)
                ol_n += len(ol)
                sy_n += len(sy)
                tp_q += q
                tp_y += y
                for (p, (c, eg)) in cs.items():
                    sinf[c]['n'] += 1
                    sinf[c]['q'] += p in mos_q
                    sinf[c]['y'] += p in mos_y
                    if c == 'NOFINIT':
                        (d, vf, _) = next((e for e in eg if e[2] == 'NOFINIT'))
                        nofin[f'{d}-{vf}'] += 1
                        nofin[f'{d}-{vf}|otkazildi'] += p not in mos_q
                for p in sy - mos_q:
                    fp[fp_sinfi(g, p, feats)] += 1
            s = {c: {'n': v['n'], 'mos_qatiy': v['q'], 'mos_yumshoq': v['y'], 'otkazildi_qatiy': v['n'] - v['q'], 'R_qatiy': round(v['q'] / v['n'], 4) if v['n'] else None, 'R_yumshoq': round(v['y'] / v['n'], 4) if v['n'] else None} for (c, v) in sinf.items()}
            otk = sum((v['otkazildi_qatiy'] for v in s.values()))
            for c in s:
                s[c]['otkazilganda_ulush'] = round(100 * s[c]['otkazildi_qatiy'] / otk, 2) if otk else None
            (F, U) = (s['FINIT'], s['ANIQLANMAGAN'])
            fin = {m: {k: round(v, 4) for (k, v) in fin_chegaralari(F[f'mos_{m}'], F['n'], U[f'mos_{m}'], U['n']).items()} for m in ('qatiy', 'yumshoq')}
            blok[tb] = {'gap': len(nq), 'qatiy': prf1(tp_q, sy_n, ol_n), 'yumshoq': prf1(tp_y, sy_n, ol_n), 'sinf': s, 'finit_recall': fin, 'nofinit_deprel_verbform': dict(sorted(((k, v) for (k, v) in nofin.items() if '|' not in k))), 'nofinit_otkazildi': {k.split('|')[0]: v for (k, v) in nofin.items() if '|' in k}, 'soxta_chegara': dict(fp.most_common()), 'soxta_jami': sum(fp.values())}
        out[oltin] = blok
    return out
