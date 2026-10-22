# KKT segmentation validity: automatic verb-bounded units vs. UD gold clauses (Uzbek)

Data and scripts for one question shared by two studies: how well do the **automatic verb-bounded units** of the
KKT analysis pipeline (rule V0: cut after every maximal run of verb codes) match the **gold clauses** of the two
Uzbek Universal Dependencies treebanks, and how much does the choice of boundaries change the Heaps-type
exponent β of the clause-pattern inventory?

Two measurements on the 1 182 treebank sentences (UD_Uzbek-UT release 2.18 test, UD_Uzbek-UzUDT release 2.18
train + test; 2 internal duplicates of UzUDT removed):

1. **Boundary precision and recall** of V0 against UD gold clause boundaries, for a broad and a narrow gold
   definition (`s11_ud_segmentation.py`).
2. **β comparison** of three clause inventories built on the same sentences (`s12_ud_gold.py`):
   A = V0 units on system tags, B = gold (narrow) clauses on the same system tags, C = gold clauses on gold tags.

`s10_ud_tagging.py` gives the treebank sizes and the tagging accuracy of the system tags against the manual UPOS
annotation, which the two measurements depend on.

## Results

### Boundary precision / recall (`results/ud_segmentation.json`)

A boundary counts only if at least one non-punctuation token lies on each side; exact match. *Broad* gold
definition (`asosiy`): `acl` and `xcomp` are clauses. *Narrow* (`xcomp_aclsiz`): they are not. "Aligned" = the
1 158 sentences whose system units could be aligned to the UD tokens; "all" = all 1 182 sentences.

| gold definition | sentences | precision | recall | F1 |
|---|---:|---:|---:|---:|
| broad | 1 158 (aligned) | 0.520 | 0.310 | 0.388 |
| broad | 1 182 (all) | 0.517 | 0.307 | 0.386 |
| narrow | 1 158 (aligned) | 0.302 | 0.285 | 0.293 |
| narrow | 1 182 (all) | 0.303 | 0.282 | 0.292 |

Gold units: broad 2 556 (2.16 per sentence), narrow 2 089 (1.77 per sentence). About half of the missed gold
boundaries have a non-finite clause head (broad 50.1 %, narrow 48.7 %, aligned sentences).

### β of the clause-pattern inventory (`results/ud_gold.json`)

Grouping B2; common grid up to N_c = 2 089 units; 20 shuffles, mean (SD).

| inventory | units | units / sentence | β | SD | types |
|---|---:|---:|---:|---:|---:|
| A — V0 units, system tags | 2 344 | 1.98 | 0.565 | 0.043 | 152 |
| B — gold clauses, system tags | 2 089 | 1.77 | 0.699 | 0.028 | 331 |
| C — gold clauses, gold tags | 2 089 | 1.77 | 0.660 | 0.031 | 231 |

**B − A = +0.134**, 95 % half-sample interval **[+0.095, +0.173]** (1 000 half-samples of 591 sentences).

| treebank | N_c | β A | β B | β C | B − A |
|---|---:|---:|---:|---:|---:|
| UD_Uzbek-UT | 742 | 0.582 | 0.712 | 0.660 | +0.129 |
| UD_Uzbek-UzUDT | 1 347 | 0.563 | 0.653 | 0.622 | +0.090 |

Also in the file: the same comparison without grouping (A 0.842, B 0.899) and for the conservative finite-only
rule set (B − A = −0.028, interval [−0.071, +0.014]).

### Tagging accuracy (`results/ud_tagging.json`)

79.5 % (8 650 / 10 874 scored tokens; UT 82.4 %, UzUDT 77.2 %), macro-F1 0.705.

## Recompute

Python 3.9 or newer, standard library only; nothing to install. From the repository root:

```
python3 scripts/s10_ud_tagging.py
python3 scripts/s11_ud_segmentation.py
python3 scripts/s12_ud_gold.py      # about 20 s
git diff --stat results/            # empty = identical
```

Each script overwrites its file in `results/`. Checked in a clean virtual environment with Python 3.9.6: all
three outputs are byte-identical to the files in `results/`. With Python 3.12 or newer, `s11` is still
byte-identical, while `s10` and `s12` differ only in the last digit of some floating-point values (float `sum()`
became compensated in Python 3.12); every printed digit stays the same.

**Limitation.** `data/ud_system_output.json` (the system's tokens, tags and V0 units on the treebank sentences,
with gold clause membership projected onto the system tokens) is the output of the KKT analyser, which is not
part of this repository. The scripts start from that file; rebuilding it requires the analyser.

## Contents

| path | content |
|---|---|
| `scripts/common.py`, `scripts/ud_tools.py` | shared helpers: β estimator, V0 rule, CoNLL-U reading, gold clause boundaries, alignment |
| `scripts/s10_ud_tagging.py` | treebank sizes, tagging accuracy |
| `scripts/s11_ud_segmentation.py` | boundary precision / recall |
| `scripts/s12_ud_gold.py` | β of inventories A / B / C, B − A interval |
| `data/ud/UD_Uzbek-UT/` | UD_Uzbek-UT release 2.18 test file, with its README and LICENSE |
| `data/ud/UD_Uzbek-UT_r2.15/` | UD_Uzbek-UT release 2.15 test file (token count of the initial release, `s10` only), with its README, `stats.xml` and LICENSE |
| `data/ud/UD_Uzbek-UzUDT/` | UD_Uzbek-UzUDT release 2.18 train and test files, with its README and LICENSE |
| `data/ud_system_output.json` | system output on the treebank sentences (see Limitation) |
| `data/ud_kkt_mapping.tsv` | mapping of UD UPOS tags to the KKT tag set |
| `data/coded_corpus_K1.json`, `data/coded_corpus_K2.json` | the two coded corpora (part-of-speech code sequences and V0 clauses only, no corpus text) |
| `results/` | outputs of `s10`, `s11`, `s12` |
| `MANIFEST.json` | sha256 of every file |

The UD treebank files are unchanged copies of the official releases. The coded corpora are the versions used in
both studies; their sha256 values equal those recorded in the Menzerath study's preregistration inventory:

- `coded_corpus_K1.json` — `b8a7033bfa5099aa962b67ebd06b9e05501e3697d02beb98e8b2dd413acdd49c`
- `coded_corpus_K2.json` — `689a0b140ec3325a224262940b21a87ee03fe0760f32b50a9a6d5013c9251d08`

They are included so that the inventories of this repository can be compared with those of the two studies; the
three scripts here do not read them.

## Note on internal references

The scripts are unchanged copies from the supplementary package of the Heaps study. Their docstrings cite
sections, tables and equations of that manuscript ("Section 4.2", "Table 5", …), and `common.py` contains helpers
that read files of that package which are not part of this repository (`unknown_positions.json`,
`heldout_sentences.json`); `s10`–`s12` do not call them. Identifiers and some labels are in Uzbek (for example
the gold definitions `asosiy` = broad and `xcomp_aclsiz` = narrow). The files have been left unedited so that their
sha256 values stay equal to the released package.

## Licences

Three layers, each with its own licence:

| layer | files | licence |
|---|---|---|
| UD-derived data | `data/ud/`, `data/ud_system_output.json` | **CC BY-SA 4.0**, the licence of the UD_Uzbek-UT and UD_Uzbek-UzUDT treebanks (see the `LICENSE.txt` files in `data/ud/`) |
| coded corpora and results | `data/coded_corpus_K1.json`, `data/coded_corpus_K2.json`, `data/ud_kkt_mapping.tsv`, `results/` | **CC BY 4.0** (`LICENSE-DATA`) |
| code | `scripts/` | **MIT** (`LICENSE-CODE`) |

`LICENSE` summarises the three layers.

## Source studies

- Sirojiddinov, Z., & Khakimov, M. *Heaps' law beyond the word: unit definition, granularity and the growth of
  part-of-speech schema inventories in Uzbek.* Submitted.
- Sirojiddinov, Z., & Khakimov, M. *What the Menzerath–Altmann exponent measures: clauses, morphemes and
  syllables in Uzbek.* In preparation. Supplementary package: https://doi.org/10.5281/zenodo.23562837

Please cite this repository with the metadata in `CITATION.cff`.
