# Glossary (hotword correction)

`glossary.py` repairs domain vocabulary on the **text** after ASR. NVIDIA Parakeet
TDT / Canary decode greedily and offer no acoustic word boosting (see
`docs/ASR-BACKENDS-PLAN.md` §13), so names and technical terms get mis-spelled and
have to be fixed deterministically afterwards. The module is stdlib-only, like
`exports.py`, so the export path can import it without touching the GPU stack.

## File format

`glossary.example.txt` is the reference. One entry per line, UTF-8 (a BOM is
tolerated); blank lines and `#` comments ignored:

```
Canonical                       # nothing known-bad to fix
Canonical<TAB>alias1;alias2     # ";"-separated known-bad ASR renderings
```

Canonical spellings are written out exactly as in the file — `MEMS` stays
uppercase, `Müller` keeps the umlaut. A canonical doubles as an implicit alias for
its own normalised form, so `mems` → `MEMS` (reported as rule `alias`). Aliases may
be multi-word (`Stress sensoren`), are matched case-insensitively and
punctuation-tolerantly, and the **longest alias wins** at a position.

## API

```python
g = load_glossary_if_exists("glossary.txt")        # None when missing/empty
text, changes = correct_text(text, g)              # or correct_segments(segments, g)
```

`load_glossary(path)` / `parse_glossary(text)` build the same `Glossary` (cheap,
build once, never mutated). `correct_segments` copies each segment and replaces
only `text` — the caller's dicts are not mutated and order/length are preserved.
Every rewrite comes back as an audit record, never silently:

```python
{"start": 4.879, "speaker": "speaker_0", "from": "Ebenmoduli", "to": "E-Modul",
 "rule": "alias" | "fuzzy"}
```

## How conservative it is, and why

* An exact canonical match (after normalisation) is **never** rewritten — which
  also makes a plain-word entry a shield for that word.
* Fuzzy matching (`difflib.SequenceMatcher` ratio on casefolded, umlaut-folded,
  hyphen/dot-stripped forms) only runs on tokens that matched no alias, only when
  the normalised token is ≥ `min_len` (default 4) and the ratio ≥ `min_ratio`
  (default 0.78), and never against multi-word canonicals.
* A rewrite that makes the word **longer** needs ratio ≥ 0.9 (`MIN_LENGTHEN_RATIO`),
  because dropping a heard suffix is far more often right than inventing one:
  `Motor` stays `Motor` next to a `Motorrad` entry.
* Repeated application is idempotent; correct output is stable under a second pass.

Measured on this repo's own `data/` with `glossary.example.txt`: at the 0.78 default
261 rewrites, of which 246 are `module`/`Module`/`Modul` → `E-Modul` (ratios
0.83–0.91) in text that meant the plain word; at `--min-ratio 0.86` 62 remain, i.e.
the alias hits plus one lengthening. The aliases carry the real fixes and ignore the
threshold, so list known-bad renderings rather than trusting the ratio, and give the
fuzzy pass a raised `min_ratio` on mixed-language corpora.

## Dry run

```
python glossary.py --glossary glossary.example.txt --check data/transcripts
```

Prints a `from -> to` table per file for `exports.py` markdown transcripts
(`**Speaker 0** `[00:00:04.879]`` + text) and `data/results/<hash>.json` sidecars,
reads a directory recursively, skips files of the wrong shape, writes nothing.
Exit code 0 unless the glossary or a named check target is unusable. Tests:
`python test_glossary.py`.
