"""Text-side glossary ("hotword") correction for ASR transcripts.

Parakeet TDT / Canary decode greedily and offer no acoustic word boosting, so
domain vocabulary (names, technical terms) has to be repaired deterministically
on the text afterwards. This module does that in one conservative pass and
reports every change it made, so a correction can always be audited instead of
silently rewriting a transcript.

Deliberately stdlib-only (re / difflib / unicodedata / json / argparse /
dataclasses / typing), like exports.py: the export path that renders transcripts
must not pull torch/nemo in just to fix spellings.

File format (`--glossary FILE`, UTF-8 with or without BOM)::

    # comments and blank lines are ignored; one entry per line
    Canonical                     # nothing else to fix
    Canonical<TAB>alias1;alias2   # aliases are ';'-separated known-bad renderings

* Canonical spellings are written out exactly as in the file: `MEMS` stays
  uppercase, `Müller` keeps the umlaut. A canonical also acts as an implicit
  alias for its own normalised form, so `mems` is repaired to `MEMS` (reported
  under rule "alias"); a token already spelled like the canonical is untouched.
* Aliases are matched case-insensitively and punctuation-tolerantly, and may be
  multi-word phrases (`Stress sensoren`). Punctuation is dropped from both sides,
  so it also splits an alias into words: `S/4 HANA` is the three-token alias
  `s 4 hana`. The longest alias wins at a position, so a phrase always beats a
  single word starting at the same token.
* Tokens that matched no alias get one fuzzy chance: `difflib` sequence ratio
  against the normalised canonical (casefolded, umlauts folded ä→ae/ö→oe/ü→ue/
  ß→ss, hyphens and dots ignored). Applied only when the token is not already a
  canonical spelling, its normalised length is >= `min_len` and the ratio is
  >= `min_ratio`. **A rewrite that would make the written word longer never
  fires fuzzily** (`MIN_LENGTHEN_RATIO = 1.0`, length counted in written letters,
  so the umlaut repair `Muller` → `Müller` still counts as the same size): dropping
  a suffix (`Kraftsensori` → `Kraftsensor`) is far more often right than inventing
  one, and every measured false positive in this repo's transcripts was a
  lengthening — `Modul`, `Module`, `model` → `E-Modul` at 0.83-0.91. Write such a
  term as an explicit alias instead (`E-Modul\tE Modul;E-Moduls`). Listing a
  plain word as its own canonical is the other shield: an exact canonical match is
  never rewritten.
* A line starting with `!` protects words rather than introducing one — e.g.
  `!Modul;Module` — and those tokens are then untouched by *every* rule, so a
  technical canonical cannot swallow an ordinary word without claiming it.

Typical use::

    g = load_glossary_if_exists("glossary.txt")          # build once
    if g:
        text, changes = correct_text(text, g)            # or correct_segments(...)

`python glossary.py --glossary glossary.example.txt --check FILE_OR_DIR` prints
what a glossary would change in the transcripts on disk, without writing.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# A rewrite that lengthens a token has to clear this bar rather than `min_ratio`.
# 1.0 = never: lengthenings must be declared as aliases (see the module docstring
# for the measured false positives that led here). Lower it to ~0.9 to allow
# close prefix expansions such as `Modul` → `E-Modul` again.
MIN_LENGTHEN_RATIO = 1.0

# Word tokens: runs of letters/digits, joined across an *internal* hyphen or
# apostrophe so `E-Modul` and `Nord-Süd` stay one token, while sentence
# punctuation stays outside the replaceable span.
_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)

# German umlauts folded the way German text itself expands them, so a transcript
# that lost the diacritics still matches. Applied after casefold(), hence lowercase.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_DROP = "-.'’_\u2019"

_MD_SEG_RE = re.compile(
    r"^\*\*(?P<spk>[^*]+)\*\*\s+`+\[(?P<ts>[0-9:,.]+)\]`+[ \t]*\n"
    r"(?P<text>[^\n]*(?:\n(?!\*\*)[^\n]*)*)",
    re.MULTILINE,
)
_TS_RE = re.compile(r"^(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)(?:[.,](?P<ms>\d+))?$")


def _norm_token(token: str) -> str:
    """Normalised comparison form of one token; '' when nothing is left."""
    s = unicodedata.normalize("NFC", token).casefold().translate(_FOLD)
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c) and c not in _DROP)


def _norm(text: str) -> str:
    """Normalised form of a whole string: space-joined normalised tokens."""
    return " ".join(n for n in (_norm_token(t) for t in _TOKEN_RE.findall(text)) if n)


def _alnum_len(s: str) -> int:
    """Written size of a word: letters and digits, ignoring case, marks, punctuation."""
    return sum(1 for c in s if c.isalnum())


@dataclass(frozen=True)
class _Cand:
    """One alias rule: a normalised token sequence → the canonical to write."""
    toks: tuple[str, ...]
    canon: str


@dataclass
class Glossary:
    """Parsed glossary. Cheap to build, cheap to share, never mutated by correct_*."""
    canons: tuple[str, ...] = ()
    by_first: dict[str, list[_Cand]] = field(default_factory=dict)
    exact: frozenset[str] = frozenset()      # every normalised canonical form
    fuzzy: tuple[tuple[str, str], ...] = ()  # (normalised, canonical) single-token only
    protected: tuple[tuple[str, ...], ...] = ()  # token runs no rewrite may touch

    def __len__(self) -> int:
        return len(self.canons)

    def __iter__(self):
        return iter(self.canons)


def _add(entries: dict[str, list[str]], canon: str, aliases: Iterable[str]) -> None:
    canon = canon.strip()
    if not canon:
        return
    known = entries.setdefault(canon, [])
    for alias in aliases:
        alias = alias.strip()
        if alias:
            known.append(alias)


def parse_glossary(text: str) -> Glossary:
    """Parse glossary text. Duplicate canonicals merge their alias lists.

    A line starting with ``!`` protects words instead of introducing one: those
    tokens are never rewritten by any rule, which is how an ordinary word
    (`Modul`, `Model`) escapes the fuzzy reach of a technical canonical.
    """
    entries: dict[str, list[str]] = {}
    protected: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        canon, _, aliases = line.partition("\t")
        canon = canon.strip()
        if canon.startswith("!"):
            protected.extend(w for w in (canon[1:] + ";" + aliases).split(";") if w.strip())
            continue
        _add(entries, canon, () if not aliases else tuple(aliases.split(";")))
    return _build(entries, protected)


def load_glossary(path: "str | os.PathLike") -> Glossary:
    """Read a glossary file (UTF-8, BOM tolerated — these files are hand-edited)."""
    with open(os.fspath(path), encoding="utf-8-sig") as fh:
        return parse_glossary(fh.read())


def load_glossary_if_exists(path: "str | os.PathLike") -> Glossary | None:
    """Glossary for `path`, or None when it is missing or holds no entries.

    Lets a caller write `g = load_glossary_if_exists(p)` and pass `None` straight
    through to correct_text/correct_segments, i.e. an absent glossary is the
    normal case, not an error. I/O errors are raised, not swallowed.
    """
    if not os.path.isfile(os.fspath(path)):
        return None
    g = load_glossary(path)
    return g if len(g) else None


def _build(entries: dict[str, list[str]], protected: Iterable[str] = ()) -> Glossary:
    canons = tuple(entries)
    norms = {c: _norm(c) for c in canons}
    exact = frozenset(n for n in norms.values() if n)
    # Fuzzy targets are the single-word canonicals only: a phrase has no shape a
    # lone mis-heard token could be aiming at. A canonical is never dropped from
    # here for being a canonical — it is the *token* side that exact matches veto.
    fuzzy = tuple((n, c, _alnum_len(c)) for c in canons for n in [norms[c]] if n and " " not in n)

    by_first: dict[str, list[_Cand]] = {}

    def add(toks: tuple[str, ...], canon: str) -> None:
        if not toks or not toks[0]:
            return
        bucket = by_first.setdefault(toks[0], [])
        if not any(c.toks == toks and c.canon == canon for c in bucket):
            bucket.append(_Cand(toks, canon))

    for canon in canons:
        cnorm = norms[canon]
        if not cnorm:
            continue
        ctoks = tuple(cnorm.split())
        add(ctoks, canon)  # the canonical doubles as its own case-repairing alias
        for alias in entries[canon]:
            atoks = tuple(_norm(alias).split())
            if not atoks or atoks == ctoks:
                continue  # already covered by the canonical itself
            add(atoks, canon)
    for bucket in by_first.values():
        # Longest phrase first, then longest text, then the canonical name: a
        # tie-break that cannot depend on the order lines were written in.
        bucket.sort(key=lambda c: (-len(c.toks), -sum(len(t) for t in c.toks), c.canon))
    prot = {p for p in (tuple(t for t in _norm(w).split() if t) for w in protected) if p}
    return Glossary(canons=canons, by_first=by_first, exact=exact, fuzzy=fuzzy,
                    protected=tuple(sorted(prot)))


def _span_protected(g: Glossary, norms: list[str], i: int, n: int) -> bool:
    """True when a protected token run overlaps the candidate span `[i, i+n)`."""
    if not g.protected:
        return False
    for p in g.protected:
        for j in range(max(0, i - len(p) + 1), min(i + n, len(norms) - len(p) + 1)):
            if tuple(norms[j:j + len(p)]) == p:
                return True
    return False


def _match_alias(g: Glossary, norms: list[str], i: int) -> "tuple[int, str] | None":
    """Longest alias starting at token `i`: (token count, canonical)."""
    for cand in g.by_first.get(norms[i], ()):
        n = len(cand.toks)
        if n > len(norms) - i:
            continue  # candidates are sorted longest-first; shorter ones still fit
        if tuple(norms[i:i + n]) == cand.toks:
            if _span_protected(g, norms, i, n):
                continue  # a `!`-protected word in the span vetoes this rewrite
            return n, cand.canon
    return None


def _best_fuzzy(
    sm: difflib.SequenceMatcher, g: Glossary, tn: str, min_ratio: float, tlen: int
) -> str | None:
    """Closest single-word canonical for `tn`, or None when nothing clears the bar."""
    best, best_ratio = None, min_ratio
    for cn, canon, clen in g.fuzzy:
        sm.set_seq2(cn)
        # A lengthening rewrite is judged against the stricter bar, so the cheap
        # pruning ratios have to use it too or a 0.8 candidate slips through.
        bar = max(min_ratio, MIN_LENGTHEN_RATIO) if clen > tlen else min_ratio
        if sm.real_quick_ratio() < bar or sm.quick_ratio() < bar:
            continue
        ratio = sm.ratio()
        if ratio >= bar and (best is None or ratio > best_ratio):
            best, best_ratio = canon, ratio
    return best


def correct_text(
    text: str, g: Glossary, *, min_ratio: float = 0.78, min_len: int = 4
) -> "tuple[str, list[dict]]":
    """Apply `g` to `text`; return the corrected text and the audit records.

    Records are `{"start": None, "speaker": None, "from": old, "to": new,
    "rule": "alias"|"fuzzy"}` — correct_text has no segment context, so those two
    keys stay None until correct_segments fills them in. An empty or absent
    glossary returns the text unchanged.
    """
    if not text or not g:
        return text, []
    spans = [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    norms = [_norm_token(t) for t, _, _ in spans]
    count = len(spans)
    repl: "list[tuple[int, str, str] | None]" = [None] * count  # index → (end, canonical, rule)
    claimed = [False] * count  # tokens an alias already took wholesale

    i = 0
    while i < count:
        hit = _match_alias(g, norms, i) if norms[i] else None
        if hit is None:
            i += 1
            continue
        n, canon = hit
        repl[i] = (i + n, canon, "alias")
        for j in range(i, i + n):
            claimed[j] = True
        i += n

    # Fuzzy gets the tokens no alias claimed, so an explicit known-bad rendering
    # can never be second-guessed by a ratio.
    sm = difflib.SequenceMatcher(None, autojunk=False)
    for i in range(count):
        if claimed[i] or _span_protected(g, norms, i, 1):
            continue
        tn = norms[i]
        if len(tn) < min_len or tn in g.exact:
            continue  # too short to judge, or already a canonical spelling
        sm.set_seq1(tn)
        best = _best_fuzzy(sm, g, tn, min_ratio, _alnum_len(spans[i][0]))
        if best is not None:
            repl[i] = (i + 1, best, "fuzzy")

    out: list[str] = []
    changes: list[dict] = []
    pos = 0
    i = 0
    while i < count:
        hit = repl[i]
        if hit is None:
            i += 1
            continue
        end, canon, rule = hit
        start, stop = spans[i][1], spans[end - 1][2]
        out.append(text[pos:start])
        out.append(canon)
        pos = stop
        old = text[start:stop]
        if old != canon:  # already spelled right: claimed, but not a change
            changes.append({"start": None, "speaker": None, "from": old, "to": canon,
                            "rule": rule})
        i = end

    out.append(text[pos:])
    return "".join(out), changes


def correct_segments(
    segments: "list[dict]", g: Glossary, **kw
) -> "tuple[list[dict], list[dict]]":
    """Correct `segments` without mutating them; return (new segments, changes).

    Each segment dict is copied and only `text` replaced, so the caller's own
    bookkeeping (word stamps, embeddings, whatever) is untouched and the result is
    a snapshot that can be published as-is. Order and length are preserved;
    anything that is not a dict, or has no string `text`, passes through unchanged.
    """
    out: list[dict] = []
    changes: list[dict] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            out.append(seg)
            continue
        new = dict(seg)
        text = seg.get("text")
        if isinstance(text, str):
            new["text"], seg_changes = correct_text(text, g, **kw)
            for c in seg_changes:
                c["start"] = seg.get("start")
                c["speaker"] = seg.get("speaker")
            changes.extend(seg_changes)
        out.append(new)
    return out, changes


# ---------------------------------------------------------------- CLI (--check)

def _ts_seconds(stamp: str) -> "float | None":
    m = _TS_RE.match(stamp.strip())
    if not m:
        return None
    parts = m.groupdict()
    return (int(parts["h"] or 0) * 3600 + int(parts["m"]) * 60 + int(parts["s"])
            + int((parts["ms"] or "0").ljust(3, "0")[:3]) / 1000)


def _segments_in_md(md: str) -> "list[dict]":
    """Segments from an exports.py-style markdown transcript."""
    segs = []
    for m in _MD_SEG_RE.finditer(md):
        segs.append({"text": m["text"].rstrip("\n"), "start": _ts_seconds(m["ts"]),
                     "speaker": m["spk"].strip()})
    return segs


def _segments_in_json(blob: str) -> "list[dict]":
    """Segments from a data/results/<hash>.json sidecar."""
    payload = json.loads(blob)
    segs = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(segs, list) or not any(isinstance(s, dict) and "text" in s for s in segs):
        raise ValueError("no 'segments' list with a 'text' field")
    return [s for s in segs if isinstance(s, dict)]


def _check_targets(path: str) -> "list[str]":
    if os.path.isdir(path):
        found = []
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                if name.endswith((".md", ".json")):
                    found.append(os.path.join(root, name))
        return sorted(found)
    return [path]


def _report(name: str, kind: str, segs: "list[dict]", changes: "list[dict]") -> None:
    print(f"{name} ({kind}, {len(segs)} segment(s))")
    for c in changes:
        where = "" if c["start"] is None else f"{c['start']:.3f} "
        who = "" if c["speaker"] is None else f"{c['speaker']}  "
        print(f"    {where}{who}{c['from']!r} -> {c['to']!r}  ({c['rule']})")
    if not changes:
        print("    no changes")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Text-side ASR glossary correction.")
    ap.add_argument("--glossary", required=True, help="glossary file (see glossary.example.txt)")
    ap.add_argument("--check", metavar="FILE_OR_DIR", required=True,
                    help="transcript (.md), result sidecar (.json) or directory to dry-run on")
    ap.add_argument("--min-ratio", type=float, default=0.78)
    ap.add_argument("--min-len", type=int, default=4)
    args = ap.parse_args(argv)

    try:
        g = load_glossary(args.glossary)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"glossary {args.glossary}: {exc}", file=sys.stderr)
        return 2
    if not len(g):
        print(f"glossary {args.glossary}: no entries", file=sys.stderr)
        return 2

    targets = _check_targets(args.check)
    if not targets:
        print(f"{args.check}: nothing to check", file=sys.stderr)
        return 2
    if not os.path.exists(args.check):
        print(f"{args.check}: no such file or directory", file=sys.stderr)
        return 2
    named = os.path.isfile(args.check)  # a directory may hold foreign .json files

    total = 0
    for target in targets:
        with open(target, encoding="utf-8") as fh:
            blob = fh.read()
        kind = "markdown" if target.endswith(".md") else "json"
        try:
            segs = _segments_in_md(blob) if kind == "markdown" else _segments_in_json(blob)
        except (ValueError, json.JSONDecodeError) as exc:
            if named:
                print(f"{target}: {exc}", file=sys.stderr)
                return 2
            print(f"{target}: skipped ({exc})")
            continue
        _, changes = correct_segments(segs, g, min_ratio=args.min_ratio, min_len=args.min_len)
        total += len(changes)
        _report(target, kind, segs, changes)
    print(f"{total} change(s) in {len(targets)} file(s), nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
