"""Behavioural tests for glossary.py — run with `python test_glossary.py`, no pytest.

Everything is in-memory or in a throwaway temp dir: the repo's data/ is never
touched, and the CLI is dry-run only, so no transcript is rewritten by these tests.
"""
import copy, json, os, pathlib, subprocess, sys, tempfile

PROJ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import glossary

# The whole point of the stdlib-only rule: an export path must not load the GPU stack.
assert not [m for m in sys.modules if m.split(".")[0] in ("torch", "nemo", "soundfile")], \
    "importing glossary pulled in a heavy module"

fails = []
COUNT = {"n": 0}


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    COUNT["n"] += 1
    if not cond:
        fails.append(label)


def fix(text, src=None, **kw):
    """correct_text -> just the text, for the assertions that don't care why."""
    return glossary.correct_text(text, src if src is not None else G, **kw)[0]


def why(text, src=None, **kw):
    return glossary.correct_text(text, src if src is not None else G, **kw)[1]


# A deliberate misspelling ("Muller" with no alias) is left to the fuzzy rule, so the
# umlaut-folding path is exercised; "Sensorik" exists only to be a longer-word temptation.
G = glossary.parse_glossary(
    "# test glossary\n"
    "\n"
    "Müller\n"
    "MEMS\tmemz\n"
    "E-Modul\tEbenmoduli;E Modul\n"
    "Kraftsensor\tKraftsensori;Kraft Sensor;Kraft\n"
    "Stresssensoren\tStress sensoren\n"
    "Motorrad\tMotorrade\n"
    "Sensorik\n"
)

print("1) explicit aliases")
check(fix("Das Ebenmoduli ist wichtig.") == "Das E-Modul ist wichtig.", "Ebenmoduli -> E-Modul (alias)")
check(why("Ebenmoduli") == [{"start": None, "speaker": None, "from": "Ebenmoduli",
                             "to": "E-Modul", "rule": "alias"}], "change record is exact")
check(fix("Der Kraftsensori fehlt.") == "Der Kraftsensor fehlt.", "Kraftsensori -> Kraftsensor (alias)")
check(fix("das mems Gerät") == "das MEMS Gerät", "mems -> MEMS (canonical doubles as alias)")
check(fix("das MEMS-Gerät") == "das MEMS-Gerät", "a correct hyphenated compound is left alone")
check(fix("Das MEMSe und das memz sind gleich.") == "Das MEMS und das MEMS sind gleich.",
      "explicit aliases memz/MEMSe -> MEMS")
check(fix("Ebenmoduli, und das E Modul nutzen.") == "E-Modul, und das E-Modul nutzen.",
      "multi-word alias 'E Modul' and punctuation tolerance")
check(fix("(Ebenmoduli).") == "(E-Modul).", "surrounding punctuation stays put")
check(fix("mems") == "MEMS" and fix("MEMS") == "MEMS", "case repaired once, then left alone")

print("2) fuzzy single-token matching")
check(fix("Herr Muller kommt spaeter.") == "Herr Müller kommt spaeter.",
      "Muller -> Müller by umlaut-folding fuzzy")
check([c["rule"] for c in why("Muller")] == ["fuzzy"], "recorded under rule 'fuzzy'")
check(fix("Kraftsensori und Sensorik und Muller") ==
      "Kraftsensor und Sensorik und Müller", "aliases, exact spellings and fuzzy in one line")
check(fix("Mueller") == "Müller", "Mueller -> Müller (folding, ratio 1.0 after norm)")

print("3) conservatism: no greedy expansion, no needless rewrites")
MOTOR = glossary.parse_glossary("Motorrad\tMotorrade\nSensorik\n")
check(fix("Der Motor läuft seit 20 Jahren.", MOTOR) == "Der Motor läuft seit 20 Jahren.",
      "Motor stays Motor next to canonical Motorrad")
check(fix("Ein Sensor ist keine Sensorik.", MOTOR) == "Ein Sensor ist keine Sensorik.",
      "a lengthening never fires fuzzily (Sensor -> Sensorik needs an alias)")
check(fix("Sensor", glossary.parse_glossary("Sensorik\n")) == "Sensor",
      "same rule with a one-entry glossary")
SHIELD = glossary.parse_glossary("E-Modul\tEbenmoduli\nModul\n")
check(fix("Das Modul ist steif.", SHIELD) == "Das Modul ist steif.",
      "listing the plain word shields it too: Modul is not swallowed by E-Modul")
check(fix("Das Modell und der model.", glossary.parse_glossary("Modul\n")) == "Das Modell und der Modul.",
      "…but the shield is itself a fuzzy target, so it is a trade, not a free win")
check(fix("Das Modul.", glossary.parse_glossary("E-Modul\tEbenmoduli\n")) == "Das Modul.",
      "ratio 0.909 notwithstanding, the plain word keeps its length")
check(fix("das E Modul und das E-Moduls nutzen.", glossary.parse_glossary(
      "E-Modul\tE Modul;E-Moduls\n")) == "das E-Modul und das E-Modul nutzen.",
      "…lengthenings that ARE wanted go through aliases")
PROT = glossary.parse_glossary("E-Modul\tEbenmoduli;E Modul\n!Modul;Module\n")
check(fix("Die Module und das Modul sind steif.", PROT) == "Die Module und das Modul sind steif.",
      "!Modul;Module protects the ordinary word from E-Modul's fuzzy reach")
check(fix("Das Ebenmoduli und das Modul.", PROT) == "Das E-Modul und das Modul.",
      "…while aliases still fire next to a protected word")
check(fix("Modul und mems.", glossary.parse_glossary("MEMS\n!Modul\n")) == "Modul und MEMS.",
      "protection covers the canonical's own case repair too, not only fuzzy")
check(fix("Wir brauchen ein E Modul.", glossary.parse_glossary(
      "E-Modul\tE Modul\n!Modul\n")) == "Wir brauchen ein E Modul.",
      "a protected word inside a multi-word alias vetoes the whole span")
check(len(PROT) == 1, "a ! line protects words, it does not add a canonical")
check(fix("E-Modul, MEMS, Müller, Motorrad, Stresssensoren, Kraftsensor") ==
      "E-Modul, MEMS, Müller, Motorrad, Stresssensoren, Kraftsensor",
      "already-correct text is untouched and reports nothing")
check(why("E-Modul, MEMS, Müller") == [], "no audit records for untouched text")

print("4) longest alias wins, multi-word before single-word")
check(fix("Kraft Sensor montieren") == "Kraftsensor montieren",
      "'Kraft Sensor' beats the shorter 'Kraft' alias at the same token")
check(fix("Kraft montieren") == "Kraftsensor montieren", "the single-word alias still applies alone")
check(fix("die Stress sensoren sind kaputt") == "die Stresssensoren sind kaputt",
      "multi-word alias Stress sensoren -> Stresssensoren")

print("5) idempotency")
CORPUS = ["Das Ebenmoduli ist wichtig.", "Herr Muller nannte memz und Stress sensoren korrekt.",
          "Kraft Sensor, Motor, Sensorik, E-Modul, SAP S/4HANA.", ""]
for line in CORPUS:
    once = fix(line)
    check(fix(once) == once and why(once) == [], f"idempotent: {line[:32]!r}")

print("6) thresholds are honoured")
check(fix("Herr Muller", min_len=8) == "Herr Muller", "min_len=8 blocks the 6-letter token")
check(fix("Herr Muller", min_ratio=0.99) == "Herr Muller", "min_ratio=0.99 blocks ratio 0.923")
check(fix("Das memz", min_ratio=0.5) == "Das MEMS", "aliases never depend on min_ratio")

print("7) parser")
P = glossary.parse_glossary(
    "# comment\n"
    "\n"
    "   \n"
    "Müller\tMuller\n"
    "Müller\tMueller\n"          # duplicate canonical: aliases merge
    "MEMS\tmemz ; meens\n"       # whitespace around aliases is trimmed
    "\t\n"
    "NoAliases\n"
)
check(len(P) == 3 and sorted(P) == ["MEMS", "Müller", "NoAliases"],
      "comments/blank lines/whitespace-only lines ignored, duplicate canonicals merged")
check(fix("Muller und Mueller und memz und meens", P) == "Müller und Müller und MEMS und MEMS",
      "aliases from both lines of a duplicated canonical are all live")
check(fix("Das NoAliases ding", P) == "Das NoAliases ding", "an alias-less canonical does not match prose")
check(len(glossary.parse_glossary("")) == 0, "empty glossary text parses to zero entries")
check(glossary.correct_text("Ebenmoduli", glossary.parse_glossary("")) == ("Ebenmoduli", []),
      "empty glossary is a no-op")

print("8) correct_segments copies, preserves order, attributes the change")
segs = [
    {"text": "Das Ebenmoduli ist wichtig.", "start": 4.879, "end": 7.0, "speaker": "speaker_0"},
    {"text": "Herr Muller spricht über MEMS.", "start": 7.0, "end": 9.5, "speaker": "speaker_1"},
    {"text": "Nichts zu tun hier.", "start": 9.5, "end": 11.0, "speaker": "speaker_0"},
]
before = copy.deepcopy(segs)
out, changes = glossary.correct_segments(segs, G)
check(segs == before, "input segments not mutated")
check(len(out) == 3 and [s["start"] for s in out] == [4.879, 7.0, 9.5], "same length, same order")
check(out is not segs and all(o is not s for o, s in zip(out, segs)), "fresh dicts, not the caller's")
check(out[0]["text"] == "Das E-Modul ist wichtig." and out[1]["text"] == "Herr Müller spricht über MEMS."
      and out[2]["text"] == "Nichts zu tun hier.", "only the two texts with errors changed")
check(all({k: v for k, v in o.items() if k != "text"} == {k: v for k, v in s.items() if k != "text"}
          for o, s in zip(out, segs)), "every other segment field survives verbatim")
check(changes[0] == {"start": 4.879, "speaker": "speaker_0", "from": "Ebenmoduli",
                     "to": "E-Modul", "rule": "alias"}, "change carries the segment's start/speaker")
check([c["start"] for c in changes] == [4.879, 7.0], "one record per rewrite, in order")
out2, ch2 = glossary.correct_segments([], G)
check(out2 == [] and ch2 == [], "no segments, no work")
out3, ch3 = glossary.correct_segments([{"start": 1.0}, {"text": 42, "start": 2.0}, "junk"], G)
check(out3 == [{"start": 1.0}, {"text": 42, "start": 2.0}, "junk"] and ch3 == [],
      "segments without string text pass through")

print("9) loading from disk, UTF-8, missing/empty files")
TMP = pathlib.Path(tempfile.mkdtemp(prefix="glossary-test-"))
GOOD = TMP / "g.txt"
GOOD.write_text("Müller\tMuller\nE-Modul\tEbenmoduli\n", encoding="utf-8")
check(len(glossary.load_glossary(GOOD)) == 2, "load_glossary reads a UTF-8 file")
check(fix("Ebenmoduli", glossary.load_glossary(GOOD)) == "E-Modul", "loaded glossary works")
with open(GOOD, "rb") as fh:
    check(b"\xc3\xbc" in fh.read(), "file really is UTF-8, not ASCII-folded")
BOM = TMP / "bom.txt"
BOM.write_text("\ufeffMüller\tMuller\n", encoding="utf-8")
check(len(glossary.load_glossary(BOM)) == 1, "a UTF-8 BOM does not break the first entry")
EMPTY = TMP / "empty.txt"
EMPTY.write_text("# nobody home\n\n", encoding="utf-8")
check(glossary.load_glossary_if_exists(EMPTY) is None, "comment-only glossary -> None")
check(glossary.load_glossary_if_exists(TMP / "missing.txt") is None, "missing file -> None")
check(glossary.load_glossary_if_exists(GOOD) is not None, "present glossary -> Glossary")
check(glossary.correct_text("Ebenmoduli", None) == ("Ebenmoduli", []), "None glossary is a no-op")
EX = glossary.load_glossary(PROJ / "glossary.example.txt")
check(len(EX) >= 5 and "E-Modul" in EX, "glossary.example.txt loads and documents real entries")
check(fix("Ebenmoduli", EX) == "E-Modul" and fix("Kraftsensori", EX) == "Kraftsensor"
      and fix("Motor", EX) == "Motor", "the shipped example behaves as advertised")
check(fix("Das Modul und die Module.", EX) == "Das Modul und die Module.",
      "…and the example's !-line shields those plain words from E-Modul's reach")

print("10) robustness: empty, huge, single-entry")
check(glossary.correct_text("", G) == ("", []), "empty text")
check(fix("…!!!???", G) == "…!!!???", "text with no tokens")
LONG = ("Das Ebenmoduli und Herr Muller und ein Motor sowie die Stress sensoren passen hier hin. " * 110)
check(len(LONG) > 5000, f"long fixture is {len(LONG)} chars")
once = fix(LONG)
check("E-Modul" in once and "Müller" in once and "Motor" in once and len(once) > 5000,
      "long text corrected")
check(fix(once) == once, "long text correction is idempotent")
check(fix("Kraftsensori", glossary.parse_glossary("Kraftsensor\n")) == "Kraftsensor",
      "single-entry glossary, fuzzy only")

print("11) CLI --check (dry run, both transcript formats)")
MD = TMP / "t.md"
MD.write_text("# Transcript — Voice.wav\n\n"
              "**Speaker 0** `[00:00:04.879]`  \nDas Ebenmoduli ist wichtig.\n\n"
              "**Speaker 1** `[00:01:02,500]`  \nHerr Muller nickt.\n", encoding="utf-8")
JS = TMP / "t.json"
JS.write_text(json.dumps({"name": "Voice.wav", "segments": [
    {"text": "Das Ebenmoduli ist wichtig.", "start": 4.879, "end": 6.0, "speaker": "speaker_0"},
    {"text": "Nichts hier.", "start": 62.5, "end": 63.0, "speaker": "speaker_1"},
]}, ensure_ascii=False), encoding="utf-8")
FOREIGN = TMP / "ledger.json"
FOREIGN.write_text(json.dumps({"not": "a transcript"}), encoding="utf-8")


def cli(*argv):
    r = subprocess.run([sys.executable, str(PROJ / "glossary.py"), *argv],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


MD_BYTES, JS_BYTES = MD.read_bytes(), JS.read_bytes()
code, out, err = cli("--glossary", str(PROJ / "glossary.example.txt"), "--check", str(TMP))
check(code == 0, f"--check on a directory exits 0 (rc={code}, {err.strip()[:80]})")
check("'Ebenmoduli' -> 'E-Modul'" in out and "'Muller' -> 'Müller'" in out,
      "change table prints from -> to")
check("(markdown, 2 segment(s))" in out and "(json, 2 segment(s))" in out,
      "both .md and .json parsed out of the directory")
check("4.879 speaker_0" in out, "json rows carry the segment start and speaker")
check("00:01:02,500" not in out and "62.500 Speaker 1" in out,
      "markdown timestamps are read as seconds, speaker label from the header")
check("skipped" in out and code == 0, "a foreign .json in the directory is skipped, not fatal")
check("nothing written" in out and MD.read_bytes() == MD_BYTES and JS.read_bytes() == JS_BYTES,
      "--check writes nothing")
code, out, _ = cli("--glossary", str(PROJ / "glossary.example.txt"), "--check", str(MD))
check(code == 0 and "(markdown, 2 segment(s))" in out and "0 change" not in out,
      "--check on one .md transcript")
code, out, _ = cli("--glossary", str(PROJ / "glossary.example.txt"), "--check", str(JS))
check(code == 0 and "(json, 2 segment(s))" in out, "--check on one .json result")
code, out, err = cli("--glossary", str(PROJ / "glossary.example.txt"), "--check", str(FOREIGN))
check(code == 2 and "segments" in err, "--check on a named file of the wrong shape exits 2")
code, out, err = cli("--glossary", str(TMP / "nope.txt"), "--check", str(MD))
check(code == 2 and "nope.txt" in err, "missing glossary exits 2 with a message")
code, out, err = cli("--glossary", str(PROJ / "glossary.example.txt"), "--check", str(TMP / "gone.md"))
check(code == 2 and "no such file" in err, "missing check target exits 2, no traceback")
QUIET = TMP / "quiet"
QUIET.mkdir()
code, out, err = cli("--glossary", str(PROJ / "glossary.example.txt"), "--check", str(QUIET))
check(code == 2 and "nothing to check" in err, "a directory with no transcripts exits 2")

print()
print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + '; '.join(fails)}")
for f in fails:
    print("  - " + f)
if fails:
    sys.exit(1)
print(f"ok {COUNT['n']} checks")
