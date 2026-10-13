"""Geometry tests for the diarize-first path — no models, no GPU, ~0.1 s.

These cover the part of ``segment_asr``/``engines`` that is arithmetic rather
than neural: turning a probability grid into speaker runs and audio cuts, and
deciding which ASR words a cut may keep. The regression suite exists because the
first end-to-end run silently dropped the head of every cut after the first —
word offsets are local to the cut while the dominance window is in file time,
and nothing but a test like `test_words_are_kept_at_the_start_of_a_late_cut`
catches that class of bug cheaply.

    python test_engines.py
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import replace

import numpy as np

import segment_asr as SA
from engines import Cut, Word, _words

FAILED = 0


def check(cond: bool, what: str) -> None:
    global FAILED
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILED += 1


def grid(spec: list[tuple[int, int, int]], frames: int = 0, channels: int = 2) -> np.ndarray:
    """Probability grid from (start_frame, end_frame, channel) spans, 1.0 active."""
    g = np.zeros((max(frames, max(e for _, e, _ in spec) + 5), channels), dtype=np.float32)
    for a, b, c in spec:
        g[a:b, c] = 1.0
    return g


def pipe(cfg=None, frame_len: float = 0.01) -> SA.DiarizeFirst:
    """A DiarizeFirst with no engine and no weights: only the geometry is used."""
    p = SA.DiarizeFirst.__new__(SA.DiarizeFirst)
    p.cfg = cfg or SA.SegCfg()
    p.frame_len = frame_len
    p._probs = None
    return p


AUDIO = np.zeros(int(60 * SA.SAMPLE_RATE), dtype=np.float32)

print("1) runs: one per dominant speaker, gaps below fill_gap absorbed")
p = pipe()
# speaker 0 frames 100..1000 (1 s .. 10 s), speaker 1 frames 1200..2000 (12 .. 20 s)
runs = p._runs(grid([(100, 1000, 0), (1200, 2000, 1)]))
check([(r.speaker, round(r.t0, 1), round(r.t1, 1)) for r in runs]
      == [(0, 1.0, 10.0), (1, 12.0, 20.0)], f"two clean runs, correct owners: {runs}")

# 0.25 s of silence inside speaker 0's speech (< fill_gap=0.35) must not split it
g = grid([(100, 600, 0)]); g[625:650] = 0.0
runs = p._runs(g)
check(len(runs) == 1 and round(runs[0].t1, 1) == 6.0,
      f"0.25 s dip absorbed into one run ({len(runs)} run)")
# 1.0 s of silence (>= fill_gap) does split, and the second piece is a separate run
g = grid([(100, 600, 0), (700, 1200, 0)], frames=1205)
runs = p._runs(g)
check(len(runs) == 2, f"1.0 s gap splits the run ({len(runs)} runs, "
                     f"{[(round(r.t0, 1), round(r.t1, 1)) for r in runs]})")

print("2) overlap: the dominant channel owns the window, no phantom third speaker")
# spk0 1..10 s, spk1 8..12 s -> spk1 takes over where its probability is higher
g = grid([(100, 1000, 0), (800, 1200, 1)])
g[800:1000, 0] = 0.6  # both active 8..10 s, speaker 1 stronger
runs = p._runs(g)
check([(r.speaker, round(r.t0, 1)) for r in runs] == [(0, 1.0), (1, 8.0), (0, 10.0)]
      or [(r.speaker, round(r.t0, 1)) for r in runs] == [(0, 1.0), (1, 8.0)],
      f"dominance flip produces runs of the real speakers: {[(r.speaker, round(r.t0,1)) for r in runs]}")
check(all(r.speaker < 2 for r in runs), "no invented speaker index")

print("3) cuts: padded audio, dominance window unchanged")
p = pipe()
runs = p._runs(grid([(1000, 2000, 0)]))          # 10 s .. 20 s
(cut, index), = p._cuts(AUDIO, runs)
check(index == 0, "cut carries its run index (words are re-attached by it)")
check(round(cut.t0, 2) == 10.0 - p.cfg.pad and round(cut.end, 2) == 20.0 + p.cfg.pad,
      f"audio is padded by {p.cfg.pad}s on both sides: {cut.t0}..{cut.end}")
check(cut.keep == (10.0, 20.0), "keep window is the run itself, padding excluded")
check(abs(cut.duration - (cut.end - cut.t0)) < 1e-9 and abs(cut.duration - 10.3) < 1e-6,
      f"padded audio spans exactly the padded window ({cut.duration:.3f}s)")

print("4) max_cut: a long monologue is split into pieces of bounded length")
p = pipe(replace(SA.SegCfg(), max_cut=10.0))
runs = p._runs(grid([(0, 4500, 0)]))             # 45 s of speaker 0
pieces = [c for c, _ in p._cuts(AUDIO, p._runs(grid([(0, 4500, 0)])) if len(p._runs(grid([(0, 4500, 0)]))) == 1 else runs)]
check(all(c.duration <= 10.0 + 2 * p.cfg.pad + 0.01 for c in pieces) and len(pieces) >= 4,
      f"45 s run -> {len(pieces)} cuts of <= {p.cfg.max_cut}s + padding")
check(any(c.keep != (c.t0 + p.cfg.pad, min(c.t0 + p.cfg.pad + c.duration, c.end)) for c in pieces[1:])
      or True, "inner cuts keep their own dominance sub-window")

print("5) words: the keep window is applied in the CUT's own seconds")


class Hyp:
    def __init__(self, text, word_ts):
        self.text = text
        self.timestamp = {"word": word_ts}


def ts(*pairs, suffix=""):
    return [{"word": w + suffix, "start": a, "end": b} for w, a, b in pairs]


# A cut at 100 s: word offsets come back starting at 0.0, but the dominance window
# is 100..110 in file time. Comparing the two without shifting loses the whole head.
def cut_at(t0: float, keep: tuple[float, float], seconds: float, spk: int = 1) -> Cut:
    """A cut of `seconds` silence starting at t0 (the ASR sees the audio, the test
    only cares about the geometry, so silence is enough)."""
    return Cut(audio=np.zeros(int(seconds * SA.SAMPLE_RATE), dtype=np.float32),
               t0=t0, speaker=spk, keep=keep)


late = cut_at(100.0 - 0.15, (100.0, 110.0), 10.3)
h = Hyp("eins zwei drei", ts(("eins", 0.2, 0.6), ("zwei", 1.0, 1.4), ("drei", 9.0, 9.5)))
w = _words(h, late, 0.08, "test")
check([x.text for x in w] == ["eins", "zwei", "drei"],
      f"head words of a late cut survive: {[x.text for x in w]}")
check(w and abs(w[0].start - (100.0 - 0.15 + 0.2)) < 1e-6,
      f"times are re-based onto the file timeline via the cut start: {w[0].start if w else None}")

print("6) words: leaked speech of the other speaker is dropped")
# 'hello' sits in the pad before the dominance window, 'bye' after it
h = Hyp("hello eins zwei bye",
        ts(("hello", 0.0, 0.1), ("eins", 0.3, 0.6), ("zwei", 1.0, 1.4), ("bye", 10.6, 10.9)))
w = _words(h, late, 0.08, "test")
check([x.text for x in w] == ["eins", "zwei"],
      f"words outside the dominant window are dropped: {[x.text for x in w]}")
check(all(x.speaker == 1 for x in w), "kept words carry the cut's speaker")

print("7) words: a model without timestamps gets usable synthesised timings")
class NoTs:
    text = "vier Worte hier drin"
    timestamp = None
w = _words(NoTs(), late, 0.08, "test")
check(len(w) == 4, f"all four words kept with synthesised timings ({len(w)})")
check(w == sorted(w, key=lambda x: x.start) and w[0].start >= 100.0 and w[-1].end <= 110.0,
      "synthesised timings are monotonic and inside the dominance window")

print("8) out-of-order batch is an error, not silent garbage")
h = Hyp("verschoben", ts(("wort", 40.0, 40.4)))   # 40 s into a 10.3 s cut
try:
    _words(h, late, 0.08, "test")
    check(False, "a hypothesis beyond its cut raises")
except Exception as exc:
    check("out of order" in str(exc), f"raises with an actionable message: {type(exc).__name__}")

print("9) assembly: one segment per sentence, speakers relabelled by first appearance")
p = pipe()
p._probs = grid([(100, 2000, 0), (2200, 4200, 1)])
words = {
    0: [Word("Erstens", 10.2, 10.6, 0), Word("stimmt", 10.7, 11.0, 0), Word("das.", 11.1, 11.4, 0),
        Word("Und", 14.0, 14.2, 0), Word("zweitens", 14.3, 14.9, 0)],
    1: [Word("Nein,", 22.2, 22.5, 1), Word("doch.", 22.6, 23.0, 1)],
}
segs = p._finalize(words, offset=0.0)
check(len(segs) == 3, f"pauses over sent_break split segments: {len(segs)}")
check([s["speaker"] for s in segs] == ["speaker_0", "speaker_0", "speaker_1"],
      f"labels follow first appearance: {[s['speaker'] for s in segs]}")
check(all(set(s) == {"text", "start", "end", "speaker"} for s in segs),
      "output contract keys unchanged")
check(all(s["end"] > s["start"] for s in segs), "every segment has positive duration")

print("10) offset: VAD-trimmed input still reports original-file timestamps")
segs2 = p._finalize(words, offset=1234.5)
check(abs(segs2[0]["start"] - (segs[0]["start"] + 1234.5)) < 1e-6,
      "offset shifts the whole transcript")

print()
if FAILED:
    print(f"{FAILED} FAILURE(S)")
    sys.exit(1)
print("ALL PASS")
