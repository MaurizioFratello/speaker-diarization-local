"""Behavioural tests for watcher.py's failure classification and queue gating.

Everything runs against a throwaway folder with the GPU stack and the on-disk
bookkeeping stubbed, so the project's real data/ and GPU are untouched.
"""
import argparse, errno, json, os, pathlib, shutil, signal, subprocess, sys, tempfile, time, types, wave

PROJ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import watcher
HEAVY_AT_START = {m for m in sys.modules if m.split(".")[0] in ("torch","nemo","soundfile")}

TMP = pathlib.Path(tempfile.mkdtemp(prefix="watcher-test-"))
TMP.mkdir(parents=True, exist_ok=True)
watcher.RESULTS_DIR = TMP / "results"
watcher.MIRROR_DIR = TMP / "mirrors"
watcher.SCRATCH_DIR = TMP / "scratch"   # else the tests litter the repo's data/scratch
watcher.LOG_PATH = TMP / "log"

A = TMP / "poison.wav"; B = TMP / "good.wav"; C = TMP / "corrupt.wav"
# Distinct bytes per file: identical content would (correctly) collapse to one
# ledger entry via the content-hash de-duplication, which would hide bugs.
for i, f in enumerate((A, B, C), start=1):
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(bytes([i * 40, 255 - i * 40] * (4000 + i * 111)))

SEGS = [{"text": "hello", "start": 0.5, "end": 1.5, "speaker": "speaker_0"}]
FAULT = {"poison": RuntimeError("CUDA out of memory"),
         "corrupt": RuntimeError("audio conversion failed (ffmpeg rc=1): Invalid data found when processing input")}


COUNT = {"init": 0, "release": 0}


class Transcriber:
    def __init__(self, **kw):
        COUNT["init"] += 1

    def transcribe(self, wav, publish=None):
        for tag, err in FAULT.items():
            if tag in str(wav):
                raise err
        if MODE["sigterm_on"] and MODE["sigterm_on"] in str(wav):
            watcher.install_signal_handlers()
            os.kill(os.getpid(), signal.SIGTERM)   # exactly what systemd sends
        if MODE["empty"]:
            return []
        return list(SEGS)


MODE = {"empty": False, "sigterm_on": None}


def prepare_audio(src, dst):
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 8000)
    return str(dst)


SILENT = {"value": False}
mod = types.ModuleType("pipeline"); mod.Transcriber = Transcriber; mod.prepare_audio = prepare_audio
mod._is_silent_wav = lambda wav, threshold=0.0: SILENT["value"]
sys.modules["pipeline"] = mod
# min_free_vram=0 / max_other_util=100 keep the GPU courtesy guard inert for the
# queueing tests; sections 12-13 exercise that guard explicitly.
args = argparse.Namespace(device="cuda", max_speakers=None, progress_every=0.0,
                         no_sidecars=True, format_list=["md"], min_free_vram=0.0,
                         max_other_util=100, gpu_time_limit=0.0, mount_wait=25.0)


def _gargs(**over):
    base = dict(device="cuda", max_speakers=None, progress_every=0.0, no_sidecars=True,
                format_list=["md"], min_free_vram=0.0, max_other_util=100,
                gpu_time_limit=0.0, mount_wait=25.0, min_age=0.0, only=None,
                limit=0, force=False, dry_run=False, ignore_open=True)
    base.update(over)
    return argparse.Namespace(**base)


def mk(path):
    st = path.stat()
    return {"path": path, "name": path.name, "size": st.st_size, "mtime": round(st.st_mtime, 3),
            "digest": watcher.file_hash(path), "duration": 0.5}


def tick(ledger, files):
    return watcher.transcribe_queue([mk(p) for p in files], args, ledger)


def entry_for(ledger, name):
    return next(e for e in ledger.data.values() if e["name"] == name)


fails = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        fails.append(label)


print("1) busy GPU must never be charged to the file")
led = watcher.Store(TMP / "l1.json"); led.data = {}
res = [tick(led, [A, B]) for _ in range(watcher.ENV_STRIKES_MAX)]
e = entry_for(led, "poison.wav")
check(all(d == 0 and f == 0 for d, f in res), "each tick aborted early, nothing counted")
check(e["status"] != "failed", f"status stayed out of 'failed' ({e['status']})")
check(e.get("tries", 0) == 0, "no retry charged for an OOM")
done, failed = tick(led, [A, B])
check(done == 1 and failed == 0, "past the strike cap the queue still transcribed the next file")
check(entry_for(led, "poison.wav")["status"] == "pending", "the sick file stays queued, not failed")

print("2) a dead share (EIO, incl. wrapped) is environmental too")
for label, err in [
    ("ffmpeg EIO text", RuntimeError("audio conversion failed (ffmpeg rc=1): av_read_frame: Input/output error")),
    ("bare OSError EIO", OSError(errno.EIO, "Input/output error")),
    ("CIFS stale handle", OSError(errno.ESTALE, "Stale file handle")),
    ("network unreachable", OSError(errno.ENETUNREACH, "Network is unreachable")),
    ("OOM via MemoryError", MemoryError()),
]:
    check(watcher._environmental(err), f"{label} -> environmental")
for label, err in [
    ("corrupt file", RuntimeError("audio conversion failed (ffmpeg rc=1): Invalid data found when processing input")),
    ("not a wav", ValueError("file did not begin with '#WAVE'")),
    ("missing file", FileNotFoundError(2, "No such file or directory")),
]:
    check(not watcher._environmental(err), f"{label} -> file's own fault")
wrapped = RuntimeError("audio conversion failed")
wrapped.__cause__ = OSError(errno.EIO, "Input/output error")
check(watcher._environmental(wrapped), "an EIO under a wrapped exception is still environmental")

print("3) endless environmental failure parks the file instead of looping forever")
led3 = watcher.Store(TMP / "l3.json"); led3.data = {}
for _ in range(watcher.ENV_GIVEUP_MAX + 2):
    tick(led3, [A])
e3 = entry_for(led3, "poison.wav")
check(e3["status"] == "stalled", f"status became 'stalled' after {e3['env_strikes']} strikes")
check(e3.get("tries", 0) == 0, "still no retries charged")

print("4) plan() gating: retried vs parked vs quarantined")


def plan_with(entries, files):
    st = watcher.Store(TMP / "state.json"); st.data = {}
    lg = watcher.Store(TMP / "ledger.json"); lg.data = entries
    for p in files:
        st.data[str(p)] = {"size": p.stat().st_size, "mtime": round(p.stat().st_mtime, 3),
                           "digest": watcher.file_hash(p)}
    a = argparse.Namespace(**vars(args)); a.limit = 0; a.force = False; a.only = None
    a.dry_run = False; a.min_age = 0; a.ignore_open = True; a.format_list = ["md"]; a.no_sidecars = True
    todo, _ = watcher.plan(TMP, a, st, lg)
    return sorted(t["name"] for t in todo)


digest = watcher.file_hash(A)
check("poison.wav" in plan_with({digest: {"name": "poison.wav", "path": str(A), "status": "failed",
                                         "tries": 1, "error": "x"}}, [A]),
      "a file that failed ONCE is retried (not quarantined early)")
check("poison.wav" not in plan_with({digest: {"name": "poison.wav", "path": str(A), "status": "failed",
                                             "tries": watcher.MAX_TRIES, "error": "x"}}, [A]),
      "a file that failed MAX_TRIES times is quarantined")
check("poison.wav" not in plan_with({digest: {"name": "poison.wav", "path": str(A), "status": "stalled",
                                             "env_strikes": 31, "error": "x"}}, [A]),
      "a parked (stalled) file is skipped")
check("poison.wav" in plan_with({digest: {"name": "poison.wav", "path": str(A), "status": "pending",
                                         "env_strikes": 6}}, [A]),
      "a pending file after the strike cap is retried on the next tick")
done_e = {digest: {"name": "poison.wav", "path": str(A), "status": "done", "result": str(B), "formats": {}}}
sideA = A.with_name(A.stem + ".transcript.md")
sideA.write_text("# Transcript — poison.wav\n\n**Speaker 0** `x`\n", encoding="utf-8")
check(plan_with(done_e, [A]) == [], "an already-transcribed file is skipped")
sideA.unlink()
check("poison.wav" in plan_with(done_e, [A]),
      "but with the transcript gone from the folder it is work again (folder = ground truth)")

print("5) identical bytes in the same tick are transcribed once, for all names")
E1 = TMP / "same_content_a.wav"; E2 = TMP / "same_content_b.wav"
with wave.open(str(E1), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\7a\3c" * 5000)
shutil.copyfile(E1, E2)
args_side = argparse.Namespace(**vars(args)); args_side.no_sidecars = False
args_side.limit = 0; args_side.force = False; args_side.only = None; args_side.min_age = 0
args_side.ignore_open = True; args_side.dry_run = False
st = watcher.Store(TMP / "s5.json"); st.data = {}
lg = watcher.Store(TMP / "l5.json"); lg.data = {}
for p in (E1, E2):
    st.data[str(p)] = {"size": p.stat().st_size, "mtime": round(p.stat().st_mtime, 3),
                       "digest": watcher.file_hash(p)}
todo5, _ = watcher.plan(TMP, args_side, st, lg)
same = [t for t in todo5 if t["path"].name.startswith("same_content")]
check(len(same) == 1, f"only one job queued for two identical files (got {len(same)})")
check(bool(same and same[0].get("copies")), "the queued job carries the second path as a copy")
watcher.transcribe_queue(same, args_side, lg)
s5 = [p for p in (E1, E2) if p.with_name(f"{p.stem}.transcript.md").is_file()]
check(len(s5) == 2, "both copies received a transcript from the single run")

print("6) empty output is only accepted for genuinely silent audio")
S = TMP / "speech.wav"          # clean file: A/B/C carry injected faults or ledger state
with wave.open(str(S), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\11\22" * 6000)
led6 = watcher.Store(TMP / "l6.json"); led6.data = {}
MODE["empty"] = True; SILENT["value"] = False
watcher.transcribe_queue([mk(S)], args, led6)
e6 = next(e for e in led6.data.values() if e["name"] == S.name)
check(e6["status"] == "failed" and e6["tries"] == 1,
      f"non-silent empty result is a failure, not a done ({e6['status']}: {e6.get('error')})")
check("EmptyResult" in (e6.get("error") or ""), f"the failure is the empty-result check: {e6.get('error')}")
check(not S.with_name(f"{S.stem}.transcript.md").is_file(), "no empty transcript written for non-silent audio")
led7 = watcher.Store(TMP / "l7.json"); led7.data = {}
SILENT["value"] = True
watcher.transcribe_queue([mk(S)], args, led7)
e7 = next(e for e in led7.data.values() if e["name"] == S.name)
check(e7["status"] == "done" and e7["segments"] == 0,
      f"silent audio is a valid empty transcript ({e7['status']})")
MODE["empty"] = False; SILENT["value"] = False

print("7) a kill between the transcript write and the ledger flush must not redo GPU work")
d8 = watcher.file_hash(B)
canon8 = watcher.result_path(d8)
canon8.parent.mkdir(parents=True, exist_ok=True)
canon8.write_text(json.dumps({"name": B.name, "segments": SEGS, "labels": {}}), encoding="utf-8")
st8 = watcher.Store(TMP / "s8.json"); st8.data = {str(B): {"size": B.stat().st_size,
              "mtime": round(B.stat().st_mtime, 3), "digest": d8}}
lg8 = watcher.Store(TMP / "l8.json"); lg8.data = {}     # ledger as if never flushed
a8 = argparse.Namespace(**vars(args)); a8.limit = 0; a8.force = False; a8.only = None
a8.min_age = 0; a8.ignore_open = True; a8.dry_run = False
sideB = B.with_name(B.stem + ".transcript.md")
from exports import render_exports as _re8
sideB.write_text(_re8({"name": B.name, "segments": SEGS, "labels": {}})["md"], encoding="utf-8")
todo8, _ = watcher.plan(TMP, a8, st8, lg8)
check(B.name not in [t["name"] for t in todo8],
      "the existing canonical transcript is trusted; the file is not requeued")
canon8.unlink()
check(B.name not in [t["name"] for t in watcher.plan(TMP, a8, st8, lg8)[0]],
      "canonical copy gone but the transcript is in the folder -> the folder is believed")
sideB.unlink()
check(B.name in [t["name"] for t in watcher.plan(TMP, a8, st8, lg8)[0]],
      "neither canonical copy nor transcript -> it is queued as work again")

print("8) SIGTERM (as systemd sends it) exits cleanly with bookkeeping intact")
led9 = watcher.Store(TMP / "l9.json"); led9.data = {}
MODE["sigterm_on"] = "good"; watcher._STOP["requested"] = False
done9, failed9 = watcher.transcribe_queue([mk(B)], args, led9)
e9 = next(iter(led9.data.values()), {})
check(watcher._STOP["requested"] and done9 == 0 and failed9 == 0,
      f"interrupted run reported cleanly (done={done9} failed={failed9})")
check(e9.get("status") == "pending", f"the interrupted file stays queued ({e9.get('status')})")
MODE["sigterm_on"] = None
watcher._STOP["requested"] = False        # a later tick starts a fresh process

print("9) GPU-contention detection and classifier sanity")
check(not watcher._environmental(RuntimeError("No such file or directory: '/data/bedroom/rec.wav'")),
      "a path containing 'room' is not mistaken for an OOM")
fake = TMP / "appsim"; fake.mkdir(exist_ok=True)
(fake / "app.py").write_text("import time\ntime.sleep(30)\n")
proc = subprocess.Popen([str(PROJ / ".venv/bin/python"), str(fake / "app.py")])
try:
    time.sleep(1.0)
    check(watcher.app_process_running(), "a live 'python app.py' on any port is detected")
finally:
    proc.terminate(); proc.wait(timeout=10)
time.sleep(0.6)
check(not watcher.app_process_running(), "and it is let go once that process exits")

print("10) scanning one folder must not forget another folder's hashes")
other = TMP / "othersub"
other.mkdir(exist_ok=True)
(O := other / "keepme.wav").write_bytes(b"\x55\x66" * 4000)
stA = watcher.Store(TMP / "sA.json")
stA.data = {str(B): {"size": B.stat().st_size, "mtime": round(B.stat().st_mtime, 3),
                    "digest": "keep-this-hash"},
            str(O): {"size": O.stat().st_size, "mtime": round(O.stat().st_mtime, 3),
                     "digest": "keep-this-too"},
            str(TMP / "vanished.wav"): {"size": 1, "mtime": 1.0, "digest": "gone"}}
watcher.scan(TMP, 0, stA, check_open=False)
check(stA.data.get(str(O), {}).get("digest") == "keep-this-too",
      "another folder's cached hash survived this scan")
check(str(B) in stA.data and str(TMP / "vanished.wav") not in stA.data,
      "files of this folder are still added/pruned correctly")


print("11) an idle tick never touches the GPU stack")
IDLE = TMP / "idle"; IDLE.mkdir(exist_ok=True)
(IDLE / "fresh.wav").write_bytes(b"\x01\x02" * 3000)      # too new: settles, nothing to do
calls = {"n": 0}
real_tq = watcher.transcribe_queue
watcher.transcribe_queue = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or (0, 0)
# baseline is taken HERE, not at import: the stubbed runs in sections 1-7 legitimately
# go through transcribe_queue, whose release_gpu() imports torch to empty the cache
baseline = set(sys.modules)
rc_idle = watcher.main(["--dir", str(IDLE), "--min-age", "999999"])
watcher.transcribe_queue = real_tq
check(rc_idle == 0 and calls["n"] == 0, "nothing pending -> the GPU queue was never invoked")
newly = sorted({m.split(".")[0] for m in set(sys.modules) - baseline
                if m.split(".")[0] in ("torch", "nemo", "soundfile", "app", "numpy", "scipy")})
check(not newly, f"idle tick loaded no GPU-stack module (new: {newly})")

print("12) a busy GPU makes it stand down instead of queueing behind your job")
orig_state, orig_rel = watcher.gpu_state, watcher.release_gpu


def tq(**over):
    """One queue run against a fresh ledger; returns (done, failed, models loaded)."""
    led = watcher.Store(TMP / f"l12-{len(COUNT)}-{time.time():.3f}.json")
    led.data = {}
    before = COUNT["init"]
    d, f = watcher.transcribe_queue([mk(B)], _gargs(**over), led)
    return d, f, COUNT["init"] - before


watcher.gpu_state = lambda: (3.0, 94)          # like sglang serving a request right now
d, f, n = tq(min_free_vram=2.0, max_other_util=25)
check(d == 0 and f == 0 and n == 0,
      "deferred at 94% util with 3 GiB free, without loading a single model")

watcher.gpu_state = lambda: (0.5, 2)           # idle card, but not enough VRAM spare
d, f, n = tq(min_free_vram=2.0, max_other_util=25)
check(d == 0 and f == 0 and n == 0, "defers when the VRAM floor is not met")

d, f, n = tq(device="cpu", min_free_vram=2.0, max_other_util=25)
check(n == 1, "--device cpu ignores the GPU guard (there is no GPU to defer for)")

watcher.gpu_state = lambda: (None, None)       # nvidia-smi stops answering
d, f, n = tq(min_free_vram=2.0, max_other_util=25)
check(n == 0, "stands down instead of assuming an idle card it cannot see (fails closed)")

watcher.gpu_state = lambda: (3.0, 94)           # the resident server is at full tilt
d, f, n = tq(min_free_vram=2.0, max_other_util=100)
check(n == 1, "--max-other-util 100 disables the advisory utilisation guard")

# A batch that opened the gate while the card was idle must run to the end of the
# queue even if utilisation climbs: sglang idles at 0 % and sits at ~95 % while
# answering prompts, yet 14 h of backlog completed alongside it. Only VRAM is a
# hard stop between files. Uses its own files so no fault armed by an earlier
# section can leak in through the stub's substring matching.
FL = TMP / "flip"; FL.mkdir(exist_ok=True)
jobs12 = []
for nm in ("first.wav", "second.wav"):
    fp = FL / nm
    with wave.open(str(fp), "wb") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000)
        wv.writeframes(b"\x07\x06" * 4000)
    jobs12.append(mk(fp))

probe = {"i": 0}


def flipping():
    probe["i"] += 1
    return (40.0, 0) if probe["i"] <= 2 else (40.0, 90)   # quiet at the gate, busy after


watcher.gpu_state = flipping
led12b = watcher.Store(TMP / "l12b.json"); led12b.data = {}
before = COUNT["init"]
d, f = watcher.transcribe_queue(list(jobs12), _gargs(min_free_vram=2.0, max_other_util=25), led12b)
check(d == 2 and f == 0 and probe["i"] >= 3 and COUNT["init"] - before == 1,
      f"utilisation climbing mid-batch does not stop it (d={d} f={f} probes={probe['i']} "
      f"models={COUNT['init'] - before}; only the VRAM floor is a hard stop)")

watcher.gpu_state = orig_state

watcher.gpu_state = orig_state

print("14b) the folder is the ground truth, and one recording cannot borrow "
       "another's transcript")
GR = TMP / "ground"; GR.mkdir(exist_ok=True)
def mkwav(dirn, nm, payload):
    fp = dirn / nm
    with wave.open(str(fp), "wb") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000)
        wv.writeframes(payload * 4000)
    return fp
r1, r2 = mkwav(GR, "Rec1_2.wav", b"\x01\x02"), mkwav(GR, "Rec12.wav", b"\x03\x04")
gargs14 = _gargs(no_sidecars=False)
def plan14(root, **over):
    aa = argparse.Namespace(**{**vars(gargs14), **over})
    stx = watcher.Store(TMP / "g14-state.json")
    stx.data = json.loads(stx.path.read_text()) if stx.path.is_file() else {}
    lgx = watcher.Store(TMP / "g14-ledger.json")
    lgx.data = json.loads(lgx.path.read_text()) if lgx.path.is_file() else {}
    out, _ = watcher.plan(root, aa, stx, lgx)
    stx.save(); lgx.save()
    return out, lgx
plan14(GR)                                        # settle
out, _ = plan14(GR)
check(sorted(t["name"] for t in out) == ["Rec12.wav", "Rec1_2.wav"],
      f"Rec1_2 and Rec12 are separate recordings, not twins ({sorted(t['name'] for t in out)})")
# do the work, then remove one transcript: it must come back as WORK, not as a copy
led14 = watcher.Store(TMP / "g14-ledger.json")
watcher.gpu_state = lambda: (40.0, 0)
watcher.transcribe_queue([mk(r1), mk(r2)], gargs14, led14)
side2 = GR / "Rec12.transcript.md"
text_before = side2.read_text(encoding="utf-8")
side2.unlink()
out, _ = plan14(GR)
check([t["name"] for t in out] == ["Rec12.wav"],
      f"a deleted transcript is queued for transcription, not reinstated ({[t['name'] for t in out]})")
check(not side2.is_file(), "and nothing was written back behind its back")
watcher.transcribe_queue([mk(r2)], gargs14, led14)
check(side2.is_file() and side2.read_text(encoding="utf-8") == text_before,
      "re-transcribing it reproduces the same transcript")
# --restore-missing is the explicit, GPU-free alternative
side2.unlink()
out, _ = plan14(GR, restore_missing=True)
check(out == [] and side2.is_file(), "--restore-missing writes the stored copy back instead")
# the mp3/wav pair still counts as one recording, and one transcript covers both
PAIR = TMP / "pair"; PAIR.mkdir(exist_ok=True)
pw = mkwav(PAIR, "Twin.wav", b"\x05\x06")
shutil.copyfile(pw, PAIR / "Twin.mp3")
plan14(PAIR)
jobs_pair, led_pair = plan14(PAIR)
check(len(jobs_pair) == 1, f"the mp3/wav pair is one job, not two ({[t['name'] for t in jobs_pair]})")
watcher.transcribe_queue(jobs_pair, gargs14, led_pair)
pair_md = PAIR / "Twin.transcript.md"
check(pair_md.is_file() and pair_md.read_text(encoding="utf-8").count("# Transcript") == 1,
      "one sidecar covers both members (same stem), written once")
pair_md.unlink()
out_r, _ = plan14(PAIR, restore_missing=True)
check(out_r == [] and pair_md.is_file(),
      "--restore-missing writes it back from the canonical result without the GPU")
pair_md.unlink()
out_w, _ = plan14(PAIR)
check([t["name"] for t in out_w] == [jobs_pair[0]["name"]],
      "by default it is re-transcribed instead (folder = ground truth)")
watcher.gpu_state = orig_state

print("13) the GPU is handed back, and never held past the time limit")
jobs = [mk(B)]
released = {"n": 0}
watcher.release_gpu = lambda *a: released.__setitem__("n", released["n"] + 1)
led13 = watcher.Store(TMP / "l13.json"); led13.data = {}
watcher.gpu_state = lambda: (40.0, 0)
watcher.transcribe_queue(list(jobs), _gargs(min_free_vram=2.0), led13)
check(released["n"] == 1, f"models dropped + allocator flushed after the run ({released['n']}x)")
led14 = watcher.Store(TMP / "l14.json"); led14.data = {}
SLOW = TMP / "slow.wav"
with wave.open(str(SLOW), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x03\x04" * 7000)
class Slow(Transcriber):
    def transcribe(self, wav, publish=None):
        time.sleep(3.0); return list(SEGS)
real_cls = sys.modules["pipeline"].Transcriber
sys.modules["pipeline"].Transcriber = Slow
watcher._STOP["requested"] = False
t0 = time.time()
d14, f14 = watcher.transcribe_queue([mk(SLOW)], _gargs(gpu_time_limit=1/60.0), led14)  # 1 second
sys.modules["pipeline"].Transcriber = real_cls
check(time.time() - t0 < 2.9 and d14 == 0 and not led14.data.get(watcher.file_hash(SLOW), {}).get("status") == "done",
      f"watchdog cut a wedged run at the limit ({time.time()-t0:.1f}s) and left it queued")

print("14) a transcript already in the folder is adopted, not recomputed")
AD = TMP / "adopt"; AD.mkdir(exist_ok=True)
V = AD / "voice.wav"
with wave.open(str(V), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x09\x08" * 5000)
SIDE = V.with_name("voice.transcript.md")
a = _gargs()

def plan_once():
    stx = watcher.Store(TMP / "ast.json")
    stx.data = json.loads(stx.path.read_text()) if stx.path.is_file() else {}
    lgx = watcher.Store(TMP / "alg.json"); lgx.data = {}
    out, _ = watcher.plan(AD, a, stx, lgx)
    return out, lgx

plan_once()                                                    # settle it first
SIDE.write_text("# Transcript — voice.wav\n\n**Speaker 0** `[00:00:00.100]`  hi\n", encoding="utf-8")
COUNT["init"] = 0
out, lgx = plan_once()
check(out == [] and sum(1 for e in lgx.data.values() if e.get("adopted")) == 1
      and COUNT["init"] == 0, "existing markdown -> adopted, zero model loads")

SIDE.unlink()
out, lgx = plan_once()
check([t["name"] for t in out] == ["voice.wav"] and not lgx.data,
      "no markdown there -> queued as normal work (not silently skipped)")

# A transcript left behind by an earlier recording that was later replaced under the
# same name must never be adopted: it predates the audio.
SIDE.write_text("# Transcript — voice.wav\n\n**Speaker 0** `[00:00:00.100]`  hi\n", encoding="utf-8")
old = time.time() - 3600
os.utime(SIDE, (old, old))
out, lgx = plan_once()
check([t["name"] for t in out] == ["voice.wav"] and not lgx.data,
      "a transcript older than the audio is stale and is NOT adopted")
# and a file that merely sits in that slot but is not our export is ignored
SIDE.write_text("somebody's notes, no header\n", encoding="utf-8")
os.utime(SIDE, (time.time(), time.time()))
out, lgx = plan_once()
check([t["name"] for t in out] == ["voice.wav"],
      "a foreign file named *.transcript.md is not adopted")
watcher.release_gpu = orig_rel


print()
print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + '; '.join(fails)}")
for f in fails:
    print("  - " + f)
sys.exit(1 if fails else 0)
