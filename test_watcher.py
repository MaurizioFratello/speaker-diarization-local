"""Behavioural tests for watcher.py's failure classification and queue gating.

Everything runs against a throwaway folder with the GPU stack and the on-disk
bookkeeping stubbed, so the project's real data/ and GPU are untouched.
"""
import argparse, errno, json, os, pathlib, shutil, signal, subprocess, sys, tempfile, time, types, wave

PROJ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import watcher

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


class Transcriber:
    def __init__(self, **kw): pass

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
args = argparse.Namespace(device="cuda", max_speakers=None, progress_every=0.0,
                         no_sidecars=True, format_list=["md"])


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
check(plan_with(done_e, [A]) == [], "an already-transcribed file is skipped")

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
todo8, _ = watcher.plan(TMP, a8, st8, lg8)
check(B.name not in [t["name"] for t in todo8],
      "the existing canonical transcript is trusted; the file is not requeued")
canon8.unlink()
check(B.name in [t["name"] for t in watcher.plan(TMP, a8, st8, lg8)[0]],
      "without it the file is queued again (the marker really is what gates it)")

print("8) SIGTERM (as systemd sends it) exits cleanly with bookkeeping intact")
led9 = watcher.Store(TMP / "l9.json"); led9.data = {}
MODE["sigterm_on"] = "good"; watcher._STOP["requested"] = False
done9, failed9 = watcher.transcribe_queue([mk(B)], args, led9)
e9 = next(iter(led9.data.values()), {})
check(watcher._STOP["requested"] and done9 == 0 and failed9 == 0,
      f"interrupted run reported cleanly (done={done9} failed={failed9})")
check(e9.get("status") == "pending", f"the interrupted file stays queued ({e9.get('status')})")
MODE["sigterm_on"] = None

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


print()
print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + '; '.join(fails)}")
for f in fails:
    print("  - " + f)
sys.exit(1 if fails else 0)
