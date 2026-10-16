"""Behavioural tests for the **Use CPU** switch in app.py — `python test_device_switch.py`.

The GPU stack (pipeline, torch) is stubbed and SPEAKER_DATA_DIR points at a throwaway
folder, so nothing here imports NeMo, touches the card, or writes to the repo's data/.
What is under test is the wiring and the reporting:

  drop with the switch ON  -> that job runs on CPU
  drop with the switch OFF -> that job runs on CUDA whenever the card can take it
  a CUDA request the VRAM gate declined is temporary, not a state until restart
  whatever answered is written into the job's provenance together with the reason
"""
import asyncio, json, os, pathlib, sys, tempfile, types, wave

PROJ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

TMP = pathlib.Path(tempfile.mkdtemp(prefix="device-switch-"))
os.environ["SPEAKER_DATA_DIR"] = str(TMP)
os.environ.pop("SPEAKER_DEVICE", None)
os.environ.pop("SPEAKER_CUDA_ONLY", None)
(TMP / "app-settings.json").write_text('{"device": "cuda"}', encoding="utf-8")

# ------------------------------------------------------------------ torch stub
torch = types.ModuleType("torch")


class _Cuda:
    available = True
    empty_cache_calls = 0

    @staticmethod
    def is_available():
        return _Cuda.available

    @staticmethod
    def empty_cache():
        _Cuda.empty_cache_calls += 1

    @staticmethod
    def get_device_name(index=0):
        return "Fake GPU"


torch.cuda = _Cuda
sys.modules["torch"] = torch

# -------------------------------------------------------------- pipeline stub
GATE = {"open": True}
LOADS = {"n": 0, "requested": []}
DECLINE = "declined at the gate: 0.50 GiB free, 3.15 GiB needed"
SEG = {"text": "hello", "start": 0.5, "end": 1.5, "speaker": "speaker_0"}


class FakeDevice:
    def __init__(self, type_):
        self.type = type_

    def __str__(self):                      # like torch.device, so str(tr.device) is honest
        return self.type


class FakeTranscriber:
    """Stands in for either pipeline class and obeys the gate like ``_pick_device``."""

    chunk_step_sec = 1.0

    def __init__(self, requested):
        LOADS["n"] += 1
        LOADS["requested"].append(requested)
        on_cuda = requested == "cuda" and GATE["open"]
        self.device = FakeDevice("cuda" if on_cuda else "cpu")
        # a deliberate cpu request gets no note; only a declined cuda request does
        self.cuda_note = DECLINE if requested == "cuda" and not on_cuda else None
        self.last_corrections = []
        self.evict_to = None                  # set by the eviction scenario

    def load(self):
        return self

    def unload(self):
        pass

    def describe(self):
        return {"label": "Fake ASR", "pipeline": "test", "language": "auto"}

    def transcribe(self, wav, publish=None, offset=0.0):
        if self.evict_to:                     # somebody else pushed us off the card
            self.device = FakeDevice(self.evict_to)
            self.cuda_note = "evicted mid-decode: tried to allocate 198.00 MiB"
        if publish:
            publish(1.0, [dict(SEG)])
        return [dict(SEG)]


def fake_gate_open(device="cuda", asr=None):
    return (True, "") if device != "cuda" or GATE["open"] else (False, DECLINE)


def fake_prepare_audio(src, dst):
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 3)      # 3 s


pipeline = types.ModuleType("pipeline")
pipeline.Transcriber = FakeTranscriber
pipeline.load_transcriber = lambda mode=None, *, device=None, lang=None, glossary=None, **kw: \
    FakeTranscriber(device or "cuda")
pipeline.gate_open = fake_gate_open
pipeline.map_spliced_time = lambda t, m: t
pipeline.splice_speech_regions = lambda wav, quantum=1.0: (wav, [[1.0, 1.0]])
pipeline.trim_edge_silence = lambda wav, quantum=1.0: (wav, 0.0)
pipeline.prepare_audio = fake_prepare_audio
sys.modules["pipeline"] = pipeline

import app                                                   # noqa: E402
import exports                                               # noqa: E402

assert not [m for m in sys.modules if m.split(".")[0] in ("nemo", "soundfile", "librosa")], \
    "the switch tests are supposed to stay off the GPU stack"

fails, N = [], 0


def check(cond, label):
    global N
    N += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        fails.append(label)


def run_one(device, name="meeting.wav"):
    """One job through the same calls POST /api/transcribe makes. Returns the job."""
    global N
    N += 1
    jid = "%012x" % (N * 7919)
    src = app.QUEUE_DIR / f"{jid}-{name}"
    src.write_bytes(b"RIFFfake")
    app.jobs[jid] = {"id": jid, "name": name, "status": "queued", "stage": None,
                     "segments": None, "vad_full": False}
    app._queue.append(jid)
    app.run_job(jid, src, device)
    return app.jobs[jid]


def req(body):
    return types.SimpleNamespace(json=lambda: asyncio.sleep(0, body))


print("1) switch OFF (cuda): the dropped file must land on the card")
app.DEVICE_PREFERENCE = "cuda"
GATE["open"] = True
LOADS.update(n=0, requested=[])
job = run_one(None)                                        # device from DEVICE_PREFERENCE
check(job["status"] == "done", f"job finished (status={job['status']}, error={job.get('error')})")
check(LOADS["requested"] == ["cuda"], "the load asked for cuda")
check(job["engine"]["device"] == "cuda", "provenance says cuda")
check("device_note" not in job["engine"], "no decline reason when the card was used")

print("2) switch ON (cpu): the dropped file must stay on CPU")
app.DEVICE_PREFERENCE = "cpu"
LOADS.update(requested=[])
job = run_one(None)
check(LOADS["requested"] == ["cpu"], "the load asked for cpu")
check(job["engine"]["device"] == "cpu", "provenance says cpu")
check("device_note" not in job["engine"], "cpu-by-choice carries no 'declined' note")

print("3) a gate-declined CUDA run used to be permanent (the actual bug)")
app.DEVICE_PREFERENCE = "cuda"
GATE["open"] = False
LOADS.update(n=0, requested=[])
job = run_one(None)
check(job["engine"]["device"] == "cpu", "ran on CPU while the card was too small")
check(job["engine"].get("device_note") == DECLINE,
      f"and said why: {job['engine'].get('device_note')}")
run_one(None)
check(LOADS["n"] == 1, f"the next decline did not reload the models again (loads={LOADS['n']})")
GATE["open"] = True
job = run_one(None)
check(LOADS["n"] == 2 and job["engine"]["device"] == "cuda",
      "once the card freed up, the next drop had CUDA back without restarting the app")
check("device_note" not in job["engine"], "the stale decline reason is gone")

print("4) eviction mid-decode is reported, and the card is re-attempted next drop")
tr = app.transcriber
tr.evict_to = "cpu"
job = run_one(None)
check(job["engine"]["device"] == "cpu", "provenance shows the device that actually answered")
check(str(job["engine"].get("device_note")).startswith("evicted mid-decode"),
      f"and the reason: {job['engine'].get('device_note')}")
tr.evict_to = None
job = run_one(None)
check(job["engine"]["device"] == "cuda", "the drop after the eviction is back on the card")

print("5) the endpoints report the truth, not the request")
app.DEVICE_PREFERENCE = "cuda"
GATE["open"] = False
res = asyncio.run(app.api_set_device(req({"device": "cpu"})))
check(res["actual_device"] == "cpu" and not res["device_note"],
      "choosing CPU deliberately is not reported as a decline")
res = asyncio.run(app.api_set_device(req({"device": "cuda"})))
check(res["actual_device"] == "cpu" and res["device_note"] == DECLINE,
      f"asking for CUDA with the gate shut says so: {res['device_note']}")
st = asyncio.run(app.api_settings())
check(st["device_preference"] == "cuda" and st["actual_device"] == "cpu",
      f"/api/settings shows preference and reality side by side: {st['device_preference']}/{st['actual_device']}")
GATE["open"] = True
res = asyncio.run(app.api_set_device(req({"device": "cuda"})))
check(res["actual_device"] == "cuda" and res["device_note"] is None,
      "flipping the switch again takes the card as soon as it is free")
check(json.loads((TMP / "app-settings.json").read_text())["device"] == "cuda",
      "the preference was persisted for the next launch")
check(app.DEVICE_PREFERENCE == "cuda" and LOADS["requested"][-1] == "cuda",
      "and the running process agrees")

print("6) provenance survives a restart and reaches the exported markdown")
app.jobs.clear()
rehydrated = app._lookup_job(job["id"])
check(rehydrated is not None and rehydrated["engine"]["device"] == "cuda",
      "a finished job read back from disk still knows which device answered")
md = exports.render_exports({"id": job["id"], "name": "meeting", "segments": job["segments"],
                             "labels": {}, "engine": job["engine"]})["md"]
check("on cuda" in md, "the transcript names the device: "
       + str([l for l in md.splitlines() if "Fake ASR" in l]))
md_cpu = exports.render_exports({"id": "x", "name": "meeting", "segments": [dict(SEG)],
                                 "labels": {},
                                 "engine": {"label": "Fake ASR", "device": "cpu",
                                            "device_note": DECLINE}})["md"]
check(f"on cpu ({DECLINE})" in md_cpu, "a forced CPU run says so in the transcript")

print("7) the tests stayed in their own folder")
check(app.DATA_DIR == TMP and app.RESULTS_DIR.parent == TMP, "SPEAKER_DATA_DIR was honoured")
check(app.DEVICE_SETTINGS_FILE.parent == TMP, "no real data/app-settings.json was touched")

print(f"\n{N - len(fails)}/{N} passed")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
