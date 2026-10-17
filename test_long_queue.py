"""Opt-in real CUDA integration test: submit two long files together via the API.

    .venv/bin/python test_long_queue.py first.mp3 second.mp3 --cuda-memory-gib 5

Runs in temporary app storage. Never edits the source audio or real data/.
CPU fallback is disabled, so a passing test genuinely used CUDA for both jobs.
"""
import argparse
import concurrent.futures
import os
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs=2, type=Path)
    parser.add_argument("--cuda-memory-gib", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    for path in args.audio:
        if not path.is_file():
            parser.error(f"audio file not found: {path}")
    if args.cuda_memory_gib <= 0 or args.timeout <= 0:
        parser.error("memory limit and timeout must be positive")

    import torch
    from fastapi.testclient import TestClient

    if not torch.cuda.is_available():
        parser.error("this integration test requires a CUDA GPU")
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = args.cuda_memory_gib * 2**30 / total
    if fraction > 1:
        parser.error("memory limit exceeds this GPU's total VRAM")
    torch.cuda.set_per_process_memory_fraction(fraction)
    os.environ["SPEAKER_CUDA_ONLY"] = "1"
    os.environ["SPEAKER_DEVICE"] = "cuda"
    with tempfile.TemporaryDirectory(prefix="speaker-long-queue-") as root:
        os.environ["SPEAKER_DATA_DIR"] = root
        # Import only after selecting isolated storage: app boot sweeps uploads/.
        import app
        app.ASR["mode"] = "balanced"
        app.ASR["lang"] = "auto"
        app.DEVICE_PREFERENCE = "cuda"
        tr = app.get_transcriber("cuda")
        model = tr._impl.engine.model
        with TestClient(app.app) as client:
            def upload(path):
                with path.open("rb") as f:
                    response = client.post("/api/transcribe", files={"file": (path.name, f)})
                response.raise_for_status()
                return response.json()["job_id"]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                ids = list(pool.map(upload, args.audio))
            print("Queued two files:", *ids, flush=True)
            deadline = time.monotonic() + args.timeout
            previous = {}
            while True:
                states = []
                for jid in ids:
                    response = client.get("/api/jobs/" + jid)
                    response.raise_for_status()
                    job = response.json()
                    states.append(job)
                    state = (job["status"], job.get("stage"))
                    if previous.get(jid) != state:
                        print(jid, *state, flush=True)
                        previous[jid] = state
                if all(s["status"] in ("done", "error") for s in states):
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("queued transcription test timed out")
                time.sleep(1)
            for job in states:
                assert job["status"] == "done", job.get("error")
                assert job["segments"], "completed with no transcript"
                assert job["engine"]["device"] == "cuda", job["engine"]
                print(job["name"], "CUDA OK; elapsed", job["elapsed"], "s", flush=True)
            assert tr._impl.engine.model is model, "ASR model was unnecessarily reloaded"
            print("Peak allocated GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 3))
            print("PASS: both queued files completed on CUDA; CPU fallback disabled")


if __name__ == "__main__":
    main()
