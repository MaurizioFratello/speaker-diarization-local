# Swappable ASR backends — investigation & plan

Status: research only, no code changed. Date: 2026-10-13.
Target: keep `Nemotron-3-Diarization` for *who spoke when*, replace/augment the
*nemotron-3.5-asr-streaming-0.6b* text layer with a stronger (or cheaper) ASR.

---

## 1. Why the ASR is not a drop-in today

`pipeline.py` runs NeMo Speech's **coupled multitalker streaming path**
(`SpeakerTaggedASR` + `CacheAwareStreamingAudioBuffer`):

* Diarization emits **frame-level speaker activity** (80 ms frames, derived from
  the 10 ms feature grid).
* For every detected speaker the pipeline opens **its own ASR stream**
  (`max_num_of_spks=8` → up to 8 concurrent decoder states), and multiplies the
  speaker's activity mask onto the mel features (`masked_asr=true`).
* The ASR encoder's chunk geometry **drives** the diarizer's chunk geometry
  (`validate_feature_frame_strides`, `configure_diar_streaming`,
  `diar_chunk_len = valid_out_len + cache_drop_size`, `feat_len_sec=0.01`).

Consequences for model swapping:

| Requirement of the current path | What it rules out |
|---|---|
| Encoder must be **cache-aware streaming** with the same 10 ms feature stride and subsampling factor 8 | every offline-only model (Parakeet TDT, Canary AED, Whisper, Moonshine, Granite Speech) |
| Model must accept a **feature mask** per speaker stream (or speaker targets) | models without a masking hook |
| Prompt-conditioned language support for 32+ locales | `multitalker-parakeet-streaming-0.6b-v1` (English only) |

Per NVIDIA's own `ASR_INTEGRATION_GUIDE.md` (model repo) there are exactly **two**
supported pairings: Multitalker Parakeet (English) and Nemotron 3.5 ASR. So for
German, **`nemotron-3.5-asr-streaming-0.6b` is currently the only model that fits
the coupled architecture at all** — there is nothing better to "drop in" on that
rail. The guide says it explicitly: *"A conventional single-speaker ASR can be
applied after diarization by cutting the audio at each speaker segment."*

→ A better transcription model means adding a **second pipeline mode**
(diarize-first), not just changing a model id.

The NeMo install already in `.deps/nemo-speech` (3.1.0+1688cc3) contains the
building blocks for that mode: `SortformerEncLabelModel.diarize()` (offline
config, SegLST output, optional frame tensors) and
`SpeakerTaggedASR.perform_offline_stt_spk()` / `merge_transcript_and_speakers()`
which tag **ASR words with the dominant diarization speaker** — see
`get_word_dict_content_offline()`. So we are not inventing anything; we are
choosing the other supported configuration.

## 2. What the current transcripts show (local evidence)

Across the 26 files in `data/transcripts/`:

* median sentence-break interval 2.7–6.6 s, and **11–26 % of segments are ≤ 2
  words** — classic streaming sentence-break artefacts
  (`sent_break_sec=1.0`, `word_window=50`).
* words split across segment boundaries: *"für die Kraftsensori / kommen wir…"*,
  sentences restarting mid-clause.
* compound/technical German mangled: *"Stress kompensieren"*, *"Ebenmoduli"*
  (E-Modul), *"Norhan induziert"*, *"Kassel Kass aktuell"*.
* no leaked `<de-DE>` tags (strip works), one job at up to 8 speakers.

So the pain is (a) wrong words for domain vocabulary and (b) chunk-boundary
fragmentation. (b) is a property of the **streaming decode**, not of the weights;
(b) alone justifies an offline pass.

## 3. Candidate models (German-focused; numbers are the vendors' own)

German WER on FLEURS test unless noted. Lower is better. Different panels are
**not strictly comparable** — treat as tiers, not a ranking.

| Model | Arch / mode | German | Size (download) | Timestamps | License | Fit here |
|---|---|---|---|---|---|---|
| `nvidia/nemotron-3.5-asr-streaming-0.6b` (**current**) | cache-aware FastConformer-RNNT, streaming | **8.31** @1.12 s chunk / 8.22 auto | 2.37 GB `.nemo` | word (frame) | OpenMDW-1.1 | baseline; also runs **offline/whole-file** (see §5a) |
| `nvidia/parakeet-tdt-0.6b-v3` | FastConformer-**TDT**, offline | **5.04** (MLS/CVoST similar) | 2.39 GB `.nemo`, 681 MB q8 GGUF | word+segment, native | CC-BY-4.0 | **best quality/effort ratio**; 25 EU langs; up to 24 min full-attention |
| `nvidia/canary-1b-v2` | FastConformer-**AED**, offline | **4.40** (FLEURS), 5.53 (CoVoST) | 6.06 GB `.nemo` | word+segment, `timestamps=True` | CC-BY-4.0 | most accurate multilingual NVIDIA model; also AST (de→en gloss!) |
| `nvidia/stt_de_fastconformer_hybrid_large_pc` | FastConformer hybrid CTC/RNNT, offline, **German-only** | MLS **3.87**, CVoICE **5.1**, VoxPopuli 8.88 | ~1.9 GB | no native (needs aligner) | CC-BY-4.0 | strongest pure-German prior; no language mixing |
| `moonshine-ai/moonshine-streaming-small-de` | Sliding-window enc–dec, streaming | **7.5** (400-clip FLEURS+MLS macro, quantized) | 431 MB `.safetensors` (112.9 M) | none | MIT | edge/CPU class, not a quality upgrade; Transformers-native |
| `moonshine-ai/moonshine-streaming-tiny-de` | same, 34 M | 12.0 | ~50…130 MB | none | MIT | NAS/CPU fallback only |
| OpenAI Whisper large-v3 / turbo (faster-whisper) | AED, offline | ≈4.0 (third-party German comparison) | 1.5–3 GB | via forced aligner | MIT | strong, but hallucinates on silence/overlap; separate runtime |
| IBM Granite Speech 4.1 2B / 5.0 470M-turboCTC | SLM, offline | supported (6 langs: en, fr, de, es, pt, ja) | 2 B / 0.5 B | CTC variant yes | Apache-2.0 | interesting if "Granite" was the model you meant; HF `generate`-based, heavier code path |
| `nvidia/multitalker-parakeet-streaming-0.6b-v1` | cache-aware, streaming, overlap-tuned | **English only** | ~2 GB | word | CC-BY-4.0 | only other model on the current coupled rail — wrong language |

Notes:
* `parakeet-tdt-0.6b-v3` and `canary-1b-v2` load with the **NeMo you already have**
  (`ASRModel.from_pretrained`) — no new runtime, no new deps.
* Moonshine's German checkpoints are MIT and runnable through the installed
  `transformers` 5.18 (`MoonshineStreamingForConditionalGeneration`), but they
  are ~10× smaller models trained for edge latency: on their own numbers they land
  near the current streaming model, not near Parakeet/Canary.
* Canary-Qwen-2.5B is **English-only** despite the docs table saying EU25 → skip.
* NVIDIA's Open-ASR-Leaderboard numbers for `parakeet-tdt-0.6b-v3` (AMI 11.31 %,
  Earnings-22 11.42 %) are meeting-domain and English; nobody publishes German
  *meeting* WER — your own recordings are the only real benchmark (see §7).

## 4. Structural trade-off you are buying

| | today (coupled streaming) | diarize-first (offline ASR) |
|---|---|---|
| live partial transcript | yes, every ~0.5 s | needs re-implementing (per-segment publish, or two-pass) |
| overlapping speech | masked per-speaker streams → both voices can be transcribed | an extracted cut still contains every voice in it → dominant voice wins, overlap words can be lost/duplicated |
| ASR compute | 1 stream × N speakers, 1.12 s chunks | one pass over the mix (batched) — usually **faster overall** |
| VRAM | ASR state × max_speakers | one model, one batch → much lower (nice for the watcher's GPU-courtesy rules) |
| sentence quality | chunk-limited (see §2) | full attention → proper sentences, fewer mid-word cuts |
| diarization | streaming config forced by ASR geometry | free to use the **offline** config (chunk_len 340 / right_ctx 40 / update 300, 30.4 s buffer) → better DER |

Middle ground for overlap (do later, only where diarization says ≥2 speakers are
active): build a per-speaker waveform/mel by zeroing other speakers' dominant
frames, ASR that cut only for those regions, keep the mix-pass words elsewhere.
That recovers most of the masked-ASR benefit while keeping an offline model.

## 5. Cheapest wins first (no re-architecture)

a) **Same weights, offline decode.** NeMo-Speech.cpp documents Nemotron 3.5 as
   supporting *whole-file recognition* as well as `--stream`; in Python that is
   `asr_model.transcribe(...)` on segments (the cache-aware encoder ignores the
   streaming accumulator offline). Expected: §2 fragmentation gone, WER somewhere
   between 8.3 and Parakeet territory, **zero downloads, ~1 h of work**.
   Must verify locally that offline decode of this streaming checkpoint works in
   NeMo 3.1 (`forward_for_asr` has no streaming guard — check by running it).
b) **Pin the language instead of `auto`.** Card numbers for de-DE are near
   identical (8.31 LangID vs 8.22 auto @1.12 s), but pinning removes the
   occasional wrong-language guess on short/code-switched clips. Needs a per-file
   `--lang de-DE|auto` knob.
c) **Sentence-breaker tuning** (`sent_break_sec`, `word_window`,
   `fix_prev_words_count`) — cosmetic but cheap; only helps the streaming mode.
d) **VRAM/precision**: on this box (RTX PRO 6000, 96 GB, but often ~4 GB free)
   the bf16-vs-fp32 auto-switch is what it is; offline Parakeet/Canary are single
   models without per-speaker streams, so they're gentler than today.

## 6. Proposed design: one contract, several engines

Keep the output contract exactly as-is — `[{"text","start","end","speaker"}]` —
so `app.py`, `exports.py`, `watcher.py`, `static/index.html` and the ledger/
hash logic do not change.

```
pipeline.py
  prepare_audio()  trim_edge_silence()  splice_speech_regions()      (unchanged)
  └─ Transcriber(asr="nemotron-streaming" | "nemotron-offline" | "parakeet-tdt" |
                            "canary" | "de-fastconformer" | "moonshine-de" | "whisper")
       ├─ StreamingDiarAsr   (today's coupled path; keeps live partials)
       └─ DiarizeFirstAsr    (new)
            1. diar_model.diarize(audio, include_tensor_outputs=True)   # offline cfg
            2. segments → speaker-majority frame probs (RTTM-like)
            3. cut audio per segment (min/max length, merge <0.4 s, cap ~20 s)
            4. engine.transcribe(batch of cuts) → text (+ word ts if available)
            5. re-stamp onto original timeline, publish per completed segment
```

* `AsrEngine` protocol: `load(device, dtype)`, `transcribe_segments(list[(wav, sr)])`
  → `list[list[(text, t0, t1)]]`, `name`, `languages`, `supports_timestamps`.
* Engines: `NemotronStreamingEngine` (wrap today's code), `NemotronOfflineEngine`,
  `ParakeetTdtEngine` (NeMo, `timestamps=True`), `CanaryEngine` (AED + timestamps,
  optional `ctx` language prompt `de` and optional **de→en translation** pass),
  `MoonshineEngine` (Transformers or the `.ort` C runtime; segment-per-cut),
  `WhisperEngine` (optional, faster-whisper).
* Local mirrors stay a first-class citizen: `models/<name>.nemo` / repo dir,
  same `_load_model()` rule as now (offline NAS boxes must not need HF at runtime).
- * Live partials in diarize-first: publish as soon as a segment finishes →
  progress bar + transcript still paint progressively (slightly lumpier). Keep
  the streaming engine as the "live" mode and add a **two-pass mode**: streaming
  draft while the GPU works → offline refine before the job is marked done
  (`--refine`), which gives both live text and the best final quality.
* UI/CLI surface: dropdown or `--asr/--lang/--refine`; persist the engine name in
  `data/results/<hash>.json` (new key `asr`) so a re-export never silently mixes
  engines; watcher gets `ASR_ENGINE=` in `setup-watcher.sh`.
* Speaker-count config: `NEMODIAR_MAX_SPKS` stays meaningful only for the coupled
  engine; diarize-first caps at the diarizer's 8 channels.

## 7. How to decide (do this before committing to a backend)

1. Pick 3 recordings (one 2-speaker clean, one 4–8-speaker noisy, one heavy
   Denglisch/technical) ≈ 10 min total; hand-correct the text → `samples/groundtruth/*.md`.
   (You already have `samples/ami_es2009a_4spk_55s.md` and
   `notsofar_mtg32000_4spk_32s.md` as seeds.)
2. Script `bench_asr.py`: run each engine over the same 16 kHz wavs, compute
   `jiwer` WER/CER vs ground truth, plus DER for speaker boundaries, plus wall-clock
   and peak VRAM (`nvidia-smi` sampling), plus segment-fragmentation stats
   (segments/file, % ≤2 words, words split across segments).
3. Run it for: current streaming, §5a offline Nemotron, Parakeet TDT v3,
   Canary-1b-v2, `stt_de_fastconformer_hybrid_large_pc`, Moonshine small-de.
   Diff the outputs on the German technical terms you care about
   (Stresssensor, E-Modul, MEMS, Abstimung, …).
4. Only then wire the winner(s) into the app; keep two engines max to start.

## 8. Phasing

| Phase | Work | Outcome |
|---|---|---|
| 0 | pin `target_lang` knob (+UI/CLI), sentence-breaker A/B, re-run 3 files | cheaper, better streaming output; no new deps |
| 1 | `AsrEngine` protocol + `DiarizeFirstAsr` with `nemotron-offline` | full-context sentences, no download, overlap caveat accepted |
| 2 | `parakeet-tdt` (+ optionally `canary`) engines, batched cuts, per-segment publish | the real accuracy step (5.0 / 4.4 vs 8.3 tier) |
| 3 | overlap-aware masked cuts, or `--refine` two-pass (streaming draft → offline final) | parity with masked-ASR on overlap, keep live UX |
| 4 | Moonshine-de (edge/CPU path for the watcher when the GPU is busy), glossary/hotwords, ITN, optional de→en gloss from Canary | polish |

## 9. Gotchas found while reading the code/docs

* NeMo couples diar↔ASR geometry **only** in the streaming mode; `diarize()` in
  offline mode needs `sortformer_modules.chunk_len=340`, `chunk_right_context=40`,
  `spkcache_update_period=300`, `_check_streaming_parameters()` before calling.
* `perform_offline_stt_spk` (NeMo's own merge) requires `hypothesis.timestamp['word']`
  → works for TDT/Canary, **not** for CTC-only or Moonshine; those need segment-cut ASR.
* `prepare_audio` output is grid-quantised by VAD trimming
  (`quantum=chunk_step_sec`); that quantum is a *streaming* concept. In
  diarize-first, cut points no longer need chunk-grid alignment — but keep the
  splicing/`map_spliced_time` bookkeeping for the streaming engine only.
* Canary/Whisper (AED) can hallucinate on music/silence; the existing Silero
  trim/splice helps, but the diarization-derived cuts should be the VAD here.
* Model size on first watcher tick: Parakeet 2.4 GB, Canary 6.1 GB, Whisper ~1.5 GB
  to `models/` — the HF download path in `_load_model()` needs the same
  "no network / mirror exists" logic it has today, plus a `--asr` flag in
  `setup-watcher.sh`'s `EXTRA_ARGS`.
* Speaker labels stay arrival-ordered and per-file (unchanged).
* `binary_diar_preds` / `min_sigmoid_val` / segment postprocessing tuning lives
  with the diarizer and is shared by both modes.

## 10. Locked decisions (from the operator, 2026-10-13)

* **Quality beats live text.** Live partial transcripts stay a nice-to-have, so the
  default path is **diarize-first + offline ASR**, publishing per completed speaker
  segment (progress bar, growing transcript, speaker legend all still work). The
  streaming draft → offline refine (two-pass) is an *opt-in later extra*, not the
  baseline.
* **Two global tiers, one setting** (single meeting at a time, no per-upload picker):
  * `balanced` → `nvidia/parakeet-tdt-0.6b-v3` (2.4 GB, de FLEURS 5.0)
  * `quality` → `nvidia/canary-1b-v2` (6.1 GB, de FLEURS 4.4)
  * `live` → today's coupled streaming path stays in the tree (fallback + two-pass
    draft + the CPU/`--device cpu` route), no behaviour change.
* **Language: detect by default, allow an override** (`--lang auto|de|en|…`).
* **Overlap is turn-taking**, mostly at sentence starts/ends — see §12, no source
  separation needed.

## 11. Language control per engine (this is what actually constrains the design)

| Engine | Auto-detect | Pin a language | How |
|---|---|---|---|
| parakeet-tdt-0.6b-v3 | **yes, built in** — "automatically detects the language of the audio and transcribes it without requiring additional prompting" | awkward: language is a prefix token (HF `forced_decoder_ids` / `get_decoder_prompt_ids`); NeMo issue #14799 reports short-clip misclassification | `ASRModel.from_pretrained(...).transcribe(files, timestamps=True)` |
| nemotron-3.5-asr-streaming (current / `live`) | yes | **first-class**: `set_inference_prompt('de-DE')` | already wired in `pipeline.py` (`target_lang`) |
| canary-1b-v2 | **no clean autodetect**: NeMo requires `source_lang=`/`target_lang=` in `transcribe()`; unsupported language raises in `CanaryTokenizer` (`_map_canary1_to_canary2_lang`) | first-class (`source_lang='de'`), and free bonus: `target_lang='en'` = de→en gloss | needs a Lang-ID stage |
| stt_de_fastconformer_hybrid_large_pc | n/a (German-only) | always German | — |
| moonshine-*-de | n/a (one model per language) | n/a | — |
| Granite Speech 4.1 2B | yes (6 langs) | `language="German"` | prompt-based |

**Consequence — the `quality` tier needs a language decision per cut.** Design:

1. one Lang-ID stage feeding the AED model: `speechbrain/lang-id-voxlingua107-ecapa`
   (ECAPA-TDNN, 107 spoken languages, **Apache-2.0**, ~760 KB graph + 85 MB weights,
   plain ONNX exports exist, CPU, milliseconds per 3 s clip);
2. detect on ≥3 s of diarized speech, aggregate per speaker-run with hysteresis
   (never flip language mid-sentence — that would be worse than code-switch
   mis-spelling);
3. `--lang de` bypasses the detector entirely; a mixed-language meeting stays on
   German (English tech terms are a *spelling* problem, not a language-ID problem,
   so the German decoder is still the right choice);
4. in two-pass mode the already-loaded Nemotron model's own `<de-DE>` tags can
   replace the classifier for free.

## 12. Overlap strategy (turn-taking at sentence edges)

A diarization cut still contains every voice inside it, so the failure mode is a
neighbour's word leaking into the neighbouring speaker's line. Fix — this is
exactly what the offline models' **word timestamps** buy us:

1. shrink each cut to the frames where the target speaker dominates
   (`p_spk > 0.5` and ≥ other speakers' `p`, ± 80 ms frame grid);
2. transcribe the (unshrunk, context-padded) region, then **drop every word whose
   timestamp falls outside the dominant window** — the other speaker's leak;
3. dedupe: if two speakers emit the same token with overlapping timestamps, keep
   the one whose dominance is higher (prevents "both say it");
4. optionally re-extend to the diarization segment boundary for the *displayed*
   start/end so the timeline keeps looking like the diarization output.

Sustained overlap (two people talking for 10 s) then yields the dominant voice per
speaker, which is what a meeting transcript needs anyway; if it ever becomes a
problem, §Phase 3's masked per-speaker cuts slot in without interface changes.

## 13. Hotwords / glossary — is it an architectural decision?

Partly. Acoustic word boosting is decoder-specific:

* **not available** on TDT (Parakeet v3) or AED (Canary, Whisper) greedy decoding;
* available on **CTC** (flashlight n-gram LM) and on **cache-aware RNNT** via
  NeMo-Speech.cpp's built-in context-biasing tree — but its boosted German-capable
  checkpoints are thin on the ground (`parakeet-ctc-1.1b` is English; the German
  `stt_de_fastconformer_hybrid_large_pc` has a CTC head);
* **IBM Granite Speech 4.1 2B** does keyword biasing natively *and* speaks German:
  `transcribe(audio, task="transcribe", language="German", keyword_biases=["Müller-Lüdenscheidt", "SAP S/4HANA"])`
  (Apache-2.0, ~2 B, HF `generate` path — heavier and slower than Parakeet).

Recommendation: **text-side glossary correction** as a post-pass (deterministic,
model-agnostic, works for both tiers, auditable diff, no new model): fuzzy
pronunciation/Levenshtein matching of out-of-vocabulary candidates against a
`glossary.txt` (names + MEMS/E-Modul/Stresssensor…), which fixes exactly the
corpus of errors seen in §2 (`Ebenmoduli` → `E-Modul`). Keep **Granite Speech**
as an optional third engine if you later want acoustic biasing, and keep
**NeMo-Speech.cpp** as the low-VRAM / CPU fallback (§15).

## 14. Build order

| Step | Deliverable | Needs download |
|---|---|---|
| 1 | `AsrEngine` protocol + `DiarizeFirstAsr` (offline `diarize()` → cuts → dominant-window filter) wired to **`nemotron-offline`** (same weights you already have) + per-segment publish; `--asr`/`--lang` flags; engine name stored in `data/results/<hash>.json` | no |
| 2 | `bench_asr.py`: WER/CER (jiwer) + DER + wall-clock + peak-VRAM + fragmentation stats on 3 hand-corrected clips | no (parakeet 2.4 GB) |
| 3 | `parakeet-tdt` engine = **balanced** tier | 2.4 GB |
| 4 | `canary` engine + Lang-ID stage = **quality** tier | 6.1 GB + ~85 MB |
| 5 | glossary post-correction + ITN; optional Granite/`moonshine-de` engines | optional |

## 15. Why not NeMo-Speech.cpp as the engine (your question)

It runs **Nemotron 3.5** (40 locales, cache-aware, whole-file *or* streaming),
**Parakeet TDT 0.6B v3** (25 langs, explicitly *full-utterance only* — streaming
requests are rejected) and **Parakeet CTC 1.1B** (English); it has **no Canary**.
So adopting the C++ runtime would cap the top tier at Parakeet (de 5.0) and lose
Canary (de 4.4). What it would buy: 681 MB q8 GGUF instead of 2.4–6.1 GB `.nemo`,
acoustic word boosting (CTC / cache-aware RNNT), rule-based ITN, and a
one-binary VAD+diar+ASR+PnC+ITN pipeline (incl. `--diarize` word-level speaker
tags with Nemotron 3 Diarization at 10 ms). Verdict: **keep it as the fallback
engine** for tight-VRAM/CPU boxes or if hotword boosting becomes non-negotiable —
not as the primary path, because it caps peak accuracy and would replace the
diarization stage we are deliberately keeping.

## Sources

* `nvidia/Nemotron-3-Diarization` README + `ASR_INTEGRATION_GUIDE.md` (HF, Oct 2026)
* `nvidia/nemotron-3.5-asr-streaming-0.6b` model card (HF; FLEURS WER per locale/chunk)
* `nvidia/parakeet-tdt-0.6b-v3` card (25 languages, de FLEURS 5.04, Open-ASR table)
* `nvidia/canary-1b-v2` card (de FLEURS 4.40, timestamps, CC-BY-4.0)
* `nvidia/stt_de_fastconformer_hybrid_large_pc` card (MLS de 3.87, CV 5.1, VoxPopuli 8.88)
* NeMo-Speech docs: *ASR Model Checkpoints*, *Canary Chunked and Streaming Decoding*
* NeMo-Speech.cpp `docs/asr/models.md`, `docs/cli.md` (Parakeet TDT v3 offline-only;
  Nemotron 3.5 whole-file + streaming; Sortformer V3 diarization; word boosting/ITN)
* Moonshine Voice docs: *Available Models*, *Accuracy (WER)* (de Small Streaming 7.5 %,
  de Tiny 12.0 %, MIT; VAD `vad_max_segment_duration` default 15 s)
* `nvidia/canary-qwen-2.5b` card (English-only)
* IBM Granite Speech docs (en/fr/de/es/pt/ja); third-party German comparison
  (iravoice parakeet-vs-whisper: Whisper large-v3 4.0 vs Parakeet TDT v3 4.12)
* Local: `.deps/nemo-speech/nemo/collections/asr/parts/utils/multispk_transcribe_utils.py`
  (`SpeakerTaggedASR.perform_offline_stt_spk`, `merge_transcript_and_speakers`,
  `get_word_dict_content_offline`, `mask_features`, `validate_feature_frame_strides`)

---

## 16. What was built (2026-10-13)

Shipped, in this order (each a separate commit, so any step can be reverted):

| commit | content |
|---|---|
| `1db071a` | `model_store.py` — local `models/*.nemo` first, Hub as fallback |
| `491c8c8` | `glossary.py` + 81-check suite — text-side vocabulary repair with an audit trail |
| `bcf53e9` | `engines.py`, `segment_asr.py`, `pipeline.py` — the diarize-first mode and its four checkpoints |
| `060e091` | `--asr/--lang/--glossary` through app, watcher and systemd unit |
| `384db0e` | glossary aliases for the renderings this corpus really produced |

Implemented as planned: one output contract, two pipeline shapes, one global
engine setting, per-segment progress instead of per-second, dominance-window leak
rejection, and a hallucination/repeat guard before text reaches the transcript.

### Measured, not assumed

- **Offline diarization works** with `spkcache_len=264, fifo_len=40, chunk_len=340,
  chunk_right_context=40, spkcache_update_period=300` + `_check_streaming_parameters()`;
  the probability grid arrives at 10 ms resolution (upsampled from 80 ms encoder
  frames), which is fine-grained enough to attribute words to speakers.
- **Speaker turns agree with the streaming path** on the same 3-minute clip:
  turn points within 0.3–1.0 s, words per speaker 217/120 vs 212/128 — the
  diarization half is not inventing a different conversation.
- **The prompt checkpoint must be called with `use_lhotse=False`.** With the lhotse
  path, the prompt index is read from `cut.supervisions[0].language`, which a bare
  file list does not carry, so roughly half of every batch dies on
  `Unknown prompt key: 'None'`. The published example uses
  `override_config=RNNTPromptTranscribeConfig(...)`; it only works because that
  bypasses the prompt dataset.
- **`nemotron-3.5-asr-streaming-0.6b` exposes no word timestamps offline**
  (`hyp.timestamp` empty with `timestamps=True`), so `fast` synthesises them.
  The other three engines report real word and segment timestamps.
- **A saturated GPU cannot even hand out a CUDA context** on this box (another
  tenant holds 92 of 96 GiB), so the free-VRAM probe is wrapped: `--device cuda`
  degrades to CPU with a log line instead of failing the job. The watcher's
  courtesy floor is derived from the selected engine (`engines.vram_floor_gib`).
- **Word offsets are local to the cut; the dominance window is in file time.**
  Comparing them without the shift dropped the head of every cut after the first.
  `test_engines.py` covers this and the rest of the geometry without loading a model.
- **`difflib` autojunk is a trap for CER**: with the default, text that was 6 %
  wrong scored 71 %. Either pass `autojunk=False` or use `jiwer`.

### Still open

1. **Spoken language ID** for `quality` + `language=auto` (plan §11). Today
   `quality` refuses to guess, which is honest but a papercut; the design (hysteresis
   over diarized speaker runs, `speechbrain/lang-id-voxlingua107-ecapa`) stands.
2. **Tuning pass** on `SegCfg` (`pad`, `threshold`, `merge_gap`, `sent_break`) against
   the benchmark; 0.15 s of context padding already visibly helps, `pad=0.35` may help more.
3. **Acoustic biasing** (plan §13): n-gram CTC/flashlight LM or `decodercontext`
   on Canary, instead of text-side repair only.
4. **Overlap**: the dominant-speaker rule is a deliberate loss of the quieter
   channel; source separation is the only way to recover it and is not planned.
