# v9 — Audio features: is a plain mel enough? (robustness + SOTA front-ends)

STATIC — v9 research note (no code changed; recommendation ladder for the audio conditioning).

Question (user): different audio files for the "same" song differ in quality, noise and
loudness even at the same title+artist. Is the plain log-mel enough, or should we use a
pretrained/SOTA audio model (MERT, CLAP, Spleeter), a denoiser, or richer features?

## 0. The current audio path (what we'd be changing)

`src/data/audio.py` + `AudioConfig` (`src/config.py`): `librosa.load(sr=22050, mono)` →
`melspectrogram(n_fft=1024, hop=256, n_mels=64, fmin=20, fmax=11025, power=2.0)` →
`power_to_db(ref=np.max)` → `(dB+40)/40`. Result `(64, T)` at **86.13 fps (~11.6 ms/frame)**,
frame-aligned to the beatmap signal. It is the **only** audio conditioning: `unet.py`
concatenates it channel-wise with the 21-ch noisy signal at `in_conv` (`cond_channels =
AUDIO.n_mels = 64`). Computed once at preprocess (`mels/<audio_id>.npy`, float16) and
recomputed identically at inference (`generate.prepare_audio` → `log_mel`, float32).

**Load-bearing consequence: any change to the conditioning tensor (more mels, extra
channels, a learned embedding) changes `in_conv`'s input width → the U-Net must be
RETRAINED FROM SCRATCH and the whole corpus RE-PREPROCESSED.** That cost dominates every
option below (the user runs all trains; v8 base-160 is a clean ~60-epoch run).

## 1. The robustness problem against THIS pipeline

Which "same song, different file" variations actually reach the model, and which the
pipeline already absorbs:

- **Loudness / volume — mostly already handled.** `power_to_db(ref=np.max)` divides the
  whole spectrogram by its own peak before the dB+offset/scale, so a file that is 6 dB
  louder lands at ~the same normalised mel. This is a per-file *global* gain invariance,
  so a quieter rip ≈ a louder rip. **Caveat:** `ref=np.max` is sensitive to a single loud
  transient (a clap, a clipped sample) — one spike pins the reference and pushes the rest
  of the song down. A perceptual loudness target (§3) is more stable than peak-referencing.
- **Codec / EQ / sample-rate differences — partially hurt.** MP3 at low bitrate, a YouTube
  rip, or a bright vs. dark master shift mel energy, especially in the highs (we cut at
  fmax=11025, which already discards the most codec-fragile band). Onsets survive
  reasonably; fine high-frequency texture does not. Low stakes for *rhythm* (onsets are
  broadband/low-mid), higher stakes only if we ever lean on timbre.
- **Broadband noise / hiss — hurts onset precision.** Tape hiss or encoder noise raises the
  spectral floor and smears onset edges. This is the variation most likely to interact with
  the two known issues below.
- **Leading silence / padding / alignment — the sharpest hazard.** Different rips have
  different leading silence. The mel is frame-aligned to the signal, and timing comes from a
  *separate* path (`--timing-from` ref BPM+offset, or `timing.py` beat-track). If the rip's
  offset differs from the reference's offset, every onset is shifted relative to the audio
  the model sees → systematic early/late notes. We do not currently trim leading silence or
  align the mel to the detected downbeat.

**Tie to the two known issues:**
- **Novel-song timing ~28% exact** (`timing.py` `librosa.beat.beat_track`). This is a
  *timing-estimation* failure (BPM/offset), **not** a mel-feature failure — a richer audio
  embedding fed to the *denoiser* does **not** fix it. The real lever is the planned timing
  model (`src/timing_model/`, designed in `v8.md`) or always passing `--timing-from`. A
  cleaner *onset* representation (HPSS / drums stem, §3–4) could marginally help the beat
  tracker, but the structural fix is the timing model. **Be honest: audio features are the
  wrong tool for the 28% number.**
- **~16% of straight-song gaps land on the 1/6 grid (onset over-firing).** Here richer
  audio features *can* plausibly help: the over-firing is the onset channel reacting to
  audio that is ambiguous between 1/4 and 1/6, and reverb tails / broadband noise make a
  clean 1/4 hit look like two close onsets. A **percussion-focused** band (drums stem or
  HPSS-percussive, §3–4) sharpens onset edges and suppresses sustained harmonic energy that
  muddies the gap, which is the single most defensible reason to enrich the audio side.

## 2. Pretrained / SOTA options (trade-offs against an 86-fps, frame-aligned, 12 GB target)

Two hard constraints filter everything: (a) the model needs **~86 fps frame-aligned**
features over the whole song; a **single clip embedding** cannot localise a hit and is
near-useless as the *only* conditioning; (b) **12 GB VRAM** must hold the front-end *plus*
the diffusion train/sample. And every one forces a from-scratch retrain (§0).

| Option | Output | fps / alignment | Dim vs 64-mel | VRAM (4070 Ti 12 GB) | Retrain? | Deps / license | Verdict |
|---|---|---|---|---|---|---|---|
| **MERT** (frame-level music SSL) | frame hidden states | ~**75 Hz** (24 kHz/320) → resample to 86 | 768 (×13 layers) | ~330 M model; feasible if features are **pre-extracted to disk** at preprocess, not run live | Yes | `transformers`; **CC-BY-NC 4.0 (non-commercial)** | Strongest music-content rep; non-commercial license is a real flag for a public release |
| **MusicHuBERT / music2vec** | frame hidden states | ~50–75 Hz | 768 | similar to MERT | Yes | HF; check per-model | MERT family; same shape/caveats |
| **EnCodec** (Meta) | discrete codes / pre-quant latents | **75 Hz** (24 kHz) or 150 Hz (48 kHz) → resample | 128-d continuous latent (pre-VQ) | small encoder (~tens of M); cheap | Yes | `encodec`/`transformers`; **MIT** | A *neutral reconstruction* codec, not music-semantic; cheap + permissive, but a learned mel ≈ what we already have. Low marginal value as a *replacement* |
| **DAC** (Descript Audio Codec) | codes / latents | ~86 Hz at 44.1 kHz config | ~1024 | small | Yes | MIT | Like EnCodec; reconstruction-oriented |
| **MusicGen encoder** | uses EnCodec front-end | (see EnCodec) | — | larger | Yes | CC-BY-NC for weights | The generative head is irrelevant to us; only its EnCodec front-end matters → use EnCodec directly |
| **CLAP** (LAION / MS) | **one clip embedding** | **NOT frame-level** | 512 | tiny | Yes | LAION CLAP: permissive | **Not frame-aligned** → cannot localise onsets. Only usable as a *global* style/genre token added to the difficulty ctx vector (like SR), never as the per-frame conditioning. Possible future style-conditioning lever, not a mel replacement |
| **Jukebox** (OpenAI) codes | frame codes | ~345/86/21 Hz (3 levels) | large | **5B model, multi-GB; impractical on 12 GB** | Yes | research-only | Historically strong but far too heavy; rule out |
| **BEATs** (audio SSL, general) | frame patches | ~50 Hz (patch grid) | 768 | ~90 M; feasible pre-extracted | Yes | MIT-ish | General audio (AudioSet), not music-specialised; weaker than MERT for music, lighter than Jukebox |
| **PaSST / AST / PANNs** | clip or coarse frame | coarse / clip | varies | small–mid | Yes | mixed | Tagging models; coarse time res — same localisation problem as CLAP |

**Source separation (rhythm-relevant — the most on-point family):**
- **Demucs (v4 / HT-Demucs)** — SOTA 4-stem (drums/bass/vocals/other). A **drums stem → mel**
  is directly rhythm-relevant: osu! onsets follow percussion. Runs **offline at preprocess**
  (cache the stem mel like any mel; no inference-time cost if we also pre-separate the user's
  song once at generate — adds a one-time per-song separation, ~seconds–minutes on GPU,
  fits 12 GB at chunked inference). MIT license. **This is the strongest separation option.**
- **Spleeter (Deezer)** — 2/4/5-stem, TensorFlow, older/lighter, lower quality than Demucs,
  pulls in a TF dependency we otherwise don't have. MIT. Demucs is the better choice unless
  the TF/torch split matters.
- **Open-Unmix (UMX)** — lighter, PyTorch, MIT; weaker than Demucs but a cheap middle ground.

**Denoisers / enhancers:**
- **DeepFilterNet** (real-time speech denoiser) — tuned for *speech*, can over-suppress
  music; risky for our broadband-music onsets. Low priority.
- **Demucs as a denoiser** — separating and dropping the "other/noise" residual is a
  music-appropriate way to clean, reusing the same model as the stem path.
- General point: a denoiser helps only if noise is the dominant corruption; for the
  "same song, different rip" problem, **loudness normalisation + a percussion band (§3)
  cover most of the benefit at a fraction of the cost.**

Common theme: every pretrained model that is actually frame-level (MERT, EnCodec, BEATs)
emits at 50–75 Hz, so it needs **linear interpolation to 86 fps** to stay frame-aligned —
straightforward (resample along time), but a source of subtle drift if done carelessly.
All add a heavy inference dependency and a multi-GB download, and all force the retrain.

## 3. Cheap front-end fixes — NO new model, NO new dependency download

These touch only `audio.py` / `AudioConfig`. The ones that change the **tensor shape**
(more mels, extra channel) still force a retrain + reprocess; the ones that only change
*values* (loudness norm, per-channel norm) need a reprocess but no architecture change.

1. **Loudness normalisation (EBU R128 / ReplayGain) via `pyloudnorm`.** Normalise each file
   to a fixed integrated LUFS *before* the mel. More robust than `ref=np.max` (perceptual,
   not peak; immune to a single clipped transient). Tiny CPU cost. **Best robustness-per-dollar
   fix for the "different volume" complaint.** Adds `pyloudnorm` (BSD, pure-python, light).
   Changes mel *values* → reprocess + retrain to be consistent train↔infer.
2. **Per-channel (per-mel-band) normalisation.** Replace the global `(dB+40)/40` with a
   per-band mean/std computed over the corpus (or per-file). Stabilises EQ/codec differences
   band-by-band. Values-only change → reprocess + retrain. (Note: also revisit the
   `ref=np.max` choice here — a fixed dB reference + per-band norm is more file-invariant.)
3. **More mels / log-frequency axis.** 64 → 96/128 mels gives finer pitch resolution.
   Marginal for *rhythm* (onsets are broadband), helpful only if we later condition on
   melody/timbre. Changes tensor shape → retrain + reprocess. Modest VRAM bump.
4. **librosa-native HPSS (harmonic-percussive separation).** `librosa.effects.hpss` (or
   `decompose.hpss` on the spectrogram) splits into harmonic + percussive with **zero new
   model / zero download**. A **percussive-band mel** is a cheap, rhythm-aligned onset
   feature — exactly what the 1/6 over-firing wants — and runs in the existing librosa
   dependency at preprocess. Add it as an **extra conditioning channel block** (e.g. a small
   percussive mel concatenated onto the harmonic mel) → tensor shape changes → retrain +
   reprocess, but **no new dependency and trivial compute**. This is the sweet spot: most of
   the "drums stem" benefit at HPSS cost.

Also worth fixing regardless of features (cheap, and already flagged in HANDOFF round-3 as a
parity bug): **leading-silence trim / downbeat alignment** so different rips present the same
audio at frame 0, and an **explicit train↔infer mel-parity check** (float16 cache vs float32
recompute; the `-1.0` vs `0.0` tail-pad skew at `generate.py:122` vs `dataset.py:77`).

## 4. Recommendation ladder (cheapest → biggest)

| Rung | Change | Cost | Expected benefit | Retrain? |
|---|---|---|---|---|
| **(0)** | **Loudness-norm (pyloudnorm) + per-channel mel norm + leading-silence trim + mel-parity test** | ~hours of code; 1 reprocess; 1 retrain | Directly kills the "different volume / different rip" inconsistency; modest but real robustness. Low risk | Yes (values change) — pairs naturally with the next train |
| **(a)** | Add a **librosa-HPSS percussive mel** as an extra conditioning channel block | ~a day; reprocess + retrain; no new dependency | Rhythm-aligned onset signal → best shot at the **1/6 over-firing**; cheap | Yes (shape change) |
| **(b)** | Replace (a)'s HPSS band with a **Demucs drums-stem mel** (pre-separated offline) | + a stem-extraction pass over the corpus (GPU hours) + a Demucs dep + per-song separation at infer | Cleaner drums than HPSS → better onsets, at real preprocess/infer cost | Yes (shape change) |
| **(c)** | **Augment** the mel with a **frozen MERT/EnCodec embedding** interpolated to 86 fps (pre-extracted to disk) | High: feature-extraction pass over corpus, multi-GB model, +768/128 channels, interpolation plumbing, infer dependency, (MERT) non-commercial license | Richest content rep; **uncertain** it beats mel+drums for a *rhythm/placement* task; biggest blast radius | Yes (shape change) |

**Honest read given 12 GB + "the user runs every train":**
- The conditioning is the input width of the net, so **nothing here is free** — every rung
  past pure-value tweaks is a full retrain + reprocess, which is the expensive resource.
- The "different-file" robustness problem the user actually asked about is **mostly a
  loudness/normalisation problem, and `ref=np.max` already half-solves it.** Rung (0) closes
  most of the remaining gap cheaply and should ride along with the *next* planned train
  regardless.
- For map *quality*, the highest-confidence audio lever is a **percussion band** (HPSS first,
  Demucs only if HPSS proves it pays). It targets the one known issue (1/6 over-firing) that
  audio features can actually move.
- **MERT/EnCodec (rung c) is not worth it yet:** large blast radius, MERT's non-commercial
  license is a release hazard, and there is no evidence a music-semantic embedding beats
  mel+drums for *where to place objects in time*. Park it as a later experiment, behind
  per-song aim conditioning + alignment (the already-prioritised v9 work).
- **CLAP / Jukebox: rule out** for per-frame conditioning (CLAP is clip-level → only a future
  global *style* token; Jukebox is too heavy for 12 GB).
- **The 28% novel-song timing is NOT an audio-feature problem** — it is BPM/offset estimation;
  fix it with the timing model or `--timing-from`, not a richer mel.

## 5. References (links)

- **MERT** — Li et al., *MERT: Acoustic Music Understanding Model with Large-Scale
  Self-supervised Training*, ICLR 2024. https://arxiv.org/abs/2306.00107 ·
  https://huggingface.co/m-a-p/MERT-v1-95M (weights **CC-BY-NC 4.0**).
- **music2vec / MusicHuBERT** — m-a-p family, https://huggingface.co/m-a-p.
- **CLAP** — Wu et al., *Large-Scale Contrastive Language-Audio Pretraining*, ICASSP 2023.
  https://arxiv.org/abs/2211.06687 · https://github.com/LAION-AI/CLAP (clip embedding).
- **EnCodec** — Défossez et al., *High Fidelity Neural Audio Compression*, 2022.
  https://arxiv.org/abs/2210.13438 · https://github.com/facebookresearch/encodec (MIT).
- **DAC** — Kumar et al., *High-Fidelity Audio Compression with Improved RVQGAN*, 2023.
  https://github.com/descriptinc/descript-audio-codec.
- **MusicGen** — Copet et al., *Simple and Controllable Music Generation*, 2023.
  https://arxiv.org/abs/2306.05284 (uses EnCodec front-end; weights CC-BY-NC).
- **Jukebox** — Dhariwal et al., 2020. https://arxiv.org/abs/2005.00341 (research-only, ~5B).
- **BEATs** — Chen et al., *Audio Pre-Training with Acoustic Tokenizers*, ICML 2023.
  https://arxiv.org/abs/2212.09058.
- **Demucs v4 / HT-Demucs** — Rouard et al., *Hybrid Transformers for Music Source
  Separation*, 2022. https://github.com/facebookresearch/demucs (MIT).
- **Spleeter** — Hennequin et al., 2020. https://github.com/deezer/spleeter (MIT, TF).
- **Open-Unmix** — Stöter et al., 2019. https://github.com/sigsep/open-unmix-pytorch (MIT).
- **DeepFilterNet** — Schröter et al., 2022. https://github.com/Rikorose/DeepFilterNet.
- **pyloudnorm** (EBU R128) — Steinmetz & Reiss. https://github.com/csteinmetz1/pyloudnorm (BSD).
- **librosa HPSS** — Fitzgerald, *Harmonic/Percussive Separation using Median Filtering*, 2010.
  https://librosa.org/doc/latest/generated/librosa.effects.hpss.html
