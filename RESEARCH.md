# Research: osu! patterns, slider shapes, and how to train for them

**Status: v8 released** (base-160, 21-channel; see `RESULTS.md` for the quality history). This is
the design + roadmap doc — the mapping concepts behind each model change and the per-version drafts
(§10.x). It started from play-test feedback (early maps were rhythm-OK but clustered and
straight-slider-heavy); most of that is now addressed (curved / reverse / red-corner sliders, SV,
jumps, kiai, hitsounds), with per-song jump *extremes* the main open item (→ v9, §10.11).

## 0. Implementation status

The v3-era "what's done" checklist is superseded — see **`RESULTS.md`** for the shipped
feature/version history (through v8) and **§10.x** below for the per-version design drafts. Still
open: per-section flow / distance-snap coupling, style / mapper conditioning, multi-section-BPM
output + downbeat tracking, and **per-song jump conditioning (v9, §10.11)**.

## 1. osu!standard pattern vocabulary

Source: [osu! wiki — Mapping techniques](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics),
[Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps).

- **Streams** — runs of circles, usually ¼-beat apart, at small *consistent*
  spacing. Shapes: straight, curved, zig-zag, "variable" (spacing grows).
- **Jumps** — consecutive objects placed *far* apart (spacing ≫ distance-snap)
  to emphasise the music. Sub-types: spaced streams, "geometry" jumps (squares,
  triangles, stars), back-and-forth, sharp-angle vs wide-angle.
- **Flow** — the cursor-path smoothness between objects. *Flow-aim* = angles
  that continue the motion; *aim/snap* = sharp direction changes. Good maps
  alternate intentionally; random angles feel like clusters (our current issue).
- **Distance snap (DS)** — the editor invariant that *time gap ∝ spatial gap*.
  Most ranked patterns hold DS roughly constant within a section. **This is the
  single biggest thing our model ignores** — it places x/y without coupling
  spacing to the inter-onset time.
- **Tech maps** — dense, irregular slider shapes with rapid SV changes; the
  hardest target, save for later.

### Why our output looks random
The model predicts `cursor_x/y` per frame independently of the *rhythm spacing
rule*. Real mappers derive position from (previous position + flow angle +
DS·beat-gap). With no such structure or conditioning, the network samples
plausible-but-uncorrelated positions → clusters.

## 2. Slider shapes

Source: [osu! file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)),
[slider docs](https://llllllllll.github.io/slider/what-is-a-beatmap.html).

Curve types in the `.osu` hit-object: `B` bezier, `C` catmull, `L` linear,
`P` perfect circle. Encoding: `curveType|x1:y1|x2:y2|...,slides,length`.

- **Linear (L)** — 2 points, straight.
- **Perfect circle (P)** — exactly 3 points, arc → semicircles.
- **Bezier (B)** — N control points; degree = N−1. *Most real sliders.*
  Repeated/coincident anchors create sharp "red-anchor" corners → waves,
  S-curves, blankets, and slider-art are all bezier with many points.
- **Catmull (C)** — legacy, passes through points; rare in modern maps.

**Status (implemented):** the encoder now traces a slider's control points into
the `cursor_x/y` channel across the hold, and the decoder samples that path and
emits a multi-point **Bezier** (`_slider_path` in `signal.py`), falling back to
linear for short/straight sliders. Real-map round-trip Bezier ratio went 5% →
27%. Remaining work is the *exact* control-point representation (§3.B proper) and
`P`/red-anchor shapes.

## 3. Concrete model/representation changes

Ordered by effort-to-impact.

### A. Couple position to rhythm (fixes "random clusters") — highest impact
- Add **distance-snap conditioning**: derive a per-section DS scalar from the
  data and feed it (and/or predict spacing as a target), so spacing tracks the
  beat gap. Cheap version: post-process — quantise onsets to the beat grid and
  re-space positions to enforce roughly constant DS per section while preserving
  the model's angles.
- Train a **flow-aware loss / representation**: instead of absolute x/y, model
  Δposition (velocity) and angle between successive objects, which is what flow
  actually is. This makes "continue the motion vs snap" learnable.
- **Rhythm snapping**: snap generated onsets to ¼/⅛ subdivisions of the
  estimated beat grid (we already estimate BPM in `timing.py`). Even as pure
  post-processing this removes the "loose" feel.

### B. Real slider shapes (fixes single-line sliders) — medium
- **[done — interim]** The decoder samples `cursor_x/y` along the slider's held
  frames and emits them as Bezier control points; the encoder now writes the
  slider's control points into the cursor channel so the shape is learnable.
- **[proposed — proper]** Add dedicated slider-body channels: encode the path as
  a few **relative control-point offsets** (fixed K≈3–4 anchors) + a curve-type
  flag at the slider-head frame, so shape is a first-class target rather than
  riding on the shared cursor channel.

### C. Conditioning for controllable style — medium/large
- Condition the diffusion on **difficulty** (star rating / CS / AR / OD) and a
  coarse **pattern/density token** so the user can ask for "stream-heavy Insane"
  vs "jump map". Star rating + object density are computable from the data. The
  manifest already stores AR/OD/HP/CS/bpm/creator, so the data side is ready.

### D. Scale & capacity — ongoing
- **[done]** Now training on 3004 difficulties (deduped) with a 97M-param
  attention U-Net, EMA, and cosine LR. Further scaling to the full 31k+ library
  is future work.

## 4. Suggested next step (smallest change, biggest felt improvement)

**B-interim** (curved sliders from the cursor signal) is **done**. The remaining
no-retrain win is **A-postprocess**: beat-snap generated onsets to the estimated
grid (and per-section DS normalisation), which directly attacks the measured
rhythm gap (`metrics.py`: ~0.70 on-¼-grid vs ~0.99 for real maps). Watch out for
triplet sections (1/3, 1/6 divisors — ~10%+ of maps). After that, move to
flow/DS-conditioned training (§3.A) and proper control-point slider channels
(§3.B) for the deeper fixes.

## 5. Player/community vocabulary & mapper signature styles

The wiki uses formal terms; players name patterns differently, and *who* mapped
something is itself a strong style signal. For a generator that "feels" like
real osu!, these named styles are effectively the **labels we'd condition on**.

### Named patterns (community usage)
- **1-2 (one-two) jumps** — alternating back-and-forth two-object jumps; the
  archetypal aim-farm pattern. So tied to **Sotarks** that "Sotarks syndrome"
  describes hearing a song only for its 1-2 potential.
  ([farm map](https://osu.miraheze.org/wiki/Farm_map))
- **Geometry jumps** — **squares, triangles, pentagons, stars** (polygon vertex
  order; "star order" cross-traverses the polygon). Note: many players *read*
  these as a sequence of single jumps, not as the shape.
  ([star patterns](https://osu.ppy.sh/community/forums/topics/292689))
- **Streams & friends** — **burst** (2–4 notes), **triple/quad**, full
  **stream** (¼ runs), **cutstream** (stream broken by spacing/NC), **kickstream**,
  **zig-zag / variable-spaced** streams.
- **Stacks / double stacks / overlaps**, **anti-jumps / anti-flow** (spacing
  that fights the natural motion for emphasis), **blankets** (a circle hugged by
  a slider curve), **divebomb / tornado / flower / honeycomb** combos.
- **Burst vs alt maps** — *alt*: flow-aim, slow spaced streams (~120–160 BPM)
  with followpoints, fast small jumps & sliders. *burst*: faster taps, less
  spaced (~160–230 BPM). ([forum](https://osu.ppy.sh/community/forums/topics/1763755))

### Mapper signatures (style = conditioning label)
- **Sotarks** — aim/jump-centric farm, 1-2 patterns, spaced bursts, blankets.
- **Tech mappers (e.g. ProfessionalBox, Yugu)** — dense **cutstreams**,
  abnormal stream shapes, rapid **SV** (slider-velocity) changes, irregular
  slider geometry; canonical tech maps cited by the wiki include *ProfessionalBox
  [Primordial Nucleosynthesis]* and *Yugu — MARENOL [Extra]*.
  ([technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps))

### How this informs training
- **Mapper-/style-conditioning**: the `.osu` `Creator` field gives a free label.
  Condition the diffusion on a **mapper embedding** (or a coarse style class:
  *farm/aim, stream, tech, alt*) so generation can target "Sotarks-like 1-2s" or
  "ProfessionalBox-like tech". Cluster maps by pattern statistics (spacing
  variance, stream %, SV variance) to derive style classes where Creator is too
  sparse.
- **Pattern-aware metrics** to *measure* style. `src/metrics.py` already computes
  density, stream/jump ratios, spacing stats, and on-¼-grid ratio. Still to add:
  count 1-2s (alternating angle ~180°), polygon-jump detection (constant spacing +
  turning angle), and SV-change rate. These double as eval metrics and as targets
  for a pattern-conditioning token.
- **Curriculum**: start by conditioning on the easy axes (density, star rating,
  burst-vs-alt), then add mapper embeddings once data is scaled — signature
  styles need many examples per mapper to learn.

## 6. Timing complexity, difficulty params & game modes (measured on the library)

Measured on a random 1500-`.osu` sample of the local library:

- **Game mode**: ~100% osu!std (1 taiko in the sample). Other modes
  (mania/taiko/catch) have *different* `.osu` semantics — e.g. mania uses `x` as
  a column index, not a playfield coordinate — so they must never be mixed into
  std training. `preprocess.py` already filters `mode == 0`; keep it.
- **Difficulty params (std)**: AR mean 8.05 (p10–p90 5.0–9.6), OD 7.17
  (4.0–9.2), HP 4.66 (3.0–6.0), CS 3.84 (3.0–4.5). Applied as fixed defaults in
  `generate.py` for now (model has no difficulty conditioning).
- **Variable BPM**: **26%** of std maps have >1 distinct BPM. Our single
  `[TimingPoints]` output and single-BPM estimate **desync on ~¼ of songs**.
- **Beat divisor**: 1/4 79%, 1/8 7%, 1/6 5%, 1/2 3%, 1/3 2%, 1/16 2%. So
  **>10% of maps need triplet/sextuplet snapping** — snapping only to 1/4 is
  wrong for those.

### Implications (short-term vs long-term)

**(1) Variable BPM / sub-beat flow** — *hard, long-term.* The frame-grid signal
representation already handles variable tempo for *placement* (everything is
absolute-ms → frame), so the model isn't blocked. What breaks is (a) writing a
single timing point and (b) rhythm-snapping. Long-term: detect multiple tempo
sections (downbeat tracker / tempo over time) and emit multiple uninherited
timing points; snap onsets per-section to the section's grid using the map's
beat divisor. Interim: keep one estimated BPM and accept desync on ~26% of
songs, documented.

**(2) Difficulty / per-song multi-difficulty** — *medium, long-term.* One song
has many difficulties (Normal→Extra) with different AR/OD/HP/CS *and* density.
The model currently collapses these into one unconditioned output. Plan:
(a) preprocess each difficulty with its params as **conditioning inputs**, then
(b) at inference accept a target difficulty (star rating + AR/OD/HP/CS) and
generate to match. If full conditioning proves too costly, fall back to training
separate models per difficulty bucket, or hardcode a single target tier (what we
do now: ~AR8/OD7 Insane-ish). Dataset means above are the sane hardcode.

**(3) Non-std game modes** — *handled.* Filtered out; flagged here so nobody
later assumes `x,y` are playfield coords for mania/taiko/catch.

## 7. Kiai, timing sections & hitsounds (control/effect data we currently drop)

A `.osu` carries timing/effect/sound metadata we don't yet model. Storing it now
(in preprocessing) is cheap and unlocks future features without re-crawling.

### Timing points & BPM sections (`[TimingPoints]`)
`time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects`.
- **Uninherited** points (red) set BPM (`beatLength` ms/beat) and can change
  mid-song → **multiple BPM sections** (26% of our maps). meter = time signature.
- **Inherited** points (green) set **slider velocity** (`beatLength = -100/SV%`)
  and **hitsound volume/sample set** per region — tech maps abuse SV heavily.
- **`effects` bitfield**: bit 0 = **kiai time** (the "chorus"/hype section,
  visual flair + emphasis), bit 3 = omit first barline. Kiai marks the musically
  intense sections mappers map most densely.

### Hitsounds (per hit object + sample-set regions)
- Hit object `hitSound` bitfield: bit0 normal, **bit1 (2) whistle**, **bit2 (4)
  finish**, **bit3 (8) clap**; plus a `hitSample` (`normal:addition:index:vol:file`).
- Sliders also have per-edge sounds (`edgeSounds`/`edgeSets`). Sample set
  (Normal/Soft/Drum) comes from the active inherited timing point.
- These encode *rhythmic accent* — claps/finishes usually land on strong beats,
  so they're a strong supervision signal for "where the emphasis is".

### How to adapt (incremental, none block current work)
1. **Extract & store** — **[done]**: the preprocess manifest records `has_kiai`,
   `n_timing_points`, `bpm`, and difficulty/creator metadata per map (and the
   parser exposes `kiai_spans()` / `TimingPoint.kiai`). No model change yet;
   future-proofs conditioning. (Full per-section SV/kiai timeseries not stored
   yet — only summary flags.)
2. **Kiai channel** (small experiment): add a 7th signal channel `kiai_hold`
   (+1 during kiai) so the model learns to ramp density/spacing in choruses;
   condition or just predict it and use it to modulate decode density.
3. **Multi-section timing on output** (medium): emit several uninherited points
   from a downbeat/tempo-over-time tracker instead of one; add inherited points
   for SV variety (tech feel).
4. **Hitsounds** (later): predict a coarse accent channel (clap/finish/normal)
   and map it to `hitSound` bits; or copy hitsounds from a reference via the
   sample-set regions. Lowest priority — cosmetic-ish but adds polish.

## 8. Reference pattern distributions (evaluation targets)

Computed with `src/corpus_stats.py` over the **entire local library — 31,362
std maps** — bucketed by **star rating** (rosu-pp, `difficulty.py`). Mappers'
difficulty *names* are arbitrary, so SR is the principled axis. Score a generated
map against its SR bucket with `python -m src.metrics --osu gen.osu --ref-stats
artifacts/reference_stats.json` (it computes the map's SR, picks the bucket, and
reports z-score + in-p10–p90 flag). Cells are `mean +/- std`.

| metric | Easy (n=1861) | Normal (n=4387) | Hard (n=5732) | Insane (n=7640) | Expert (n=7181) | Expert+ (n=4561) |
|---|---|---|---|---|---|---|
| `star_rating` | 1.794 +/- 0.158 | 2.336 +/- 0.183 | 3.445 +/- 0.337 | 4.728 +/- 0.354 | 5.848 +/- 0.334 | 8.181 +/- 7.95 |
| `density_per_s` | 1.125 +/- 0.252 | 1.72 +/- 0.352 | 2.66 +/- 0.498 | 3.608 +/- 0.675 | 4.424 +/- 0.897 | 5.764 +/- 3.317 |
| `circle_ratio` | 0.347 +/- 0.104 | 0.393 +/- 0.104 | 0.443 +/- 0.12 | 0.56 +/- 0.13 | 0.612 +/- 0.123 | 0.669 +/- 0.146 |
| `slider_ratio` | 0.64 +/- 0.104 | 0.599 +/- 0.104 | 0.553 +/- 0.12 | 0.438 +/- 0.13 | 0.386 +/- 0.123 | 0.326 +/- 0.143 |
| `bezier_slider_ratio` | 0.202 +/- 0.213 | 0.152 +/- 0.177 | 0.14 +/- 0.172 | 0.145 +/- 0.163 | 0.167 +/- 0.159 | 0.192 +/- 0.17 |
| `new_combo_ratio` | 0.288 +/- 0.064 | 0.237 +/- 0.058 | 0.236 +/- 0.057 | 0.254 +/- 0.106 | 0.257 +/- 0.077 | 0.257 +/- 0.095 |
| `mean_spacing_px` | 145.9 +/- 24.4 | 130.5 +/- 20.3 | 123.3 +/- 20.2 | 140.0 +/- 31.5 | 152.0 +/- 39.1 | 154.6 +/- 54.9 |
| `std_spacing_px` | 62.9 +/- 11.8 | 62.5 +/- 10.8 | 67.7 +/- 11.3 | 79.2 +/- 13.6 | 88.7 +/- 14.5 | 97.0 +/- 20.8 |
| `stream_ratio` | 0.003 +/- 0.022 | 0.009 +/- 0.038 | 0.081 +/- 0.097 | 0.15 +/- 0.13 | 0.212 +/- 0.16 | 0.308 +/- 0.22 |
| `jump_ratio` | 0.195 +/- 0.12 | 0.133 +/- 0.09 | 0.129 +/- 0.08 | 0.239 +/- 0.138 | 0.322 +/- 0.161 | 0.343 +/- 0.196 |
| `on_quarter_grid_ratio` | 0.944 +/- 0.187 | 0.943 +/- 0.187 | 0.938 +/- 0.187 | 0.926 +/- 0.195 | 0.917 +/- 0.213 | 0.851 +/- 0.277 |
| `mean_turn_angle_deg` | 81.8 +/- 11.1 | 75.1 +/- 10.4 | 82.8 +/- 13.8 | 101.8 +/- 18.5 | 103.5 +/- 19.9 | 96.8 +/- 24.3 |
| `reversal_ratio` | 0.074 +/- 0.054 | 0.046 +/- 0.042 | 0.077 +/- 0.068 | 0.188 +/- 0.094 | 0.218 +/- 0.092 | 0.214 +/- 0.111 |
| `sv_changes_per_min` | 0.867 +/- 4.294 | 1.946 +/- 9.202 | 6.704 +/- 18.88 | 14.118 +/- 30.668 | 19.579 +/- 23.369 | 24.651 +/- 39.639 |

(Bucket split: Easy 1861, Normal 4387, Hard 5732, Insane 7640, Expert 7181,
Expert+ 4561 = 31,362. Raw per-bucket mean/std/p10/p90 for every metric is in
`artifacts/reference_stats.json`. The Expert+ `star_rating` std is inflated by a
few extreme-SR outliers / non-ranked joke maps.)

### What this tells us (targets for the generator)

- **Everything scales monotonically with star rating** — density, circle ratio,
  streams, jumps, spacing spread, turn angle, SV changes. This is strong evidence
  that **SR is a good single conditioning axis** (§9.1): the model can interpolate
  difficulty along it.
- **Circle↔slider trade-off**: harder = more circles (Easy 0.35 → Expert+ 0.67),
  sliders inversely.
- **Streams scale hard** (Easy 0.003 → Expert+ 0.31); **jumps dip mid (Hard
  0.13) then climb** to 0.34 at Expert+ — Hard maps lean on sliders/rhythm, top
  diffs on aim.
- **On-¼-grid ~0.85–0.94** everywhere — real maps are tight but not perfect
  (triplets, 1/8, sub-beat). Our bounded beat-snap target is ~0.9, not 1.0.
- **`sv_changes_per_min` 0.9 → 25** — SV variety strongly marks difficulty, and
  our single-timing-point output (0) is the biggest systematic gap.
- **`bezier_slider_ratio` ~0.14–0.20** across buckets — a concrete target for the
  curved-slider work.

## 9. Proposed conditioning & extended outputs (design)

Three asks, one unifying observation: **difficulty is an *input* the model should
be told; kiai and hitsounds are *outputs* the model should generate.** So (A) adds
a context vector to the denoiser, while (B) and (C) add channels to the generated
signal — and a single re-preprocess + retrain delivers all three.

### 9.1 Difficulty conditioning — generate to a target star rating

**Idea.** Tell the denoiser the difficulty of the map it should produce, as a
small **context vector** `c = [SR, AR, OD, HP, CS, log(density)]`. SR is the exact
rosu-pp value (Section: `difficulty.py`), computed once in preprocessing and
stored in the manifest.

**Model.** Embed `c` with an MLP and **add it to the timestep embedding** so every
residual block is modulated by *(diffusion step + difficulty)* — the same FiLM
path the timestep already uses:

$$ \mathbf{e} = \mathrm{MLP}_t(\gamma(t)) + \mathrm{MLP}_c(\mathbf{c}). $$

**Make it actually bite — classifier-free guidance (CFG).** During training,
drop `c` to a learned *null* embedding with probability ~0.15. At inference,
sample with

$$ \hat\epsilon = \epsilon_\theta(\mathbf{x}_t,\mathbf{m},\varnothing) + w\,\big(\epsilon_\theta(\mathbf{x}_t,\mathbf{m},\mathbf{c}) - \epsilon_\theta(\mathbf{x}_t,\mathbf{m},\varnothing)\big),\quad w\approx 2\text{–}4, $$

which pushes the sample toward the requested difficulty. Without CFG a weak scalar
condition is often ignored.

**Inference.** User passes a target SR (and optional AR/OD/HP/CS); the output
`[Difficulty]` is set to match (not the current hardcoded AR8/OD7). Because rosu
gives a cheap SR read-out, we can **verify**: compute the generated map's SR and,
if off, nudge `w` or resample — a closed feedback loop. Feasibility is supported
by §8: every metric varies smoothly/monotonically with SR, so the mapping is
learnable. "Difficulty names" are just SR bands, so this subsumes them.

### 9.2 Kiai zones (1–3 blocks)

**Idea.** Add a 7th signal channel **`kiai_hold`** (+1 during kiai, else −1),
generated jointly with the map. It is naturally **music-aware**: kiai ≈ the
chorus/drop, and the denoiser is already conditioned on the mel, so it can learn
"energy peak → kiai."

**Decode (sparse → clean blocks).** The raw channel is noisy, but kiai is highly
structured (1–3 long spans). Post-process: threshold → contiguous runs → enforce a
minimum length (a few seconds / whole measures) → merge near runs → keep the top-K
(K≤3) by mean activation → snap edges to downbeats → write the timing-point
`effects` kiai bit.

**Controllable / alternative.** The user can override or lock kiai spans. A cheap
alternative is a tiny **mel → kiai segmentation** CNN (binary mask over time)
whose output both sets kiai *and* feeds back as conditioning so the map ramps
density inside kiai.

### 9.3 Hitsounds (rhythmic accents)

**Idea.** osu! hitsounds (`whistle=2, finish=4, clap=8`) mark musical accents
(claps on snares, finishes on cymbals). Add **three impulse channels**
(`whistle`, `finish`, `clap`) shaped like the onset channel (+1 at objects that
carry that addition). Generated jointly; **decode** by reading the three channels
at each onset frame and OR-ing the `hitSound` bitfield. Alignment is free: the
same mel conditioning that drives onsets also exposes the percussion.

**Lower priority bits.** Sample set (Normal/Soft/Drum) is a per-section choice —
default it, or pick per region with a small classifier. **No-retrain interim**:
detect percussive onsets (snare/cymbal bands via the spectrogram) and assign
clap/finish to the nearest object.

### 9.4 Unified plan

- **Channels 6 → 10**: `onset, slider_hold, spinner_hold, new_combo, cursor_x,
  cursor_y, kiai_hold, whistle, finish, clap`.
- **One preprocess** that also stores SR (manifest) and encodes kiai + accent
  channels; **one retrain** with the difficulty context vector + CFG.
- **Decode** gains post-processing for the new sparse channels (kiai blocks,
  accent bits) — no model change beyond the extra channels.
- **Risks / notes**: sparse channels need careful thresholding (false
  positives); more channels slightly enlarge the model; tag the dataset +
  representation with a version (`std-v2-10ch`) so older checkpoints stay
  interpretable; SR conditioning is a *correlation* the model learns, not a
  guarantee — hence the rosu verification loop.

### 9.5 Suggested order

1. **Re-preprocess** with the curve-aware encoder **+ SR in manifest + kiai +
   accent channels** (one pass).
2. **Retrain** with difficulty conditioning (`c`) + CFG. This alone gives:
   curved sliders (from §3.B), difficulty control (9.1), kiai (9.2), hitsounds
   (9.3).
3. Add decode post-processing for kiai/accents; set output difficulty params
   from the request; wire the rosu SR verification loop.

## 10. Roadmap: v3 status → v4 → v5

### 10.0–10.1 v3/v4 — shipped (summary; details in RESULTS.md)

v3 (10-ch signal + difficulty conditioning + CFG) and v4/**v4b** (ranked-only data
+ crop 4096 + attn_levels 3 + flip aug, current release) are done. Decode/post wins
shipped and in the code: slider-end snap + RDP simplify, slider-tail clamp, beat-snap
+ trim, **C** `accent_threshold=0.85` (hitsounds → ~0.33), **D-iii** `[Events]` breaks
(cosmetic), `--match-sr`.

**Still open, referenced below:**
- **B. SR-offset bake** into `target_context` — fit target-vs-achieved from one
  `evaluate.py` sweep so one pass hits the SR (today `--match-sr` iterates).
- **D-i/ii. Density / breaks (model-side)** — condition density harder, or suppress
  onsets in low-mel-energy sections; the `[Events]` writer only marks existing gaps.
- **E. Style / mapper conditioning** *(wider ctx)* — coarse style class or `creator`
  embedding + CFG. **Deferred** from the v5 batch as too speculative (§10.3).
- **F. Slider-shape + `slides` channels** — **implemented as the v5 batch → §10.3.**

### 10.2 v5 — pattern realism & timing fidelity

- **Flow / distance-snap modelling** (§3.A) — the deepest fix for "clusters" and
  odd circle placement (#6): model Δposition + flow angle so spacing tracks the
  beat gap and streams curve/zig-zag like real maps, instead of independent x/y.
  Likely needs an autoregressive or relative-position representation.
- **Multi-section BPM timing** (decode) — tempo-over-time / downbeat tracker →
  multiple uninherited points + per-section snap (fixes the 26% variable-BPM
  desync; also tightens kiai/break edges to downbeats).
- **Kiai/break as a learned segmentation** — a small mel→{kiai,break} head whose
  output both writes timing and feeds back as conditioning.
- **Capacity** — only revisit base ≥160 / latent diffusion if bf16 stability is
  solved (QK-norm wasn't enough at 160; try lower LR/fp32-attention/EMA-warmup).
- **Spinners** — model spinner end position (currently fixed centre).
- **Architecture (2026 survey)** — field moved U-Net→**DiT** (pure transformer,
  adaLN-zero conditioning, RoPE, QK-norm). We already have MHSA + QK-norm; keep
  the U-Net **long skip connections** (ablations show they're the critical part;
  even U-ViT/DiT add them). Full DiT shines at large scale we don't have, and the
  2026 consensus is "data + training > arch tricks" (matches our ranked-data win).
  *Contained upgrade worth trying:* **adaLN-zero conditioning** to replace the
  additive FiLM time+ctx path (`unet.py`); RoPE in attention is a smaller second.
- **Attention compute / FlashAttention** *(perf, conditioned)* — FA3 is Hopper-only
  (our 4070 Ti is Ada sm_89 → never). FA2 *is* Ada-compatible but **absent from the
  Windows torch 2.6 wheel** ("not compiled with flash attention"); SDPA already
  uses the fused **cuDNN/memory-efficient** backend, so we're not on the slow math
  path. Building a `flash-attn` wheel (against the existing torch, no torch upgrade)
  would give only a **single-digit % end-to-end** gain for the current *conv* U-Net
  (attention is a minority of FLOPs, runs only at coarse levels). **Only worth it
  if v5 goes DiT/attention-heavy** (then attention dominates and FA2 matters). Free
  now: pin `sdpa_kernel([CUDNN_ATTENTION, EFFICIENT_ATTENTION])` to force the best
  fused backend.

## 10.3 v5 slider-shape + style — design (ACTIVE BUILD, branch `feat/v5-slider-style`)

The next batch (chosen 2026-06-14; **refined to proven-only**). **Scope: §10.1.F
slider channels only.** Targets the two most visible, highest-confidence
complaints: **sliders are straight lines (60–80%)** and **no reverse sliders**.
Validated direction: Mapperatorinator's `osu_diffusion` makes slider anchors +
repeat counts first-class typed points (see [[reference-mapperatorinator]]).

**§10.1.E style/mapper conditioning is DEFERRED** (not in this batch): KMeans
cluster quality, whether a coarse label captures real "style", and whether
conditioning on it actually helps are all uncertain and indirect — too speculative
to bundle. It stays a separate later experiment; `CONTEXT_DIM` is unchanged here.

### Why the current encoding fails
Slider shape is traced into the shared `cursor_x/cursor_y` channels across the
hold (`signal.encode_beatmap`), so it competes with the global cursor path and
denoises to mush → the decoder's straight-vs-curve test usually picks a line. And
`slides` (repeat count) is dropped entirely: a reverse slider is encoded as a slow
forward slider, so the decoder always emits `slides=1`.

### Slider-shape channels (the F fix) — "hold-box" anchor encoding
Move shape into **dedicated channels that carry the control-point offsets as a
constant value over the slider's hold span** (like `slider_hold`, but valued).
Hold-boxes are smooth and denoise-friendly; decode reads the span mean (robust to
noise). Fixed **K=3 anchor slots** (covers waves/arcs; RDP already simplifies real
sliders to ≤4 control points):

- For anchor *i* ∈ {1..K}: two channels `slider_dx_i`, `slider_dy_i` = the offset
  of control point *i* from the slider head, normalised by playfield size
  (dx/512, dy/384) into ~[-1,1], held constant over the hold span (baseline 0
  elsewhere). Sliders with < K control points repeat the last anchor (decode
  dedupes consecutive identical anchors).
- **`slides` channel**: hold-box valued by repeat count, e.g. `clip((slides-1)/3,
  0,1)*2-1` → 1→-1, 2→-0.33, 3→+0.33; decode rounds the span mean back.

Encode: from each slider's `curve_points` (first K, head-relative) + `slides`.
Decode: at a slider head, read the K anchor channels' hold-span means → control
points = head + offsets; read `slides` → repeat count; `length` = anchor path
length (one traversal); osu! derives duration from length×slides. The cursor
channels keep carrying the *circle/jump* path; sliders no longer pollute them.

Cursor channels also change: stop tracing slider control points into
`cursor_x/cursor_y` (that's the noise source). The cursor keeps only the object
**head** keyframe (+ the slider **end** keyframe for flow continuity); slider body
shape lives entirely in the anchor channels.

### Channel layout: 10 → 17 (new channels APPENDED, indices 0–9 unchanged)
`[0 onset, 1 slider_hold, 2 spinner_hold, 3 new_combo, 4 cursor_x, 5 cursor_y,
6 kiai_hold, 7 whistle, 8 finish, 9 clap, 10 slider_dx1, 11 slider_dy1,
12 slider_dx2, 13 slider_dy2, 14 slider_dx3, 15 slider_dy3, 16 slides]`.
`N_SIGNAL_CHANNELS=17`. Appending (not reordering) keeps the v4/ranked 10-ch
models + existing tests valid; anchors baseline 0, `slides` baseline -1.

### Implementation checklist (hermetic-testable before any retrain)
1. `config.py`: append 7 channels (10→17) + `CH_*` indices.
2. `signal.py`: `encode_beatmap` writes anchor (RDP≤K, head-relative, normalised,
   held over span) + `slides` channels, and only head/end cursor keyframes;
   `decode_signal` reads anchors (span-mean) + `slides` (round) for slider geometry,
   replacing `_slider_path`'s cursor trace.
3. `dataset.py`: **flip aug must negate `slider_dx_i` (h-flip, ch 10/12/14) and
   `slider_dy_i` (v-flip, ch 11/13/15)** — else augmentation corrupts slider
   geometry; `slides` (16) is flip-invariant. Pad anchors to 0, `slides` to -1.
4. `model`/`train`: input channels follow config; **fresh train** (17-ch can't load
   the 10-ch v4 ckpt). `CONTEXT_DIM` unchanged (no style this batch).
5. Tests: round-trip a curved + a **reverse** slider through encode→decode; flip-aug
   negates the new channels. Update `TECH_REPORT.md` channel table.
6. `conditioning.py`: **unchanged** (style deferred).

### Risks / mitigations
- Anchor channels are sparse (active only during sliders) → noisier. Mitigation:
  hold-box (not spikes) + span-mean decode + RDP cleanup retained.
- Reverse-slider length: anchors describe **one** traversal; `length` = that path,
  `slides` = repeats, hold span = full out-and-back; let osu! derive duration.
- 17-ch model is a fresh train (no resume from v4) — gate it on the current
  ranked run's eval first, so we know the baseline it must beat.

### Sequencing
Analysis (this doc) → sync memory/docs → implement code + hermetic tests (safe,
no GPU) → **after the running ranked train finishes + is evaluated** → re-preprocess
`ranked-v5` → fresh train → eval/package `[AI-v5-sliders]`. Inference-side wins
(§10.1.B SR-bake, §10.1.D-ii energy-gated breaks, kiai-snap) ship independently.

## 10.4 v4b play-feedback action items (2026-06-14)

From the in-game v4b comparison (see RESULTS.md). Ranked+context+aug validated for
patterns/kiai/hitsounds; the open items:

- **Rhythm regression (NEW — top decode priority, testable on v4b, no retrain).**
  v4b puts some notes off the ¼ grid (look like 1/6 or 1/8) and leaves occasional
  strange 0.5–2 s pauses; v4 rhythm was tighter. Two hypotheses, not exclusive:
  (a) the snap loosening (45→60 ms / 40→50 %, fb #5) now pulls onsets onto the
  *wrong* ¼ line — **test reverting to 45 ms**; (b) ranked maps use 1/6·1/8
  subdivisions that `snap_to_grid`'s ¼(+⅓) grid can't place, so they land
  off-grid — **add 1/6, 1/8 divisors (per-section)**. Also probe the pauses
  (density / `onset_threshold`). A/B these decode params on the v4b ckpt.
- **Kiai ends 1–3 s early** — extend the decoded kiai end (pad to the next
  downbeat) in `decode_kiai`.
- **No spinners** — `spinner_hold` rarely fires at decode; check the spinner
  encode/threshold, or accept (low priority).
- **Curve sliders still low / streams still weak** — the representation fixes:
  v5 slider channels (§10.3, implemented) for curves+reverse; flow/distance-snap
  (§10.2) for streams. Need the v5 train + likely v6 flow work.

## 10.5 v5 play feedback (2026-06-15) + the gold-data plan

v5 (17-ch) in-game: kiai 9/10; **reverse sliders work**; streams "way better, not
ranked level"; some patterns good, some nasty.

**Fixed this pass (no retrain):**
- **AR** was terrible (`4+0.7·sr` → AR5–7). Now `target_settings` AR `7.75+0.25·sr`
  (AR9 ≈ median player; SR≤3→8–8.5, ≤5→8.5–9, >5→9–10) — also matches real ranked AR.
- **Straight vs curved sliders** — v5 emitted *every* slider as a 3-point bezier
  (a "line built from curve points"). Decode now emits a **linear** slider when the
  anchor polygon is near-collinear (`SLIDER_STRAIGHT_RATIO`); verified ~77% B / 23% L.
- `package_map` was overriding the generated AR/OD with the original's (so all prior
  in-game tests ran at the original's AR) — now keeps the generated settings.

**Still open:**
- **Rhythm** — ✅ **off-¼ notes FIXED (task #8)**: diagnosed that the model places
  notes on the 1/8 & 1/6 grids (1/4-only snap caught 68%; 1/8 caught 100%), so the
  ¼-only snapper dragged them off — `generate` now snaps to `divisors=(4,8,6)` →
  100% on a clean 1/4|1/8|1/6 line (36/33/31 split). **Still open (model-side):** the
  occasional 1–2 s pauses over steady music are a *density* gap, not a snap issue —
  fold into density conditioning / gold-data (§10.1.D), not a decode fix.
- **Hitsounds** below ranked level (≈v4) — model under-places vs ranked; gold-data
  (hitsound≥10% filter) + maybe an accent-density condition.
- **Slider velocity (SV)** — *we currently ignore SV (everything SV=1)*; real maps
  vary it. Two levels: (a) **decode-side per-slider SV** (task #9 / §11 5.1) — emit
  an inherited point per slider so geometry length + intended duration both hold,
  making SV non-trivial for free; (b) **learn SV from data** (encode an SV channel +
  condition) — a v6 representation change. Do (a) first.
- **Patterns** (some nasty) — flow/distance-snap modelling (§10.2, v6).

**Round 2 fixes (2026-06-15, decode-side, no retrain) — `[AI-v5c]`:**
- **Slider "imposter line"** — decoder kept clustered/redundant anchors (a straight
  slider with 2 bunched points near the end). Now RDP-simplifies decoded anchors →
  real lines + genuine curves (~76% L / 24% B). `_slider_from_anchors`.
- **Intro junk** (out-of-bounds note + downbeat note + 8 s silence) — `trim_isolated_ends`
  now drops a leading *cluster* before a big gap, not just a single lead note.
- **Overlapping spinners** — `decode_signal` merges spinners within 800 ms (a split
  spinner showed as two overlapping).
- **Timing** — `generate --timing-from <ref.osu>` uses a known map's exact BPM+offset
  (fixes 198→202 BPM + wrong offset). *Novel songs still use the ~28%-accurate librosa
  estimate → needs a better timing model (e.g. Mapperatorinator's infer-20×-and-average
  "super timing"), a v6+ item.*
- **Still model-side:** kiai is inconsistent per-generation (gold-data `--require-kiai`
  for v6); 0.6–0.8 s density gaps (density conditioning); SV/patterns (v6).

**Gold-data filter (next dataset, user spec).** `preprocess --gold` =
`--ranked-only --require-kiai --single-bpm --min-hitsound-frac 0.1 --min-sr 1
--max-sr 10`. New manifest fields `n_uninherited` (BPM-change detection) +
`hitsound_frac`. Rationale: ranked/loved + kiai + single-BPM (the model can't do
multi-BPM timing yet) + real hitsound density + sane SR removes the weakest training
signal. User has new ranked/loved maps to add → refresh `osu!.db` (open osu!) then
`preprocess --gold` → retrain v6.

## 10.6 v6 batch — design (ACTIVE, branch `feat/v6-sv-adaln`)

One re-preprocess + fresh train. Targets the v5 model-side residue: same-speed sliders,
inconsistent kiai, weak hitsounds, density gaps. Three changes:

### A. Slider velocity (SV) — REVERTED (the decode-side approach was wrong)
First attempt derived SV per-slider from geometry/duration and "sectioned" it — but that gave
~24 SVs / 77 points (user: *"terrible"*). **Why it's wrong:** SV is **not** a per-slider
geometric consequence; it's a **structural/stylistic** choice mappers make in a few **coarse
sections, like kiai** (per user, an experienced mapper): *slow part → low SV; drop → SV≈1 or
above; most of the map ≈1; occasional **fast 1–6 s burst** when the song calls for fast
sliders.* Real maps have **few** SV sections, tied to **song structure**, not per-slider noise.
Reverted to clean SV=1 (`snap_slider_ends`, v5 behaviour).

**The right way (future v6+):** SV must be **learned/structural**, e.g. (a) an **SV channel**
the model learns (it reproduces where real mappers put coarse SV changes — naturally sparse),
or (b) a **structural heuristic** tied to mel energy / kiai (low SV in quiet intros/breakdowns,
≈1 elsewhere, rare fast burst). Either way: **coarse, few sections, structure-aligned.** Analyse
real SV maps for the typical section count/diversity first. Not in this v6 batch.

### B. adaLN-zero conditioning
Replace the additive FiLM (`h += time(t_emb)`) with **DiT adaLN-zero**: each `ResBlock1d`
predicts `(scale, shift, gate)` from the conditioning embedding (`SiLU→Linear`, gate
**zero-init** so blocks start as identity), modulating `h = norm(h)·(1+scale)+shift` and gating
the residual `x + gate·block(...)`. Gives difficulty multiplicative control; the 2026-survey
contained upgrade. Model-only change (hermetic-testable); fresh train (can't load v5 ckpt).

### C. Gold data
`preprocess --gold` (code DONE), 17-ch → `ranked-v6`.

### Order / status
✅ **B (adaLN-zero)** DONE. ❌ **A (SV)** reverted (structural-SV deferred, see above).
✅ Re-preprocessed `ranked-v6` (gold, 17-ch) → trained (adaLN) → eval/packaged `[AI-v6]`. DONE.

## 10.7 v7 batch — "patterns" (ACTIVE, design 2026-06-16)

**v6 play feedback:** (1) rhythm better than v5, still imperfect; (2) hitsounds 5/10
(persistent every version); (3) **patterns now the #1 issue** — beginner-level jumps/
streams; (4) kiai fluctuates run-to-run; (5) too many straight line-sliders (user wants
~50-65% curved).

### Phase 1 analysis — measure before building (`analyze_phase1.py`, real 397 vs v6 sweep)
| measure | real | v6 | |
|---|---|---|---|
| mean / std spacing px | 133.5 / 77.3 | 119.7 / 66.2 | compressed |
| jump_ratio / stream_ratio | 0.205 / 0.149 | 0.119 / 0.101 | ~½–⅔ of real |
| turn angle° / reversal | 88.4 / 0.123 | 85.1 / 0.118 | **≈ real** |
| visible-curve % (sagitta≥10px) | 38.1 | 13.4 | ~⅓ of real |
| median slider sagitta px | 4.8 | **0.0** | collapsed straight |
| SV distinct vals / changes per map | 5 / 10 | 1 / 0 | v6 = none |

**Root cause (key):** patterns (#3) and straight sliders (#5) are the **same bug —
under-dispersion from ε-MSE** (the model regresses spatial outputs toward the mean →
compressed spacing *and* collinear slider anchors). **Flow angles are already ≈ real**, so
**attention is NOT the bottleneck** — objective/representation > attention. Decode can't fix
sliders (the type-B classifier already matches visible curvature; the model just makes few
curves). Real SV is **structured + sparse** (~5 vals / ~10 changes per map, mostly ≤1.2×
with a rare 10× burst) → validates a **learned SV channel**. Curve target set to **38-45%**
visibly-curved (user's choice, just above the corpus ~38% upper bound; v6 is ~13%). Added
`metrics.curved_slider_ratio` (sagitta-based) to track it.

### Phased plan (one variable at a time; P1 gates the rest)
- **P0** ✅ slider-mix decode probe → concluded decode-bound is wrong lever; added curvature metric.
- **P1** ✅ analysis above.
- **P2** ✅ **v-prediction + zero-terminal-SNR** (`--objective v --zero-snr`, §11 5.3) — attacks
  under-dispersion directly + doubles as the base-160 unblock. Done + hermetic-tested +
  GPU-smoked (v-loss O(1), ~100× ε scale → not comparable to v6 0.003; LR/grad-clip may
  want retuning). **NEXT: user trains base-128 v-pred, eval vs Phase-1 baselines.**
- **P3** attention (RoPE / up-path S-5 / one finer level + grad-checkpointing) — *demoted*
  by the flow-angle finding; cheap A/B only. Build flash-attn-2 wheel **only if** this makes
  attention dominate FLOPs (else SDPA's fused backend already suffices, §10.2). fp8/fp4 not
  worth it (weights ≈0.25 GB; activations are the cost → grad-checkpointing instead).
- **P4** representation reprocess (ONE pass): flow/Δpos channels + **learned SV channel** +
  optional curvature cue → `gold-v7` (~18-20 ch); retrain best P2/P3 config.
- **P5** parallel tracks: kiai segmentation head (#4); hitsound musicality (#2); BPM/offset
  (try pretrained beat-trackers before a bespoke net).

### P3 — attention upgrade ✅ CODE DONE (6459ee7); gate on whether it beats P2 in play
Build nothing exotic. RoPE + up-path attention + grad-checkpointing implemented behind flags
(`--rope --up-attn --grad-checkpoint`), backward-compatible. **v7 draft memory (base128, b16,
crop4096, 19-ch):** baseline 5.30 GB · +rope 5.31 · +up_attn 9.83 · **+up_attn+grad_ckpt 5.02**
(grad-checkpointing makes up-attention ~free) · attn4 (full-res O(T²)) OOMs → not viable.
**Recommended config: `--rope --up-attn` (+`--grad-checkpoint` for headroom).** Details:
- **RoPE in `AttnBlock1d`** — attention is currently pure content-based (no positional
  signal beyond conv locality); rotary embeddings inject *relative time* so the model can
  attend "N frames ago" (beats recur at a fixed frame stride, ~26 fr at 200 BPM/86 fps).
  Apply rotary to q,k after QK-norm (rotation preserves unit norm); head_dim is even.
  Use the per-level local frame index. Cost negligible; flag `--rope`, store in ckpt args.
- **Up-path attention (audit S-5)** — attention currently only on down-path + mid; add it
  to the symmetric up blocks for extra refinement during upsampling. Moderate cost.
- **Finer-level attention** — the finest (full-res 4096) level has no attention; adding it
  is O(T²) memory — *only* here is grad-checkpointing/FA relevant. Defer unless RoPE+up-path
  move metrics.
- **Gradient checkpointing (`--grad-checkpoint`)** — wrap down/up res+attn blocks in
  `torch.utils.checkpoint`; the real memory enabler for finer attention / base-160 within
  12 GB (FA2 standalone is redundant — `AttnBlock1d` already uses SDPA's fused flash kernel).
- **Measure:** A/B (P2) vs (P2+RoPE+up-path) on jump/stream/curvature via `analyze_phase1.py`.

### Memory & precision for scaling — the fp8/fp4 question (MEASURED 2026-06-16)
Probed the real training footprint (base 128, batch 16, crop 4096, bf16 autocast, fused AdamW):
| component | size | share of peak |
|---|---|---|
| **activations** (transient, fwd+bwd) | **~4.26 GB** | **~80%** |
| Adam states (m,v fp32) | 0.55 GB | 10% |
| weights (fp32 master) | 0.25 GB | 5% |
| grads (fp32) | 0.25 GB | 5% |
| **peak total** | **5.3 GB** / 12 | |

**Conclusion — weight quantization is the wrong lever here.** fp32→fp8 weights saves 0.18 GB
(3.5% of peak), fp4 saves 0.22 GB — negligible, because weights are 5% of memory; **activations
are 80%**. Implications:
- **base-160 is NOT memory-blocked** (5.3 GB peak leaves ~6.7 GB free) — it's **stability**-blocked
  (bf16 divergence, §7). The real "scaling" fix is **P2 (v-pred/zero-SNR) + per-channel
  standardisation (§11 5.2)**, not quantization.
- To cut the 80% that matters (activations, e.g. when P3 adds full-res O(T²) attention): **gradient
  checkpointing** (lossless), gradient accumulation (`--accum`, already present), or smaller crop —
  never weight dtype.
- **fp8 *training*** on Ada (sm_89): immature (PyTorch fp8 is Hopper-focused, needs amax scaling)
  and it would compound our existing bf16 instability → **not now**. **fp4** is Blackwell-era → N/A.
- Minor real option: **8-bit Adam** (bitsandbytes) trims the 0.55 GB optimizer state to ~0.15 GB,
  but it's Windows-finicky and irrelevant at base 128 (plenty of headroom). Revisit only if a much
  bigger net is memory-bound.
- **dtype policy that helps:** keep bf16 autocast for bulk matmuls but **fp32 for norms / attention
  softmax / the v-target** (stability), which we largely already do via autocast + QK-norm.
- **Inference quantization** (fp8/int8 ckpt) is feasible and mature-ish but inference isn't a
  bottleneck (ckpt ~1 GB, generation fast) → low priority.

### P4 design draft — representation + SV channel (ONE reprocess → `gold-v7`)
Decide the final channel set *after* P2 numbers (P2 may already fix dispersion). Order by
confidence:
- **A. Learned SV channel (the user's insight; highest priority). ✅ CODE DONE (b49709f).**
  Per-frame continuous
  channel = the effective SV-multiplier *timeline* (from green lines + SliderMultiplier),
  piecewise-constant. Encode **log2(SV)** scaled+clamped to ~[0.25×,4×]→[-1,1] (multiplicative;
  the rare 10× burst is clipped so it doesn't compress the useful range). 17→18 ch. **Decode —
  stability over fidelity (user choice 2026-06-16): target ~6-8 sections** (between coarse ~4 and
  real ~10), never noise. Robustness chain: (1) median-filter the raw channel; (2) quantise SV to
  a coarse grid (~0.1); (3) **hysteresis** — only open a new section on ΔSV ≥ ~0.15 (kills tiny
  wobbles); (4) **min section length ~1-2 s**; (5) hard cap (~8 sections) keeping the largest
  changes if over budget. Then **slider duration = pixel_len/(100·SliderMultiplier·SV)·beat** —
  resolves the geometry-vs-duration tension (same anchors, SV sets speed → duration follows).
  Hermetic-testable (SV timeline encode→decode round-trip; a noisy channel must collapse to ≤8
  stable sections). Naturally sparse because trained on real sparse SV.
- **B. Flow/Δposition (conditional on P2).** If v-pred doesn't fully widen spacing, add Δx/Δy
  (velocity) as **auxiliary** prediction targets (extra supervision; decode keeps absolute x/y)
  rather than replacing the representation. +2 ch. Skip if P2 suffices.
- **C. Curvature cue. ✅ CODE DONE (0c78502)** (sliders stayed flat after P2, so confirmed).
  Per-slider intended-
  sagitta scalar (held over the span) so the model signals curve-vs-straight intent decoupled
  from anchor MSE; decode scales L/B + displacement from it. **Target 38-45% visibly-curved**
  (user's choice, just above the corpus ~38% upper bound) → bias this cue / decode threshold
  once curvature is honest. +1 ch.
- Bundle whichever of A/B/C survive → `gold-v7` (18-21 ch); retrain best P2/P3 config.

### P5 design draft — parallel tracks (independent; fit between trains)
- **Stream density (measured 2026-06-18, stream test on "Everything will freeze" [Extra], a
  deathstream).** v7.5 is **stream-shy**: on a 53%-stream / 7.1-dens song it made stream_ratio
  0.17, density 5.06, slider_ratio 0.61 (vs ref 0.535 / 7.1 / 0.30) — i.e. an *average* SR6 map,
  not the song's character. **Decode levers added + tested** (`generate --density`, `--onset-threshold`):
  raising the conditioned density 4.8→7.5 lifted streams **0.17→0.28** *and* fixed the slider bias
  (0.61→0.38 toward real); lower onset threshold added a little more (→0.305) at an on-grid cost.
  So streams are **partly gated by conditioning (fixable now via `--density`)** — but the model
  **caps ~6.2 dens even when asked 7.5**, so the rest needs model-side work: **density conditioning**
  (condition on a per-song density inferred from the audio's onset rate, not the SR default) +
  raising the **onset ceiling** (the same under-firing as the intro-gap/dropped-note #1/#6). Also a
  **circle/slider-balance** nudge (the slider bias crowds out stream circles). Highest-leverage for
  "match stream songs"; queue alongside the onset-decode fix.
- **Extreme per-song STYLE, not just streams (jump test 2026-06-18, "Happppy song" [happy birthday
  to me.], a jump map).** Same shape as the stream test on the other axis: ref jump_ratio 0.387 /
  spacing 167, v7.5 made 0.129 / 116 — it matched aggregate stats (density 4.57≈4.12, slider 0.38≈0.38,
  streams 0.17≈0.15) but **not the song's jump-spam**. General finding: **the model regresses to the
  average map for the conditioned SR**; per-song extremes (deathstream, jump-spam, heavy-curve) are
  all under-produced. **Jumps have no decode lever** (unlike density) — spacing magnitude *is* the
  under-dispersion ceiling (~116px cap) → needs P4-B flow channels / representation work. Cheap
  no-retrain nudge to try: higher CFG `--guidance` (3-4, more committed/extreme outputs).
- **Trailing-outro phantom (bug, v7.5 play feedback).** On "Happppy song" the gen put a circle at
  318.82 s (audio 318.85 s; the ref mapper stopped at 316.9 s, leaving ~2 s outro) → autobot fails
  on the last note. The model over-maps the low-energy outro; `trim_isolated_ends` only trims
  ≥~2.2 s trailing gaps so ~1 s-gap tail notes survive. **Fix (decode, no retrain):** trim trailing
  notes that fall after the last *dense* cluster / in the low-mel-energy tail (mirror of the #1
  intro-empty issue — the model is unreliable in low-energy sections, under-firing in intros and
  over-firing in outros). Cheap P0.
- **Kiai segmentation head (#4 fluctuation).** Kiai is one stochastic channel → borderline
  sections flip per sample (eval saw kiai=0.00 at SR4). Train a small **supervised mel→kiai**
  1D-conv head (BCE vs real kiai labels) and use its *deterministic* output at decode (and/or as
  conditioning), replacing the noisy generated channel. Analyse real kiai vs mel energy first.
- **Hitsound musicality (#2, persistent 5/10).** whistle/finish/clap are per-frame independent
  accents, uncorrelated with percussion. Analyse where each type lands vs beat phase + audio
  onset bands (claps on backbeats, finish on downbeats/cymbal onsets). Start **rule-based** from
  beat phase + per-band onsets (deterministic, no retrain); model-side percussion conditioning
  is the fallback.
- **BPM/offset detection for novel songs (expanded).** The timing problem decomposes into
  **tempo** (BPM; ~74% of gold maps single-BPM, ~26% variable), **offset** (the ms position of the
  beat grid's phase — the hard part: a correct BPM with a wrong offset shifts *every* object;
  osu! wants ~5 ms accuracy), and **downbeat/meter** (measure starts, for kiai/section edges).
  librosa gives tempo+beats but rough phase (~28% exact). **Our unfair advantage: ground truth** —
  the corpus has human-verified BPM+offset in every `.osu` timing point → both a *benchmark* and a
  *training set*. Plan, cheap→involved:
  1. **Benchmark first** (mirrors Phase 1 "measure before building"): score pretrained trackers
     against corpus ground truth — BPM within 0.1, offset within ~10 ms — on osu!'s music
     distribution (anime/electronic/fast; general trackers train on other genres). Candidates:
     **beat_this** (2024 ISMIR, transformer, SOTA, pip+torch), **BeatNet** (CRNN+particle filter,
     joint beat/downbeat/tempo). *madmom is strong but has Python-3.12/Windows install pain — try
     last.* Decisive and cheap; tells us if any tool is good enough as-is.
  2. **Offset refinement** (regardless of tracker): cross-correlate the onset-strength envelope
     with a click train at the estimated BPM; the lag maximising correlation = offset. Plus
     **"super timing"** (§10.2): aggregate estimates over many windows / runs → median for stability.
  3. **Bespoke downbeat net** *only if* pretrained falls short on osu! music: a small TCN/CRNN
     `mel → (beat, downbeat) activation` trained on the ~25 k-song labelled corpus (reuse our mel
     pipeline); extract BPM via tempogram/autocorrelation + offset via the activation phase.
  4. **Novel self-consistent angle:** our model already emits an **onset channel on the audio frame
     grid**, and generated onsets cluster on beats — so we can *fit* a BPM+offset grid to the
     model's own onset output as a post-hoc refinement (no extra net). Worth a quick test.
  Keep `--timing-from <ref.osu>` for known songs (always exact). Independent of the patterns work.

## 10.8 Timing model — BPM + offset for novel songs (design, 2026-06-17)

Replaces the librosa estimate (~28% exact) for songs with no reference `.osu`. `--timing-from`
stays the exact path for known songs. This is the P5 timing track, fully specified here.

### The problem (and what "offset" is)
osu! timing = an **uninherited (red) timing point** `(time, beat_length, meter)`: `beat_length =
60000/BPM`, beats at `time + k·beat_length`, downbeats every `meter` beats. So we need three
things: **tempo** (BPM), **phase/offset** (the ms of the anchor **downbeat** — convention: first
strong downbeat near the song start; needs ~5-10 ms accuracy or every object drifts), and
**downbeat** (which beat is bar-1, for meter + to put the offset on a downbeat). I.e. classic
**beat + downbeat + tempo tracking**. Gold maps are single-BPM (one grid); variable-BPM (~26%)
is a later extension — filter to single-BPM + meter 4 for v1.

### Our unfair advantage: a huge in-distribution labelled set
The corpus (~6k unique audios, ~25k difficulties) ships **human-verified BPM+offset+meter** in
every map's timing points → free, large, and **in osu!'s distribution** (anime/electronic/
J-core, often 180-250 BPM) where general trackers (trained on Ballroom/GTZAN/Hainsworth) may be
out of distribution. Use it as both **benchmark** and **training set**. Split by `audio_id`
(difficulties share audio → leakage otherwise; we already dedup mels per audio_id).

### SOTA survey (2024-25; confirmed via web search)
- **Beat This!** (Foscarin et al., ISMIR 2024) — **transformer, current SOTA, no DBN
  post-processing** (minimal peak-pick); generalises across styles incl. tempo changes. The
  reference arch. Key tricks: mel front-end, **wide-tolerance** framewise targets, heavy
  augmentation (esp. tempo-stretch), shift-tolerant BCE.
- **Beat Transformer** (Zhao et al. 2022) — dilated self-attention + demixed (HPSS/stem)
  spectrograms.
- **BEAST** (2023) — online streaming transformer (beat F1 ~80, downbeat ~47, <50 ms latency).
- **madmom** RNN/TCN + **DBN bar-pointer** — long-standing strong baseline; Python-3.12/Windows
  install pain → benchmark last. **BeatNet** CRNN + particle filter (pip+torch). **librosa**
  (onset env + DP + tempogram) = our current ~28%-exact baseline.

### Recommended architecture
Two viable routes; we have the mel pipeline + RoPE + 1D-conv infra to reuse:
- **SOTA upside — small transformer (Beat This!-style):** conv mel front-end → transformer
  encoder (reuse our **RoPE**) → two linear heads (beat, downbeat) → peak-pick. Generalises best,
  data-hungry → lean on augmentation.
- **Pragmatic — TCN (Davies & Böck 2019):** dilated 1D convs (large receptive field) → beat/
  downbeat heads. Lighter, robust, less data/compute; the recommended **first** bespoke model.
- **Input:** log-mel (reuse `src/data/audio.py`) but at a **finer hop** than the diffusion 86 fps
  (offset needs ~5-10 ms; use ~100 fps / 10 ms hop) + **sub-frame parabolic interpolation** of the
  activation peak for the final offset.
- **Targets:** two framewise activations in [0,1], `beat[t]`/`downbeat[t]`, each a **widened
  window** (Gaussian / ±1-2 frames) around the ground-truth times. **Loss:** framewise BCE
  (positive-weighted/focal — beats are sparse), shift-tolerant.
- **Training:** AdamW + cosine (our infra). **Augmentation is load-bearing** — tempo-stretch
  (transforms BPM *and* labels → BPM-range coverage + invariance), pitch-shift (label-invariant),
  noise, mix. Eval F-measure on val each epoch.

### Inference (osu-specific) — global grid fit
For single-BPM maps the cleanest output is **one (BPM, offset)** that best aligns the grid to the
beat activation: tempogram/autocorrelation → BPM candidates; for each, phase-align by
cross-correlating a click train with the activation; pick the best; take the offset on a
**downbeat** (downbeat head) and refine sub-frame. This beats per-beat peak-picking for the
constant-tempo case and yields a precise pair. **"Super timing":** ensemble over windows / a few
stochastic passes → median, for stability. A DBN bar-pointer (madmom) is the heavier alternative.
A neat **self-consistent fallback:** our diffusion model already emits an onset channel on the
frame grid — fit a grid to *its* onsets when no timing model is available.

### Correctness checking
- **MIR (`mir_eval.beat`):** F-measure (±70 ms), Cemgil, CMLt/CMLc, **AMLt/AMLc** (tolerates
  octave 2×/½× errors → separates "metrically plausible" from phase errors).
- **osu exact-match (the one that matters; we have ground truth):** BPM within **0.1** (count
  octave errors 2×/½×/1.5×/⅔× separately — plausible but wrong for osu); **offset error mod
  beat_length** (the grid repeats, so phase is what matters) bucketed **<5 ms excellent / <10 good
  / <20 playable**; **whole-song grid drift** (max deviation of predicted vs true beats — compounds
  BPM+offset error). Test split by `audio_id`.
- **Practical:** drop the predicted timing into a generated map, eyeball in the osu! editor.

### Plan (CPU/no-GPU first; train only if needed)
1. ✅ **Dataset extractor + eval harness DONE** (CPU, `src/timing_model/`, 1d71d60): `labels.py`
   (osu timing → beat/downbeat/BPM/offset) + `metrics.py` (native F-measure + osu exact-match) +
   6 tests. Separate package from the diffusion sources.
2. **Benchmark pretrained** (Beat This! → BeatNet → librosa; madmom last) on the corpus → know the
   real exact-match on osu! music. Decisive — may make training unnecessary. *(Needs the tracker
   libs + `mir_eval`; light inference. Next step when GPU/libs available.)*
3. **Train bespoke only if pretrained falls short** — TCN first, transformer (+ our RoPE) for
   upside. Plenty of labelled data; augmentation-heavy.

## 10.9 v7.5 — recover jumps + red/white slider points (design, 2026-06-17)

### Why `--up-attn` killed jumps (analysis)
v7-full bundled v-pred + SV + curve + **RoPE + up-path attention**; jumps collapsed
(jump_ratio 0.145→0.048, mean-spacing 129.6→102.8, turn 88→77). Mechanism: **self-attention
is a weighted *average* across time** — it pulls each frame's cursor value toward an attended
mean, i.e. it *smooths* the spatial channels and reduces their variance. Jumps need exactly the
opposite: high-variance, confident position changes. **Up-path** attention sits right before the
output, so its smoothing hits the final positions directly. Under ε/v-MSE (which already rewards
mean-prediction) the extra attention capacity is spent *averaging* → lower loss, less dispersion.
The turn-angle drop (88→77 = more clustered) is the fingerprint. This matches the Phase-1 finding:
flow angles were already ≈ real, so attention was **not** the bottleneck — adding it *actively
hurt*. (The down+mid attention from v5/v6 was fine; only the new up-path layer is the suspect.)

**Will ablating it just revert to v7-vpred?** No. Dropping `--rope --up-attn` returns attention
to the proven v6/vpred config **while keeping the v-pred objective + the SV channel + the curve
channel**. So it's vpred's jumps **plus** v7's working SV + curved sliders — strictly ahead of
vpred (which had no SV and flat sliders), not a revert. The bundle confound means the SV/curve
channels *could* share minor blame; the ablation is the definitive test, but the smoothing
mechanism + metric direction strongly implicate up-attn.

### Red vs white slider points (measured: real ranked, 250 maps)
osu! sliders have **white** control points (smooth bezier) and **red** points (sharp corners).
In the file a **red point = a doubled consecutive control point** (`B|a|b|b|c` → corner at `b`,
splitting the bezier). **Data:** **12.7% of all sliders have ≥1 red point**, and since only ~17%
are bezier, **~75% of bezier sliders are red-point/angular, not smooth** — i.e. most osu "curves"
are *angular*, which our smooth-only decoder cannot produce. Median 1 red point per red-slider
(mean 1.56, max 87 for zig-zags). So red points matter as much as smooth curves for slider style.

### v7.5 design (one retrain)
1. **Drop `--rope --up-attn`** → recover jumps (keep SV + curve channels + v-pred). A/B vs v7-full
   + v7-vpred on jump_ratio/mean-spacing.
2. **Corner cue channel** (+1, 19→20; mirrors the curvature cue): per-slider scalar = "angular"
   (set from real red-point detection). **Encode:** detect doubled control points *before* RDP
   (the current `_rdp` collapses duplicates → destroys corners — must preserve/mark them).
   **Decode:** when the corner cue is high, emit the sharp-angle anchors as **red** (doubled
   control points → corners); gentle anchors stay white/smooth (existing curve-cue bow). `write_osu`
   already writes `curve_points` verbatim, so doubling them is enough — no writer change.
3. Net v7.5 = jumps back + smooth *and* angular sliders. (Curve target 38-45% now splits into
   smooth-curve vs red-angular; revisit the metric to count both.)

## 10.10 Loss function — options & analysis (2026-06-17)

**Framing.** Diffusion trains on **MSE between predicted and true noise** (ε, or v for v-pred).
This isn't an arbitrary "choice" — it's the denoising score-matching objective (the ELBO
reduction), and it's what makes diffusion train stably. So we don't *replace* it wholesale; we
either swap the pointwise distance (L1/Huber) or add **auxiliary** terms. Important: the loss is
on the *noise*, not on an image, and our target is a 19-ch time-series, not pixels — so
image-specific perceptual losses don't transfer directly.

**Per the user's ideas:**
- **LPIPS (perceptual) — doesn't transfer.** LPIPS = distance in a pretrained *image* CNN's
  feature space (VGG on ImageNet). There's no pretrained "beatmap-signal" perceptual net, and our
  target isn't an image. The domain analog would be matching our hand-crafted `metrics.py` stats
  (density/spacing/…), but those are computed on *decoded* objects → non-differentiable → can't
  backprop. Skip, unless we ever train a learned beatmap encoder (overkill).
- **Gram-matrix style loss — literal form doesn't transfer, but the *idea* does.** Gram = feature
  correlations (image texture/style). The useful kernel: match **statistical moments** rather than
  pointwise values. Our version = a **variance/auto-correlation matching** auxiliary on the
  predicted x0: penalise the cursor channels' *variance* being below real (directly = more jumps),
  or match the onset channel's autocorrelation (rhythm regularity). This attacks under-dispersion,
  but it's the highest-risk term (adds a non-standard objective; tune carefully).
- **Weighted MSE — the most promising, and domain-appropriate.** Two distinct flavours:
  (a) **per-channel weighting** — our 19 channels differ in *difficulty*, not scale (true ε is
  unit-Gaussian per channel). The easy piecewise channels (SV, curve, hitsounds) are "solved"
  early while the hard **cursor/anchor** channels stay underfit → the model hedges to the mean →
  under-dispersion. Up-weighting the spatial channels focuses gradient where patterns live.
  (b) **active-frame weighting** — most frames are empty baseline; weighting frames near onsets
  concentrates learning on the notes.

**Other principled options (not mentioned, but better fits):**
- **L1 / Pseudo-Huber on ε/v** — cheapest, lowest-risk swap; some diffusion/consistency works
  report sharper, more robust results than L2. Easy A/B (`--loss {mse,huber}`). **Try first.**
- **Min-SNR-γ timestep weighting** (Hang et al. 2023) — reweight per-timestep loss to balance the
  multi-task denoising; faster convergence + better quality. Established, low-risk.
- **Per-channel input standardisation** (§11 #5.2, already queued) — different from (a): zero-mean/
  unit-var the *input* channels for stability (a base-160 unblock candidate), not loss weighting.

**Honest priority.** The loss is a *secondary* lever for our dispersion problem — the data showed
representation (flow channels) and *avoiding over-smoothing* (drop attention, §10.9) matter more,
and v-pred already moved the objective. But cheap, stable nudges are worth A/B-ing, one at a time
(we value stability — base-160 already diverges): **(1) Pseudo-Huber, (2) min-SNR-γ, (3) per-channel
loss weighting on cursor/anchors.** Skip LPIPS/Gram-literal/adversarial (don't transfer / destabilise).

## 10.11 v8 — P4-B: break the jump under-dispersion ceiling (design, 2026-06-18)

**The gap.** v7.5 is the best model but **regresses to the *average* map for the conditioned
SR**: on jump-spam songs it caps at mean-spacing ~116 px / jump_ratio ~0.13 vs the song's
167 px / 0.39 (Happppy test, §10.7 P5). Spacing is object *position*, untouched by any decode
lever → it is the headline remaining quality gap and needs model/representation work.

### Why absolute x/y under-disperses (mechanism, precise)
The target carries position as **absolute** `cursor_x/y` (ch 4/5, signal.py). Under ε/v-MSE the
model learns `E[x0 | x_t, cond]`; for position the conditional mean ≈ playfield centre (a jump
can go anywhere, so the mean of a broad distribution is the middle). Spacing is the **difference**
of two such mean-regressed positions → it collapses to far below even the real *average*. Two
compounding facts make the spatial channels the worst-hit:
- **2-of-20 channels, high-frequency content.** The piecewise channels (SV, curve, hitsounds,
  holds) are "solved" early and dominate the averaged loss; the cursor channels' jump detail is a
  tiny fraction of the mean MSE → underfit → the model hedges to the mean (this is exactly the
  §10.10(a) per-channel-difficulty argument).
- **Spacing is a 2nd-order statistic of a 1st-order target.** Differences of under-dispersed,
  correlated variables are *even more* under-dispersed. (turn-angle is already ≈ real → the path
  *shape* is fine; only the *scale* = segment length = spacing is compressed.)
- v-pred (P2) closed ~70 % of the mean gap and ~28 % of the jump gap but **not variety**;
  dropping up-attn (v7.5) recovered the rest of v-pred's level — both confirm the ceiling is the
  *representation*, not attention or the objective.

### Which channels collapse, and why — *not all of them* (the three buckets)
Mean-regression pulls **every** channel toward its conditional mean given (audio + difficulty);
whether that hurts depends on what that mean *is* and how decode reads it. The governing rule:
**a channel is endangered in proportion to how much its correct value depends on information NOT
in the audio.** Three buckets:
- **(1) Safe — the mean is the right answer (audio-determined).** `onset`, `slider_hold`,
  `spinner_hold`, hitsound *count*. The mel strongly determines note *times*, so the onset bumps
  stay at musical onsets (they don't wash to the −1 baseline *at notes*). This is why rhythm is the
  model's best dimension (7/10) and slider/circle mix ≈ real. (Hitsound *quantity* is fine→over —
  decode needs `accent_threshold` 0.85 because the channel **over-fires**, ~0.52→0.33; signal.py.
  The 4/10 is placement/musicality + stability, a *different* problem.)
- **(2) Collapse — a free creative choice the audio doesn't pin down (P4-B's family).**
  `cursor_x/y` → centre → spacing collapses (116 vs 167). slider anchors → collinear (median
  sagitta 0.0) → straight (partly rescued by the curve cue → 28%). `corner` → a **rare binary →
  regresses to ~base-rate 0.13 → decode-thresholds to ~2%** — the clearest "washed to baseline"
  case, and exactly *why* the corner cue under-fires. One shared root: a high-variance target the
  audio doesn't determine.
- **(3) Hurt by a *different* mechanism (not magnitude collapse).** `kiai` — structural,
  few-count, boundary-ambiguous → under-confident → borderline sections flip per sample (eval saw
  kiai 0.00 at SR4); fix = the supervised kiai head, not P4-B. `spinner` — under-produced from data
  sparsity, not cancellation.

**Takeaway:** bucket 2 is the single biggest gap and the shared cause of *both* the pattern and the
straight-slider complaints — but it is **not** a whole-signal collapse. Bucket-1 channels working is
exactly why the maps are playable. v8 targets bucket 2; bucket-3 items stay on their own tracks.

### The key correction to the §10.7-B draft (signed Δ does NOT fix it)
The old draft proposed "add Δx/Δy (velocity) auxiliary targets." **Signed Δx/Δy mean-regress to
≈ 0** the same way absolute x/y regress to centre — jump *directions* are ~uniform, so the mean
displacement vector cancels. Predicting Δ instead of x moves the zero, it doesn't remove it.
What survives mean-regression is a **non-negative magnitude**: the spacing *distance* has a
conditional mean ≈ the true typical spacing (a large positive number, no cancellation).

This is the project's own evidence, twice over:
- **Curve cue worked** (13→28 % visible): a non-negative scalar (sagitta) whose mean is a useful
  "typical bow." A spacing-magnitude scalar is the same kind of object.
- **Corner cue under-fired** (~2 % vs 13 %): a *rare binary* whose mean = base-rate, then thresholded
  to ≈ 0. The lesson: encode dense positive scalars, not sparse binaries.

**Reframe:** moving magnitude into its own channel changes the *fallback of mean-regression*
from "centre → ~0 spacing" to "→ the correct average spacing." The floor rises to real-average
(~134 px) for free; per-song extremes (167) still need the model to capture the deviation
around that mean — which the cue + loss up-weighting help but (by the curve-cue precedent)
likely won't fully reach. State this honestly: expect a solid lift toward the average and a
partial recovery of the per-song extreme, not a full fix in one shot.

### Primary design — a spacing-magnitude channel (+1, 20→21 ch; `CH_SPACING`=20)
- **Encode** (signal.py): for each hit object, `s = dist(prev object's *end* pos, this object's
  start pos)` (release→press = what the player's aim actually traverses; slider end keyframe
  already exists). Store `s / SPACING_PX_SCALE` clipped to [0,~1.2], **held over the inter-object
  gap** (dense supervision, like curve/sv), baseline 0. `SPACING_PX_SCALE ≈ 256` (≈ playfield
  half-diagonal; tune so the common 0–300 px range fills [0,~1.2] and the rare 500+ clips).
  *Flip-aug invariant* — distance is unchanged by h/v mirror, so the augment code needs **no**
  negation for it (contrast: signed Δ channels would need the cursor negation — another reason
  to prefer magnitude). Index-appended → old ckpts still load; dataset pad/baseline = 0.
- **Decode** (signal.py `decode_signal`, post-step after objects are built): rebuild positions by
  **accumulating displacement = (model's own direction) × (channel magnitude)**, re-anchored per
  new-combo:
  - first object & every new-combo head → snap to the model's **absolute** `cursor` position
    (keeps global structure, bounds drift);
  - within a combo, `q_{k+1} = q_k + s_k · dir_k`, where `dir_k = normalise(cursor[p_{k+1}] −
    cursor[p_k])` (the angle the model already gets right) and `s_k` = windowed mean of the
    spacing channel at the onset (same read pattern as the curve cue);
  - **reflect at playfield walls** (mirror the offending component) rather than clamp — clamping
    re-compresses spacing, reflection preserves magnitude and mimics real edge-aim. NC re-anchoring
    keeps reflections rare.
  - knobs (no retrain): a `--spacing-scale` blend α between the model's raw positions and the
    reconstructed ones (α=0 = today's behaviour; α=1 = full magnitude), so we can dial intensity
    and A/B without retraining.

### De-risk the decode BEFORE spending a train (cheap, no retrain)
The decode half is testable on the **existing v7.5** output: synthesise a target magnitude per gap
(e.g. from SR/density, or just multiply v7.5's own spacings ×1.4) and run the reconstruction
(direction-preserving rescale + wall reflection + NC re-anchor). If that yields clean, in-bounds
jump maps, the reconstruction math + reflection are validated and the only open question is whether
the *channel* learns honest per-song magnitudes — which is what the train tests. Also A/B the
free nudge: CFG `--guidance 3–4` (more committed/extreme) on the jump song.

### Complementary levers (ride the same train; one variable tracked at a time)
- **Per-channel loss up-weighting** (§10.10(a)) — now co-primary, not a nudge: `--spatial-loss-weight`
  weights cursor/anchors/**spacing**/**corner** above the easy piecewise channels in `_diffusion_loss`
  (was an unweighted mean over all channels), renormalised to mean 1 so the overall scale is unchanged.
  Directly counters the "2-of-21 underfit" mechanism (and corner's under-fire). Small, reversible.
- **Per-channel target standardisation** (§11 5.2) — zero-mean/unit-var the channels so the spatial
  channels aren't drowned by the −1-baseline binaries; also a base-160 stability candidate.
- **Variance-matching auxiliary** (§10.10 Gram-idea) — penalise predicted cursor-channel variance
  below real. Highest risk (non-standard term); hold unless the channel + weighting underperform.
- **distance-snap variant** (refinement, not v8-primary): encode `spacing / beat-gap` (the DS
  multiplier mappers actually use) instead of raw px — a more *stationary* target, but couples to
  timing at decode. Note for v9 if raw magnitude proves noisy.

### v8 scope — cursor only, or anchors + corners too?
All three bucket-2 channels collapse, but they need different amounts of *new* work, and the
v7-full "bundling lost attribution" lesson has a sharper reading: **bundle changes whose metrics
are DISJOINT (attribution survives); isolate changes that move the SAME metric.** v7-full's mistake
was up-attn fighting SV/curve over the *same* spatial-dispersion metric. Here the candidate fixes
touch disjoint metrics and most are **decode-tunable**, so a bundle is recoverable at eval time.
- **Cursor x/y — core, new representation.** Spacing-magnitude channel + decode reconstruction.
  Owns the headline metric (jump_ratio / mean-spacing). Decode-tunable via `--spacing-scale` α
  (α=0 → today's behaviour) → its effect is attributable post-train *without a retrain*.
- **Slider anchors — NO new representation; helped for free.** Curvature *magnitude* is already
  the curve cue (the non-negative-scalar fix, working at 28%); the per-channel **loss up-weighting**
  (a complementary v8 lever) also up-weights the anchor channels → extra push on straightness,
  measured by `curved_slider_ratio` (disjoint from jump_ratio). A richer anchor-*shape* rep
  (waves/blankets geometry) shares metrics with the curve cue and is a harder problem → **defer to
  v9.**
- **Corner — loss up-weight, NOT count re-encode (corrected during implementation 2026-06-19).**
  The drafted "scale by red-point count" would make the under-fire *worse*. Under mean-regression a
  generated slider's corner value ≈ `e·P(angular|context)` where `e` is the encoded value; decode
  fires when that ≥ `CORNER_DECODE_THRESHOLD` (0.25). The binary uses `e=1.0`; count-scaling drops a
  1-red slider (the median) to `e=0.33`, so it clears the threshold *less* often → **fewer** corners,
  not more. The binary's higher margin is exactly why it fires more. So the firing rate is raised the
  same way as spatial dispersion — **up-weight the corner channel in the loss** (sharper fit →
  P(angular|context) less hedged → more high-confidence fires) — plus post-train threshold tuning.
  Encoding stays binary; corner joins the `--spatial-loss-weight` set. Orthogonal metric
  (angular-slider ratio), decode-tunable threshold → attribution intact.

**Decision: v8 = spacing-magnitude channel (cursor) + per-channel loss up-weighting on the under-fit
channels (cursor/anchors/spacing/corner), one reprocess → `ranked-v8` (21-ch), one train.** Anchors
+ corner get fixed by the loss lever (no new/changed encoding); a dedicated slider-shape rep waits
for v9. Attribution holds because the metrics are disjoint *and* the spacing effect is decode-tunable
(set α=0 to isolate the loss-weighting's contribution from the channel's at eval, no retrain).

### Eval & acceptance
`analyze_phase1.py` (real vs v7.5 vs v8): **mean-spacing toward 134, jump_ratio toward 0.20, std
toward 77** without collapsing streams/turn-angle. Then the **Happppy jump-song A/B** (target
0.39 / 167) and the deathstream A/B (guard streams didn't regress). Hermetic test: spacing
encode→decode round-trip (a held magnitude reconstructs the spacing within tolerance; a flat-0
channel reproduces today's positions).

### Cost / sequencing / risk
One reprocess → `ranked-v8` (21-ch) + one ~6 h train (USER). Risks: (1) decode drift/bounds —
mitigated by NC re-anchor + reflection + the de-risk pass above; (2) over-spacing on *average*
songs if the channel over-fires — mitigated by the `--spacing-scale` blend (decode-tunable);
(3) stream interaction (don't inflate 1/4 spacing) — the magnitude is per-gap so streams (small
gaps) keep small spacing by construction. Build order: spacing channel + decode reconstruction
first (the representation fix), then per-channel loss up-weighting on cursor/anchors/spacing/corner
bundled into the same reprocess/train; A/B vs v7.5 on disjoint metrics (jump_ratio, angular-slider
ratio, curved_slider_ratio), isolating the spacing channel via its `--spacing-scale` knob.

### Outcome (base-160 train, 2026-06-20) — partial; the channel regresses to the SR-average
Trained base-160 v8 (val 0.041, clean — RESULTS; + the base-160 stability win, §7). **The core bet
only half held.** `eval_spacing_channel` on a jump song (Happppy, real spacing 173 / jump 0.42): the
spacing channel predicts only ~120–127 px (ratio 1.03–1.04 over the cursor) — it mean-regresses to
`E[spacing | audio, SR]` = the **SR-average**, NOT the per-song extreme. **Why the §10.11 prediction
was wrong:** a magnitude scalar does mean-regress to the *correct* value — but "correct" here is the
SR *average*, and the channel **shares the cursor's audio+SR conditioning**, so it has no extra
information about whether *this* song is jump-heavy. Both channel and cursor → the same SR-average;
moving magnitude into a channel re-encoded the average, it didn't recover the per-song extreme (the
channel is itself under-dispersed: chan_p90 ~200 vs real ~340). The curve cue worked only because
decode *forces* a bow; respace faithfully reproduces the channel's compressed magnitude.
- **What worked:** base-160 stability (headline); no regression (curves 0.369, SV intact); and
  `--spacing-scale >1` as a **manual** global jump dial (Happppy raw 0.116 → 2.5 → 0.297) — useful
  but not automatic (over-spaces calm songs; uniform scaling can't make the bimodal stream+jump
  structure).
- **The real per-song fix (→ v9):** condition on an **audio-inferred aim-intensity / target spacing**
  (compute per-song from onset-energy/spectral-flux, feed like `--density`; the §10.7-P5 stream-
  density idea on the spacing axis). A passive channel can't beat the conditioning it shares — the
  lever must be *new information at the input*, or an objective that samples extremes rather than
  regressing to the mean (the deeper under-dispersion problem persists).

## 10.12 v9 — alignment + postprocess (design, 2026-06-21)

Three workstreams researched in parallel (detailed reports in `docs/v9/`): (1) postprocess
1/4-snap correctness, (2) refresh the stale corpus stats, (3) a "ranked-map" reward + RL/post-train
alignment. Branch `feat/v9-align` off `feat/v8-flow`. The connective theme is **model alignment**:
v8 ships decent quality but mean-regresses to the SR-average; v9 attacks that from both the *input*
(per-song conditioning, already roadmapped) and the *output distribution* (RL toward a reward), and
cleans up the decode/postprocess errors that survive into the editor.

### 10.12.1 Postprocess 1/4-snap — a real BPM-dependent bug, FIXED (`docs/v9/task1_postprocess.md`)
The user's "I have to nudge notes onto 1/4 in the editor" was a genuine defect. `snap_to_grid` stored
`o.time + int(round(grid - o.time))`; on a **half-tie delta** Python's banker's rounding sends
`round(±0.5) → 0`, so the note **doesn't move** and stays 1 ms off the editor's own tick (`round(grid)`),
which the editor then flags **unsnapped**. Measured on real outputs (`artifacts/generated/*.osu`, already
through v8 postprocess) the editor-unsnapped rate was **1.8%–52.9% depending on BPM/offset** (worst:
205 BPM); the fix (store `int(round(grid))` directly — the editor's exact tick) drops it to **0%** on
every sample. This BPM-dependence is why the user saw it only on some songs. **Fixed** (commit `bc3c80a`):
snap stores the rounded grid line; `_clamp` made NaN/inf-safe; new `clamp_objects_to_playfield` guards
*all* heads (circles too, not just slider bodies) called in `generate._one_pass`; +8 hermetic tests.
- **Ruled OUT** (correctly, not bugs): the bounded snap (`max_snap ≈ beat/16`) — 0% of objects are off
  *all* of {1/4,1/8,1/6} by >1 ms in any sample, so the bound is right (kept it; it protects against a
  wrong BPM estimate). And circle OOB via postprocess can't happen (`_reflect` and the blend stay in-range).
- **Open → v9 model/decode (NOT postprocess):** ~16% of gaps on a *straight* song land on the 1/6
  (triplet) grid — that's the model's onset noise being caught by the 1/6 divisor, not a snap bug. Did
  **not** force-snap to 1/4 (would wreck real triplets/streams). Candidate fixes: per-song divisor
  selection or tighter onset precision. Also flagged: stale `artifacts/generated/*.osu` still carry old
  ≤1 ms offsets (regenerate to clear); possible **doubled 240 BPM red lines** on the `audio_*` maps
  (timing — investigate separately).

### 10.12.2 Refresh the corpus stats — parallelized; full run is the user's to fire (`docs/v9/task2_data_stats.md`)
`artifacts/reference_stats.json` (the per-SR-bucket gold distribution behind `score_against_reference`
and the v9 reward) was **n=31362, dated ~Jun 14**, from an older parser + older Songs snapshot → stale.
- **`corpus_stats` parallelized** (commit `113ab26`): was single-threaded; the rosu SR call dominates and
  parallelises cleanly → `--workers` (default `cpu_count-1`; `1` = serial), same idiom as `preprocess.py`.
  Aggregation is order-independent (`_summary` sorts) so parallel ≡ serial — 5 hermetic equivalence tests.
  Full ~95k-file refresh is now cheap.
- **Diagnosis (1k-probe under current code vs old 31k):** most large single-bucket %s are sampling noise,
  but three shifts are **systematic in every bucket** = a real parser change, not library growth:
  `hitsound_ratio` **+14…37%**, `kiai_ratio` **−9…−27%**, `mean_spacing_px` **−2…−7%** (−3…−10 px).
  Matters for the reward: `mean_spacing_px` is weighted 1.5; hitsound/kiai are 0.25 (lower-stakes, and get
  their own v9 heads). **The full refresh must run before the reward is used in anger** — band p10/p90 need
  large n. Old file preserved at `artifacts/reference_stats_v8_old.json`.
  - **USER action:** `uv run python -m src.corpus_stats --songs "C:/osu!/Songs" --out artifacts/reference_stats.json`

### 10.12.3 "Ranked-map" reward + RL alignment — design (`docs/v9/task3_rl_alignment.md`)
A computable reward `R(map, target_SR)` on the machinery we already trust (`compute_metrics` +
per-SR-bucket reference dists + rosu SR), prototyped pure/hermetic at `src/eval/reward.py` (+7 tests).
- **Reward (one line):** `R = 0.65·(weighted-mean band-membership of the map's metrics inside the real
  per-SR p10–p90 bands) + 0.35·(rosu SR-closeness)`. The per-metric sub-score is a **flat-topped tent**:
  `1.0` *anywhere inside* the real band, falloff only *outside*. This is the anti-reward-hacking core —
  **zero gradient for going more extreme**, so the optimum is "land in the ranked distribution", not
  "maximise jumps". Directly encodes the v8 `--spacing-scale` lesson (maximising a spacing metric played
  *worse*). Mapper-weighted metrics (grid-snap 2.0; spacing/density/streams 1.5; flow 1.0; cosmetic 0.25).
  Convex blend (not product) so a momentary SR miss doesn't zero a good map. Reads the stats schema at call
  time → survives the §10.12.2 refresh.
- **RL method survey → cheapest-first plan.** Reward is non-differentiable (rosu + parser/decode), so
  **DRaFT/AlignProp and reward-guided sampling are blocked** (would need a learned differentiable surrogate
  — too many moving parts for 12 GB). Recommended ladder: **best-of-N reward ranking** (no train, immediate,
  also the data source) → **RWR / best-of-N distillation** fine-tune of `best.pt` (short train, standard
  denoising loss on reward-filtered self-gen) → **Diffusion-DPO** (preference pairs from ranked-vs-gen; most
  12 GB-friendly *training* method, no in-loop rollout) → **DDPO/DPOK** (LoRA + few-step rollout) only if
  needed and only with a hardened reward. Every phase gated on **in-game play feedback**, not just metrics.
- **The strategic call — conditioning first, then RL (synergistic, not competing).** Both attack the same
  root (mean-regression / under-produced per-song extremes) from opposite ends. §10.11's own diagnosis says
  the missing lever is *new information at the input* = the planned **v9 per-song aim-intensity conditioning**;
  RL is the *other* lever (an objective that samples extremes). **Conditioning is the higher-leverage *first*
  move**: RL can only amplify a tail the model can already sample, and without a per-song signal RWR/DPO would
  push spacing up **globally** — the exact uniform `--spacing-scale` failure. Conditioning makes "more jumps
  *on the jumpy song*" expressible; RL then makes the model commit to that tail. **Best-of-N can run now**
  (free) to measure how much tail already exists to select from.

### 10.12.4 Sequenced v9 plan
1. **Postprocess snap fix — DONE** (`bc3c80a`), ships immediately (no retrain).
2. **Refresh `reference_stats.json` — DONE** (USER ran the parallel scan): **n=31362 → 94639** (3× the
   library). Confirmed the stale-stats shifts at full scale (every bucket): hitsound_ratio +16…28%,
   kiai_ratio −9…−16%, mean_spacing_px −2…−6%, sv_changes_per_min +12…72%, reversal_ratio −7…−30%. Bands
   now reliable. (Expert+ SR-mean 8.18→7.68 = library composition, not a metric change.)
3. **Best-of-N reward ranking — DONE** (`2263443`, `src/best_of_n.py`, `main.py bestofn`): sample N per
   (song, SR), score with `reward.py`, keep the best; one model/audio load reused across all N+SRs, seeded
   per-candidate. Writes `<out>.bon.json` (full per-candidate breakdown). Validated end-to-end on the v8
   ckpt + refreshed reward (candidates vary, reward discriminates, winner promoted). **Next: USER runs a
   real N (e.g. `--n 8 --sr 5 6 7`) on a jump song and gives play feedback** — if best-of-N satisfies, RL
   may be unnecessary.
   - **Reward caveat found (validation):** `on_quarter_grid_ratio` (and so the reward via it) assumes a
     **single BPM** — it under-measures *variable-BPM* maps (a real ranked accel-map scored 0.28 vs the
     0.78–1.0 band). **Does NOT affect best-of-N** (generated maps are single-BPM by construction; gold
     training data is single-BPM-filtered) — but note it before using the reward on arbitrary real maps.
4. **v9 per-song aim-intensity conditioning** (reprocess + train, USER) — the diagnosed primary fix.
5. **RWR / DPO on the conditioned model** (short/moderate trains, USER) — commit to the per-song tail,
   using best-of-N's reward-ranked self-generations as the corpus. Parallel, no big train: hitsound
   musicality + kiai head (HANDOFF §6).

## 11. Audit follow-ups (external review 2026-06-14)

A separate auditor read every `src/` file + re-derived the diffusion math (all
correct). Defects were at the encode/decode/writer boundary + config hygiene.

**Fixed** (commits `44f8a80`, `6444f3e` on `feat/v5-slider-style`): **C-1** spec-correct
slider `edgeSounds`/`edgeSets`/`hitSample` in `write_osu` (was a malformed, shifted
field that lazer could reject); **C-2** `generate` writes AR/OD/HP/CS from
`conditioning.target_settings(sr)` instead of a hardcoded AR8/OD7 — file difficulty
(and the rosu SR read-back) now match what the model was conditioned on; **C-3** a
slider that lost its curve points is rewritten as a circle (not type&2 with no path);
**S-8** short-song mel pad uses −1.0 (silence) not 0.0 (≈−40 dB); **S-16** `corpus_stats`
parses + mode-filters before the expensive rosu call; **S-6** `AttnBlock1d` asserts
`ch % heads == 0`; **S-9** `item_id` gets a source-path hash (no silent npz overwrite);
**S-3** removed dead `skip_chs`; **S-1** noted `p_sample` reference-only; checkpoints
store `sig_channels`; deleted dead `src/utils/` + unused `h5py`/`pyyaml`/`colorlog`;
synced `requirements.txt`; fixed TECH_REPORT §9 (it listed the *diverged* base-160/0.5
config as "current" → now base-128/0.3) + the v5 decode (§8.2) + README channel count.

**Deferred (tasks created / future work):**
- **Per-slider SV** (5.1) — emit an inherited timing point per slider so geometry
  length *and* model-intended duration are both honoured (solves the SV=1
  shape-vs-rhythm tension; makes `sv_changes_per_min` non-zero). *Task.*
- **Per-channel target standardisation** (5.2) — train on `(x−μ_c)/σ_c` (corpus
  per-channel stats) to zero-centre the −1-baseline channels; a suspected
  contributor to the base-160 bf16 divergence. *Task.*
- **Zero-terminal-SNR β + v-prediction** (5.3) — the principled fix likely needed
  to unblock base ≥160 (higher leverage than just lowering LR). *Future / pairs with 5.2.*
- **Batched CFG** (5.4) — one concatenated forward instead of two → ~2× faster
  sampling, identical output. **Done** (`diffusion.ddim_sample`, 2026-06-20): batch-2 forward,
  second half `ctx_drop=True` (== null embedding, bit-identical; hermetic-tested). **Memory tradeoff:
  the batch-2 forward ~doubles peak activations → OOMs marathon-length songs at base-160 (e.g. the
  8-min ICDD song hit 11.4/12 GB at batch-1), so `generate --no-batch-cfg` keeps the low-memory
  two-forward path.** Plus inference `generate --compile` (opt-in, stacks) + a tqdm progress bar over
  the DDIM steps. (Long-song speedup is still memory-bound — chunked/windowed generation is the real
  lever there; future.)
- **Attention on the up-path** (S-5) / fuse the top skip (S-4) — architecture A/B
  vs the 17/19 metric; needs a retrain.
- Minor (cosmetic/negligible, left as-is): `package_map` re-parse drops `[Events]`
  breaks (S-17); `_validate` last-batch over-weighting (S-14); `snap_slider_ends`
  SV=1 (S-11, no bug for single-timing generated maps; subsumed by 5.1);
  `compute_breaks` ordering (S-13); slider 1-frame demotion (S-7, rare).
- Investigations (need runs/data): base-160 divergence root cause (U-4, grad-norm
  trace); long-song attention-length transfer (U-5); `osu!.db` size-prefix robustness
  (U-1, fine for the current client). Moot after fixes: lazer slider-field tolerance
  (U-2 → C-1), short-song pad frequency (U-6 → S-8).

## References
- [Mapping techniques (Basics)](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics)
- [Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps)
- [Making good sliders](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Making_good_sliders)
- [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [slider library — what is a beatmap](https://llllllllll.github.io/slider/what-is-a-beatmap.html)
- [Farm map (1-2 jumps, Sotarks)](https://osu.miraheze.org/wiki/Farm_map)
- [Star patterns (forum)](https://osu.ppy.sh/community/forums/topics/292689)
- [Burst vs alt maps (forum)](https://osu.ppy.sh/community/forums/topics/1763755)
- Timing model (§10.8): Beat This! (accurate beat tracking w/o DBN, ISMIR 2024 —
  [researchgate](https://www.researchgate.net/publication/382739081_Beat_this_Accurate_beat_tracking_without_DBN_postprocessing))
  · [Beat Transformer (dilated self-attn), 2022](https://arxiv.org/pdf/2209.07140)
  · [BEAST — online streaming transformer, 2023](https://arxiv.org/abs/2312.17156)
  · Davies & Böck, *TCN for beat tracking*, EUSIPCO 2019 · `mir_eval.beat` metrics
- Loss options (§10.10): Hang et al., *Efficient Diffusion Training via Min-SNR Weighting*,
  ICCV 2023 · Pseudo-Huber loss (Song & Dhariwal, *Improved Techniques for Consistency
  Models*, 2023) · Gatys et al., *Neural Style Transfer* (Gram matrices) — image-only, doesn't transfer.
