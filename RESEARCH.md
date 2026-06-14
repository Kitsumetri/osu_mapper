# Research: osu! patterns, slider shapes, and how to train for them

Notes driven by play-test feedback: the maps are *somewhat* playable and follow
the rhythm, but (a) patterns look like random clusters rather than real osu!
patterns, and (b) early output had only straight 2-point sliders. This doc
summarises the relevant mapping concepts and how each maps to a concrete model
change.

## 0. Implementation status

What of the plan below is already in the codebase vs still proposed:

| Item | Status |
|------|--------|
| Curved Bezier sliders (encode shape into cursor signal, decode body) — §3.B interim | **done** (`signal.py`) |
| Eval metrics (density, stream/jump, spacing, on-grid) — §3/§5 | **done** (`metrics.py`) |
| Scale + capacity: 3004 maps, 97M attention U-Net, EMA, cosine LR — §3.D | **done** |
| Store kiai / timing / difficulty / creator metadata — §7.1 | **done** (manifest) |
| Difficulty defaults (AR8/OD7/HP5/CS4) — §6 | **done** (`generate.py`) |
| Rhythm snapping (bounded 1/4-grid beat-snap) — §3.A / §4 | **done v1** (`postprocess.py`; on-grid 0.70→0.82) |
| **Difficulty conditioning** (SR context vector + classifier-free guidance) — §3.C / §9.1 | **done (code, v3)**; needs trained v3 model |
| **Kiai signal channel** (ch 6) + decode to 1–3 spans — §7.2 / §9.2 | **done (code, v3)** |
| **Hitsound accent channels** (ch 7–9, whistle/finish/clap) — §7.4 / §9.3 | **done (code, v3)** |
| Star-rating bucketing of the corpus (rosu-pp) — §8 | **done** (`difficulty.py`, 31k maps) |
| Per-section / triplet snapping, flow/DS coupling — §3.A | *next* |
| Style / mapper conditioning — §5 | proposed |
| Multi-section BPM timing on output, downbeat tracking — §6 / §7.3 | proposed |

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

### 10.0 v3 status (shipped)
10-channel signal + difficulty conditioning + CFG; heavy model
(`std-v3-heavy2`, base 128, loss 0.0056) generates difficulty-controllable maps
(in-game SR ≈ target ±0.5), kiai, hitsounds, curved sliders. **Decode fixes from
play feedback**: slider ends snapped to the ¼-grid (55%→0% off-grid), bezier
control points simplified via RDP (6–8→≤4), slider time-overlap clamp,
beat-snap, dangling-end trim, `--match-sr` calibration loop.

**Still open (from play feedback) — mostly model/conditioning, not decode:**
| # | feedback | nature | plan |
|---|----------|--------|------|
| 1 | no breaks (too dense) | model leaves no gaps | `[Events]` breaks shipped (§10.1.D-iii) but only mark existing gaps; real fix is density conditioning (§10.1.D-i/ii) |
| 2 | kiai lags the drop ~10–12 s, 1/3 coverage | kiai channel alignment | more data + downbeat-snap kiai edges — v4 |
| 6 | some circle placement odd | pattern quality | flow/DS modelling — v5 §10.2 |
| — | streams slightly low, SR drift at extremes | undertrained tails | more data (v4) + bake SR-offset (§10.1.B) |
| — | hitsounds slightly high (~0.5 vs 0.33) | accent threshold | ✅ DONE — `accent_threshold=0.85` → ~0.33 (§10.1.C) |

### 10.1 v4 — scale + control (next; some need a re-preprocess)

The current full-library preprocess (`std-v3-all`, ~28k maps, curated SR≤12)
feeds this. Batch the representation-changing items into one re-preprocess+retrain.

- **A. Scale (running)** — train on the full library (≈28k vs 6k), base 128,
  **batch 32** (VRAM was only ~half used), more epochs. Biggest single lever;
  expected to lift streams, calibrate SR tails, sharpen patterns. *No repr change.*
- **B. SR-offset bake (cheap, post-eval)** — measure target-vs-achieved over an
  `evaluate.py` sweep; fit a correction into `target_context` so one pass hits the
  target (today `--match-sr` iterates). Revisit `density≈0.8·sr` vs §8.
- **C. Hitsound + accent threshold (cheap, decode)** — ✅ **DONE (2026-06-14)**:
  `decode_signal(accent_threshold=0.85)` brings hitsound usage to ~0.33 (real),
  from ~0.52 at threshold 0. Calibrated by sweeping on real generated output —
  the accent channels saturate near +1, so only a high cut thins them.
- **D. Density / breaks control** — the model fills everything (no gaps → no
  breaks). Options: (i) condition density more strongly / separately so quiet
  target → sparser; (ii) suppress onsets where the mel energy is low
  (intro/break/outro detection); (iii) write explicit `[Events]` break periods
  for gaps >~3.5 s. (i)+(ii) are the real fixes; (iii) is cosmetic. **(iii) DONE
  (2026-06-14)**: `postprocess.compute_breaks` + `write_osu(breaks=…)`. But (iii)
  only marks gaps that already exist — dense songs still produce 0 breaks, so
  (i)/(ii) remain the open root-cause fix.
- **E. Style / mapper conditioning** *(repr: wider ctx)* — append a coarse style
  class (farm/stream/tech/alt, clustered from `metrics.py` pattern stats) or a
  learned `creator` embedding to `c`, with the same CFG. Targets "Sotarks 1-2
  farm" vs "tech". Manifest already stores `creator`.
- **F. Slider-shape + repeats channels** *(repr: +channels)* — move slider shape
  off the shared, noisy cursor channel into **dedicated K-anchor offset channels**
  at the slider head, so curves are a first-class denoised target (RDP becomes a
  clean-up, not a crutch). **Also encode `slides` (repeat count)**: today the
  representation loses it — a reverse slider (`slides≥2`, ~8% of real sliders)
  is encoded as a *slow forward* slider (hold box over the full out-and-back
  duration + forward-only cursor trace), so the decoder always emits `slides=1`
  and we never generate reverse sliders. Add a small per-slider-head repeat
  signal (e.g. a channel whose value encodes slides 1/2/3) and decode it.

*v4 batch*: one re-preprocess adding (E style label + F slider channels), retrain
with the wider context at scale; B/C/D-ii/D-iii are inference-side, ship anytime.

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

## References
- [Mapping techniques (Basics)](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics)
- [Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps)
- [Making good sliders](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Making_good_sliders)
- [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [slider library — what is a beatmap](https://llllllllll.github.io/slider/what-is-a-beatmap.html)
- [Farm map (1-2 jumps, Sotarks)](https://osu.miraheze.org/wiki/Farm_map)
- [Star patterns (forum)](https://osu.ppy.sh/community/forums/topics/292689)
- [Burst vs alt maps (forum)](https://osu.ppy.sh/community/forums/topics/1763755)
