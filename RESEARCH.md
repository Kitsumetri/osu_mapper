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
| Per-section / triplet snapping, flow/DS coupling — §3.A | *next* |
| DS / flow-aware position coupling — §3.A | proposed |
| Difficulty / style / mapper conditioning — §3.C / §5 | proposed |
| Kiai signal channel — §7.2 | proposed |
| Multi-section timing on output, downbeat tracking — §6 / §7.3 | proposed |
| Hitsound accent channel — §7.4 | proposed |

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

Computed with `src/corpus_stats.py` over a **12,000-map** random sample of the
local std library (`src/metrics.py` per map), bucketed by object density. These
are the "what real maps look like" targets — score a generated map against the
matching bucket with `python -m src.metrics --osu gen.osu --ref-stats
artifacts/reference_stats.json` (z-score + in-p10–p90 flag). Cells are
`mean +/- std`.

| metric | Easy (n=2238) | Normal (n=2385) | Hard (n=4760) | Insane (n=1858) | Extra (n=759) |
|---|---|---|---|---|---|
| `density_per_s` | 1.459 +/- 0.344 | 2.533 +/- 0.292 | 3.736 +/- 0.421 | 5.074 +/- 0.415 | 7.616 +/- 6.842 |
| `circle_ratio` | 0.376 +/- 0.114 | 0.447 +/- 0.118 | 0.562 +/- 0.124 | 0.644 +/- 0.123 | 0.752 +/- 0.134 |
| `slider_ratio` | 0.615 +/- 0.113 | 0.548 +/- 0.118 | 0.435 +/- 0.124 | 0.354 +/- 0.123 | 0.243 +/- 0.126 |
| `bezier_slider_ratio` | 0.18 +/- 0.198 | 0.14 +/- 0.161 | 0.149 +/- 0.152 | 0.172 +/- 0.158 | 0.165 +/- 0.189 |
| `new_combo_ratio` | 0.261 +/- 0.065 | 0.237 +/- 0.061 | 0.255 +/- 0.095 | 0.256 +/- 0.081 | 0.226 +/- 0.099 |
| `mean_spacing_px` | 139.5 +/- 23.6 | 129.4 +/- 25.2 | 149.0 +/- 36.5 | 146.3 +/- 45.1 | 116.0 +/- 50.9 |
| `std_spacing_px` | 64.3 +/- 12.3 | 69.5 +/- 14.1 | 83.2 +/- 15.6 | 91.5 +/- 17.3 | 87.2 +/- 26.6 |
| `stream_ratio` | 0.012 +/- 0.053 | 0.072 +/- 0.092 | 0.141 +/- 0.111 | 0.268 +/- 0.154 | 0.5 +/- 0.227 |
| `jump_ratio` | 0.172 +/- 0.111 | 0.156 +/- 0.118 | 0.281 +/- 0.167 | 0.304 +/- 0.173 | 0.217 +/- 0.176 |
| `on_quarter_grid_ratio` | 0.931 +/- 0.207 | 0.936 +/- 0.186 | 0.916 +/- 0.213 | 0.907 +/- 0.225 | 0.861 +/- 0.276 |
| `mean_turn_angle_deg` | 79.8 +/- 11.6 | 85.4 +/- 16.9 | 103.6 +/- 18.3 | 98.2 +/- 20.0 | 76.1 +/- 27.3 |
| `reversal_ratio` | 0.066 +/- 0.057 | 0.093 +/- 0.084 | 0.2 +/- 0.103 | 0.204 +/- 0.097 | 0.148 +/- 0.116 |
| `sv_changes_per_min` | 1.99 +/- 10.8 | 6.88 +/- 11.0 | 15.5 +/- 29.3 | 22.2 +/- 31.0 | 24.8 +/- 44.6 |

(Bucket split of the 12k sample: Easy 2238, Normal 2385, Hard 4760, Insane 1858,
Extra 759. Raw per-bucket mean/std/p10/p90 for every metric is in
`artifacts/reference_stats.json`.)

### What this tells us (targets for the generator)

- **Density defines difficulty** — these buckets *are* density bins, but the
  other metrics shift monotonically with them, confirming density is a good
  conditioning axis.
- **Circle↔slider trade-off**: harder maps are more circle-heavy
  (Easy 0.38 → Extra 0.75 circles); slider ratio falls inversely. A generator
  with no difficulty conditioning will sit at one fixed point — another argument
  for difficulty conditioning.
- **Streams scale hard with difficulty** (0.01 → 0.50); **jumps peak at
  Insane** (~0.30) then dip for Extra (which is stream-dominated).
- **On-¼-grid ~0.86–0.94** everywhere — real maps are tight but *not* perfectly
  on-grid (sub-beat detail, triplets, 1/8). Our bounded beat-snap target should
  be ~0.9, not 1.0.
- **`sv_changes_per_min` rises steeply** (2 → 25) — SV variety is a real marker
  of harder/tech maps, and our single-timing-point output (0) is the largest
  systematic gap.
- **`bezier_slider_ratio` ~0.14–0.18** across all buckets — a concrete target
  for the new curved-slider decoder.

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

## References
- [Mapping techniques (Basics)](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics)
- [Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps)
- [Making good sliders](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Making_good_sliders)
- [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [slider library — what is a beatmap](https://llllllllll.github.io/slider/what-is-a-beatmap.html)
- [Farm map (1-2 jumps, Sotarks)](https://osu.miraheze.org/wiki/Farm_map)
- [Star patterns (forum)](https://osu.ppy.sh/community/forums/topics/292689)
- [Burst vs alt maps (forum)](https://osu.ppy.sh/community/forums/topics/1763755)
