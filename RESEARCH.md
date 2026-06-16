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

### P3 design draft — attention upgrade (cheap A/B; demoted by the flow-angle finding)
Gate on whether it adds anything *on top of* P2; build nothing exotic.
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
- **A. Learned SV channel (the user's insight; highest priority).** Per-frame continuous
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
- **C. Curvature cue (conditional).** If sliders stay flat after P2, add a per-slider intended-
  sagitta scalar (held over the span) so the model signals curve-vs-straight intent decoupled
  from anchor MSE; decode scales L/B + displacement from it. **Target 38-45% visibly-curved**
  (user's choice, just above the corpus ~38% upper bound) → bias this cue / decode threshold
  once curvature is honest. +1 ch.
- Bundle whichever of A/B/C survive → `gold-v7` (18-21 ch); retrain best P2/P3 config.

### P5 design draft — parallel tracks (independent; fit between trains)
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
  sampling, identical output. *Task.*
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
