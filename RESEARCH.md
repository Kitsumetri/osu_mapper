# Research: osu! patterns, slider shapes, and how to train for them

Notes driven by play-test feedback: the maps are *somewhat* playable and follow
the rhythm, but (a) patterns look like random clusters rather than real osu!
patterns, and (b) sliders are only straight 2-point lines. This doc summarises
the relevant mapping concepts and how each maps to a concrete model change.

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

- **Linear (L)** — 2 points, straight (what we generate now).
- **Perfect circle (P)** — exactly 3 points, arc → semicircles.
- **Bezier (B)** — N control points; degree = N−1. *Most real sliders.*
  Repeated/coincident anchors create sharp "red-anchor" corners → waves,
  S-curves, blankets, and slider-art are all bezier with many points.
- **Catmull (C)** — legacy, passes through points; rare in modern maps.

Our decoder reconstructs only the start + end frame → a straight `L|end`. We
throw away the *body shape* entirely because the signal has no channel for it.

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
- Add slider-body channels to the signal: e.g. encode the slider path as a few
  **relative control-point offsets** (a fixed K anchors, K≈3–4) plus a curve-type
  flag, sampled at the slider-head frame. Decode → `B|p1|p2|...`.
- Simpler interim win **without retraining**: in the decoder, sample the
  `cursor_x/y` signal *along the slider's held frames* and emit those as bezier
  control points (`B|...`). The cursor channel already moves during the hold, so
  the body shape is partially there — we currently discard it. This alone yields
  curved/multi-direction sliders.

### C. Conditioning for controllable style — medium/large
- Condition the diffusion on **difficulty** (star rating / CS / AR / OD) and a
  coarse **pattern/density token** so the user can ask for "stream-heavy Insane"
  vs "jump map". Star rating + object density are computable from the data.

### D. Scale & capacity — ongoing
- Train on far more than 601 difficulties (31k+ available); dedupe shared audio
  to cut storage. More data + bigger U-Net should sharpen pattern structure.
- EMA weights + cosine LR for cleaner samples.

## 4. Suggested next step (smallest change, biggest felt improvement)

Implement **B-interim** (decode slider body from the existing cursor signal) and
**A-postprocess** (beat-snap onsets + per-section DS normalisation). Both are
*decoder/post-processing only* — no retraining — and directly target the two
play-test complaints. Then move to flow/DS-conditioned training (A) and
control-point slider channels (B) for the real fix.

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
- **Pattern-aware metrics** to *measure* style: count 1-2s (alternating angle
  ~180°), detect polygon jumps (constant spacing + turning angle), stream ratio
  (¼-spaced runs), SV-change rate. These double as eval metrics and as targets
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
1. **Extract & store now** (this session): per map, save kiai spans, all timing
   points (BPM sections + SV), and difficulty/creator metadata in the preprocess
   manifest. No model change; future-proofs conditioning.
2. **Kiai channel** (small experiment): add a 7th signal channel `kiai_hold`
   (+1 during kiai) so the model learns to ramp density/spacing in choruses;
   condition or just predict it and use it to modulate decode density.
3. **Multi-section timing on output** (medium): emit several uninherited points
   from a downbeat/tempo-over-time tracker instead of one; add inherited points
   for SV variety (tech feel).
4. **Hitsounds** (later): predict a coarse accent channel (clap/finish/normal)
   and map it to `hitSound` bits; or copy hitsounds from a reference via the
   sample-set regions. Lowest priority — cosmetic-ish but adds polish.

## References
- [Mapping techniques (Basics)](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics)
- [Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps)
- [Making good sliders](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Making_good_sliders)
- [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [slider library — what is a beatmap](https://llllllllll.github.io/slider/what-is-a-beatmap.html)
- [Farm map (1-2 jumps, Sotarks)](https://osu.miraheze.org/wiki/Farm_map)
- [Star patterns (forum)](https://osu.ppy.sh/community/forums/topics/292689)
- [Burst vs alt maps (forum)](https://osu.ppy.sh/community/forums/topics/1763755)
