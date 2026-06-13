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

## References
- [Mapping techniques (Basics)](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics)
- [Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps)
- [Making good sliders](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Making_good_sliders)
- [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [slider library — what is a beatmap](https://llllllllll.github.io/slider/what-is-a-beatmap.html)
- [Farm map (1-2 jumps, Sotarks)](https://osu.miraheze.org/wiki/Farm_map)
- [Star patterns (forum)](https://osu.ppy.sh/community/forums/topics/292689)
- [Burst vs alt maps (forum)](https://osu.ppy.sh/community/forums/topics/1763755)
