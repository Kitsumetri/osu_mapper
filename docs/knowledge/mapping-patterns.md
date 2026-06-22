# osu! mapping patterns & domain knowledge

**Purpose:** the osu!standard domain knowledge the model is trying to reproduce — the pattern
vocabulary (streams/jumps/flow/distance-snap), slider shapes, community/mapper style language, the
timing/difficulty/game-mode facts measured on the library, and the kiai/hitsound control data. This
is the *why* behind the channels and the conditioning. | **STATIC** (osu! domain facts; rarely
changes).

Sources: the [osu! wiki](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics) and the
[.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)). Links collected
in [references.md](references.md). For the *encoding* that realises these, see
[signal-encoding.md](signal-encoding.md); for the proposed conditioning/extended-output **design**, see
§7 below.

---

## 1. osu!standard pattern vocabulary

- **Streams** — runs of circles, usually ¼-beat apart, at small *consistent* spacing. Shapes:
  straight, curved, zig-zag, "variable" (spacing grows).
- **Jumps** — consecutive objects placed *far* apart (spacing ≫ distance-snap) to emphasise the
  music. Sub-types: spaced streams, "geometry" jumps (squares, triangles, stars), back-and-forth,
  sharp-angle vs wide-angle.
- **Flow** — the cursor-path smoothness between objects. *Flow-aim* = angles that continue the motion;
  *aim/snap* = sharp direction changes. Good maps alternate intentionally; random angles feel like
  clusters.
- **Distance snap (DS)** — the editor invariant that *time gap ∝ spatial gap*. Most ranked patterns
  hold DS roughly constant within a section. This is the single biggest thing the model historically
  ignored — it placed x/y without coupling spacing to inter-onset time, hence "random clusters".
- **Tech maps** — dense, irregular slider shapes with rapid SV changes; the hardest target.

**Why early output looked random:** the model predicted `cursor_x/y` per frame independently of the
rhythm spacing rule. Real mappers derive position from (previous position + flow angle + DS·beat-gap);
with no such structure or conditioning, the network sampled plausible-but-uncorrelated positions →
clusters. The fixes (flow/spacing channels, per-song conditioning) are tracked in the version docs.

## 2. Slider shapes

Curve types in the `.osu` hit-object: `B` bezier, `C` catmull, `L` linear, `P` perfect circle.
Encoding: `curveType|x1:y1|x2:y2|...,slides,length`.

- **Linear (L)** — 2 points, straight.
- **Perfect circle (P)** — exactly 3 points, arc → semicircles.
- **Bezier (B)** — N control points; degree = N−1. *Most real sliders.* Repeated/coincident anchors
  create sharp "red-anchor" corners → waves, S-curves, blankets, slider-art are all bezier with many
  points.
- **Catmull (C)** — legacy, passes through points; rare in modern maps.

**Red vs white control points (measured: 250 real ranked maps).** A **red point = a doubled
consecutive control point** (`B|a|b|b|c` → corner at `b`). **12.7% of all sliders have ≥1 red point**;
since only ~17% are bezier, **~75% of bezier sliders are red-point/angular, not smooth** — i.e. most
osu "curves" are *angular*. Median 1 red point per red-slider (mean 1.56, max 87 for zig-zags). So red
points matter as much as smooth curves for slider style — hence the v7.5 `corner` channel.

## 3. Player/community vocabulary & mapper signature styles

The wiki uses formal terms; players name patterns differently, and *who* mapped something is itself a
strong style signal. For a generator that "feels" like real osu!, these named styles are effectively
the **labels we'd condition on**.

### Named patterns (community usage)
- **1-2 (one-two) jumps** — alternating back-and-forth two-object jumps; the archetypal aim-farm
  pattern (tied to **Sotarks**).
- **Geometry jumps** — squares, triangles, pentagons, stars (polygon vertex order; "star order"
  cross-traverses the polygon). Many players *read* these as a sequence of single jumps, not as the
  shape.
- **Streams & friends** — **burst** (2–4 notes), **triple/quad**, full **stream** (¼ runs),
  **cutstream** (broken by spacing/NC), **kickstream**, **zig-zag / variable-spaced** streams.
- **Stacks / double stacks / overlaps**, **anti-jumps / anti-flow** (spacing that fights the natural
  motion for emphasis), **blankets** (a circle hugged by a slider curve), **divebomb / tornado /
  flower / honeycomb** combos.
- **Burst vs alt maps** — *alt*: flow-aim, slow spaced streams (~120–160 BPM) with followpoints, fast
  small jumps & sliders. *burst*: faster taps, less spaced (~160–230 BPM).

### Mapper signatures (style = conditioning label)
- **Sotarks** — aim/jump-centric farm, 1-2 patterns, spaced bursts, blankets.
- **Tech mappers (e.g. ProfessionalBox, Yugu)** — dense cutstreams, abnormal stream shapes, rapid SV
  changes, irregular slider geometry.

### How this informs training
- **Mapper-/style-conditioning**: the `.osu` `Creator` field is a free label; condition on a mapper
  embedding or a coarse style class (*farm/aim, stream, tech, alt*). Cluster maps by pattern
  statistics where Creator is too sparse. **Deferred** — KMeans cluster quality + whether a coarse
  label captures real "style" are uncertain (see [versions/v5.md](../versions/v5.md)).
- **Pattern-aware metrics** to *measure* style (`src/metrics.py`): density, stream/jump ratios,
  spacing stats, on-¼-grid, turn-angle, curvature, SV-change rate. Double as eval metrics and as
  conditioning targets.
- **Curriculum**: condition on the easy axes (density, star rating, burst-vs-alt) first, add mapper
  embeddings once data is scaled.

## 4. Timing complexity, difficulty params & game modes (measured on the library)

Measured on a random 1500-`.osu` sample of the local library:

- **Game mode**: ~100% osu!std. Other modes (mania/taiko/catch) have *different* `.osu` semantics —
  e.g. mania uses `x` as a column index, not a playfield coordinate — so they must **never** be mixed
  into std training. `preprocess.py` filters `mode == 0`; keep it.
- **Difficulty params (std)**: AR mean 8.05 (p10–p90 5.0–9.6), OD 7.17 (4.0–9.2), HP 4.66 (3.0–6.0),
  CS 3.84 (3.0–4.5). These are the sane hardcode/defaults when not conditioning.
- **Variable BPM**: **26%** of std maps have >1 distinct BPM. A single `[TimingPoints]` output +
  single-BPM estimate **desyncs on ~¼ of songs** — gold-data is single-BPM-filtered for this reason.
- **Beat divisor**: 1/4 79%, 1/8 7%, 1/6 5%, 1/2 3%, 1/3 2%, 1/16 2%. So **>10% of maps need
  triplet/sextuplet snapping** — snapping only to 1/4 is wrong for those (decode snaps to (4,8,6)).

## 5. Kiai, timing sections & hitsounds (control/effect data)

A `.osu` carries timing/effect/sound metadata. Storing it in preprocessing is cheap and unlocks
features.

### Timing points (`[TimingPoints]`)
`time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects`.
- **Uninherited** (red) points set BPM (`beatLength` ms/beat) and can change mid-song → multiple BPM
  sections (26%). meter = time signature.
- **Inherited** (green) points set **slider velocity** (`beatLength = -100/SV%`) and hitsound
  volume/sample set per region — tech maps abuse SV heavily.
- **`effects` bitfield**: bit 0 = **kiai time** (the "chorus"/hype section), bit 3 = omit first
  barline. Kiai marks the densest-mapped sections.

### Hitsounds
- Hit object `hitSound` bitfield: bit0 normal, **bit1 (2) whistle**, **bit2 (4) finish**, **bit3 (8)
  clap**; plus a `hitSample` (`normal:addition:index:vol:file`).
- Sliders also have per-edge sounds (`edgeSounds`/`edgeSets`). Sample set (Normal/Soft/Drum) comes
  from the active inherited timing point.
- Hitsounds encode *rhythmic accent* — claps/finishes usually land on strong beats — so they are a
  strong supervision signal for "where the emphasis is".

## 6. Reference pattern distributions (evaluation targets)

The per-SR-bucket gold distribution (the "what ranked maps look like" target behind
`metrics.score_against_reference` and the v9 reward) is documented separately, with the full table and
current `n`, in [corpus-stats.md](corpus-stats.md). The headline finding: **every metric scales
monotonically with star rating** (density, circle ratio, streams, jumps, spacing spread, turn angle,
SV changes), which is strong evidence that **SR is a good single conditioning axis**.

## 7. Proposed conditioning & extended outputs (the original design)

The unifying observation: **difficulty is an *input* the model should be told; kiai and hitsounds are
*outputs* the model should generate.** Difficulty adds a context vector to the denoiser; kiai and
hitsounds add channels to the generated signal. Most of this is now shipped (see the version docs); it
is recorded here as the durable design rationale.

### 7.1 Difficulty conditioning — generate to a target star rating
Tell the denoiser the difficulty as a context vector $c = [\mathrm{SR},\mathrm{AR},\mathrm{OD},
\mathrm{HP},\mathrm{CS},\log(\text{density})]$ (SR = exact rosu-pp value). Embed with an MLP and add it
to the timestep embedding; make it bite with **classifier-free guidance** (drop $c$ to a learned null
~15% of the time; sample with guidance weight $w\approx2$–4). Because rosu gives a cheap SR read-out
of the generated map, requested-vs-achieved SR can be **verified** and $w$ tuned — a closed loop.
*(Shipped v3+; math in [diffusion-math.md](diffusion-math.md) §6.4.)*

### 7.2 Kiai zones (1–3 blocks)
A `kiai_hold` channel (+1 during kiai), generated jointly and naturally music-aware (kiai ≈ the
chorus/drop, and the denoiser sees the mel). Decode: threshold → contiguous runs → enforce min length
→ merge near runs → keep top-K (K≤3) → snap edges to downbeats → write the `effects` kiai bit. A
cheap alternative is a small **mel→kiai segmentation** head whose deterministic output replaces the
noisy generated channel (the planned v9 kiai head). *(Channel shipped v3+; supervised head pending.)*

### 7.3 Hitsounds (rhythmic accents)
Three impulse channels (`whistle`, `finish`, `clap`) shaped like the onset channel (+1 at objects
carrying that addition). Decode by reading the three channels at each onset frame and OR-ing the
`hitSound` bitfield. Persistent weakness (placement/musicality ~4–5/10); the planned fix is
rule-based placement from beat-phase + per-band onsets (claps on backbeats, finishes on
downbeats/cymbals). *(Channels shipped v3+; musicality fix pending — see the roadmap.)*

### 7.4 Channel evolution
The signal grew from 10 (v3) → 17 (v5: +6 slider-anchor dx/dy, +slides) → 19 (v7: +sv, +curve) → 20
(v7.5: +corner) → 21 (v8: +spacing). Each set was **appended** so old checkpoints still load. The
full per-channel spec is in [signal-encoding.md](signal-encoding.md); the per-version rationale is in
[docs/versions/](../versions/README.md).
