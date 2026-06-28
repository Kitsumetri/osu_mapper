# Signal encoding & decoding: the beatmap ⇄ signal representation

**Purpose:** the data-representation half of the model — how a `.osu` beatmap is encoded into the
multi-channel diffusion target $\mathbf{x}_0$, the per-channel meaning of the 21-channel signal, and
how a sampled signal is decoded back into hit objects (onset peak-picking, slider reconstruction,
timing, beat-snap). | **STATIC** (changes only on a representation/encode-decode change — i.e. a new
channel set or decode rule).

This is the data half of the old `TECH_REPORT.md`; the diffusion/architecture half is in
[diffusion-math.md](diffusion-math.md). Code: `src/data/signal.py` (`encode_beatmap`,
`decode_signal`), `src/config.py` (channel layout), `src/parsing/beatmap.py` (`.osu` parse/write).
Math is GitHub-flavored LaTeX.

---

## 1. The beatmap signal — the diffusion target

The beatmap is encoded as a continuous multi-channel signal $\mathbf{x}_0\in[-1,1]^{C\times T}$ with
$C=21$ channels (v8). Channels 0–9 are the v3 set, 10–16 the v5 slider-shape additions, 17–20 the
v7/v7.5/v8 cue channels. Everything lives on the audio's spectrogram **frame grid** (~86 fps), so
column $j$ of the signal and column $j$ of the mel refer to the same instant — this removes the
audio↔map alignment problem.

| ch | name | meaning |
|----|------|---------|
| 0 | `onset` | impulse at each circle / slider head |
| 1 | `slider_hold` | $+1$ while a slider is held, else $-1$ |
| 2 | `spinner_hold` | $+1$ during a spinner, else $-1$ |
| 3 | `new_combo` | impulse at new-combo objects |
| 4 | `cursor_x` | normalised playfield $x$, interpolated over time |
| 5 | `cursor_y` | normalised playfield $y$, interpolated over time |
| 6 | `kiai_hold` | $+1$ during kiai (chorus) sections, else $-1$ |
| 7–9 | `whistle`/`finish`/`clap` | impulse at objects carrying that hitsound |
| 10–15 | `slider_dx/dy_{1..3}` | $K{=}3$ control-point offsets from the slider head, normalised by $(W,H)$, **held constant over the slider span** (baseline 0) |
| 16 | `slides` | repeat count held over the slider span ($1{\to}{-}1,\,2{\to}{-}\tfrac13,\dots$); recovers reverse sliders |
| 17 | `sv` | slider-velocity timeline, $\log_2(\text{SV})$ scaled, piecewise-constant like green lines (v7) |
| 18 | `curve` | per-slider sagitta (bow) cue, held over the span (v7) |
| 19 | `corner` | per-slider red/angular-corner flag, held over the span (v7.5) |
| 20 | `spacing` | head-to-head distance to the next object, $\text{dist}/300$, held over the gap (v8) |

`src/config.py` exposes `CH_SV`=17, `CH_CURVE`=18, `CH_CORNER`=19, `CH_SPACING`=20. Channel checks
are index-based **and** `generate.load_model` builds from the ckpt's own `sig_channels`, so older
checkpoints (10/17/19/20-ch) still load.

## 2. Encoding the channels

**Onset / new-combo** channels place a Gaussian bump at each object's frame $c = \text{time}/\Delta\tau$:

$$
b(i) = \exp\!\Big(-\tfrac{(i-c)^2}{2\sigma^2}\Big),\quad \sigma = 1.2,\qquad
\mathbf{x}_0[\text{ch}] = 2\,b - 1 .
$$

Using a smooth bump (rather than a one-hot spike) gives the regression target a gradient around each
onset and makes decoding robust to sub-frame jitter.

**Hold** channels (`slider_hold`, `spinner_hold`, `kiai_hold`) are box functions set to $+1$ on
$[\,t_\text{start}, t_\text{end}\,]$ and $-1$ elsewhere.

**Cursor** channels store object positions, normalised by the playfield $(W,H)=(512,384)$,

$$
\tilde{x} = \frac{2x}{W}-1,\qquad \tilde{y} = \frac{2y}{H}-1,
$$

and **linearly interpolated** between objects so the path is smooth. The cursor stores only object
**heads** (and each slider's **end**, for flow continuity).

**Slider-shape (v5)** channels carry the body geometry off the shared cursor path. For each slider,
its control polygon is RDP-simplified to $K{=}3$ anchors; each anchor's head-relative offset
$(\Delta x/W,\,\Delta y/H)$ is **held constant over the slider span** (a valued box, like
`slider_hold`). The `slides` channel holds the repeat count likewise. Holding a constant value over
the span (rather than a single-frame spike) makes these a smooth, denoise-friendly regression
target; decoding reads the **span mean** (robust to noise) — see §4. This replaced v3's
cursor-traced shape, which competed with the global cursor path and decoded to near-straight lines,
and recovers reverse sliders ($\text{slides}\ge 2$) that v3 dropped entirely.

**Cue channels (v7/v7.5/v8)** all use the same dense "held over the span/gap" idiom (chosen because
dense non-negative scalars survive mean-regression where sparse spikes/binaries wash to baseline —
see [versions/v8.md](../versions/v8.md)):
- `sv` — the effective SV-multiplier timeline, encoded as $\log_2(\text{SV})$ scaled+clamped to
  ~$[0.25\times,4\times]\to[-1,1]$, piecewise-constant like green lines.
- `curve` — per-slider intended sagitta (bow), held over the slider span.
- `corner` — per-slider red/angular-corner flag (binary), held over the slider span. A **red point =
  a doubled consecutive control point** in the `.osu`; corners must be detected *before* RDP (which
  collapses duplicates).
- `spacing` — head-to-head distance to the next object, $\text{dist}/\text{SPACING\_PX\_SCALE}$
  (≈300), clipped, held over the inter-object gap. Flip-aug invariant (distance is unchanged by
  mirror), unlike the cursor / slider-anchor channels which the flip augment must negate.

## 3. Decoding: signal → beatmap

The sampled $\hat{\mathbf{x}}_0$ is converted to discrete objects by a deterministic decoder
$\mathcal{D}$ (`decode_signal`).

### 3.1 Onset peak-picking

Object times are the local maxima of the onset channel above a threshold $\theta_o$, separated by at
least $g$ frames:

$$
\mathcal{P} = \big\{\,i\ :\ x_0[i] \ge \theta_o,\ x_0[i]\ge x_0[i\pm1],\ i - i_\text{prev} \ge g \,\big\}.
$$

Each peak's song time is $\hat\tau(i) = i\,\Delta\tau$. The cursor channels at $i$ give the object
position; the hold channels classify circle vs. slider vs. spinner.

### 3.2 Slider reconstruction and duration

**v5 (live decoder, `_slider_from_anchors`):** the $K{=}3$ dedicated anchor channels are read as the
**span mean** over $[i_0,i_1]$, denormalised to head-relative control points
$p_k = (x + \mathrm{d}x_k W,\ y + \mathrm{d}y_k H)$, deduplicated, and emitted as a **Bézier** control
polygon (linear if a single distinct point, or if the anchor polygon is near-collinear). The
`slides` channel's span mean rounds to the repeat count, recovering reverse sliders. The written
**pixel length** is the polyline length $\ell = \sum_k \lVert p_k - p_{k-1}\rVert$. Decoded anchors
are RDP-simplified to remove redundant/clustered points (the "imposter line" fix).

*(Legacy v4 decoder `_slider_path`, kept as a 10-channel fallback: samples the shared cursor path at
up to $K{=}8$ anchors, RDP-simplified to ≤4 — which decodes near-straight because the cursor is just a
head→end interpolation during a slider. The v5 anchor channels exist precisely to fix this.)*

osu! derives a slider's *duration* from its length, slider velocity $v$, and the local beat length
$\beta_\text{ms}$:

$$
\Delta t_\text{slider} = \frac{\ell}{v}\,\beta_\text{ms}\cdot(\text{slides}),
\qquad v = \text{SliderMultiplier}\times 100\times \mathrm{SV}.
$$

To prevent a long slider from overlapping the next object, the writer **clamps** $\ell$ so
$\Delta t_\text{slider}$ fits the gap $\Delta$ to the next object:

$$
\ell \le \frac{0.9\,\Delta\,v}{\beta_\text{ms}\cdot(\text{slides})}.
$$

### 3.3 SV, curve, corner, spacing decode

- **SV** (`decode_sv`) — median-filter the raw channel, quantise to a coarse grid, apply hysteresis
  (only open a new section on a large ΔSV), enforce a min section length (~1–2 s), and hard-cap to
  ~6–8 sections — robustness over fidelity (target ~6–8, between coarse ~4 and real ~10, never
  noise). Each section becomes an inherited (green) timing point. **SV-aware slider snapping is
  required** — slider length depends on the SV at its time, else every slider drifts off the grid.
- **Curve** — the per-slider sagitta cue scales the L↔B decision and the bow displacement, decoupling
  curve-vs-straight intent from the anchor MSE.
- **Corner** — when the corner cue exceeds `CORNER_DECODE_THRESHOLD`, sharp-angle anchors are emitted
  as **red** (doubled control points → corners); gentle anchors stay white/smooth. `write_osu` writes
  `curve_points` verbatim, so doubling them is enough — no writer change.
- **Spacing** (`respace_by_magnitude`, post-step) — rebuild positions by accumulating
  displacement = (model's own direction) × (channel magnitude), re-anchored per new-combo, reflecting
  off the walls. Decode-tunable via `--spacing-scale` (α=0 = raw model positions). **Note: shelved in
  practice — relocated objects play worse in-game; generate with `--spacing-scale 0`** (see
  [versions/v8.md](../versions/v8.md)).

### 3.4 Timing estimation (BPM + offset)

The output `.osu` needs a timing point. From the onset-strength envelope, beat tracking gives beat
times $\{b_m\}$; tempo is the **median inter-beat interval**, octave-folded into a plausible band
(now $[89,205)$ — the old $[125,250)$ doubled genuinely-slow songs, see
[versions/v9.md](../versions/v9.md)):

$$
\text{BPM} = \mathrm{fold}\!\Big(\frac{60000}{\mathrm{median}_m(b_{m+1}-b_m)}\Big),\qquad
\text{offset} = 1000\,b_0 .
$$

For known songs, `--timing-from <ref.osu>` uses the reference map's exact BPM+offset (always exact).
The librosa estimate is only ~28% phase-exact on novel songs → the planned timing model
([versions/v8.md](../versions/v8.md) §10.8 design).

### 3.5 Beat snapping (rhythm post-process)

Onsets are optionally nudged onto the estimated grid. With beat length $\beta_\text{ms}$, offset $o$,
and subdivision interval $\iota = \beta_\text{ms}/d$ (default divisors $(4,8,6)$), the snap target for
time $\tau$ is

$$
g(\tau) = o + \iota\,\Big\lfloor \frac{\tau - o}{\iota} + \tfrac12 \Big\rfloor,
$$

applied **only if** $|g(\tau)-\tau|\le \delta_\text{max}$ (a bound of a few tens of ms). Bounding the
move means a wrong BPM estimate cannot drag the whole map onto a bad grid. The snapped time is stored
as `int(round(grid))` — **the editor's own integer tick** — not `time + round(delta)`, which left a
banker's-rounding half-tie note 1 ms off and editor-unsnapped (the v9 snap bug, see
[versions/v9.md](../versions/v9.md)).

**Slider-end snapping.** Snapping onsets leaves slider *ends* off-grid (their duration comes from
§3.2). Each slider's duration is rounded to the nearest $1/d$-beat multiple $k\,\iota$ ($k\ge 1$,
capped to fit the gap to the next object) and its **length recomputed**
$\ell = (k\iota/\beta_\text{ms})\,v$. This moved slider ends from ~55% off the ¼-grid to ~0%.

## 4. Why this representation (design rationale)

- **Frame grid, not milliseconds** — everything aligned on the audio frame grid avoids drift and
  removes the audio↔map alignment problem.
- **Smooth bumps / held boxes, not spikes** — dense targets give a regression gradient and survive
  mean-regression; sparse binaries wash to base-rate (the corner-cue under-fire lesson,
  [versions/v8.md](../versions/v8.md)).
- **Dedicated slider-anchor channels** — slider shape on the shared cursor path denoised to mush; a
  dedicated valued-box representation made shape a first-class, denoise-friendly target.
- **Append, don't reorder** — every new channel set was appended at the end (indices 0–9 unchanged,
  etc.) so older checkpoints + existing tests stay valid.

See [mapping-patterns.md](mapping-patterns.md) for the osu! domain knowledge behind these choices
(why DS/flow/SV/hitsounds matter), and [diffusion-math.md](diffusion-math.md) for the model.
