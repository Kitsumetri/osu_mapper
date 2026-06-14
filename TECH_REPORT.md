# Technical Report: A Conditional Diffusion Model for osu! Beatmap Generation

This report describes the mathematical model used to train `osu_mapper`: a
**spectrogram-conditioned denoising diffusion probabilistic model (DDPM)** that
generates osu!standard beatmaps from raw audio. It covers the data
representation, the diffusion process and training objective, the sampling
schemes, the denoiser architecture, the optimisation procedure, and the
deterministic decode that turns the model output back into a `.osu` file.

> Math is written in GitHub-flavored LaTeX (`$ … $` inline, `$$ … $$` block).

---

## 1. Problem formulation

A beatmap is a time-ordered set of hit objects placed over a song. We cast
generation as **conditional sampling**: given an audio clip, draw a beatmap from
the conditional distribution

$$
\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{c}),
$$

where $\mathbf{x}\in\mathbb{R}^{C\times T}$ is a *map signal* (Section 3.2) and
$\mathbf{c}\in\mathbb{R}^{F\times T}$ is the audio's log-mel spectrogram
(Section 3.1). Both live on a shared, fixed **time-frame grid**, which removes
the audio↔map alignment problem: column $j$ of $\mathbf{x}$ and column $j$ of
$\mathbf{c}$ refer to the same instant.

We model $p_\theta(\mathbf{x}\mid\mathbf{c})$ with a diffusion model. A discrete
decoder $\mathcal{D}$ (Section 8) then maps a sampled signal to hit objects.

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| $T$ | number of time frames (sequence length) |
| $C=17$ | map-signal channels (v5; 10 in v3) |
| $F=64$ | mel bands |
| $N$ | number of diffusion steps ($N=1000$) |
| $t \in \{1,\dots,N\}$ | diffusion timestep (not song time) |
| $\mathbf{x}_0$ | clean map signal; $\mathbf{x}_t$ noised version |
| $\boldsymbol{\epsilon}$ | standard Gaussian noise |
| $\epsilon_\theta$ | the neural denoiser (a 1D U-Net) |
| $f_s,\ h$ | audio sample rate ($22050$), hop length ($256$) |

The frame rate is $f_s/h = 22050/256 \approx 86.13$ Hz, i.e.
$\Delta\tau = 1000\,h/f_s \approx 11.61$ ms per frame.

---

## 3. Data representation

### 3.1 Audio conditioning — log-mel spectrogram

The mono waveform $y$ (resampled to $f_s$) is turned into a mel spectrogram with
$n_\text{fft}=1024$, hop $h=256$, $F=64$ mel filters over $[20, 11025]$ Hz. With
power spectrogram $S = |\mathrm{STFT}(y)|^2$ and mel filterbank
$\mathbf{W}\in\mathbb{R}^{F\times(n_\text{fft}/2+1)}$,

$$
M = \mathbf{W}\,S \in \mathbb{R}^{F\times T},\qquad
M^{\text{dB}} = 10\log_{10}\!\frac{M}{\max M}.
$$

It is then affinely normalised to roughly $[-1,1]$:

$$
\mathbf{c} = \frac{M^{\text{dB}} + 40}{40}.
$$

### 3.2 Beatmap signal — the diffusion target

The beatmap is encoded as a continuous multi-channel signal
$\mathbf{x}_0\in[-1,1]^{C\times T}$ with $C=17$ channels (v5; channels 0–9 are the
v3 set, 10–16 are the v5 slider-shape additions):

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

**Onset / new-combo** channels place a Gaussian bump at each object's frame
$c = \text{time}/\Delta\tau$:

$$
b(i) = \exp\!\Big(-\tfrac{(i-c)^2}{2\sigma^2}\Big),\quad \sigma = 1.2,\qquad
\mathbf{x}_0[\text{ch}] = 2\,b - 1 .
$$

Using a smooth bump (rather than a one-hot spike) gives the regression target a
gradient around each onset and makes decoding robust to sub-frame jitter.

**Hold** channels are box functions set to $+1$ on $[\,t_\text{start},
t_\text{end}\,]$ and $-1$ elsewhere.

**Cursor** channels store object positions, normalised by the playfield
$(W,H)=(512,384)$,

$$
\tilde{x} = \frac{2x}{W}-1,\qquad \tilde{y} = \frac{2y}{H}-1,
$$

and **linearly interpolated** between objects so the path is smooth. The cursor
stores only object **heads** (and each slider's **end**, for flow continuity).

**Slider-shape (v5)** channels carry the body geometry off the shared cursor
path. For each slider, its control polygon is RDP-simplified to $K{=}3$ anchors;
each anchor's head-relative offset $(\Delta x/W,\,\Delta y/H)$ is **held constant
over the slider span** (a valued box, like `slider_hold`). The `slides` channel
holds the repeat count likewise. Holding a constant value over the span (rather
than a single-frame spike) makes these a smooth, denoise-friendly regression
target; decoding reads the **span mean** (robust to noise) — see Section 8.2.
This replaces v3's cursor-traced shape, which competed with the global cursor
path and decoded to near-straight lines, and recovers reverse sliders ($\text{slides}\ge 2$)
that v3 dropped entirely.

---

## 4. Denoising diffusion (DDPM)

### 4.1 Forward (noising) process

A fixed Markov chain gradually adds Gaussian noise to $\mathbf{x}_0$ over $N$
steps with a **linear variance schedule**
$\beta_t$ from $\beta_1=10^{-4}$ to $\beta_N=2\times10^{-2}$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) =
\mathcal{N}\!\big(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\big).
$$

With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$, the chain
admits a closed form for any $t$ (the "nice property"):

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) =
\mathcal{N}\!\big(\mathbf{x}_t;\ \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I}\big),
$$

so we can sample $\mathbf{x}_t$ directly (this is `q_sample`):

$$
\boxed{\ \mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I}).\ }
$$

### 4.2 Reverse (denoising) process

Generation runs the chain backwards. The reverse transitions are modelled as
Gaussians whose mean is predicted from a noise estimate $\epsilon_\theta$:

$$
p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t, \mathbf{c}) =
\mathcal{N}\!\big(\mathbf{x}_{t-1};\ \boldsymbol{\mu}_\theta(\mathbf{x}_t,\mathbf{c},t),\ \sigma_t^2 \mathbf{I}\big),
$$

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t,\mathbf{c},t) =
\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(\mathbf{x}_t,\mathbf{c},t)\Big),
$$

with the posterior variance (the closed-form variance of
$q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$):

$$
\sigma_t^2 = \tilde\beta_t = \beta_t\,\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.
$$

### 4.3 Training objective

The full variational bound reduces (Ho et al., 2020) to a simple **denoising
score-matching** loss: predict the noise that was added. We use its conditional
form,

$$
\boxed{\ \mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0,\mathbf{c},\,t\sim\mathcal{U}\{1,N\},\,\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})}
\Big[\big\lVert \boldsymbol{\epsilon} - \epsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}}_{\mathbf{x}_t},\ \mathbf{c},\ t)\big\rVert_2^2\Big].\ }
$$

Each training step: sample a crop $\mathbf{x}_0$ and its aligned $\mathbf{c}$,
draw $t$ uniformly and $\boldsymbol{\epsilon}$, form $\mathbf{x}_t$, and minimise
the MSE between $\boldsymbol{\epsilon}$ and the network's prediction.

### 4.4 Conditioning

Conditioning on audio is done by **channel-wise concatenation**: the denoiser
receives $[\mathbf{x}_t;\mathbf{c}]\in\mathbb{R}^{(C+F)\times T}$ as input, so
every output frame attends to the local spectrogram content. The timestep $t$
is injected through a FiLM-style embedding (Section 6.2).

---

## 5. Sampling

### 5.1 Ancestral sampling (full DDPM)

Starting from $\mathbf{x}_N\sim\mathcal{N}(0,\mathbf{I})$, iterate for
$t=N,\dots,1$:

$$
\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t,\mathbf{c},t) + \mathbf{1}[t>1]\,\sigma_t\,\mathbf{z},\qquad \mathbf{z}\sim\mathcal{N}(0,\mathbf{I}).
$$

This uses all $N$ steps. **Note:** naively *skipping* steps while reusing the
per-step coefficients above is **incorrect** — it under-denoises toward the mean.
For fast sampling use DDIM.

### 5.2 DDIM (accelerated, deterministic)

DDIM (Song et al., 2021) defines a non-Markovian process with the same training
marginals, so the *same* trained $\epsilon_\theta$ can be sampled on any
increasing subsequence $\{\tau_1<\dots<\tau_S\}\subseteq\{1,\dots,N\}$. First
form the predicted clean signal,

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_{\tau_i} - \sqrt{1-\bar\alpha_{\tau_i}}\;\epsilon_\theta(\mathbf{x}_{\tau_i},\mathbf{c},\tau_i)}{\sqrt{\bar\alpha_{\tau_i}}},
$$

then step to the previous subsequence index:

$$
\mathbf{x}_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\;\hat{\mathbf{x}}_0
+ \sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}\;\epsilon_\theta
+ \sigma_{\tau_i}\,\mathbf{z},
$$

$$
\sigma_{\tau_i} = \eta\sqrt{\frac{1-\bar\alpha_{\tau_{i-1}}}{1-\bar\alpha_{\tau_i}}\Big(1-\frac{\bar\alpha_{\tau_i}}{\bar\alpha_{\tau_{i-1}}}\Big)}.
$$

With $\eta=0$ the process is **deterministic** and high quality at $S\approx
50$–$100$ steps (≈10× faster than full DDPM). We clamp $\hat{\mathbf{x}}_0$ to
$[-1.5, 1.5]$ for stability.

---

## 6. Denoiser architecture $\epsilon_\theta$

A **1D conditional U-Net** over the time axis.

### 6.1 U-Net backbone

Input $[\mathbf{x}_t;\mathbf{c}]$ is projected to $B$ base channels, then passed
through $L$ down stages with channel multipliers $(1,2,4,8)$, each a residual
block followed by a stride-2 convolution (halving $T$); a bottleneck; and a
symmetric up path with transposed convolutions and skip connections. A residual
block is

$$
\mathrm{Res}(h) = h + W_2\,\phi\!\big(\mathrm{GN}(W_1\,\phi(\mathrm{GN}(h)) + \mathbf{s}_t)\big),
$$

where $\phi=\mathrm{SiLU}$, $\mathrm{GN}$ is GroupNorm, and $\mathbf{s}_t$ is the
time shift (Section 6.2).

### 6.2 Timestep embedding and FiLM injection

$t$ is mapped to a sinusoidal embedding and an MLP,

$$
\gamma_j(t) = \big[\cos(t\,\omega_j),\ \sin(t\,\omega_j)\big],\quad
\omega_j = \exp\!\Big(-\frac{\ln 10000}{d/2}\,j\Big),\quad
\mathbf{e}_t = \mathrm{MLP}(\gamma(t)),
$$

and each residual block adds a learned, per-channel **shift**
$\mathbf{s}_t = W\,\mathbf{e}_t$ (a FiLM layer with unit scale), broadcasting the
timestep information across time.

### 6.3 Self-attention with QK-normalisation

At the two coarsest resolutions and the bottleneck, a multi-head self-attention
block models long-range structure (necessary for coherent patterns). For head
$h$ with queries/keys/values $Q,K,V\in\mathbb{R}^{T\times d_h}$,

$$
\hat Q = \frac{Q}{\lVert Q\rVert_2},\quad \hat K = \frac{K}{\lVert K\rVert_2},\quad
A = \mathrm{softmax}\!\big(\tau\,\hat Q\hat K^{\top}\big),\quad
O = A\,V,
$$

where $\tau = \exp(s)$ is a learnable temperature (clamped to $\tau\le 100$) and
$s$ is a trainable scalar. **QK-normalisation** bounds the logits in
$[-\tau,\tau]$ regardless of activation scale; without it, plain dot-product
attention diverged under bf16 mixed precision. The output projection is
**zero-initialised**, so each attention block starts as an identity map and
eases in during training.

### 6.4 Difficulty conditioning & classifier-free guidance

The denoiser is additionally conditioned on a **difficulty context vector**
$\mathbf{d}\in\mathbb{R}^{6}$ — a normalised
$[\mathrm{SR}, \mathrm{AR}, \mathrm{OD}, \mathrm{HP}, \mathrm{CS}, \text{density}]$
(SR = the rosu-pp star rating). It is embedded and **added to the timestep
embedding**, so every residual block is modulated by *(diffusion step +
difficulty)*:

$$ \mathbf{e} = \mathrm{MLP}_t(\gamma(t)) + g,\qquad
g = \begin{cases} \mathrm{MLP}_d(\mathbf{d}), & \text{conditioned},\\ \mathbf{n}, & \text{null (dropped)},\end{cases} $$

where $\mathbf{n}$ is a **learned null embedding**. During training the context is
dropped to $\mathbf{n}$ with probability $p_\text{drop}=0.15$, which trains both
the conditional and unconditional models in one network.

**Classifier-free guidance.** At inference we combine the two predictions to push
the sample toward the requested difficulty with strength $w$:

$$ \hat\epsilon = \epsilon_\theta(\mathbf{x}_t,\mathbf{c},t,\mathbf{n}) + w\big(\epsilon_\theta(\mathbf{x}_t,\mathbf{c},t,\mathbf{d}) - \epsilon_\theta(\mathbf{x}_t,\mathbf{c},t,\mathbf{n})\big),\quad w\approx 2. $$

$w=1$ recovers plain conditional sampling; $w>1$ trades diversity for stronger
adherence to the target star rating. Because rosu-pp gives a cheap SR read-out of
the *generated* map, the requested vs achieved SR can be checked and $w$ tuned.

---

## 7. Optimisation

### 7.1 Loss, precision, gradients

The objective is the MSE of Section 4.3, optimised with **AdamW** under
**bf16 autocast**. Gradients are clipped to a max global norm $g_\text{max}=0.3$,

$$
\mathbf{g}\leftarrow \mathbf{g}\cdot\min\!\Big(1,\ \frac{g_\text{max}}{\lVert\mathbf{g}\rVert_2}\Big),
$$

with optional gradient accumulation over $A$ micro-batches (effective batch
$A\cdot B$).

### 7.2 Exponential moving average (EMA)

A shadow copy of the weights is tracked for inference,

$$
\bar\theta \leftarrow \rho\,\bar\theta + (1-\rho)\,\theta,\qquad \rho = 0.999,
$$

updated each optimiser step. Sampling uses $\bar\theta$, which gives smoother,
higher-quality outputs than the raw weights.

### 7.3 Learning-rate schedule

Linear warmup for $W$ steps to a peak $\eta_0$, then cosine decay to $0$ over the
remaining $S_\text{tot}$ steps:

$$
\eta(s) =
\begin{cases}
\eta_0\,\dfrac{s}{W}, & s < W,\\[2mm]
\dfrac{\eta_0}{2}\Big(1 + \cos\pi\,\dfrac{s-W}{S_\text{tot}-W}\Big), & s \ge W.
\end{cases}
$$

---

## 8. Decoding: signal → beatmap

The sampled $\hat{\mathbf{x}}_0$ is converted to discrete objects by a
deterministic decoder $\mathcal{D}$.

### 8.1 Onset peak-picking

Object times are the local maxima of the onset channel above a threshold
$\theta_o$, separated by at least $g$ frames:

$$
\mathcal{P} = \big\{\,i\ :\ x_0[i] \ge \theta_o,\ x_0[i]\ge x_0[i\pm1],\ i - i_\text{prev} \ge g \,\big\}.
$$

Each peak's song time is $\hat\tau(i) = i\,\Delta\tau$. The cursor channels at
$i$ give the object position; the hold channels classify circle vs. slider vs.
spinner.

### 8.2 Slider reconstruction and duration

**v5 (live decoder, `_slider_from_anchors`):** the $K{=}3$ dedicated anchor
channels are read as the **span mean** over $[i_0,i_1]$, denormalised to
head-relative control points $p_k = (x + \mathrm{d}x_k W,\ y + \mathrm{d}y_k H)$,
deduplicated, and emitted as a **Bézier** control polygon (linear if a single
distinct point). The `slides` channel's span mean rounds to the repeat count,
recovering reverse sliders. The written **pixel length** is the polyline length
$\ell = \sum_k \lVert p_k - p_{k-1}\rVert$.

*(Legacy v4 decoder `_slider_path`, kept as a 10-channel fallback: samples the
shared cursor path at up to $K{=}8$ anchors, RDP-simplified to ≤4 — which decodes
near-straight because the cursor is just a head→end interpolation during a slider.
The v5 anchor channels exist precisely to fix this.)*

Crucially, osu! derives a slider's *duration* from its length, slider velocity
$v$, and the local beat length $\beta_\text{ms}$:

$$
\Delta t_\text{slider} = \frac{\ell}{v}\,\beta_\text{ms}\cdot(\text{slides}),
\qquad v = \text{SliderMultiplier}\times 100\times \mathrm{SV}.
$$

To prevent a long slider from overlapping the next object, the writer **clamps**
$\ell$ so $\Delta t_\text{slider}$ fits the gap $\Delta$ to the next object:

$$
\ell \le \frac{0.9\,\Delta\,v}{\beta_\text{ms}\cdot(\text{slides})}.
$$

### 8.3 Timing estimation (BPM + offset)

The output `.osu` needs a timing point. From the onset-strength envelope we run
beat tracking to obtain beat times $\{b_m\}$, then estimate tempo from the
**median inter-beat interval** and fold octave errors into the osu! range
$[125, 250)$:

$$
\text{BPM} = \mathrm{fold}\!\Big(\frac{60000}{\mathrm{median}_m(b_{m+1}-b_m)}\Big),\qquad
\text{offset} = 1000\,b_0 .
$$

### 8.4 Beat snapping (rhythm post-process)

Onsets are optionally nudged onto the estimated grid. With beat length
$\beta_\text{ms}$, offset $o$, and subdivision interval $\iota = \beta_\text{ms}/d$
(default $d=4$), the snap target for time $\tau$ is

$$
g(\tau) = o + \iota\,\Big\lfloor \frac{\tau - o}{\iota} + \tfrac12 \Big\rfloor,
$$

applied **only if** $|g(\tau)-\tau|\le \delta_\text{max}$ (a bound of a few tens
of ms). Bounding the move means a wrong BPM estimate cannot drag the whole map
onto a bad grid; durations are preserved by shifting $t_\text{end}$ by the same
delta.

**Slider-end snapping.** Snapping onsets leaves slider *ends* off-grid (their
duration comes from §8.2). So each slider's duration is rounded to the nearest
$1/d$-beat multiple $k\,\iota$ ($k\ge 1$, capped to fit the gap to the next
object) and its **length recomputed** $\ell = (k\iota/\beta_\text{ms})\,v$. This
moved slider ends from ~55% off the ¼-grid to ~0%.

---

## 9. Hyperparameters (current run)

| Group | Value |
|-------|-------|
| Audio | $f_s=22050$, $n_\text{fft}=1024$, hop $=256$, $F=64$ mels, $[20,11025]$ Hz |
| Signal | $C=17$ channels (v5), frame rate $\approx 86$ Hz, crop $T=4096$ ($\approx 48$ s) |
| Diffusion | $N=1000$, linear $\beta\in[10^{-4}, 2\times10^{-2}]$, $\epsilon$-prediction |
| Sampler | DDIM, $S\approx100$, $\eta=0$ |
| U-Net | base $=128$, mults $(1,2,4,8)$, $t$-dim $256$, attention (4 heads, QK-norm, `attn_levels=3`), $\approx$ 63 M params |
| Optim | AdamW (fused), peak LR $1.2\times10^{-4}$, weight decay $10^{-4}$, warmup $1000$, cosine decay |
| Stability | bf16 autocast, grad-clip $0.3$, EMA $\rho=0.999$. **base 160 + bf16 diverges — do not use** |

---

## 10. References

- J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models (DDIM).* ICLR 2021.
- A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters* (QK-normalisation). 2023.
- jaswon. *osu!dreamer* — signal + diffusion approach for osu! maps.
- osu! wiki — [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)).
