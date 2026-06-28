# TECH_REPORT — moved into `docs/`

The technical/math report was split into two small files under [`docs/`](docs/INDEX.md):

- **The diffusion math** — problem formulation, DDPM forward/reverse + training objective (ε and
  v-prediction + zero-terminal-SNR), DDIM sampling, the U-Net / adaLN / QK-norm-attention
  architecture, difficulty conditioning + CFG, optimisation, hyperparameters:
  → [`docs/knowledge/diffusion-math.md`](docs/knowledge/diffusion-math.md)
- **The data representation** — the 21-channel beatmap signal, per-channel encode, and the
  deterministic decode (onset peak-picking, slider reconstruction & duration, SV/curve/corner/spacing
  decode, timing estimation, beat-snap):
  → [`docs/knowledge/signal-encoding.md`](docs/knowledge/signal-encoding.md)

Start at the navigation map: [`docs/INDEX.md`](docs/INDEX.md).
