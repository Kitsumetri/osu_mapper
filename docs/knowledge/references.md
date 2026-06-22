# References & prior art

**Purpose:** the external links and papers the project draws on — osu! domain docs, the diffusion/ML
papers behind the model and the alignment work, and the sibling/prior-art repos. | **STATIC**
(reference list; append as new sources are used).

## osu! domain
- [Mapping techniques (Basics)](https://osu.ppy.sh/wiki/en/Mapping_Techniques/Basics)
- [Technical maps](https://osu.ppy.sh/wiki/en/Beatmap/Technical_maps)
- [Making good sliders](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Making_good_sliders)
- [.osu file format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
- [slider library — what is a beatmap](https://llllllllll.github.io/slider/what-is-a-beatmap.html)
- [Farm map (1-2 jumps, Sotarks)](https://osu.miraheze.org/wiki/Farm_map)
- [Star patterns (forum)](https://osu.ppy.sh/community/forums/topics/292689)
- [Burst vs alt maps (forum)](https://osu.ppy.sh/community/forums/topics/1763755)

## Diffusion / model
- J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models (DDIM).* ICLR 2021.
- A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- T. Salimans, J. Ho. *Progressive Distillation for Fast Sampling* (v-prediction). ICLR 2022.
- S. Lin et al. *Common Diffusion Noise Schedules and Sample Steps are Flawed* (zero-terminal-SNR).
  WACV 2024.
- Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters* (QK-normalisation). 2023.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE). 2021.

## Loss / training (§ loss options)
- Hang et al. *Efficient Diffusion Training via Min-SNR Weighting.* ICCV 2023.
- Song & Dhariwal. *Improved Techniques for Consistency Models* (Pseudo-Huber loss). 2023.
- Gatys et al. *Neural Style Transfer* (Gram matrices) — image-only, **does not transfer** to our
  time-series target.

## Timing / beat tracking (timing-model design)
- Foscarin et al. *Beat this! Accurate beat tracking without DBN postprocessing.* ISMIR 2024
  ([researchgate](https://www.researchgate.net/publication/382739081_Beat_this_Accurate_beat_tracking_without_DBN_postprocessing)).
- [Beat Transformer (dilated self-attn), 2022](https://arxiv.org/pdf/2209.07140).
- [BEAST — online streaming transformer, 2023](https://arxiv.org/abs/2312.17156).
- Davies & Böck. *TCN for beat tracking.* EUSIPCO 2019. · `mir_eval.beat` metrics.

## RL / alignment (reward + RL design)
- Black, Janner, Du, Kostrikov, Levine. *Training Diffusion Models with Reinforcement Learning*
  (DDPO). 2023.
- Fan et al. *DPOK: RL for Fine-tuning Text-to-Image Diffusion Models* (DDPO + KL). 2023.
- Wallace et al. *Diffusion Model Alignment Using Direct Preference Optimization* (Diffusion-DPO). 2023.
- Clark et al. *Directly Fine-tuning Diffusion Models on Differentiable Rewards* (DRaFT). 2023.
- Prabhudesai et al. *Aligning Text-to-Image Diffusion Models with Reward Backpropagation*
  (AlignProp). 2023.
- Peters & Schaal. *Reinforcement Learning by Reward-Weighted Regression* (RWR). 2007.

## Prior art / sibling repos
- [jaswon/osu-dreamer](https://github.com/jaswon/osu-dreamer) — the signal + diffusion approach.
- [OliBomby/osu-diffusion](https://github.com/OliBomby/osu-diffusion) ·
  [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) — DiT-style coordinate diffusion
  (slider anchors + repeat counts as typed points; flips aug, style conditioning — techniques to
  adopt).
- [gyataro/osuT5](https://github.com/gyataro/osuT5) · [kotritrona/osumapper](https://github.com/kotritrona/osumapper)
