# MorphGen — Conditional Face Generation with WGAN-GP

A conditional deep convolutional GAN trained on CelebA to generate faces from semantic attribute vectors. Built with PyTorch, deployed with Streamlit.

**Live demo → [morphgen.streamlit.app](https://morphgen.streamlit.app)**

---

## Problem

Most generative face models produce random outputs with no user control. MorphGen lets a user specify semantic attributes — hair color, facial hair, expression, accessories — and generates a face that reflects those choices.

---

## Approach

- **Architecture:** Conditional DCGAN — Generator and Critic both conditioned on CelebA's 40 binary attributes
- **Training objective:** WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training and meaningful loss curves
- **Conditioning:** Attribute vector concatenated to noise vector `z` in the Generator; projected to a spatial map and concatenated channel-wise in the Critic
- **Dataset:** CelebA — 200K+ aligned face images with 40 binary attribute labels per image
- **Resolution:** 64×64 RGB (center-crop + resize)

### Architecture

```
Generator
  z (128-dim) + attrs (40-dim) → ConvTranspose2d ×5 → 3 × 64 × 64 (Tanh)

Critic
  image (3 × 64 × 64) + attr_map (1 × 64 × 64) → Conv2d ×4 → scalar score
```

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Image size | 64 × 64 |
| Latent dim | 128 |
| Batch size | 128 |
| Generator LR | 1e-4 |
| Critic LR | 4e-4 (TTUR) |
| Critic steps per G step | 5 |
| Gradient penalty weight λ | 10 |
| Optimizer | Adam (β1=0.0, β2=0.9) |
| Epochs | 50 |

---

## Results

- Visually coherent face generation across 40 attribute combinations
- Stable WGAN-GP loss curves — no mode collapse observed
- Controlled attribute transfer: fix a seed, toggle attributes, observe the same base face change

---

## Demo Features

The Streamlit app exposes four modes:

| Tab | Description |
|---|---|
| **Generate** | Select attributes, optionally fix a seed, generate and download a face |
| **Compare** | Side-by-side comparison of two independently configured faces |
| **Interpolate** | Smooth morph between two attribute sets on a fixed base identity |
| **Batch** | Generate N faces with the same attributes but different seeds; download as ZIP |

---

## Project Structure

```
morphgen/
├── app.py                   # Streamlit deployment
├── config.py                # Hyperparameters
├── requirements.txt
├── .gitignore
├── data/
│   ├── __init__.py
│   └── dataset.py           # CelebA loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── generator.py         # Conditional DCGAN Generator
│   └── discriminator.py     # WGAN-GP Critic
├── training/
│   ├── __init__.py
│   ├── losses.py            # Wasserstein loss + gradient penalty
│   └── train.py             # Training loop
├── utils/
│   ├── __init__.py
│   └── visualize.py         # Image grids, loss curves
└── outputs/
    └── checkpoints/         # Model weights (generator_final.pt)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CelebA

CelebA downloads automatically on first run via `torchvision`. If Google Drive rate-limits the download:

1. Download manually from [mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Place files at:

```
data/celeba/img_align_celeba/     ← all .jpg images
data/celeba/list_attr_celeba.txt
data/celeba/list_eval_partition.txt
```

### 3. Train

```bash
# Fresh training
python training/train.py

# Resume from checkpoint
python training/train.py --resume outputs/checkpoints/ckpt_epoch005_step03200.pt
```

Training logs print every 100 steps. Image grids are saved to `outputs/samples/` every 500 steps.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Key Design Decisions

**Why WGAN-GP over vanilla GAN**
Vanilla GAN uses Binary Cross-Entropy. When the Discriminator becomes too confident early in training, gradients vanish and the Generator stops learning. WGAN replaces BCE with Wasserstein distance — a smoother metric that always provides a useful gradient signal. WGAN-GP enforces the required Lipschitz constraint by penalizing gradient norms on interpolated samples rather than clipping weights.

**Why InstanceNorm in the Critic, not BatchNorm**
WGAN-GP computes per-sample gradients for the gradient penalty. BatchNorm couples samples within a batch, which corrupts those per-sample gradients. InstanceNorm normalizes per sample, per channel — making it safe to use here without interfering with the penalty computation.

**Why TTUR (Two Time-scale Update Rule)**
The Critic learns at 4× the Generator's rate (`4e-4` vs `1e-4`). A well-calibrated Critic is a prerequisite for a meaningful Generator loss signal. TTUR keeps the Critic ahead without relying solely on critic-steps-per-update tuning.

**Conditioning via spatial projection in the Critic**
The 40-dim attribute vector is projected to a `64×64` spatial map and concatenated as an extra input channel. This allows the Critic to evaluate whether image content actually matches the requested attributes — not just whether it looks realistic in isolation.

---

## Samples

The model was trained at 64×64 resolution. Generated faces show coherent structure and attribute alignment across combinations — hair color, facial hair, expression, and accessories transfer visibly between configurations. Full-resolution sample grids are viewable in the [live demo](https://morphgen.streamlit.app).

---

## Tech Stack

- PyTorch
- torchvision
- Transformers (Swin2SR — 4× super-resolution on generated faces)
- Streamlit
- Matplotlib
- Pillow

---

## References

- [Radford et al., 2015 — Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- [Arjovsky et al., 2017 — Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Gulrajani et al., 2017 — Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Liu et al., 2015 — CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
