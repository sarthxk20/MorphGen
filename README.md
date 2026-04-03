# FaceGen — Conditional Face Generation with WGAN-GP

A conditional deep convolutional GAN trained on CelebA to generate photorealistic faces from semantic attribute vectors. Built with PyTorch, deployed with Streamlit.

---

## Problem

Generative models for faces typically produce random outputs with no user control. The goal here was to build a system where a user can specify semantic attributes — hair color, facial hair, expression, accessories — and receive a generated face that reflects those choices.

## Approach

- **Architecture:** Conditional DCGAN — a Generator and Discriminator (Critic) both conditioned on CelebA's 40 binary attributes
- **Training objective:** WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training and meaningful loss curves
- **Conditioning mechanism:** Attribute vector concatenated to the noise vector `z` in the Generator; projected to a spatial map and concatenated channel-wise in the Critic
- **Dataset:** CelebA — 200K+ aligned face images with 40 binary attribute labels per image
- **Input resolution:** 64x64 RGB after center-crop and resize

### Architecture summary

```
Generator
  z (128-dim) + attrs (40-dim) → ConvTranspose2d x5 → 3 x 64 x 64 image (Tanh)

Discriminator (Critic)
  image (3 x 64 x 64) + attr_map (1 x 64 x 64) → Conv2d x4 → scalar score (unbounded)
```

### Training details

| Hyperparameter | Value |
|---|---|
| Image size | 64 x 64 |
| Latent dim | 128 |
| Batch size | 128 |
| Generator LR | 1e-4 |
| Critic LR | 4e-4 (TTUR) |
| Critic steps per G step | 5 |
| Gradient penalty weight λ | 10 |
| Optimizer | Adam (β1=0.0, β2=0.9) |
| Epochs | 50 |

## Result

- Visually coherent face generation across 40 attribute combinations
- Stable training loss curves with no mode collapse
- Interactive Streamlit demo: fix a random seed, toggle attributes, observe controlled attribute transfer on the same base face
- FID score: *(add after training)*

---

## Project Structure

```
celebagan/
├── app.py                   # Streamlit deployment
├── config.py                # all hyperparameters
├── requirements.txt
├── data/
│   └── dataset.py           # CelebA loading and preprocessing
├── models/
│   ├── generator.py         # Conditional DCGAN Generator
│   └── discriminator.py     # WGAN-GP Critic
├── training/
│   ├── losses.py            # Wasserstein loss + gradient penalty
│   └── train.py             # training loop
├── utils/
│   └── visualize.py         # image grids, loss curves
└── outputs/
    ├── checkpoints/         # model weights per epoch
    └── samples/             # generated image grids during training
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
Vanilla GAN uses Binary Cross-Entropy. When the Discriminator becomes too confident early in training, gradients vanish and the Generator stops learning. WGAN replaces BCE with Wasserstein distance — a smoother measure that always provides a useful gradient signal. WGAN-GP enforces the required Lipschitz constraint by penalizing gradient norms on interpolated samples rather than clipping weights.

**Why InstanceNorm in the Critic, not BatchNorm**
WGAN-GP computes per-sample gradients for the gradient penalty. BatchNorm couples samples within a batch, which corrupts those gradients. InstanceNorm normalizes per sample, per channel — safe to use here.

**Why TTUR (Two Time-scale Update Rule)**
The Critic learns at 4x the Generator's learning rate (`4e-4` vs `1e-4`). A well-calibrated Critic is required before the Generator loss is meaningful. TTUR keeps the Critic ahead without manual tuning of critic steps alone.

**Conditioning via spatial projection in the Critic**
The 40-dim attribute vector is projected to a `64x64` spatial map and concatenated as an extra input channel. This lets the Critic evaluate whether the content of the image actually matches the requested attributes — not just whether it looks realistic.

---

## Samples

*(Add generated image grids from `outputs/samples/` here after training)*

---

## Tech Stack

- PyTorch
- torchvision
- Streamlit
- Matplotlib
- Pillow

---

## References

- [Radford et al., 2015 — Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- [Arjovsky et al., 2017 — Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Gulrajani et al., 2017 — Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Liu et al., 2015 — CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
