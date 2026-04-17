# config.py

import torch

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_ROOT       = "./data"
IMAGE_SIZE      = 64          # resize all CelebA images to 64x64
NUM_WORKERS     = 4

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
LATENT_DIM      = 128         # size of the noise vector fed to the Generator
NUM_ATTRIBUTES  = 40          # CelebA has 40 binary attributes
CHANNELS        = 3           # RGB
FEATURES_G      = 64          # base feature map size in Generator
FEATURES_D      = 64          # base feature map size in Discriminator

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE      = 128
NUM_EPOCHS      = 50
LR_G            = 1e-4        # Generator learning rate
LR_D            = 4e-4        # Discriminator learns faster (TTUR trick)
BETA1           = 0.0         # Adam beta1 (0.0 recommended for WGAN-GP)
BETA2           = 0.9         # Adam beta2

# WGAN-GP specific
LAMBDA_GP       = 10          # gradient penalty weight
CRITIC_STEPS    = 5           # train Discriminator 5x per Generator step

# ---------------------------------------------------------------------------
# Logging & Checkpointing
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL = 500         # save a grid of generated images every N batches
CHECKPOINT_DIR  = "./outputs/checkpoints"
SAMPLES_DIR     = "./outputs/samples"
LOG_DIR         = "./outputs/logs"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
