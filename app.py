# app.py

import torch
import numpy as np
import streamlit as st
from PIL import Image
import os

from config import LATENT_DIM, NUM_ATTRIBUTES, DEVICE, CHECKPOINT_DIR
from models.generator import Generator
from data.dataset import denormalize


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FaceGen — Conditional GAN",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #0d0d0d;
            color: #e0e0e0;
        }
        h1, h2, h3 {
            font-family: 'Space Mono', monospace;
            color: #00ffe0;
        }
        .stButton > button {
            background-color: #00ffe0;
            color: #0d0d0d;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1.5rem;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #ffd700;
            color: #0d0d0d;
        }
        .stCheckbox > label {
            font-size: 0.85rem;
        }
        .metric-box {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CelebA attribute names (official order from list_attr_celeba.txt)
# ---------------------------------------------------------------------------

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

DISPLAY_NAMES = [a.replace("_", " ").title() for a in ATTR_NAMES]

# Attributes grouped for cleaner UI
ATTR_GROUPS = {
    "Face Shape": ["Attractive", "Chubby", "Oval Face", "High Cheekbones", "Pale Skin", "Young"],
    "Hair":       ["Bald", "Bangs", "Black Hair", "Blond Hair", "Brown Hair", "Gray Hair",
                   "Receding Hairline", "Straight Hair", "Wavy Hair"],
    "Facial Hair":["5 O Clock Shadow", "Goatee", "Mustache", "No Beard", "Sideburns"],
    "Features":   ["Arched Eyebrows", "Bags Under Eyes", "Big Lips", "Big Nose",
                   "Bushy Eyebrows", "Double Chin", "Eyeglasses", "Narrow Eyes", "Pointy Nose",
                   "Rosy Cheeks"],
    "Expression": ["Mouth Slightly Open", "Smiling"],
    "Gender":     ["Male", "Heavy Makeup"],
    "Accessories":["Wearing Earrings", "Wearing Hat", "Wearing Lipstick",
                   "Wearing Necklace", "Wearing Necktie"],
}


# ---------------------------------------------------------------------------
# Load Generator
# ---------------------------------------------------------------------------

@st.cache_resource
def load_generator():
    G = Generator().to(DEVICE)
    weights_path = os.path.join(CHECKPOINT_DIR, "generator_final.pt")
    if not os.path.exists(weights_path):
        st.error(
            f"Trained weights not found at `{weights_path}`. "
            "Run `python training/train.py` first to train the model."
        )
        st.stop()
    G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    G.eval()
    return G


# ---------------------------------------------------------------------------
# Generate image
# ---------------------------------------------------------------------------

def generate_image(G, attr_vector, seed=None):
    """
    Generate a single face image given a 40-dim binary attribute vector.

    Args:
        G           : trained Generator
        attr_vector : list of 40 floats (0.0 or 1.0)
        seed        : optional int for reproducible generation

    Returns:
        PIL Image
    """
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z    = torch.randn(1, LATENT_DIM, device=DEVICE)
        attrs = torch.tensor([attr_vector], dtype=torch.float32, device=DEVICE)
        fake = G(z, attrs)                       # [1, 3, 64, 64]
        fake = denormalize(fake).squeeze(0)      # [3, 64, 64], range [0, 1]
        fake = fake.permute(1, 2, 0).cpu().numpy()  # HWC
        fake = (fake * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(fake).resize((256, 256), Image.NEAREST)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("FaceGen")
st.markdown("**Conditional GAN trained on CelebA** — select attributes to control face generation.")
st.markdown("---")

G = load_generator()

col_controls, col_output = st.columns([1, 1], gap="large")

with col_controls:
    st.markdown("### Attributes")

    selected_attrs = {}

    for group, attrs in ATTR_GROUPS.items():
        with st.expander(group, expanded=(group in ["Hair", "Expression", "Gender"])):
            for attr in attrs:
                key = attr.replace(" ", "_").upper()
                selected_attrs[attr] = st.checkbox(attr, key=key, value=False)

    st.markdown("---")
    st.markdown("### Randomness")
    use_fixed_seed = st.toggle("Fix random seed", value=False)
    seed_val = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1) \
               if use_fixed_seed else None

    st.markdown("---")
    generate_btn = st.button("Generate Face")


with col_output:
    st.markdown("### Generated Face")

    # Build attribute vector in the official CelebA order
    attr_vector = []
    for name in DISPLAY_NAMES:
        attr_vector.append(1.0 if selected_attrs.get(name, False) else 0.0)

    if generate_btn or "last_image" not in st.session_state:
        img = generate_image(G, attr_vector, seed=seed_val)
        st.session_state["last_image"] = img

    st.image(
        st.session_state["last_image"],
        caption="Generated face",
        use_column_width=False,
        width=256
    )

    # Active attributes summary
    active = [n for n, v in selected_attrs.items() if v]
    if active:
        st.markdown("**Active attributes:** " + ", ".join(active))
    else:
        st.markdown("*No attributes selected — pure random generation.*")

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown("""
    - A random noise vector **z** is sampled from a standard normal distribution.
    - The selected attributes are encoded as a binary vector and concatenated with **z**.
    - The Generator (DCGAN + WGAN-GP) maps this input to a 64x64 RGB face image.
    - The same seed with different attributes will produce the same base face with different characteristics.
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<p style='font-family: Space Mono, monospace; font-size: 0.75rem; color: #555;'>"
    "Conditional DCGAN · WGAN-GP · CelebA · PyTorch"
    "</p>",
    unsafe_allow_html=True
)
