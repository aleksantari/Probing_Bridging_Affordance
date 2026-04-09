#!/usr/bin/env python
"""Validate that all SigLIP So400m variants produce distinct features.

Loads all 4 variants (raw, pg1, pi0, pi05), runs a forward pass on 3 test
images from UMD, and prints a pairwise L2 distance matrix. Any zero
off-diagonal entry means two checkpoints are secretly identical.

Run after extract_vision_towers.py:
    python scripts/validate_encoders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / ".." / ".." / "models"
DATASET_ROOT = PROJECT_ROOT / ".." / ".." / "datasets" / "UMD" / "part-affordance-dataset"

# Add project root to path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VARIANTS = {
    "raw": "google/siglip-so400m-patch14-384",
    "pg1": str(MODELS_DIR / "siglip_so400m_pg1"),
    "pi0": str(MODELS_DIR / "siglip_so400m_pi0"),
    "pi05": str(MODELS_DIR / "siglip_so400m_pi05"),
}

# Use layer 26 (last hooked layer) for comparison
TARGET_LAYER = 26


def get_test_images(n: int = 3) -> list[torch.Tensor]:
    """Load n test images from UMD dataset."""
    from PIL import Image

    tools_dir = DATASET_ROOT / "tools"
    if not tools_dir.exists():
        raise FileNotFoundError(f"UMD tools directory not found at {tools_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = []
    for img_path in sorted(tools_dir.rglob("*_rgb.jpg")):
        img = Image.open(img_path).convert("RGB")
        images.append(transform(img))
        if len(images) >= n:
            break

    if len(images) < n:
        raise RuntimeError(f"Only found {len(images)} images, need {n}")

    return images


def extract_features(model_id: str, images: list[torch.Tensor]) -> torch.Tensor:
    """Extract last-layer features from a SigLIP variant."""
    from src.models.siglip_so400m import SigLIPSo400mBackbone

    backbone = SigLIPSo400mBackbone(
        source="raw",
        model_id=model_id,
        device="cuda",
        precision="fp32",
        layers_to_hook=[TARGET_LAYER],
    )

    batch = torch.stack(images).cuda()
    with torch.no_grad():
        features = backbone(batch)

    # features[TARGET_LAYER] shape: (B, C, H, W) → flatten to (B, C*H*W)
    feat = features[TARGET_LAYER].cpu().float()
    feat = feat.reshape(feat.shape[0], -1)

    del backbone
    torch.cuda.empty_cache()
    return feat


def main() -> None:
    print("Loading test images...")
    images = get_test_images(3)
    print(f"Loaded {len(images)} images, shape: {images[0].shape}")

    # Extract features from each variant
    all_features: dict[str, torch.Tensor] = {}
    for name, model_id in VARIANTS.items():
        path = Path(model_id)
        if not path.exists() and not model_id.startswith("google/"):
            print(f"[{name}] SKIPPED — {model_id} not found")
            continue
        print(f"\n[{name}] Extracting features from {model_id}...")
        all_features[name] = extract_features(model_id, images)
        print(f"  Feature shape: {all_features[name].shape}")

    if len(all_features) < 2:
        print("\nERROR: Need at least 2 variants to compare.")
        sys.exit(1)

    # Compute pairwise L2 distances (averaged over the 3 test images)
    names = list(all_features.keys())
    n = len(names)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff = all_features[names[i]] - all_features[names[j]]
            # Mean L2 distance across the batch of test images
            distances[i, j] = diff.norm(dim=1).mean().item()

    # Print matrix
    print("\n" + "=" * 60)
    print("Pairwise L2 Distance Matrix (mean over 3 test images)")
    print("=" * 60)

    # Header
    header = f"{'':>8}" + "".join(f"{name:>10}" for name in names)
    print(header)
    print("-" * len(header))

    for i, name in enumerate(names):
        row = f"{name:>8}" + "".join(f"{distances[i, j]:>10.2f}" for j in range(n))
        print(row)

    # Assertions
    print("\n--- Checks ---")
    ok = True
    for i in range(n):
        if distances[i, i] != 0.0:
            print(f"FAIL: diagonal [{names[i]}] is {distances[i, i]:.4f}, expected 0")
            ok = False

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] == 0.0:
                print(f"FAIL: {names[i]} and {names[j]} are IDENTICAL (L2 = 0)")
                ok = False
            else:
                print(f"OK: {names[i]} vs {names[j]} = {distances[i, j]:.2f}")

    if ok:
        print("\nAll checks passed. All variants are distinct.")
    else:
        print("\nSOME CHECKS FAILED. Investigate extraction.")
        sys.exit(1)


if __name__ == "__main__":
    main()
