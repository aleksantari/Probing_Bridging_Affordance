#!/usr/bin/env python
"""Extract SigLIP So400m vision towers from parent models and save as standalone HF checkpoints.

Run once before training to pre-extract vision towers:
    python scripts/extract_vision_towers.py

Saves towers to ../../models/siglip_so400m_{variant}/ as HF-compatible directories
that can be loaded via SiglipVisionModel.from_pretrained(local_path).
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

MODELS_DIR = Path(__file__).resolve().parents[1] / ".." / ".." / "models"

VARIANTS = {
    "pg1": {
        "parent_id": "google/paligemma-3b-pt-224",
        "description": "PaliGemma stage-1 SigLIP",
    },
    "pi0": {
        "parent_id": "lerobot/pi0_base",
        "description": "π0 SigLIP",
    },
    "pi05": {
        "parent_id": "lerobot/pi05_base",
        "description": "π0.5 SigLIP",
    },
}


def extract_paligemma(model_id: str) -> torch.nn.Module:
    from transformers import PaliGemmaForConditionalGeneration

    print(f"  Loading PaliGemma from {model_id} (CPU)...")
    full_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu",
    )
    tower = full_model.vision_tower
    del full_model
    gc.collect()
    return tower


def extract_pi0(model_id: str) -> torch.nn.Module:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    print(f"  Loading π0 policy from {model_id}...")
    policy = PI0Policy.from_pretrained(model_id)
    tower = policy.model.paligemma_with_expert.paligemma.vision_tower
    # Hold reference before deleting parent
    tower_ref = tower
    del policy
    gc.collect()
    return tower_ref


def extract_pi05(model_id: str) -> torch.nn.Module:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print(f"  Loading π0.5 policy from {model_id}...")
    policy = PI05Policy.from_pretrained(model_id)
    tower = policy.model.paligemma_with_expert.paligemma.vision_tower
    tower_ref = tower
    del policy
    gc.collect()
    return tower_ref


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    extractors = {
        "pg1": extract_paligemma,
        "pi0": extract_pi0,
        "pi05": extract_pi05,
    }

    for variant, info in VARIANTS.items():
        output_dir = MODELS_DIR / f"siglip_so400m_{variant}"
        if output_dir.exists() and (output_dir / "config.json").exists():
            print(f"[{variant}] Already extracted at {output_dir}, skipping.")
            continue

        print(f"\n[{variant}] Extracting {info['description']}...")
        tower = extractors[variant](info["parent_id"])

        n_params = count_params(tower)
        print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

        # Compute L2 norm of all parameters as a fingerprint
        with torch.no_grad():
            l2_norm = sum(p.float().norm().item() ** 2 for p in tower.parameters()) ** 0.5
        print(f"  Parameter L2 norm: {l2_norm:.4f}")

        # Save as HF-compatible checkpoint
        output_dir.mkdir(parents=True, exist_ok=True)
        tower.save_pretrained(output_dir)
        print(f"  Saved to {output_dir}")

        # Free memory before next extraction
        del tower
        gc.collect()
        torch.cuda.empty_cache()

    # Verify all extracted towers load correctly
    print("\n--- Verification ---")
    from transformers import SiglipVisionModel

    for variant in VARIANTS:
        output_dir = MODELS_DIR / f"siglip_so400m_{variant}"
        tower = SiglipVisionModel.from_pretrained(output_dir, torch_dtype=torch.float32)
        n_params = count_params(tower)
        print(f"[{variant}] Loaded OK — {n_params:,} params")
        del tower

    print("\nDone. All towers extracted and verified.")


if __name__ == "__main__":
    main()
