"""SigLIP So400m backbone for linear probing.

Supports loading the vision tower from multiple sources:
- "raw": standalone google/siglip-so400m-patch14-384
- "paligemma": extracted from PaliGemma-3B
- "pi0": extracted from LeRobot pi0 policy
- "pi0_5": extracted from LeRobot pi0.5 policy

All sources share the same architecture (ViT-So400m/14, 27 layers, 1152 dim).
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

__all__ = ["SigLIPSo400mBackbone"]

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _parse_precision(precision: Optional[str]) -> Optional[torch.dtype]:
    if precision is None:
        return None
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    if precision in {"fp32", "float32"}:
        return torch.float32
    return None


def _load_vision_tower(source: str, model_id: str, dtype: torch.dtype) -> nn.Module:
    """Load a SiglipVisionModel from the specified source."""

    if source == "raw":
        from transformers import SiglipVisionModel

        return SiglipVisionModel.from_pretrained(model_id, torch_dtype=dtype)

    elif source == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration

        full_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cpu",
        )
        tower = full_model.vision_tower
        del full_model
        gc.collect()
        torch.cuda.empty_cache()
        return tower

    elif source == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        policy = PI0Policy.from_pretrained(model_id)
        tower = policy.model.paligemma_with_expert.paligemma.vision_tower
        # Detach reference before deleting parent
        tower_ref = tower
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        return tower_ref

    elif source == "pi0_5":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(model_id)
        tower = policy.model.paligemma_with_expert.paligemma.vision_tower
        tower_ref = tower
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        return tower_ref

    else:
        raise ValueError(
            f"Unknown source {source!r}. Must be one of: raw, paligemma, pi0, pi0_5"
        )


class SigLIPSo400mBackbone(nn.Module):
    """Wraps SigLIP So400m vision encoder for dense patch token extraction."""

    def __init__(
        self,
        *,
        source: str = "raw",
        model_id: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        precision: str = "bf16",
        layers_to_hook: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.source = source
        self.requested_precision = precision

        dtype = _parse_precision(precision) or torch.float32

        # Load the vision tower from the appropriate source
        vision_tower = _load_vision_tower(source, model_id, dtype)
        self.vision_tower = vision_tower.to(self.device)
        self.vision_tower.eval()
        for param in self.vision_tower.parameters():
            param.requires_grad_(False)

        # Grab the post-LayerNorm to apply to intermediate hidden states,
        # matching DINOv2's norm=True behavior in get_intermediate_layers().
        self.layer_norm = self.vision_tower.vision_model.post_layernorm

        # Get normalization stats from the canonical So400m image processor
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        image_mean = torch.tensor(
            image_processor.image_mean, dtype=torch.float32
        )
        image_std = torch.tensor(
            image_processor.image_std, dtype=torch.float32
        )
        self.register_buffer(
            "siglip_mean", image_mean.view(1, -1, 1, 1).to(self.device), persistent=False
        )
        self.register_buffer(
            "siglip_std", image_std.view(1, -1, 1, 1).to(self.device), persistent=False
        )
        self.register_buffer(
            "imagenet_mean", _IMAGENET_MEAN.view(1, -1, 1, 1).to(self.device), persistent=False
        )
        self.register_buffer(
            "imagenet_std", _IMAGENET_STD.view(1, -1, 1, 1).to(self.device), persistent=False
        )

        self.patch_size = 14

        # Determine layer count
        encoder_layers = self.vision_tower.vision_model.encoder.layers
        self.total_blocks = len(encoder_layers)

        if layers_to_hook:
            resolved = []
            for layer in layers_to_hook:
                idx = layer if layer >= 0 else self.total_blocks + layer
                if idx < 0 or idx >= self.total_blocks:
                    raise ValueError(
                        f"Layer index {layer} out of bounds for "
                        f"SigLIP So400m encoder with {self.total_blocks} layers."
                    )
                if idx not in resolved:
                    resolved.append(idx)
            self.layer_order = resolved
        else:
            self.layer_order = [self.total_blocks - 1]
        self.default_layer = self.layer_order[-1]

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        autocast_precision: Optional[str] = None,
    ) -> OrderedDict[int, torch.Tensor]:
        device = self.imagenet_mean.device
        dtype = (
            _parse_precision(autocast_precision)
            or _parse_precision(self.requested_precision)
            or next(self.vision_tower.parameters()).dtype
        )

        images = images.to(device, dtype=torch.float32, non_blocking=True)
        height, width = images.shape[-2:]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        # Denormalize from ImageNet range back to [0, 1]
        pixels = images * self.imagenet_std + self.imagenet_mean
        pixels = pixels.clamp(0.0, 1.0)

        # Renormalize with SigLIP-specific stats
        pixel_values = (pixels - self.siglip_mean) / self.siglip_std
        pixel_values = pixel_values.to(dtype=dtype)

        # Forward through vision tower
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )

        if isinstance(vision_outputs, dict):
            hidden_states = vision_outputs.get("hidden_states")
        else:
            hidden_states = getattr(vision_outputs, "hidden_states", None)

        if hidden_states is None:
            raise ValueError(
                "SigLIP So400m vision tower did not return hidden states. "
                "Ensure output_hidden_states=True is supported."
            )

        outputs = OrderedDict()
        expected_tokens = grid_h * grid_w

        for layer_idx in self.layer_order:
            sequence = hidden_states[layer_idx]

            seq_len = sequence.shape[1]
            if seq_len == expected_tokens + 1:
                # Strip CLS token
                patch_tokens = sequence[:, 1:, :]
            elif seq_len == expected_tokens:
                patch_tokens = sequence
            else:
                raise ValueError(
                    f"Expected {expected_tokens} (or +1) tokens, "
                    f"received {seq_len}. Check image size or model config."
                )

            # Apply LayerNorm to match DINOv2's norm=True extraction
            patch_tokens = self.layer_norm(patch_tokens)
            patch_tokens = patch_tokens.to(torch.float32)
            batch_size = patch_tokens.shape[0]
            feature_dim = patch_tokens.shape[-1]
            patch_tokens = (
                patch_tokens.reshape(batch_size, grid_h, grid_w, feature_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            outputs[layer_idx] = patch_tokens

        return outputs
