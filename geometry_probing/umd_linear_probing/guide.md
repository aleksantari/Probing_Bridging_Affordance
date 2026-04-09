# Affordance Probing Guide

## 1. Overview

This guide documents the adapted Zhang et al. probing pipeline (`Probing_Bridging_Affordance`) and how we use it to characterize geometric affordance representations across vision encoders in the SigLIP → PaliGemma → π0 → π0.5 VLA lineage, with DINOv2 as an external reference.

**Research question (Phase 1):** At comparable parameter scale and identical patch geometry, do encoders trained with different objectives produce measurably different geometric affordance representations on UMD? Does VLA fine-tuning shift SigLIP's representations relative to the raw checkpoint?

**What this pipeline does:** Freeze a vision encoder, extract intermediate-layer patch tokens, train a lightweight linear probe (BatchNorm + 1×1 Conv) to predict per-patch affordance classes on the UMD dataset, and report patch-grid mIoU. No decoder, no fine-tuning of the encoder.

---

## 2. Repository Structure

```
Probing_Bridging_Affordance/
├── geometry_probing/
│   └── umd_linear_probing/       ← OUR WORKING DIRECTORY
│       ├── configs/               13 experiment configs + Zhang et al.'s originals
│       ├── scripts/               Entry points (train, eval, extract, validate)
│       ├── src/
│       │   ├── data/              Dataset, transforms, collation
│       │   ├── engine/            Trainer and evaluation loop
│       │   ├── models/            Backbone wrappers + probe heads
│       │   └── utils/             Config loading, metrics, seeding, logging
│       ├── metadata/splits/       Train/val/test split JSONs
│       └── outputs/               Experiment artifacts (generated at runtime)
├── interaction_probing/           FLUX Kontext cross-attention probing (not used in Phase 1)
├── auxiliary_analysis/            Patch-level analysis (not used in Phase 1)
├── fusion_zero_shot/              Geometry-interaction fusion on AGD20K (not used in Phase 1)
├── datasets/
│   └── UMD/part-affordance-dataset/   17 tool categories, ~2.5k frames
└── models/
    ├── dinov2/                    Local torch.hub repo for DINOv2
    ├── dinov2_vitb14_pretrain.pth DINOv2-B checkpoint (346 MB)
    ├── dinov2_vitl14_pretrain.pth DINOv2-L checkpoint (1.2 GB)
    ├── siglip_so400m_pg1/         Pre-extracted PaliGemma vision tower (after extraction)
    ├── siglip_so400m_pi0/         Pre-extracted π0 vision tower (after extraction)
    └── siglip_so400m_pi05/        Pre-extracted π0.5 vision tower (after extraction)
```

---

## 3. How the Pipeline Works

### 3.1 End-to-End Training Flow

```
Config YAML
    │
    ▼
LinearProbeExperiment.__init__()
    ├── Instantiate backbone (frozen, no grad)
    ├── Build train/val/test datasets from UMD split JSON
    ├── Infer feature shape (one forward pass on first batch)
    └── Build probe head (BatchNorm2d + 1×1 Conv2d)
    │
    ▼
experiment.train()
    ├── For each (lr, wd) in grid:
    │     ├── Fresh head + Adam optimizer
    │     ├── For each epoch (max_epochs=2):
    │     │     ├── Forward: backbone(images) → features dict
    │     │     ├── Fuse: concat 4 layers → (B, C_fused, H_patch, W_patch)
    │     │     ├── Head: BN → 1×1 Conv → logits at patch grid
    │     │     ├── Loss: CrossEntropy on patch-level targets
    │     │     ├── Upsample logits → pixel space → confusion matrix → mIoU
    │     │     └── Val eval every eval_interval epochs
    │     └── Early stopping on val mIoU (patience=1)
    └── Evaluate best checkpoint on test split → save summary JSON
```

### 3.2 Patch-Grid Probing (Not Pixel-Level)

This pipeline operates at **patch-grid resolution**, not pixel resolution. This is a critical distinction:

- **Ground truth:** UMD masks are downsampled from pixel space to a patch grid via `downsample_affordance_mask()`. Each `patch_size × patch_size` block is assigned the majority class, but only if that class covers ≥55% of valid pixels in the block. Otherwise the patch is marked `ignore_index=255` and excluded from loss and mIoU.

- **Prediction:** The probe head outputs logits at the patch grid (e.g., 16×16 for 224×224 images with patch_size=14). These logits are bilinearly upsampled to pixel space only for metric computation.

- **Why patch-grid:** The downstream consumer of these features (PaliGemma's LLM, π0's action expert) cross-attends to patch tokens at patch resolution. Measuring at patch resolution is closer to what the VLA pipeline actually sees. A pixel-level decoder could mask differences in raw feature quality.

- **Consequence:** Our mIoU numbers are NOT comparable to pixel-level UMD segmentation papers. Only to other patch-grid probing results computed with this exact protocol.

### 3.3 Multi-Layer Feature Fusion

The `MultiLayerLinearHead` concatenates features from 4 equally-spaced intermediate layers:

1. Each hooked layer produces a tensor of shape `(B, C, H_patch, W_patch)`
2. All layers are bilinearly resized to match the **primary layer's** spatial grid (primary = first hooked layer, per Zhang et al.'s convention)
3. Concatenated along channel dimension → `(B, 4*C, H, W)`
4. BatchNorm2d → 1×1 Conv2d → `(B, num_classes, H, W)`

| Encoder | Layers Hooked | Hidden Dim | Fused Dim |
|---------|--------------|-----------|-----------|
| DINOv2-B/14 (12 blocks) | [2, 5, 8, 11] | 768 | 3,072 |
| DINOv2-L/14 (24 blocks) | [5, 11, 17, 23] | 1,024 | 4,096 |
| SigLIP-So400m/14 (27 blocks) | [6, 13, 20, 26] | 1,152 | 4,608 |

### 3.4 UMD Dataset

17 tool categories (bowl, cup, hammer, knife, ladle, mallet, mug, pot, saw, scissors, scoop, shears, shovel, spoon, tenderizer, trowel, turner) with ~2,500 frames total.

**8 affordance classes** (mapped from per-pixel labels):
Classes include grasp, cut, scoop, contain, pound, support, wrap-grasp, and a background class. `metric_ignore_indices: [0]` excludes background from mIoU.

**Split:** `category_split_seed42_v20.json` — train/val/test by tool category (not by instance), ensuring the probe generalizes across unseen tools.

**Images:** ~480×640 RGB JPEGs. Masks in `.mat` files.

---

## 4. What We Adapted

### 4.1 New Backbone: `SigLIPSo400mBackbone`

**File:** `src/models/siglip_so400m.py`

Wraps the SigLIP-So400m/14 vision encoder for dense patch token extraction. Supports loading from 4 sources, all sharing the same architecture (ViT-So400m/14, 27 layers, 1152-dim):

| Source | Checkpoint | What it represents |
|--------|-----------|-------------------|
| `raw` | `google/siglip-so400m-patch14-384` | Contrastive-only pretraining |
| `pg1` | Pre-extracted from PaliGemma-3B | + multimodal stage-1 at 224×224 |
| `pi0` | Pre-extracted from π0 policy | + VLA flow-matching fine-tuning |
| `pi05` | Pre-extracted from π0.5 policy | + knowledge-insulated VLA training |

**Normalization:** Input arrives ImageNet-normalized (from the shared transform pipeline). The backbone denormalizes back to [0, 1] pixel space, then renormalizes with SigLIP-specific stats read from the canonical processor (`google/siglip-so400m-patch14-384`).

**Positional encoding:** Uses `interpolate_pos_encoding=True` to handle input resolutions different from the model's native training resolution (384 for raw, 224 for VLA variants).

**CLS token:** Stripped if present (seq_len == expected_tokens + 1).

### 4.2 Dataset Resize Support

**File:** `src/data/dataset.py`

Added `target_image_size` parameter to `UMDAffordanceDataset`. When set (e.g., `[224, 224]`), images are resized before patch-multiple padding:

- **Images:** Bicubic interpolation (`cv2.INTER_CUBIC`) — higher quality for downscaling
- **Masks:** Nearest-neighbor interpolation (`cv2.INTER_NEAREST`) — preserves class label integrity

This resize happens in `__getitem__` after label remapping and before `_pad_to_patch_multiple`.

### 4.3 Relaxed Image Size Validation

**File:** `src/engine/trainer.py`, `_infer_feature_shape()`

Zhang et al.'s original code rejected any image not `(3, 480, 640)`. We relaxed this to accept any `(B, 3, H, W)` input where H and W are divisible by `patch_size` (or `pad_to_patch_multiple` is enabled).

### 4.4 Trainer Model Branch

**File:** `src/engine/trainer.py`

Added `elif self.model_target == "siglip_so400m":` branch that:
1. Uses the standard ImageNet transform (shared with DINOv2/SigLIP2)
2. Passes config params to `SigLIPSo400mBackbone`
3. Sets `self.patch_size = 14` and `self.target_layer` from the backbone

### 4.5 Seed Override

**File:** `scripts/train.py`

Added `--seed` CLI argument. When provided, overrides `config["training"]["seed"]` after config loading. Enables multi-seed runs without duplicating configs:

```bash
python scripts/train.py --config configs/dinov2_b14_res224.yaml --seed 42
```

### 4.6 Pre-Extraction Script

**File:** `scripts/extract_vision_towers.py`

One-time script that extracts the SigLIP vision tower from each parent model (PaliGemma, π0, π0.5) and saves as standalone HuggingFace checkpoints to `models/siglip_so400m_{variant}/`. This avoids loading full 4-6 GB parent models on every probe training run.

After extraction, configs point to local dirs with `source: raw, model_id: ../../models/siglip_so400m_{variant}` — the backbone simply calls `SiglipVisionModel.from_pretrained(local_path)`.

### 4.7 Encoder Validation Script

**File:** `scripts/validate_encoders.py`

Loads all 4 SigLIP variants, runs forward on 3 UMD test images, prints a pairwise L2-distance matrix of last-layer features. Catches the critical risk that two extracted checkpoints are secretly identical (off-diagonal distance = 0).

### 4.8 Geometry Disabled

All our configs explicitly set `use_depth: false, use_normal: false`. Zhang et al.'s geometry augmentation (Metric3D-predicted depth/normals) is out of scope for Phase 1 — we are probing the encoder's intrinsic representations, not augmented representations. The Metric3D prediction files do not exist on disk in our setup.

Zhang et al.'s original configs (`dinov2.yaml`, `siglip.yaml`) remain untouched with geometry enabled.

---

## 5. Configuration Files

All 13 experiment configs live in `configs/`. Naming convention: `{encoder}_{resolution}.yaml`.

### 5.1 DINOv2 Configs (4 files)

| Config | Model | Resolution | Layers | Notes |
|--------|-------|-----------|--------|-------|
| `dinov2_b14_resumd.yaml` | ViT-B/14 (86M) | UMD native | [2,5,8,11] | **Validation gate** — must produce reasonable mIoU |
| `dinov2_b14_res224.yaml` | ViT-B/14 (86M) | 224×224 | [2,5,8,11] | "Dominate from below" — if 86M DINO beats 400M SigLIP, capacity argument dies |
| `dinov2_l14_resumd.yaml` | ViT-L/14 (300M) | UMD native | [5,11,17,23] | Scale-matched reference |
| `dinov2_l14_res224.yaml` | ViT-L/14 (300M) | 224×224 | [5,11,17,23] | Scale-matched reference at fair resolution |

### 5.2 SigLIP-So400m Configs (9 files)

| Config | Source | Resolution | Model ID |
|--------|--------|-----------|----------|
| `siglip_so400m_raw_resumd.yaml` | raw | UMD native | `google/siglip-so400m-patch14-384` |
| `siglip_so400m_raw_res224.yaml` | raw | 224×224 | `google/siglip-so400m-patch14-384` |
| `siglip_so400m_raw_res384.yaml` | raw | 384×384 | `google/siglip-so400m-patch14-384` |
| `siglip_so400m_pg1_resumd.yaml` | pg1 | UMD native | `../../models/siglip_so400m_pg1` |
| `siglip_so400m_pg1_res224.yaml` | pg1 | 224×224 | `../../models/siglip_so400m_pg1` |
| `siglip_so400m_pi0_resumd.yaml` | pi0 | UMD native | `../../models/siglip_so400m_pi0` |
| `siglip_so400m_pi0_res224.yaml` | pi0 | 224×224 | `../../models/siglip_so400m_pi0` |
| `siglip_so400m_pi05_resumd.yaml` | pi05 | UMD native | `../../models/siglip_so400m_pi05` |
| `siglip_so400m_pi05_res224.yaml` | pi05 | 224×224 | `../../models/siglip_so400m_pi05` |

The `res384` config is an ablation: raw SigLIP at its native pretraining resolution, to confirm the raw baseline isn't penalized by downscaling to 224.

### 5.3 Shared Training Settings

All configs use identical training hyperparameters (matching Zhang et al.):

```yaml
training:
  batch_size: 4
  max_epochs: 2
  patience: 1            # early stopping after 1 epoch with no val improvement
  lr_grid: [0.001]
  weight_decay_grid: [0.01]
  precision: bf16
  seed: 1337             # override with --seed for multi-seed runs
```

---

## 6. The Experimental Matrix

### 6.1 Encoders (6 total)

| Code | Params | Hidden Dim | Patch | Training Objective |
|------|--------|-----------|-------|--------------------|
| DINOv2-B/14 | 86M | 768 | 14 | iBOT (self-supervised, dense) |
| DINOv2-L/14 | 300M | 1,024 | 14 | iBOT (self-supervised, dense) |
| SigLIP-So400m (raw) | 400M | 1,152 | 14 | Contrastive (image-text) |
| SigLIP-So400m (PG1) | 400M | 1,152 | 14 | + multimodal stage-1 |
| SigLIP-So400m (π0) | 400M | 1,152 | 14 | + VLA flow-matching |
| SigLIP-So400m (π0.5) | 400M | 1,152 | 14 | + knowledge-insulated VLA |

### 6.2 Resolutions (2 + 1 ablation)

| Resolution | Patch Grid | Purpose |
|-----------|-----------|---------|
| 224×224 | 16×16 (256 tokens) | **Primary/headline** — fairness-matched, all encoders near native |
| UMD native (~480×640) | 35×46 (1,610 tokens) | **Replication** — validates against Zhang et al.'s protocol |
| 384×384 (raw only) | 28×28 (784 tokens) | **Ablation** — raw SigLIP at native pretraining resolution |

### 6.3 Run Matrix

6 encoders x 2 resolutions + 1 ablation = **13 probe training runs** (single seed).
For final numbers: 13 x 3 seeds (1337, 42, 2024) = **39 runs** total.

Each run takes a few minutes on a 5090.

### 6.4 Fairness Summary

**Asymmetries favoring SigLIP (a SigLIP loss is therefore robust):**
- 33% more parameters than DINOv2-L (400M vs 300M)
- ~70x more training data (10B image-text pairs vs 142M images)
- Wider features (1152 vs 1024)

**Asymmetries favoring DINOv2:**
- iBOT objective directly trains patch-level features — sympathetic to the probe
- LVD-142M curation aligns better with UMD's kitchen-tool distribution than WebLI
- Register tokens (if used) clean up patch artifacts

**The "dominate from below" argument:** If DINOv2-B (86M) beats SigLIP-So400m (400M), the capacity counter-argument collapses. A reviewer cannot say "SigLIP lost because it had less capacity" when the DINO model that beat it had 4.5x fewer parameters.

**The within-SigLIP trajectory** (raw -> PG1 -> pi0 -> pi0.5) holds all cross-architecture asymmetries constant and is the cleanest measurement regardless of how the DINOv2 comparison shakes out.

---

## 7. Next Steps: Running the Experiments

### Step 0: Environment Setup

Activate the appropriate conda environment with dependencies:
- `torch`, `torchvision`
- `transformers` (for SigLIP loading)
- `lerobot` (for pi0/pi0.5 extraction only)
- `scipy` (for loading .mat label files)
- `opencv-python` (for image/mask resizing)

Verify the UMD dataset exists at `datasets/UMD/part-affordance-dataset/tools/`.

### Step 1: Download DINOv2-B Checkpoint

If not already present:
```bash
cd Probing_Bridging_Affordance/models/
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

DINOv2-L checkpoint (`dinov2_vitl14_pretrain.pth`) is already downloaded.

### Step 2: Extract Vision Towers

One-time extraction of PaliGemma, pi0, and pi0.5 vision towers:

```bash
cd Probing_Bridging_Affordance/geometry_probing/umd_linear_probing/
python scripts/extract_vision_towers.py
```

This creates:
- `models/siglip_so400m_pg1/` (~1.5 GB)
- `models/siglip_so400m_pi0/` (~1.5 GB)
- `models/siglip_so400m_pi05/` (~1.5 GB)

### Step 3: Validate Encoders

Confirm all 4 SigLIP variants produce distinct features:

```bash
python scripts/validate_encoders.py
```

Expected output: a 4x4 L2-distance matrix with zeros on the diagonal and nonzero off-diagonal entries. If any off-diagonal is zero, an extraction failed.

### Step 4: Validation Gate

Run DINOv2-B/14 at UMD native resolution to establish baseline and validate pipeline correctness:

```bash
python scripts/train.py --config configs/dinov2_b14_resumd.yaml
```

Check `outputs/dinov2_b14_resumd_*/` for results. This establishes our backbone-only baseline (no geometry augmentation). Note: our number will be lower than Zhang et al.'s ~0.670 since they used geometry augmentation which we've disabled.

**Do not proceed until this run converges to a reasonable mIoU.** If the training loss doesn't decrease or mIoU is near zero, debug before running any other config.

### Step 5: 224x224 Baseline

```bash
python scripts/train.py --config configs/dinov2_b14_res224.yaml
```

This verifies the resize pipeline works and establishes the resolution delta (how much mIoU changes between UMD native and 224x224).

### Step 6: DINOv2-L Reference Points

```bash
python scripts/train.py --config configs/dinov2_l14_resumd.yaml
python scripts/train.py --config configs/dinov2_l14_res224.yaml
```

These are the reference points all SigLIP runs get compared against.

### Step 7: SigLIP Trajectory

Run raw SigLIP first (includes the 384 ablation):
```bash
python scripts/train.py --config configs/siglip_so400m_raw_resumd.yaml
python scripts/train.py --config configs/siglip_so400m_raw_res224.yaml
python scripts/train.py --config configs/siglip_so400m_raw_res384.yaml
```

Then the trajectory (PG1 -> pi0 -> pi0.5):
```bash
python scripts/train.py --config configs/siglip_so400m_pg1_res224.yaml
python scripts/train.py --config configs/siglip_so400m_pg1_resumd.yaml
python scripts/train.py --config configs/siglip_so400m_pi0_res224.yaml
python scripts/train.py --config configs/siglip_so400m_pi0_resumd.yaml
python scripts/train.py --config configs/siglip_so400m_pi05_res224.yaml
python scripts/train.py --config configs/siglip_so400m_pi05_resumd.yaml
```

### Step 8: Multi-Seed Replication

For final reported numbers, re-run each of the 13 configs with seeds 42 and 2024:

```bash
for config in configs/*.yaml; do
    # Skip Zhang et al.'s original configs
    case "$config" in
        *dinov2.yaml|*dinov2.local*|*dinov3*|*dino.yaml|*clip*|*sam*|*sd21*|*siglip.yaml) continue ;;
    esac
    python scripts/train.py --config "$config" --seed 42
    python scripts/train.py --config "$config" --seed 2024
done
```

Report mean +/- std over the 3 seeds (1337, 42, 2024).

### Step 9: Analysis

Results for each run are saved in `outputs/{output_dir_name}_{timestamp}/`:
- `summary.json` — final test mIoU, best val mIoU, hyperparameters
- `linear_probe.test_metrics.json` — per-class IoU breakdown
- `training_history.json` — epoch-by-epoch loss and mIoU curves

**Primary deliverables:**
1. **Table 1 (res_224):** 6 encoders x mIoU (mean +/- std), param count, training objective
2. **Table 2 (res_umd_native):** Same rows, robustness check at Zhang et al.'s original resolution
3. **Within-SigLIP trajectory figure:** x = training stage (raw -> PG1 -> pi0 -> pi0.5), y = mIoU
4. **Validation-gate confirmation:** DINOv2-B/14 at UMD native produces reasonable mIoU

---

## 8. Key Files Reference

| File | Purpose |
|------|---------|
| `src/models/siglip_so400m.py` | SigLIP-So400m backbone wrapper (our addition) |
| `src/models/dinov2.py` | DINOv2 backbone wrapper (Zhang et al.'s) |
| `src/models/linear_head.py` | MultiLayerLinearHead probe architecture |
| `src/data/dataset.py` | UMDAffordanceDataset with resize support |
| `src/engine/trainer.py` | LinearProbeExperiment orchestrator |
| `src/engine/eval.py` | evaluate_linear_probe() function |
| `src/utils/config.py` | Config loading with path resolution |
| `src/utils/metrics.py` | IoU computation from confusion matrix |
| `scripts/train.py` | Training entry point (--config, --seed) |
| `scripts/eval.py` | Evaluation entry point |
| `scripts/extract_vision_towers.py` | One-time tower extraction |
| `scripts/validate_encoders.py` | L2-distance encoder validation |
