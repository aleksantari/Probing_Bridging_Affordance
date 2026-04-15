# geometry_probing — Agent Guide

Our adaptation of the **geometry-side** linear probe from *Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models* (Zhang et al.), repurposed to characterize VLA vision towers rather than survey generic VFMs. Freezes a vision encoder, trains a lightweight `BatchNorm2d → 1×1 Conv` head on the UMD part-affordance dataset, and reports mIoU **at patch-grid resolution** (not pixel space).

The paper's original goal was probing geometry vs. interaction cues across a zoo of foundation models. **Our goal** is narrower: use the same probing methodology to measure what happens to a SigLIP-So400m vision tower as it is fine-tuned into VLA policies. Current focus is **Phase 1**: comparing DINOv2-B/L (as a non-VLA reference) against the 4-point SigLIP-So400m trajectory (raw → PaliGemma PG1 → π0 → π0.5) at res224 and UMD-native resolutions. The upstream repo's interaction-side probing and fusion zero-shot pipelines live in sibling directories (`interaction_probing/`, `fusion_zero_shot/`) and are out of scope here.

## Read these before touching anything

Do not duplicate what these already cover — index into them instead.

- [umd_linear_probing/PHASE1_ZHANG_ADAPTATION_PLAN.md](umd_linear_probing/PHASE1_ZHANG_ADAPTATION_PLAN.md) — research plan, epistemology, encoder & resolution rationale, execution phases, risk register.
- [umd_linear_probing/guide.md](umd_linear_probing/guide.md) — pipeline walkthrough, env setup, tower extraction, per-file reference table.
- [umd_linear_probing/results.md](umd_linear_probing/results.md) — current numbers, pre/post-fix comparison, per-class IoU, interpretation.

## Layout

```
geometry_probing/
├── train.py / eval.py              # thin wrappers → umd_linear_probing/scripts/
└── umd_linear_probing/
    ├── configs/                    # 13 Phase-1 YAMLs
    ├── scripts/                    # train.py, eval.py, extract_vision_towers.py, validate_encoders.py
    ├── metadata/splits/            # category_split_seed42_v20.json
    ├── outputs/                    # {model_target}_{resolution}_{timestamp}/
    └── src/
        ├── data/                   # UMDAffordanceDataset, patch-grid downsampling, collate
        ├── engine/                 # LinearProbeExperiment (trainer.py), evaluate_linear_probe
        ├── models/                 # per-backbone wrappers + MultiLayerLinearHead
        ├── utils/                  # config loader, logging, metrics, seed
        └── visualization/
```

## Running experiments

- Preferred launcher: `python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/<name>.yaml`
- Direct: `cd geometry_probing/umd_linear_probing && python scripts/train.py --config configs/<name>.yaml [--seed 42]`
- Eval an existing checkpoint: `python scripts/eval.py --config ... --checkpoint ...`
- **Environment**: activate with `use_conda <env>` from the lane shell — never `conda activate` directly. Hardware target: RTX 5090 (32 GB), 9950X (32 threads), 60 GiB RAM. Default `batch_size: 4` is calibrated for the frozen-backbone regime; tune `num_workers` if dataloader-bound.

## Config convention

- Naming: `{encoder_code}_{resolution_code}.yaml` (e.g. `dinov2_b14_res224.yaml`, `siglip_so400m_pi0_resumd.yaml`).
- Paths inside configs resolve **relative to the config file's parent directory** (see [umd_linear_probing/src/utils/config.py](umd_linear_probing/src/utils/config.py)).
- Key fields: `dataset.{root, split_path, patch_size, num_classes}`, `model.{target, params.layers_to_hook, params.primary_layer}`, `training.{batch_size, max_epochs, patience, lr_grid, weight_decay_grid, precision, seed, output_root}`.
- Defaults that matter: `precision: bf16`, `seed: 1337`, `max_epochs: 2`, `patience: 1`, `lr=1e-3`, `wd=1e-2`.

## Output layout

`outputs/{model_target}_{resolution}_{timestamp}/`
- `master.log`, `summary.json`, `training_history.json`
- optional `val_examples.pt`, `test_examples.pt`
- `lr{..}_wd{..}/` — per-hyperparam subdir with `linear_probe.pth`, `train.log`, `config_snapshot.yaml`

## Non-obvious things that bite people

These are the details a fresh session will not infer from code alone.

1. **Metrics are at patch-grid resolution, not pixel space.** Ground-truth masks are majority-voted onto the patch grid in [umd_linear_probing/src/data/dataset.py](umd_linear_probing/src/data/dataset.py) (`downsample_affordance_mask`). Patches with <55% coverage of the winning class → `ignore_index=255` (excluded from loss and mIoU). **Do not** compare these numbers to pixel-level UMD probing papers.

2. **Primary layer is the *first* hooked layer, not the last.** Feature fusion bilinearly resizes all hooked layers to the primary layer's spatial grid before concat. Zhang et al.'s convention; preserved deliberately. See `guide.md` §3.3.

3. **LayerNorm on intermediate features matters.** DINOv2 features are extracted with `norm=True`. SigLIP features must have `vision_model.post_layernorm` applied per hooked layer — earlier runs missed this and produced unstable rankings (see `results.md` §7, §10).

4. **Rankings differ between res224 and resumd — cause is not established.** Observation: SigLIP variant orderings are not consistent across the two resolutions, and raw SigLIP has the largest resolution delta. `results.md` §12 proposes a positional-encoding-asymmetry story (raw native=384, PG1/π0/π0.5 native=224), but **this has not been isolated experimentally.** The one run that would have tested it — raw SigLIP at res384 — crashed during feature-shape inference and was never recovered (§1, §10). Until that run exists (and ideally multi-seed), the PE hypothesis is a post-hoc narrative, not a finding. Do not repeat the causal framing in new write-ups without flagging it as unverified.

5. **Single-seed results; no noise floor.** All current numbers are seed=1337, n=1. Inter-model gaps in the 0.1–3pp range (e.g. π0 vs DINOv2-L at res224 differs by **0.001**, "PG1 ≥ raw" by 0.9pp at resumd) are reported in `results.md` as if they rank models, but we have no variance estimate to distinguish them from seed jitter. The training regime (`max_epochs=2, patience=1`) makes ep1/ep2 val swings comparable to the inter-model deltas being interpreted. Treat any SigLIP-trajectory ordering as provisional until multi-seed (42, 2024) is run.

6. **Val/test discrepancy is unexplained.** Several runs have test mIoU *much higher* than val mIoU (e.g. DINOv2-L resumd: val 0.521, test 0.679). This is split-distribution behavior we haven't characterized and it weakens any claim that leans on absolute mIoU.

7. **Patch-coverage threshold (55%) was not ablated.** Changing the `downsample_affordance_mask` coverage threshold moves the mIoU denominator (how many patches are kept vs. marked `ignore_index=255`), and we don't know how sensitive rankings are to it.

8. **What `results.md` currently supports vs. overclaims.** Defensible: DINOv2-B/14 at resumd hits 0.666, passing the Zhang et al. validation gate of ~0.670 — the pipeline reproduces the reference number. Not yet defensible (despite being stated in §16–§17): "PG1 ≥ raw everywhere", "π0.5 ≤ π0 everywhere", "VLA fine-tuning helps at res224", "DINOv2 dominates from below". These are single-seed readings dressed up as trends.

9. **Phase 1 explicitly disclaims proxy validity.** Affordance mIoU is **not** a proxy for downstream VLA policy performance — see the plan doc. Frame results as upstream encoder characterization only, and even that only after the above caveats are addressed.

6. **Feature shapes are inferred at runtime** via one forward pass inside `LinearProbeExperiment.__init__` — no need to hand-specify head dimensions in configs.

7. **SigLIP-So400m PG1/π0/π0.5 towers are extracted offline** via `scripts/extract_vision_towers.py` from PaliGemma / LeRobot π0 / π0.5 parent checkpoints. `scripts/validate_encoders.py` asserts the four SigLIP variants produce pairwise-distinct features via an L2-distance matrix — run it after any re-extraction.

8. **Duplicate plan file at the repo root.** `geometry_probing/PHASE1_ZHANG_ADAPTATION_PLAN.md` (currently untracked) is a working copy; the canonical version lives under `umd_linear_probing/`. Prefer the canonical one.

## Conventions

- Default seed `1337`; multi-seed runs (`42`, `2024`) deferred to final reporting.
- Logging via `create_logger()` in `src/utils/logging.py` — dual console/file handlers, written into the experiment output dir.
- mIoU in `src/utils/metrics.py` **excludes background class 0** (classes 1–7 only).
