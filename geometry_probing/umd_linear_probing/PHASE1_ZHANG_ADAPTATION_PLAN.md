# Phase 1: Affordance Probing Adaptation Plan

Adapting Zhang et al.'s probing repository (`QZhang2111/Probing_Bridging_Affordance`) to characterize how geometric affordance representations evolve through the SigLIP → PaliGemma → π0 → π0.5 pipeline.

---

## 1. Research context and epistemic framing

### 1.1 The long-term hypothesis

The overarching research program tests whether **affordance perception — decomposed into geometric perception and interaction perception (Zhang et al., arXiv 2602.20501) — is causally related to action generation quality in generalist robotic policies**, and whether the dominant VLA stack (SigLIP → PaliGemma → action expert) is leaving substantial downstream performance on the table because its visual front-end suppresses affordance-relevant features.

This hypothesis has multiple load-bearing claims that must be tested independently:

1. **Architectural claim** (well-supported, not our contribution): Contrastive objectives are mechanically incapable of supervising fine-grained spatial structure in patch tokens. VLM backbones (Fu et al., "Hidden in Plain Sight," COLM 2025) further degrade spatial features through the LLM. VLAs inherit these degraded features.
2. **Upstream empirical claim** (what Phase 1 addresses): Measurable affordance representation quality differs between encoder families, and VLA training measurably shifts SigLIP's affordance representations in some direction.
3. **Proxy validity claim** (unproven, deferred to Phase 2+): Affordance probing scores correlate with downstream policy performance on real manipulation tasks.
4. **Intervention claim** (deferred to full paper): Architectural changes to the visual front-end (e.g., dual-stream geometric + interaction world model) improve downstream policy performance.

### 1.2 What Phase 1 is and is not

**Phase 1 is an evidence-accumulation step, not a proof.** It addresses claim (2) only. It does not attempt to prove claim (3) or (4). The deliverable is a characterization of feature evolution along the SigLIP → PaliGemma → π0 → π0.5 trajectory, with DINOv2 as an external reference point.

**Claims Phase 1 can support:**
- At comparable parameter scale and identical patch geometry, encoders trained with different objectives produce measurably different geometric affordance representations on UMD.
- VLA training (π0/π0.5 fine-tuning) shifts SigLIP's affordance representations relative to the raw checkpoint in a measurable direction.
- The observed gaps/shifts are not trivially explained by parameter count, training data scale, or patch geometry.

**Claims Phase 1 cannot support:**
- That affordance probing scores predict downstream manipulation performance.
- That SigLIP is the "wrong" encoder for VLAs.
- That swapping encoders would improve π0/π0.5.
- Any causal claim about what a hypothetical better encoder would enable.

### 1.3 The honest framing paragraph for the writeup

> We probe geometric affordance representations across a controlled set of vision encoders to characterize how training objective and VLA fine-tuning affect dense feature quality. Within-architecture, we measure how π0 and π0.5 fine-tuning shifts SigLIP's affordance representations relative to the raw checkpoint. Cross-architecture, we compare against frozen DINOv2 at two scales as an external reference. We deliberately grant the contrastive models capacity and data advantages — SigLIP-So400m has 33% more parameters than DINOv2-L and was trained on roughly 70× more images — so that any negative result is robust to the standard counter-arguments. We do not claim that affordance probe scores directly predict downstream policy performance; we treat this as upstream evidence motivating downstream investigation in subsequent work.

### 1.4 What a hostile reviewer would say, and our preemptive response

**Anticipated attack:** "UMD mIoU is a static segmentation metric and does not predict manipulation success. π0 and π0.5 achieve frontier results on real manipulation despite the SigLIP features this paper characterizes as deficient, suggesting the deficit does not bottleneck downstream performance. The paper provides no downstream evidence."

**Response:** We explicitly scope the contribution as upstream characterization and do not make downstream claims. The within-SigLIP trajectory (raw → π0 → π0.5) is a clean within-architecture measurement that is informative regardless of proxy validity. The DINOv2 comparison is a reference point, not the load-bearing result. Downstream validation is Phase 2 work.

This framing is not optional. It must be in the introduction, the results section, and the limitations.

---

## 2. Key concepts and decisions from the design conversation

### 2.1 What Zhang et al.'s repo actually does (verified by code inspection)

- **Probe architecture**: `MultiLayerLinearHead` = `BatchNorm2d + 1×1 Conv2d(C_fused → num_classes)`. No nonlinearities, no decoder.
- **Feature fusion**: four equally-spaced intermediate layers are hooked, each bilinearly resized to the "primary" layer's spatial grid, concatenated along channels, batch-normed, and projected. For DINOv2-B/SigLIP-base (12 blocks), layers `[2, 5, 8, 11]` are used with layer 2 as primary. For DINOv3-7B (40 blocks), layers `[9, 19, 29, 39]` with layer 9 as primary.
- **Primary = first layer** (unusual): the probe's spatial grid is sized to the *first* hooked layer, not the last. Every other layer is resized to match.
- **Probe operates at patch-grid resolution, not pixel resolution**: ground-truth masks are **downsampled** from pixel space to a patch grid via `downsample_affordance_mask`, which assigns each `patch_size × patch_size` block the class that dominates it, but only if that class covers ≥55% of the valid pixels in the block (otherwise the patch is marked `ignore_index` and excluded from loss and mIoU). **This is not pixel-level segmentation.** Loss and mIoU are computed at the patch grid (e.g. 16×16 or 24×24).
- **No global image resize**: images are loaded at UMD native resolution (~640×480), optionally padded to a patch-size multiple. Encoders are run at this high resolution via positional-embedding interpolation. The repo's transforms do only `ToTensor()` + ImageNet normalization — no Resize anywhere.
- **Training**: 2 epochs, batch 4, Adam LR 1e-3, weight decay 0.01, bf16. Single LR/WD point in grid. Single seed (1337). A full probe trains in a few minutes on a 5090.
- **Shipped SigLIP config uses `google/siglip-base-patch16-384`** — base-sized, patch 16. The class default is `siglip2-giant-opt-patch16-384`. **Neither of these is what π0/π0.5 use.** π0/π0.5 use SigLIP-So400m/14, which none of Zhang et al.'s shipped configs cover. We will need to write a new config and confirm the SigLIP2 wrapper works with patch-14 So400m.

### 2.2 Why patch-grid probing is the right choice for our question

Patch-grid probing measures the intrinsic spatial information content of the encoder's patch tokens, without a bilinear-upsampling decoder laundering the features. This is **better aligned with our research question than pixel-level probing** because:

- The downstream consumer (PaliGemma's LLM, π0's action expert) cross-attends to patch tokens at patch resolution. It does not upsample them to pixel space. Measuring at patch resolution is closer to what the VLA pipeline actually sees.
- The pixel-level probing protocol's bilinear upsampler is a zero-parameter decoder that can mask differences in raw feature quality. Patch-grid probing removes this confound.
- Boundary-patch exclusion (≥55% coverage threshold) prevents the probe from being rewarded for guessing on ambiguous patches, which again aligns with measuring intrinsic information content.

**Consequence:** Our numbers are NOT directly comparable to pixel-level UMD segmentation papers. The only valid comparisons are to other patch-grid probing results computed with this exact protocol. This must be explicit in the writeup.

### 2.3 The resolution decision

**Verified facts from the conversation:**

- `google/siglip-so400m-patch14-384` was originally pretrained at 384×384.
- PaliGemma stage 1 further trains SigLIP-So400m at **224×224** on multimodal data (1B examples). The PaliGemma-pt-224 checkpoint contains a SigLIP that has been adapted to 224 through this process.
- π0 and π0.5 both apply `ResizeImages(224, 224)` to all camera inputs (verified in `openpi/src/openpi/training/config.py`). They inherit PaliGemma-pt-224's SigLIP and further fine-tune it at 224.
- **DINOv2 (all sizes) was pretrained at 224×224.**

Therefore, **the vast majority of encoders we care about are natively 224**, with only the pre-PaliGemma raw SigLIP-So400m checkpoint being natively 384.

**Decision: run every encoder at two resolutions** (cheap enough that picking one is the wrong call):

1. **Primary resolution: 224×224** (Table 1, headline numbers). Every encoder at or near its training distribution. Fairness-matched. Resize UMD images with bicubic, masks with nearest-neighbor, then apply patch-multiple padding.
2. **Replication resolution: UMD native (~640×480)** (Table 2, robustness check). Reproduces Zhang et al.'s exact protocol. Required for the validation gate: DINOv2-B/14 at UMD native must hit ~0.670 mIoU. Without this, our pipeline is unvalidated.
3. **Optional: raw SigLIP-So400m at 384×384** — one ablation run, confirms the raw SigLIP baseline isn't penalized by being downscaled to 224.

**Rhetorical ordering:** Lead the paper with Table 1 (224-matched fair comparison), present Table 2 (UMD-native replication) as the robustness check. This preempts the "you picked a resolution that favors your conclusion" objection at the start, before it comes up.

### 2.4 The SigLIP lineage — there are four distinct checkpoints, not two

A subtle point that changes the experimental design: "the SigLIP inside π0" is not the same as "the SigLIP checkpoint on HuggingFace." There are four distinct stages:

| Stage | Checkpoint | Training history |
|---|---|---|
| Raw SigLIP | `google/siglip-so400m-patch14-384` | Contrastive only, WebLI ~10B pairs, 384 |
| PaliGemma stage 1 | Extract from `google/paligemma-3b-pt-224` | + multimodal pretraining on 1B examples at 224 |
| π0-SigLIP | Extract from `lerobot/pi0_base` | + VLA flow-matching fine-tuning at 224 |
| π0.5-SigLIP | Extract from `lerobot/pi05_base` | + knowledge-insulated VLA training at 224 |

**Running all four gives a 4-point trajectory** rather than a 2-point comparison. This lets us separately measure what PaliGemma-stage-1 training and what VLA training each do to the SigLIP representations. This is a meaningfully stronger experimental design than just raw-vs-VLA, and it directly connects to the Fu et al. "Hidden in Plain Sight" finding (which is about the PaliGemma stage specifically).

**All four stages must be in the experiment.** Skipping PaliGemma-stage-1 confounds "PaliGemma training" with "VLA training" in any observed delta.

### 2.5 The fairness story, stated honestly

The "SigLIP-So400m vs DINOv2-L" comparison is **not** fair in any strict sense. Specifically:

**Asymmetries that favor SigLIP** (a SigLIP loss is therefore robust to them):
- 33% more parameters (400M vs 300M)
- Wider features (1152 vs 1024 hidden dim) → larger probe input and more probe capacity
- ~70× more training data (10B image-text pairs vs 142M images)
- More pretraining compute
- Shape-optimized architecture tuned for its objective

**Asymmetries that favor DINOv2:**
- LVD-142M's curated data distribution (ImageNet-22k-style seeds) aligns better with UMD's kitchen-tool distribution than WebLI does. This is a real confound we cannot fully eliminate.
- iBOT objective directly trains patch-level features — i.e., it explicitly optimizes the thing the probe measures. This isn't "unfair" (it's the entire point of the comparison) but the probe is sympathetic to DINO's training signal.
- Register tokens in `-with-registers` variants clean up patch artifacts that disproportionately affect dense probing.

**The defensible fairness claim is not "matched compute."** It is: *we compare encoders that produce comparable dense feature interfaces (patch-14, comparable patch grids at matched resolution, frozen backbone), and we deliberately grant the contrastive models capacity and data advantages so that a negative result is robust to those counter-arguments. The within-SigLIP trajectory holds all of these asymmetries constant and is the cleanest within-architecture measurement.*

### 2.6 Why we dominate from below, not from above

**The strongest experimental move is to add DINOv2-B/14 (86M params) to the comparison.** If DINOv2-B beats SigLIP-So400m on UMD — and Zhang et al.'s 0.670 reference number strongly suggests it will — then the capacity argument dies: a 4.5× smaller DINO model beats SigLIP. A reviewer cannot say "SigLIP lost because it had less capacity" when the DINO model that beat it had a quarter of SigLIP's parameters. This is worth more than any amount of matched-compute rhetoric.

---

## 3. Experimental matrix

### 3.1 Encoders (6 total)

| Code | Checkpoint source | Params | Hidden dim | Layers | Layers hooked |
|---|---|---|---|---|---|
| `dinov2_b14` | `facebookresearch/dinov2` hub, `dinov2_vitb14` + `.pth` | 86M | 768 | 12 | `[2,5,8,11]` |
| `dinov2_l14` | `facebookresearch/dinov2` hub, `dinov2_vitl14` + `.pth` | 300M | 1024 | 24 | `[5,11,17,23]` |
| `siglip_so400m_raw` | `google/siglip-so400m-patch14-384` (HF) | 400M | 1152 | 27 | `[6,13,20,26]` |
| `siglip_so400m_pg1` | Extracted from `google/paligemma-3b-pt-224` vision tower | 400M | 1152 | 27 | `[6,13,20,26]` |
| `siglip_so400m_pi0` | Extracted from `lerobot/pi0_base` PaliGemma vision tower | 400M | 1152 | 27 | `[6,13,20,26]` |
| `siglip_so400m_pi05` | Extracted from `lerobot/pi05_base` PaliGemma vision tower | 400M | 1152 | 27 | `[6,13,20,26]` |

**Layer selection principle:** equally-spaced quartiles ending at the last block, following Zhang et al.'s convention. Primary layer = first of four (matches the repo's behavior). For DINOv2-L with 24 blocks: `[5, 11, 17, 23]`. For SigLIP-So400m with 27 blocks: `[6, 13, 20, 26]`.

### 3.2 Resolutions

| Config | Input size | Purpose |
|---|---|---|
| `res_224` | 224×224 (resize) | Primary/headline — fairness-matched |
| `res_umd_native` | ~640×480 (patch-multiple pad) | Replication of Zhang et al., validation gate |
| `res_384` (SigLIP-raw only) | 384×384 (resize) | Ablation, native resolution of raw SigLIP |

### 3.3 Run matrix

6 encoders × 2 resolutions = 12 headline runs + 1 ablation = **13 probe training runs total**.

Plus validation gate: `dinov2_b14` at `res_umd_native` must match Zhang et al.'s ~0.670 mIoU before any other run is trusted. This is the first run to execute and the gate for everything downstream.

### 3.4 Per-run seeds and statistics

The repo defaults to a single seed (1337). For final numbers, run each cell **3 seeds** (1337, 42, 2024) and report mean ± std. This is cheap (~3× the headline cost, still an afternoon of compute) and catches high-variance cells where a single-seed result would be misleading.

Do not bother with seeds until the entire matrix has been validated single-seed. Seed replication is the last step, not the first.

---

## 4. Implementation plan



### 4.2 Dataset setup

Download UMD part-affordance dataset and place at `datasets/UMD/part-affordance-dataset/` following the repo's expected structure. Use the repo's shipped split: `metadata/splits/category_split_seed42_v20.json`.

No Metric3Dv2 depth/normal manifests are required for Phase 1 — we are not running the depth-augmentation diagnostic from Zhang et al. If we choose to add it later, we will need to generate the `metric3d_predictions.json` manifest, but this is out of scope for the core experiment.

In the configs, either provide an empty geometry manifest or set `use_depth: false` and `use_normal: false` to skip depth augmentation entirely.

### 4.3 Model setup

**DINOv2 setup:**

```bash
mkdir -p models
cd models
git clone https://github.com/facebookresearch/dinov2.git
# Download weights
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

(If the `-with-registers` variants are available and we choose to use them, also download `dinov2_vitb14_reg4_pretrain.pth` and `dinov2_vitl14_reg4_pretrain.pth`. Phase 1 default is without registers to stay strictly comparable to Zhang et al.'s reported number.)

**Raw SigLIP-So400m setup:**
**PaliGemma stage-1 SigLIP extraction:**
**π0 / π0.5 SigLIP extraction:**
we got these models work in /VLA-affordance so use this as a reference for correct setup

### 4.4 Config files to create

All configs live in `geometry_probing/umd_linear_probing/configs/` (copy the shipped `dinov2.yaml` and `siglip.yaml` as templates).

**Naming convention:** `{encoder_code}_{resolution_code}.yaml`

Required configs (12 + 1 ablation):

```
dinov2_b14_res224.yaml
dinov2_b14_resumd.yaml              # <- VALIDATION GATE: must match ~0.670
dinov2_l14_res224.yaml
dinov2_l14_resumd.yaml
siglip_so400m_raw_res224.yaml
siglip_so400m_raw_resumd.yaml
siglip_so400m_raw_res384.yaml       # <- ablation, raw SigLIP at native resolution
siglip_so400m_pg1_res224.yaml
siglip_so400m_pg1_resumd.yaml
siglip_so400m_pi0_res224.yaml
siglip_so400m_pi0_resumd.yaml
siglip_so400m_pi05_res224.yaml
siglip_so400m_pi05_resumd.yaml
```

### 4.5 Code changes needed

**1. Add a Resize transform option to the dataset pipeline.**

The repo currently has no Resize step — images go in at UMD native size. For the `res_224` and `res_384` configs, we need to resize both image and mask before patch-multiple padding.

Add a `resize_to` field to the dataset config block:

```yaml
dataset:
  resize_to: [224, 224]  # null or omit for UMD native
  ...
```

And update `UMDAffordanceDataset.__getitem__` to apply the resize before `_pad_to_patch_multiple`:
- Image: bicubic interpolation.
- Mask: nearest-neighbor interpolation (critical — bilinear corrupts class labels).

This is a ~20-line change in `src/data/dataset.py`.

**2. Verify `SigLIP2Backbone` handles patch-14 So400m correctly.**

The shipped wrapper hardcodes `self.patch_size = 16` in `__init__`. **This must be overridden from config.** The repo reads `patch_size` from the model params block but the wrapper's `__init__` ignores it. Fix: change `self.patch_size = 16` to `self.patch_size = patch_size` with `patch_size` as a constructor argument passed from the trainer.

This is the single most important code correctness issue. A forgotten fix here silently runs So400m with wrong patch geometry and produces nonsense.

**3. Handle local-path SigLIP checkpoints.**

The `SigLIP2Backbone` uses `AutoModel.from_pretrained(model_id)`, which already accepts local paths. No change needed — just point `model_id` to the extracted checkpoint directory. Verify by setting `model_id: models/siglip_so400m_pi0` in a test config and confirming it loads.

**4. Add a per-encoder numerical validation script.**

After extraction, before any probing, run a sanity script that:
- Loads each of the four SigLIP variants.
- Runs a forward pass on three fixed test images.
- Compares the L2 distance of final-layer features across variants.
- Prints a matrix. Raw-vs-raw should be 0. Raw-vs-PG1 should be nonzero. PG1-vs-π0 should be nonzero (but likely smaller). Any zero off-diagonal entry means an extraction failed or two "different" checkpoints are secretly the same file.

This is 30 lines of code and catches 90% of extraction bugs before they pollute the experiment.

### 4.6 Execution order

**Phase A — validation gate (DO NOT SKIP):**

1. Set up env + repo + UMD dataset + DINOv2-B/14 weights.
2. Run `dinov2_b14_resumd.yaml` with Zhang et al.'s exact protocol (no resize, patch-multiple padding, layers `[2,5,8,11]`, primary layer 2).
3. **Confirm test mIoU ≈ 0.670.** If off by more than a few percent, stop and debug. Do not proceed to any other runs until this validates.
4. Also run `dinov2_b14_res224.yaml` to get the 224 baseline for DINOv2-B, and note the delta between 224 and UMD-native mIoU — this characterizes how much the resolution change costs in absolute terms.

**Phase B — DINOv2 reference points:**

5. Run `dinov2_l14_res224.yaml` and `dinov2_l14_resumd.yaml`.

These are the reference points all SigLIP runs get compared against.

**Phase C — raw SigLIP:**

6. Run `siglip_so400m_raw_res224.yaml`, `siglip_so400m_raw_resumd.yaml`, `siglip_so400m_raw_res384.yaml`.
7. Before trusting results: confirm the patch-size fix in `SigLIP2Backbone` is in place (section 4.5 item 2). Check that the probe's fused input dim is 4×1152 = 4608, not 4×1152 with wrong patch arithmetic.

**Phase D — extraction and downstream SigLIPs:**

8. Run the SigLIP extraction scripts (section 4.3).
9. Run the per-encoder numerical validation script (section 4.5 item 4). Confirm L2 distances are nonzero between variants.
10. Run `siglip_so400m_pg1_res224.yaml`, `siglip_so400m_pg1_resumd.yaml`.
11. Run `siglip_so400m_pi0_res224.yaml`, `siglip_so400m_pi0_resumd.yaml`.
12. Run `siglip_so400m_pi05_res224.yaml`, `siglip_so400m_pi05_resumd.yaml`.

**Phase E — statistical replication:**

13. For any cell in the matrix that will appear in the final table, re-run with seeds 42 and 2024. Report mean ± std.

**Phase F — analysis and writeup:**

14. Generate Table 1 (res_224) and Table 2 (res_umd_native).
15. Compute the within-SigLIP trajectory: raw → PG1 → π0, raw → PG1 → π0.5. Is the feature quality monotonic? Where are the biggest deltas? Does π0.5's knowledge insulation preserve more affordance information than π0's naive fine-tuning?
16. Compute the DINO reference deltas at matched resolution.
17. Write up with the framing from section 1.3.

---

## 5. What to report

### 5.1 Primary table (headline, res_224)

Rows: the 6 encoders. Columns: mIoU (mean ± std over 3 seeds), param count, training data scale, pretraining objective.

Expected shape of result (based on Zhang et al. priors, not guaranteed):
- DINOv2-L ≳ DINOv2-B > all SigLIP variants.
- Within SigLIP: raw ≈ PG1 > π0 ≈ π0.5, OR raw > PG1 > π0 ≈ π0.5 (Fu et al.-consistent), OR some other ordering (also informative).

### 5.2 Secondary table (robustness, res_umd_native)

Same rows, same columns. Goal: show that whatever the ordering is in Table 1, it persists (or changes in an interesting way) under Zhang et al.'s original protocol.

### 5.3 Within-SigLIP trajectory figure

Line plot: x-axis = training stage (raw → PG1 → π0 → π0.5), y-axis = mIoU. One line at res_224, one at res_umd_native. This is the cleanest visualization of the within-architecture delta, which is our strongest result regardless of how the DINO comparison shakes out.

### 5.4 Validation-gate confirmation

One sentence in the methods section confirming that our DINOv2-B/14 at UMD native reproduces Zhang et al.'s reported ~0.670 mIoU within some tolerance. This is the evidence that our pipeline is correct.

### 5.5 Explicit limitations section

Must include:
- Patch-grid probing, not pixel-level segmentation. Numbers are not comparable to pixel-level papers.
- UMD is static geometric affordance only — no interaction, contact, temporal, or multi-object content.
- Proxy validity (affordance mIoU → manipulation performance) is unproven and not claimed.
- Architecture and training-data asymmetries between SigLIP and DINOv2 are noted but not controlled for; we rely on the capacity inversion (DINOv2-B beats SigLIP-So400m) and the within-SigLIP trajectory to defend the interpretation.
- iBOT is sympathetic to the probe's dense-feature metric by construction. This is the mechanism we are trying to characterize, not a bug, but must be named.

---

## 6. What this enables for Phase 2

If the Phase 1 results point in the expected direction (DINOv2 > SigLIP, VLA fine-tuning shifts SigLIP features in a measurable way), Phase 2 work becomes:

1. **Downstream sanity check.** Train a small behavior-cloning policy on a single LIBERO task with two encoder variants (one SigLIP, one DINOv2) held frozen, otherwise identical architecture. Does upstream mIoU correlate with downstream success rate? This is the proxy-validity experiment that Phase 1 cannot do.
2. **Deeper probing into the VLM.** Probe PaliGemma's LLM intermediate activations for the same affordance signal. Does the geometric information that exists in SigLIP's patch tokens survive the LLM forward pass? This connects directly to Fu et al.'s finding.
3. **Axis 2 — interaction probing.** Extend to generative models (Flux, Cosmos Predict2) using Zhang et al.'s interaction probing module. This is the other half of Zhang et al.'s affordance decomposition and is independently informative about what generative backbones get that SigLIP doesn't.
4. **Dual-stream architecture work.** Only if (1) shows the proxy has some validity should we commit to the full dual-stream world model architecture.

If Phase 1 results *don't* point in the expected direction — if SigLIP matches DINOv2, or if VLA fine-tuning recovers affordance information rather than degrading it — that is **also publishable and more interesting**, because it would be evidence against the prevailing narrative (Fu et al. + Zhang et al.) that contrastive and VLM stages erode dense features. The entire Phase 1 design treats all outcomes as informative, which is why it's a good experiment to run regardless of which direction the result points.

---

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| DINOv2-B/14 does not reproduce ~0.670 at UMD native | Entire pipeline unvalidated | Validation gate is explicit; stop and debug before any other run |
| SigLIP extraction from LeRobot yields identical weights to raw checkpoint | π0/π0.5 probes are secretly the raw SigLIP probe | L2-distance sanity check (section 4.5 item 4); must see nonzero deltas |
| `SigLIP2Backbone` patch-size bug silently uses wrong geometry | All SigLIP results are wrong | Fix explicitly before any SigLIP run; verify patch grid size in `_infer_feature_shape` log output |
| UMD mask nearest-neighbor resize not applied, bilinear corrupts labels | 224 runs produce noisy mIoU | Audit the resize code change; test with a dummy mask and assert all output values are valid class indices |
| Training-set contamination: the HF SigLIP checkpoint happens to have been tuned on UMD at some point | Inflates raw SigLIP number | Low probability; SigLIP-So400m was trained on WebLI, not on UMD-like data. Note as an assumption, do not test. |
| Primary layer = first hooked layer bias is inappropriate for our encoders | Probe spatial grid mismatch | Run one ablation with primary=last layer on DINOv2-B to confirm the headline ordering is preserved. If it flips, report both and discuss. |
| 2-epoch training is too short to reveal real differences between encoders | Underpowered comparison | The probe is a single linear layer on frozen features — 2 epochs is usually enough for convergence. If training loss is still dropping at epoch 2, extend to 5-10 epochs. Monitor loss curves on the first run. |

---

## 8. Deliverables

By end of Phase 1:

1. **Adapted repo fork** with new configs, wrapper fixes, resize support, and extraction scripts. All reproducible from a single `run.py` invocation per config.
2. **Validation-gate confirmation** (DINOv2-B/14 at UMD native reproduces ~0.670).
3. **Results matrix** (6 encoders × 2 resolutions × 3 seeds + raw-SigLIP-384 ablation).
4. **Trajectory figure** (within-SigLIP feature evolution).
5. **Class-term writeup** framed per section 1.3, with the limitations section from 5.5.
6. **Phase 2 experimental proposal** grounded in Phase 1 findings, scoped to start with the downstream sanity check (section 6, item 1).

The class-term deliverable is the writeup + repo + results. The writeup should be honest about what it does and does not show, should lead with the within-SigLIP trajectory as the strongest result, and should explicitly disclaim proxy validity while motivating Phase 2 investigation of it.
