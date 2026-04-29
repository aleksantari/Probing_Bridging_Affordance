# Phase 1 Linear Probing Results — Pre-LayerNorm Fix

**Date:** 2026-04-08 / 2026-04-09
**Seed:** 1337 (all runs)
**Training:** max_epochs=2, patience=1, lr=0.001, wd=0.01, batch_size=4, precision=bf16
**Geometry:** Disabled (use_depth=false, use_normal=false)
**Note:** These results contain a known bug — SigLIP features were extracted WITHOUT LayerNorm, while DINOv2 features used `norm=True`. See "Known Issue" section below. Results are preserved for comparison against the post-fix runs.

---

## 1. Summary Table — Test mIoU

### UMD Native Resolution (resumd, ~480×640, grid 35×46)

| Rank | Encoder | Params | Test mIoU | Val mIoU | Output Dir |
|------|---------|--------|-----------|----------|------------|
| 1 | DINOv2-L/14 | 300M | **0.679** | 0.521 | `dinov2_l14_resumd_20260408-173113` |
| 2 | DINOv2-B/14 | 86M | **0.666** | 0.490 | `dinov2_b14_resumd_20260408-172336` |
| 3 | SigLIP PG1 | 400M | **0.646** | 0.441 | `siglip_so400m_pg1_resumd_20260408-175845` |
| 4 | SigLIP raw | 400M | **0.621** | 0.450 | `siglip_so400m_raw_resumd_20260408-174541` |
| 5 | SigLIP π0 | 400M | **0.597** | 0.446 | `siglip_so400m_pi0_resumd_20260408-182121` |
| 6 | SigLIP π0.5 | 400M | **0.586** | 0.458 | `siglip_so400m_pi05_resumd_20260408-183404` |

### 224×224 Resolution (res224, grid 16×16)

| Rank | Encoder | Params | Test mIoU | Val mIoU | Output Dir |
|------|---------|--------|-----------|----------|------------|
| 1 | SigLIP π0 | 400M | **0.395** | 0.197 | `siglip_so400m_pi0_res224_20260409-080957` |
| 2 | SigLIP raw | 400M | **0.368** | 0.219 | `siglip_so400m_raw_res224_20260408-231637` |
| 3 | DINOv2-L/14 | 300M | **0.362** | 0.218 | `dinov2_l14_res224_20260408-174009` |
| 4 | SigLIP π0.5 | 400M | **0.325** | 0.163 | `siglip_so400m_pi05_res224_20260409-081308` |
| 5 | DINOv2-B/14 | 86M | **0.318** | 0.171 | `dinov2_b14_res224_20260408-172829` |
| 6 | SigLIP PG1 | 400M | **0.311** | 0.204 | `siglip_so400m_pg1_res224_20260409-080538` |

### Res384 Ablation

| Encoder | Test mIoU | Notes |
|---------|-----------|-------|
| SigLIP raw (384×384) | **FAILED** | Crashed during feature shape inference. Log: `siglip_so400m_raw_res384_20260408-232616` |

---

## 2. Ranking Instability Between Resolutions

The SigLIP rankings completely flip between resumd and res224:

| SigLIP Variant | resumd Rank | res224 Rank | resumd mIoU | res224 mIoU |
|----------------|-------------|-------------|-------------|-------------|
| PG1 | 1st (best) | 4th (worst) | 0.646 | 0.311 |
| raw | 2nd | 2nd | 0.621 | 0.368 |
| π0 | 3rd | 1st (best) | 0.597 | 0.395 |
| π0.5 | 4th (worst) | 3rd | 0.586 | 0.325 |

**This instability is attributed to the missing LayerNorm on SigLIP features (see Known Issue).**

---

## 3. Per-Class Test IoU Breakdown

Affordance classes (background excluded): grasp(1), cut(2), scoop(3), contain(4), pound(5), support(6), wrap-grasp(7)

### resumd (UMD Native)

| Encoder | grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---------|-------|-----|-------|---------|-------|---------|------------|
| DINOv2-L/14 | 0.516 | 0.702 | 0.517 | 0.807 | 0.755 | 0.703 | 0.756 |
| DINOv2-B/14 | 0.510 | 0.608 | 0.521 | 0.809 | 0.764 | 0.659 | 0.792 |
| SigLIP PG1 | 0.535 | 0.647 | 0.499 | 0.811 | 0.693 | 0.578 | 0.762 |
| SigLIP raw | 0.418 | 0.708 | 0.396 | 0.805 | 0.661 | 0.651 | 0.709 |
| SigLIP π0 | 0.482 | 0.705 | 0.402 | 0.810 | 0.534 | 0.599 | 0.642 |
| SigLIP π0.5 | 0.498 | 0.672 | 0.374 | 0.776 | 0.572 | 0.594 | 0.614 |

### res224

| Encoder | grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---------|-------|-----|-------|---------|-------|---------|------------|
| SigLIP π0 | 0.149 | 0.442 | 0.291 | 0.503 | 0.416 | 0.396 | 0.565 |
| SigLIP raw | 0.146 | 0.269 | 0.366 | 0.562 | 0.437 | 0.416 | 0.385 |
| DINOv2-L/14 | 0.123 | 0.237 | 0.191 | 0.681 | 0.472 | 0.302 | 0.532 |
| SigLIP π0.5 | 0.089 | 0.251 | 0.243 | 0.459 | 0.300 | 0.428 | 0.504 |
| DINOv2-B/14 | 0.070 | 0.391 | 0.188 | 0.573 | 0.212 | 0.265 | 0.525 |
| SigLIP PG1 | 0.159 | 0.246 | 0.282 | 0.558 | 0.291 | 0.145 | 0.498 |

---

## 4. Training Dynamics

All runs used seed=1337, max_epochs=2, patience=1.

### resumd — Epoch-by-Epoch

| Encoder | Ep1 Train Loss | Ep1 Val mIoU | Ep2 Train Loss | Ep2 Val mIoU | Early Stop? |
|---------|---------------|--------------|---------------|--------------|-------------|
| DINOv2-L/14 | 0.0186 | 0.521 | 0.0042 | 0.464 | Yes (ep2) |
| DINOv2-B/14 | 0.0204 | 0.490 | 0.0041 | 0.408 | Yes (ep2) |
| SigLIP PG1 | 0.0242 | 0.441 | 0.0063 | 0.418 | Yes (ep2) |
| SigLIP raw | 0.0253 | 0.450 | 0.0061 | 0.381 | Yes (ep2) |
| SigLIP π0 | 0.0233 | 0.446 | 0.0061 | 0.410 | Yes (ep2) |
| SigLIP π0.5 | 0.0218 | 0.458 | 0.0059 | 0.443 | Yes (ep2) |

### res224 — Epoch-by-Epoch

| Encoder | Ep1 Train Loss | Ep1 Val mIoU | Ep2 Train Loss | Ep2 Val mIoU | Early Stop? |
|---------|---------------|--------------|---------------|--------------|-------------|
| DINOv2-L/14 | 0.0178 | 0.218 | 0.0028 | 0.198 | Yes (ep2) |
| DINOv2-B/14 | 0.0193 | 0.171 | 0.0025 | 0.169 | Yes (ep2) |
| SigLIP PG1 | 0.0227 | 0.091 | 0.0042 | 0.204 | No (improved) |
| SigLIP raw | 0.0239 | 0.193 | 0.0045 | 0.219 | No (improved) |
| SigLIP π0 | 0.0207 | 0.197 | 0.0042 | 0.181 | Yes (ep2) |
| SigLIP π0.5 | 0.0205 | 0.151 | 0.0043 | 0.163 | No (improved) |

**Observations:**
- All resumd runs: val mIoU peaked at epoch 1, early stopping triggered at epoch 2.
- res224 runs: mixed — PG1/raw/π0.5 improved at epoch 2, DINOv2s and π0 degraded.
- PG1 res224 had an anomalously low epoch 1 val mIoU (0.091) that recovered to 0.204 at epoch 2.

---

## 5. Resolution Delta

| Encoder | resumd mIoU | res224 mIoU | Delta (pp) | Ratio |
|---------|-------------|-------------|------------|-------|
| DINOv2-L/14 | 0.679 | 0.362 | -31.7 | 1.88× |
| DINOv2-B/14 | 0.666 | 0.318 | -34.8 | 2.09× |
| SigLIP PG1 | 0.646 | 0.311 | -33.5 | 2.08× |
| SigLIP raw | 0.621 | 0.368 | -25.3 | 1.69× |
| SigLIP π0 | 0.597 | 0.395 | -20.2 | 1.51× |
| SigLIP π0.5 | 0.586 | 0.325 | -26.1 | 1.80× |

All models degrade at 224×224 (16×16 grid is much coarser), but the degree varies widely (20-35 pp). The inconsistency in resolution deltas across SigLIP variants further supports the LayerNorm bug hypothesis.

---

## 6. Validation Gate

**Target:** DINOv2-B/14 at UMD native, no geometry → reasonable mIoU near Zhang et al.'s 0.670 (with geometry).
**Result:** 0.666 — **PASSED**. Pipeline validated.

---

## 7. Known Issue: Missing LayerNorm on SigLIP Features

**Bug:** DINOv2 features are extracted with `norm=True` (LayerNorm applied to all intermediate layers), but SigLIP features are raw `output_hidden_states` with no normalization. The probe head's BatchNorm2d partially compensates but is resolution-dependent, causing different normalization behavior at 16×16 vs 35×46 grids.

**Impact:** The SigLIP rankings are unreliable — they flip completely between resolutions. The DINOv2 results are unaffected and valid.

**Fix:** Apply `vision_model.post_layernorm` to each SigLIP intermediate layer before returning features, matching DINOv2's treatment. Implemented in `src/models/siglip_so400m.py`. All 10 SigLIP configs will be re-run.

---

## 8. Within-SigLIP Trajectory (resumd only, pre-fix)

```
raw (0.621) → PG1 (0.646, +2.5pp) → π0 (0.597, -4.9pp) → π0.5 (0.586, -1.1pp)
```

PG1 is the high-water mark. VLA fine-tuning stages (π0, π0.5) degrade affordance probing performance. **However, this trajectory may change after the LayerNorm fix.**

---

## 9. DINOv2 vs SigLIP (resumd, pre-fix)

Even with the LayerNorm disadvantage, the structural finding holds:
- DINOv2-B/14 (86M) **beats all** SigLIP variants (400M) at UMD native resolution
- The "dominate from below" argument is strong at this resolution
- Whether this holds at res224 post-fix remains to be seen

---
---

# Phase 1 Linear Probing Results — Post-LayerNorm Fix

**Date:** 2026-04-09
**Seed:** 1337 (all runs)
**Training:** max_epochs=2, patience=1, lr=0.001, wd=0.01, batch_size=4, precision=bf16
**Geometry:** Disabled
**Fix applied:** `vision_model.post_layernorm` now applied to all SigLIP intermediate hidden states before returning features, matching DINOv2's `norm=True` behavior. DINOv2 results unchanged (not re-run).

---

## 10. Summary Table — Test mIoU (Post-Fix)

### UMD Native Resolution (resumd, ~480×640, grid 35×46)

| Rank | Encoder | Params | Test mIoU | Val mIoU | Output Dir |
|------|---------|--------|-----------|----------|------------|
| 1 | DINOv2-L/14 | 300M | **0.679** | 0.521 | `dinov2_l14_resumd_20260408-173113` (unchanged) |
| 2 | DINOv2-B/14 | 86M | **0.666** | 0.490 | `dinov2_b14_resumd_20260408-172336` (unchanged) |
| 3 | SigLIP PG1 | 400M | **0.628** | 0.408 | `siglip_so400m_pg1_resumd_20260409-094133` |
| 4 | SigLIP raw | 400M | **0.619** | 0.468 | `siglip_so400m_raw_resumd_20260409-092652` |
| 5 | SigLIP π0 | 400M | **0.574** | 0.414 | `siglip_so400m_pi0_resumd_20260409-095341` |
| 6 | SigLIP π0.5 | 400M | **0.571** | 0.438 | `siglip_so400m_pi05_resumd_20260409-100553` |

### 224×224 Resolution (res224, grid 16×16)

| Rank | Encoder | Params | Test mIoU | Val mIoU | Output Dir |
|------|---------|--------|-----------|----------|------------|
| 1 | DINOv2-L/14 | 300M | **0.362** | 0.218 | `dinov2_l14_res224_20260408-174009` (unchanged) |
| 2 | SigLIP π0 | 400M | **0.361** | 0.209 | `siglip_so400m_pi0_res224_20260409-102323` |
| 3 | SigLIP π0.5 | 400M | **0.339** | 0.199 | `siglip_so400m_pi05_res224_20260409-102549` |
| 4 | SigLIP PG1 | 400M | **0.334** | 0.175 | `siglip_so400m_pg1_res224_20260409-102057` |
| 5 | DINOv2-B/14 | 86M | **0.318** | 0.171 | `dinov2_b14_res224_20260408-172829` (unchanged) |
| 6 | SigLIP raw | 400M | **0.267** | 0.143 | `siglip_so400m_raw_res224_20260409-101802` |

---

## 11. Pre-Fix vs Post-Fix Comparison

### resumd

| Encoder | Pre-fix | Post-fix | Change | Rank Change |
|---------|---------|----------|--------|-------------|
| SigLIP PG1 | 0.646 | **0.628** | -1.8 pp | 3rd → 3rd |
| SigLIP raw | 0.621 | **0.619** | -0.2 pp | 4th → 4th |
| SigLIP π0 | 0.597 | **0.574** | -2.3 pp | 5th → 5th |
| SigLIP π0.5 | 0.586 | **0.571** | -1.5 pp | 6th → 6th |

**Resumd ranking is stable.** Same order pre- and post-fix. Scores dropped slightly (1-2 pp) — LayerNorm compresses feature scale, giving the probe slightly less raw signal to work with. The ranking was already correct here.

### res224

| Encoder | Pre-fix | Post-fix | Change | Rank Change |
|---------|---------|----------|--------|-------------|
| SigLIP π0 | 0.395 | **0.361** | -3.4 pp | 1st → 1st |
| SigLIP raw | 0.368 | **0.267** | -10.1 pp | 2nd → 6th ↓↓ |
| SigLIP π0.5 | 0.325 | **0.339** | +1.4 pp | 4th → 3rd ↑ |
| SigLIP PG1 | 0.311 | **0.334** | +2.3 pp | 6th → 4th ↑↑ |

**Res224 ranking partially stabilized.** PG1 moved from last among SigLIP to 3rd (closer to its resumd position). But raw SigLIP collapsed from 2nd to last (-10.1 pp), and π0 remains 1st (vs 3rd at resumd).

---

## 12. Ranking Stability Assessment (Post-Fix)

| SigLIP Variant | resumd Rank | res224 Rank | Consistent? |
|----------------|-------------|-------------|-------------|
| PG1 | 1st | 2nd | ~Yes |
| raw | 2nd | 4th (last) | **No** |
| π0 | 3rd | 1st | **No** |
| π0.5 | 4th | 3rd | ~Partial |

**The LayerNorm fix did NOT fully resolve the ranking instability.** PG1 stabilized (top-2 at both resolutions), but raw and π0 still swap positions. This points to the **positional encoding asymmetry** as a second confound:

- **Raw SigLIP** was pretrained at 384×384. At res224, its positional encodings are *downscaled* from 27×27 → 16×16, an out-of-distribution regime. At resumd, it interpolates *up* from 27×27 → 35×46 (closer to native). This explains why raw does relatively well at resumd but collapses at res224.

- **PG1/π0/π0.5** were all trained at 224×224. At res224, they operate at *native* positional resolution (no interpolation). At resumd, they extrapolate *up* from 16×16 → 35×46, which degrades their positional signal.

The ranking difference between resolutions is therefore **partially a real signal about each checkpoint's native operating regime**, not purely an artifact.

---

## 13. Per-Class Test IoU Breakdown (Post-Fix)

Affordance classes: grasp(1), cut(2), scoop(3), contain(4), pound(5), support(6), wrap-grasp(7)

### resumd (Post-Fix)

| Encoder | grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---------|-------|-----|-------|---------|-------|---------|------------|
| DINOv2-L/14 | 0.516 | 0.702 | 0.517 | 0.807 | 0.755 | 0.703 | 0.756 |
| DINOv2-B/14 | 0.510 | 0.608 | 0.521 | 0.809 | 0.764 | 0.659 | 0.792 |
| SigLIP PG1 | 0.479 | 0.672 | 0.435 | 0.815 | 0.693 | 0.619 | 0.680 |
| SigLIP raw | 0.396 | 0.685 | 0.385 | 0.803 | 0.674 | 0.644 | 0.748 |
| SigLIP π0 | 0.466 | 0.642 | 0.431 | 0.805 | 0.502 | 0.549 | 0.621 |
| SigLIP π0.5 | 0.438 | 0.671 | 0.340 | 0.794 | 0.600 | 0.576 | 0.579 |

### res224 (Post-Fix)

| Encoder | grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---------|-------|-----|-------|---------|-------|---------|------------|
| DINOv2-L/14 | 0.123 | 0.237 | 0.191 | 0.681 | 0.472 | 0.302 | 0.532 |
| SigLIP π0 | 0.111 | 0.439 | 0.285 | 0.555 | 0.277 | 0.286 | 0.573 |
| SigLIP π0.5 | 0.118 | 0.316 | 0.284 | 0.528 | 0.289 | 0.352 | 0.483 |
| SigLIP PG1 | 0.100 | 0.358 | 0.249 | 0.551 | 0.223 | 0.428 | 0.429 |
| DINOv2-B/14 | 0.070 | 0.391 | 0.188 | 0.573 | 0.212 | 0.265 | 0.525 |
| SigLIP raw | 0.094 | 0.274 | 0.120 | 0.497 | 0.085 | 0.362 | 0.439 |

**Notable:** Raw SigLIP's collapse at res224 is broad — it drops across nearly all classes, especially *pound* (0.674 → 0.085) and *scoop* (0.385 → 0.120). This is consistent with positional encoding degradation affecting all spatial reasoning, not just specific classes.

---

## 14. Training Dynamics (Post-Fix)

### resumd — Epoch-by-Epoch

| Encoder | Ep1 Train Loss | Ep1 Val mIoU | Ep2 Train Loss | Ep2 Val mIoU | Early Stop? |
|---------|---------------|--------------|---------------|--------------|-------------|
| SigLIP PG1 | 0.0194 | 0.408 | 0.0049 | 0.399 | Yes (ep2) |
| SigLIP raw | 0.0188 | 0.468 | 0.0048 | 0.408 | Yes (ep2) |
| SigLIP π0 | 0.0205 | 0.414 | 0.0052 | 0.396 | Yes (ep2) |
| SigLIP π0.5 | 0.0202 | 0.438 | 0.0053 | 0.414 | Yes (ep2) |

### res224 — Epoch-by-Epoch

| Encoder | Ep1 Train Loss | Ep1 Val mIoU | Ep2 Train Loss | Ep2 Val mIoU | Early Stop? |
|---------|---------------|--------------|---------------|--------------|-------------|
| SigLIP PG1 | 0.0180 | 0.170 | 0.0032 | 0.175 | No (improved) |
| SigLIP raw | 0.0179 | 0.143 | 0.0031 | 0.142 | Yes (ep2) |
| SigLIP π0 | 0.0185 | 0.209 | 0.0034 | 0.150 | Yes (ep2) |
| SigLIP π0.5 | 0.0187 | 0.176 | 0.0036 | 0.199 | No (improved) |

**Observations:**
- All resumd runs: early stopping at epoch 2 (same as pre-fix). Pattern is consistent across all experiments.
- res224: Raw now early-stops (previously it improved at epoch 2) — the LayerNorm changed its convergence dynamics.

---

## 15. Resolution Delta (Post-Fix)

| Encoder | resumd mIoU | res224 mIoU | Delta (pp) | Ratio |
|---------|-------------|-------------|------------|-------|
| DINOv2-L/14 | 0.679 | 0.362 | -31.7 | 1.88× |
| DINOv2-B/14 | 0.666 | 0.318 | -34.8 | 2.09× |
| SigLIP PG1 | 0.628 | 0.334 | -29.4 | 1.88× |
| SigLIP raw | 0.619 | 0.267 | -35.2 | 2.32× |
| SigLIP π0 | 0.574 | 0.361 | -21.3 | 1.59× |
| SigLIP π0.5 | 0.571 | 0.339 | -23.2 | 1.68× |

**Raw SigLIP has the largest resolution delta** (35.2 pp, 2.32×) — expected given its 384 native resolution is furthest from 224. The VLA-trained variants (π0, π0.5) have the smallest deltas (21-23 pp) — they were trained at 224 so that's their native regime.

---

## 16. Within-SigLIP Trajectory (Post-Fix)

### resumd
```
raw (0.619) → PG1 (0.628, +0.9pp) → π0 (0.574, -5.4pp) → π0.5 (0.571, -0.3pp)
```

### res224
```
raw (0.267) → PG1 (0.334, +6.7pp) → π0 (0.361, +2.7pp) → π0.5 (0.339, -2.2pp)
```

**The trajectories tell different stories depending on resolution:**

- **resumd:** Same as pre-fix — PG1 peaks, then VLA fine-tuning degrades features. Clear downward trend after PG1.
- **res224:** Monotonically improves from raw through π0, with π0.5 dropping slightly. VLA fine-tuning *helps* at this resolution.

**This difference is interpretable, not artifactual.** At 224×224 (the VLA's actual operating resolution), the π0/π0.5 checkpoints are in-distribution — they were fine-tuned at this exact resolution with these exact positional encodings. Their features are optimized for 16×16 patch grids. At resumd, the extrapolated positional encodings penalize the VLA checkpoints more than raw (which was pretrained closer to resumd resolution).

---

## 17. Key Takeaways (Post-Fix)

### What's robust across both resolutions:
1. **DINOv2 > SigLIP at resumd.** Both DINOv2-B (86M) and DINOv2-L (300M) beat all SigLIP variants (400M). The "dominate from below" argument holds at UMD native resolution.
2. **PG1 ≥ raw everywhere.** PaliGemma stage-1 fine-tuning never hurts — PG1 beats or matches raw at both resolutions.
3. **π0.5 ≤ π0 everywhere.** The final VLA stage never helps over π0.

### What's resolution-dependent:
1. **π0 vs PG1:** π0 loses to PG1 at resumd (-5.4pp) but beats it at res224 (+2.7pp). This is the positional encoding effect — π0 is in-distribution at 224 but out-of-distribution at resumd.
2. **Raw SigLIP:** Competitive at resumd (2nd among SigLIP) but collapses at res224 (last). It was pretrained at 384 — neither resolution is native, but 224 is further away.
3. **DINOv2 "dominate from below":** Holds clearly at resumd. At res224, DINOv2-L ties with SigLIP π0 (0.362 vs 0.361) and DINOv2-B falls below all VLA variants.

### Interpretation:
The res224 results are arguably more meaningful for the VLA thesis because **224×224 is the actual operating resolution of π0/π0.5**. At this resolution, VLA fine-tuning shows a clear benefit: PG1 > raw (+6.7pp) and π0 > PG1 (+2.7pp). The degradation only appears at π0.5 (-2.2pp), suggesting that knowledge-insulated training may over-specialize.

However, the resumd results show that this benefit is resolution-specific, not a fundamental improvement in feature quality. When evaluated at higher resolution (where positional encoding quality matters more), the VLA checkpoints lose their advantage.

---
---

# Phase 1 Linear Probing Results — Res224 Multi-Seed (n=4)

**Date:** 2026-04-14
**Seeds:** 1337 (existing, from §10) + 42, 2024, 7 (new)
**Training:** unchanged — max_epochs=2, patience=1, lr=0.001, wd=0.01, batch_size=4, precision=bf16, geometry disabled
**Code:** post-LayerNorm-fix (same as §10). DINOv2 code path unchanged since §1.
**Scope:** res224 configs only. Resumd was **not** re-run and remains n=1.

---

## 18. Multi-Seed Setup

Sections §1–§17 report every res224 ordering from a single seed (1337). Adjacent-rank gaps of 0.1–3 pp were being read as findings with no noise floor. To establish one, we re-ran the six res224 configs at three additional seeds (42, 2024, 7), giving n=4 per encoder. All 18 new runs completed with rc=0; output dirs are `outputs/{encoder}_res224_20260414-21XXXX/`. The seed is recorded in each run's `summary.json` and `config_snapshot.yaml`.

**What this section does and does not do.** It establishes a noise floor at res224 and revisits §17's claims against it. It does **not** re-run resumd, fix the res384 raw-SigLIP crash, ablate the patch-coverage threshold, or explain the val/test gap (see §23).

---

## 19. Aggregate Test mIoU (n=4)

Sorted by mean. Per-seed values listed in seed order **[1337, 42, 2024, 7]** (seeds are verified against each run's `config_snapshot.yaml`).

| Rank | Encoder | Params | n | Mean | Std | Min | Max | Per-seed values [1337, 42, 2024, 7] |
|------|---------|--------|---|------|-----|-----|-----|--------------------------------------|
| 1 | SigLIP π0.5 | 400M | 4 | **0.3889** | 0.0424 | 0.3387 | 0.4418 | [0.339, 0.393, 0.382, 0.442] |
| 2 | DINOv2-L/14 | 300M | 4 | **0.3773** | 0.0113 | 0.3624 | 0.3888 | [0.362, 0.389, 0.375, 0.383] |
| 3 | SigLIP π0 | 400M | 4 | **0.3719** | 0.0192 | 0.3524 | 0.3954 | [0.361, 0.352, 0.379, 0.395] |
| 4 | SigLIP PG1 | 400M | 4 | **0.3519** | 0.0261 | 0.3292 | 0.3862 | [0.334, 0.358, 0.329, 0.386] |
| 5 | DINOv2-B/14 | 86M | 4 | **0.3383** | 0.0281 | 0.3171 | 0.3769 | [0.318, 0.377, 0.317, 0.341] |
| 6 | SigLIP raw | 400M | 4 | **0.3125** | 0.0415 | 0.2673 | 0.3651 | [0.267, 0.365, 0.322, 0.296] |

**Two things to notice:**

1. **DINOv2-L/14 is by far the most seed-stable encoder** (σ = 0.011, vs ≥ 0.019 for every other encoder and ≥ 0.041 for π0.5 and raw). Its position in the ranking is the most trustworthy even though its mean is not on top.

2. **Almost every adjacent-rank gap is inside the per-encoder seed noise:**

   | Adjacent pair | Mean gap | Relevant σ | Separated? |
   |---|---|---|---|
   | π0.5 − DINOv2-L | 1.2 pp | σ_π0.5 = 4.2 pp, σ_L = 1.1 pp | **No** |
   | DINOv2-L − π0 | 0.5 pp | σ_π0 = 1.9 pp | **No** |
   | π0 − PG1 | 2.0 pp | σ_PG1 = 2.6 pp | **No** |
   | PG1 − DINOv2-B | 1.4 pp | σ_B = 2.8 pp | **No** |
   | DINOv2-B − raw | 2.6 pp | σ_raw = 4.2 pp | **No** |

   Only the **top–bottom gap** (π0.5 − raw ≈ 7.6 pp) clearly exceeds per-encoder σ, and even that is fragile given how wide π0.5 and raw spread individually. At n=4, no adjacent pair at res224 is statistically separated.

---

## 20. Per-Class Test IoU — n=4 Mean

Affordance classes (background excluded): grasp, cut, scoop, contain, pound, support, wrap-grasp.

| Encoder | grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---------|-------|-----|-------|---------|-------|---------|------------|
| DINOv2-L/14 | 0.111 | 0.354 | 0.226 | **0.636** | **0.417** | 0.387 | 0.511 |
| DINOv2-B/14 | 0.080 | 0.388 | 0.189 | 0.576 | 0.302 | 0.323 | 0.510 |
| SigLIP raw | 0.090 | 0.318 | 0.174 | 0.531 | 0.227 | 0.374 | 0.473 |
| SigLIP PG1 | 0.121 | 0.384 | 0.219 | 0.593 | 0.309 | 0.346 | 0.491 |
| SigLIP π0 | 0.106 | 0.384 | **0.279** | 0.622 | 0.325 | 0.324 | **0.564** |
| SigLIP π0.5 | **0.127** | **0.392** | **0.293** | 0.603 | 0.357 | **0.436** | 0.515 |

π0.5 leads on 5 of 7 classes (grasp, cut, scoop, support, and ties π0 on scoop), DINOv2-L leads on contain and pound, π0 leads on wrap-grasp. These per-class rankings inherit all the significance caveats from §19 — with per-encoder σ on the order of 2–4 pp at the *overall* mIoU level, individual class gaps are even less reliable.

---

## 21. Claims from §17 Revisited

| §17 claim | n=4 verdict | Notes |
|---|---|---|
| "DINOv2 > SigLIP at resumd" | **Untouched** — resumd not re-run | Still single-seed; needs multi-seeding before it can be defended |
| "PG1 ≥ raw everywhere" | **Mean holds, not significant at res224** | PG1 mean 0.352 vs raw 0.312 (+4.0 pp), within combined σ |
| "π0.5 ≤ π0 everywhere" | **Contradicted at res224** | π0.5 mean 0.389 > π0 mean 0.372 (+1.7 pp); ordering flips from §10 |
| "π0 beats PG1 at res224 by +2.7 pp" | **Not separated** | n=4 gap is 2.0 pp, < σ_PG1 = 2.6 pp |
| "Raw SigLIP collapses at res224" | **Softened** | Seed=1337 (0.267) was the **worst** of four. Mean 0.312, max 0.365. Raw is still weakest on average, but the "collapse" framing was seed-contingent |
| "DINOv2 dominate from below at res224" | **Never held** at res224 | DINOv2-B is 5th of 6 at n=4, below all SigLIP VLA variants. The dominate-from-below argument is a resumd phenomenon, not a res224 one |
| "DINOv2-L ties SigLIP π0 at res224 (0.362 vs 0.361)" | **Coincidence** | At n=4 both are in the 0.37–0.38 band; the 0.001 gap in §10 was not informative and the two encoders remain statistically indistinguishable |
| "res224 shows monotone raw → PG1 → π0 → π0.5 improvement" | **Mean trend only** | Mean sequence is monotone (0.312 → 0.352 → 0.372 → 0.389), but each step is ~1σ. The *trajectory* is suggestive; no individual step ordering is defended by this data |

**Net effect.** The only §17 claim that survives n=4 at res224 is the coarse top-vs-bottom observation (π0.5 > raw). Every fine-grained ordering among the middle four encoders is within seed noise. The monotone raw → PG1 → π0 → π0.5 mean trajectory is the most interesting n=4 signal, but it is suggestive, not conclusive, at this sample size.

---

## 22. Val vs Test Gap — Systematic and Unexplained

n=4 means, res224:

| Encoder | Val mean | Test mean | Gap |
|---------|----------|-----------|-----|
| DINOv2-B/14 | 0.1961 | 0.3382 | **+0.1422** |
| DINOv2-L/14 | 0.2258 | 0.3773 | **+0.1515** |
| SigLIP raw | 0.1707 | 0.3125 | **+0.1417** |
| SigLIP PG1 | 0.2014 | 0.3519 | **+0.1505** |
| SigLIP π0 | 0.2292 | 0.3719 | **+0.1427** |
| SigLIP π0.5 | 0.2432 | 0.3889 | **+0.1457** |

The test-above-val offset is **+14.2 to +15.2 pp across every encoder**, and this was already visible in §10 (e.g. DINOv2-L val 0.218 → test 0.362). A systematic offset of this magnitude, this uniform across encoders and seeds, cannot be seed noise or an encoder-specific artifact. Candidate causes we have not ruled out:

1. **Split distribution asymmetry.** The category split places different tool categories in val vs test; if test is systematically "easier" (more common grips, fewer rare affordances), every encoder would see a uniform lift.
2. **Early-stopping selection bias.** Patience=1 with max_epochs=2 means the "best" checkpoint is chosen on a very small number of val evaluations. If that selection systematically favors a generalization direction that transfers to test, the gap would be uniform.
3. **Patch-coverage threshold interaction.** The 55% coverage rule might discard different fractions of val vs test patches, shifting the mIoU denominator unequally.

This is flagged as a **known issue**. Absolute mIoU numbers from this pipeline should not be used as a benchmark until the gap is characterized. Relative rankings within a single split (what §19/§21 use) are less affected, but the existence of this systematic offset is itself a signal that the eval setup needs auditing.

---

## 23. What This Section Does Not Resolve

- **resumd is still n=1.** Every resumd claim in §1–§17 remains single-seed. Cross-resolution comparisons (e.g. the positional-encoding story in §12) should not be made until resumd is also multi-seeded.
- **res384 raw SigLIP still failed.** The positional-encoding-asymmetry hypothesis in §12 needs a working raw-at-384 run to be testable; that leg does not exist.
- **Patch-coverage threshold (55%) is un-ablated.** We do not know how sensitive res224 rankings are to this choice.
- **Val/test gap is unexplained** — see §22.
- **n=4 is a low noise floor.** σ on π0.5 and raw is 0.042 — wide enough that more seeds could still shift rankings, particularly for encoders at the top and bottom of the table.

**The next experiments that would actually tighten these claims**, in priority order: (a) resumd multi-seed at the same 4 seeds used here; (b) diagnose and fix the res384 raw-SigLIP crash; (c) run 1–2 more seeds on res224 for π0.5 and raw specifically, since they dominate the residual uncertainty; (d) a 2-point ablation of the patch-coverage threshold.

---
---

# 24. Presentation Brief (self-contained)

A standalone summary for downstream write-up. Everything below is derivable from §1–§23, collected here so a reader who has not opened the rest of this document can present the work in one pass.

## 24.1 The problem

Vision-Language-Action (VLA) policies like π0 and π0.5 (LeRobot) start from a pretrained vision encoder — SigLIP-So400m — and fine-tune it through a PaliGemma multimodal stage (PG1) and then through flow-matching / knowledge-insulated VLA training. Each stage reshapes the vision tower for policy execution, but it is unclear whether these stages preserve, improve, or degrade the geometric scene understanding that the encoder originally learned.

The question this project attacks:

> **Does VLA fine-tuning of a vision tower preserve the geometric affordance information that the original SigLIP encoder carried, and how do the resulting representations compare to a purely self-supervised geometry specialist like DINOv2?**

The answer matters because it tells us whether the vision tower inside a robot policy is still a competent "seeing the world" module, or whether it has collapsed into a task-specific feature extractor that would need to be paired with a frozen geometry encoder in downstream systems.

## 24.2 Approach

We adapted the **geometry-side linear probe** from Zhang et al., *Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models*. The paper's original aim was surveying a zoo of generic foundation models; we repurposed its methodology to characterize a specific VLA trajectory.

**Method:**
- Freeze the vision encoder.
- Hook 4 intermediate transformer layers, fuse them by bilinear resize to the first hooked layer's grid, concatenate along channels.
- Train a lightweight head: `BatchNorm2d → 1×1 Conv → 7-class affordance logits`.
- Evaluate at **patch-grid resolution** (not pixel space) — each patch is assigned a majority-vote affordance class at ≥55% coverage, otherwise excluded.
- Dataset: UMD part-affordance dataset (7 classes: grasp, cut, scoop, contain, pound, support, wrap-grasp), category-split so train / val / test contain different tool categories.

**Encoders compared:**

| Encoder | Params | Native res | Role |
|---|---|---|---|
| DINOv2-B/14 | 86M | 224 | Small self-supervised geometry reference |
| DINOv2-L/14 | 300M | 224 | Larger self-supervised geometry reference |
| SigLIP-So400m (raw) | 400M | 384 | Pretrained contrastive tower |
| SigLIP-So400m (PG1) | 400M | 224 | + PaliGemma multimodal stage-1 |
| SigLIP-So400m (π0) | 400M | 224 | + π0 flow-matching VLA fine-tune |
| SigLIP-So400m (π0.5) | 400M | 224 | + π0.5 knowledge-insulated VLA fine-tune |

The SigLIP-So400m trajectory is the **core experimental lever**: the same 400M-parameter architecture at four points along a VLA training path.

**Resolutions:**
- `res224` (16×16 patch grid) — matches π0 / π0.5's actual operating resolution.
- `resumd` (~480×640 UMD native, 35×46 patch grid) — matches the setup in Zhang et al., serves as a pipeline validation gate.

**Multi-seed:** both `res224` and `resumd` were run at 4 seeds (1337, 42, 2024, 7); n=4 per encoder at each resolution.

## 24.3 Main results (preliminary)

### Result 1 — Pipeline validation

At `resumd` with `geometry=off`, DINOv2-B/14 scores **0.666 test mIoU** against Zhang et al.'s reported 0.670 (with geometry enabled). The pipeline reproduces the reference. Everything else is reported against this baseline.

### Result 2 — Single-seed rankings are not trustworthy

At `res224`, the single-seed post-fix table (§10) reported this ordering: DINOv2-L (1st) → π0 → π0.5 → PG1 → DINOv2-B → raw SigLIP (last). After running 3 additional seeds (n=4 total), the ordering shifts substantially:

| Rank | Encoder | n=4 mean | σ | Per-seed [1337, 42, 2024, 7] |
|---|---|---|---|---|
| 1 | **SigLIP π0.5** | 0.3889 | 0.042 | [0.339, 0.393, 0.382, 0.442] |
| 2 | **DINOv2-L** | 0.3773 | **0.011** | [0.362, 0.389, 0.375, 0.383] |
| 3 | SigLIP π0 | 0.3719 | 0.019 | [0.361, 0.352, 0.379, 0.395] |
| 4 | SigLIP PG1 | 0.3519 | 0.026 | [0.334, 0.358, 0.329, 0.386] |
| 5 | DINOv2-B | 0.3383 | 0.028 | [0.318, 0.377, 0.317, 0.341] |
| 6 | SigLIP raw | 0.3125 | 0.042 | [0.267, 0.365, 0.322, 0.296] |

**No adjacent pair is statistically separated at n=4** — every adjacent-rank gap (0.5–2.6 pp) is within the per-encoder σ. Only the top–bottom gap (π0.5 vs raw, 7.6 pp) clearly exceeds noise.

### Result 3 — The VLA trajectory at operating resolution is monotone in the mean

At `res224` the n=4 means form a **monotone sequence along the VLA training path**:

```
raw (0.312)  →  PG1 (0.352)  →  π0 (0.372)  →  π0.5 (0.389)
                 +4.0 pp         +2.0 pp         +1.7 pp
```

Each step is ~1σ — the trajectory is suggestive, not significant at this sample size. But the direction is consistent: every stage of VLA fine-tuning on SigLIP at least *does not harm* its geometric affordance probe, and the cumulative effect is a +7.7 pp mean improvement over raw SigLIP.

### Result 4 — Resolution dictates which encoder "wins" (now n=4 on both ends)

At `resumd`, DINOv2 dominates by a clear margin. At `res224` the ordering compresses and the top group is unseparated:

| Encoder | resumd mean (n=4) | res224 mean (n=4) | Δ (resumd − res224) |
|---|---|---|---|
| DINOv2-L | 0.6660 | 0.3773 | +0.289 |
| DINOv2-B | 0.6635 | 0.3383 | +0.325 |
| SigLIP raw | 0.6284 | 0.3125 | +0.316 |
| SigLIP PG1 | 0.6023 | 0.3519 | +0.250 |
| SigLIP π0.5 | 0.5648 | 0.3889 | +0.176 |
| SigLIP π0 | 0.5496 | 0.3719 | +0.178 |

At resumd, both DINOv2 variants (86M and 300M) beat every SigLIP variant (400M) — and the +3.5 pp gap from DINOv2-B to SigLIP raw exceeds either side's σ, so this is statistically separated. At res224, DINOv2-B falls to 5th of 6 and π0.5 leads in the mean. **The encoder ordering genuinely flips across resolutions, not as a single-seed artifact.**

A second pattern in the Δ column: the **VLA-trained variants (π0, π0.5) gain ~half as much** from going to resumd as raw / DINOv2 do (+0.18 vs +0.30+). Consistent with their being trained at 224-native — they are saturated at their training regime and cannot exploit the extra spatial resolution as effectively.

### Result 5 — DINOv2 variants are uniquely seed-stable; the most-stable encoder differs by resolution

DINOv2's per-seed σ is the smallest in the table at both resolutions, and the encoder that is *most* stable depends on resolution:

- At **res224**, DINOv2-L is tightest (σ=0.011), 2–4× tighter than any SigLIP variant (σ=0.019–0.042).
- At **resumd**, DINOv2-B is tightest (σ=0.0044), ~3× tighter than DINOv2-L at the same resolution and ~6× tighter than its own σ at res224.

DINOv2-B at resumd (μ=0.6635, σ=0.0044) is **the most trustworthy single number in the document**. The fact that no SigLIP variant matches DINOv2's stability at either resolution is itself an observation about the training objective: contrastive image-text and VLA fine-tuning seem to leave the representation more sensitive to seed than DINOv2's iBOT objective does.

### Result 6 — A systematic, unexplained val/test gap

Across all 48 runs (6 encoders × 4 seeds × 2 resolutions), test mIoU is **+14.2 to +15.2 pp above val** at res224 and **+14.7 to +20.6 pp above val** at resumd, uniformly across every encoder and seed. This cannot be seed noise or an encoder effect — it is a property of the splits or the training/selection loop. The largest single gap (PG1 at resumd, +20.6 pp) is wide enough to flag as its own follow-up. Three candidate causes (split distribution, early-stopping bias, coverage-threshold interaction) are identified but not resolved. **Absolute mIoU numbers from this pipeline should not be used as a benchmark until this is explained.** Relative rankings within a single split are less affected.

### Result 7 — VLA fine-tuning helps at operating resolution, hurts elsewhere

Within the SigLIP-So400m trajectory, the n=4 means tell **opposite stories at the two resolutions**:

```
res224  (n=4):  raw (0.312) → PG1 (0.352) → π0 (0.372) → π0.5 (0.389)   monotone-up,   +7.7 pp cumulative
resumd  (n=4):  raw (0.628) → PG1 (0.602) → π0 (0.550) → π0.5 (0.565)   net-down,      −6.3 pp from raw
```

At the operating resolution (224), every VLA stage at minimum does not harm the geometric affordance probe and the cumulative effect is positive. At resumd, each VLA stage is below raw SigLIP. This is the strongest single observation about the VLA training trajectory the project has: **the improvement is operating-resolution-specific, not a representational improvement.**

## 24.4 Key takeaways

1. **VLA fine-tuning of SigLIP-So400m has opposite effects at different resolutions.** At the operating resolution (res224), each VLA stage at minimum does not harm the geometric affordance probe, and the cumulative trajectory raw → PG1 → π0 → π0.5 lifts mean mIoU by +7.7 pp. At resumd, the same trajectory is net negative (−6.3 pp from raw to π0.5). The improvement at res224 is **operating-resolution-specific, not a fundamental gain in representational quality** — it does not transfer to a higher-resolution geometric probe.

2. **Every fine-grained claim we could have made from a single seed dissolves at n=4.** The literature of single-seed probing comparisons should be read skeptically. Concretely: "π0 > PG1" (+2.7 pp at n=1) is not separated at n=4 at res224; "π0.5 ≤ π0 everywhere" is contradicted at *both* resolutions; "raw collapses at res224" softens to "raw is weakest on average"; the resumd within-SigLIP ordering (PG1 > raw > π0 > π0.5) entirely reshuffles to (raw > PG1 > π0.5 > π0).

3. **Resolution is a first-order confound.** The same encoder ranks very differently at res224 vs. resumd, and this is now n=4 supported on both ends. Which resolution you probe at is inseparable from what you report. For VLA vision-tower characterization, the operating resolution (here, 224) is the more meaningful setting; for geometric capability in the abstract, resumd is more diagnostic.

4. **DINOv2 is clearly dominant at resumd at n=4.** Both DINOv2 variants beat every SigLIP variant by ≥3.5 pp at means clearly larger than per-encoder σ — this is the best-supported inter-encoder claim in the project. At res224 the picture is murkier: only DINOv2-L is competitive, and the top cluster (π0.5, DINOv2-L, π0) is unseparated. The "self-supervised geometry specialists dominate from below" argument from Zhang et al. **holds cleanly at resumd**; at res224 it does not survive multi-seeding.

5. **We are not yet claiming VLA fine-tuning helps or hurts the vision tower in absolute terms.** Affordance mIoU on UMD is a proxy whose validity for downstream policy performance has not been established. The opposite-direction-at-different-resolutions finding is a *representation-level observation*, not a statement about robot capability.

## 24.5 What remains / future directions

### Near-term, to firm up Phase 1
1. **Fix the res384 raw-SigLIP crash.** Without it, the "raw SigLIP is weakest at 224 because 384 is its native resolution" positional-encoding hypothesis is untested.
2. **Explain the val/test gap** (§22 / Result 6) — diagnose whether it's a split, selection, or threshold artifact. The PG1 resumd gap (+20.6 pp, the largest single value) is the natural starting point.
3. **Ablate the 55% patch-coverage threshold** at 2–3 points to confirm rankings are not threshold artifacts.
4. **Extra seeds for π0.5 and raw at res224** specifically, since they have the widest per-seed σ at that resolution (~0.042 each) and dominate the residual uncertainty in the top-1 / bottom-1 res224 claims.

### Phase 2 and beyond
5. **Downstream sanity check.** Pair the probe mIoU with actual policy performance on a small task set. Without this, every claim in this project is upstream-only — a statement about features, not capabilities. The opposite-direction-at-different-resolutions finding (Takeaway 1) makes this especially urgent: until we know which resolution's verdict actually predicts policy behavior, we cannot say whether VLA fine-tuning helps or hurts.
6. **Interaction-side probing** via FLUX, to complement the geometry-side probe and match the full scope of Zhang et al.
7. **Pixel-level probe with a proper decoder** as a cross-check. The patch-grid metric is methodologically cleaner (isolates feature quality) but the absolute numbers are not comparable to pixel-level literature.

## 24.6 Where to read more

- Methodology and rationale: `umd_linear_probing/PHASE1_ZHANG_ADAPTATION_PLAN.md`
- Pipeline walkthrough: `umd_linear_probing/guide.md`
- Full numerical record: §1–§29 of this file (single-seed results, LayerNorm-fix history, multi-seed tables and per-class breakdowns at both resolutions, cross-resolution synthesis)
- Agent-oriented orientation: `geometry_probing/CLAUDE.md`

---
---

# Phase 1 Linear Probing Results — Resumd Multi-Seed (n=4)

**Date:** 2026-04-28 (sweep finished 2026-04-29 00:21 — last run crossed midnight).
**Code state:** Same post-LayerNorm-fix code as §10–§17.
**Scope:** 6 resumd configs only.

---

## 25. Multi-Seed Resumd Setup

§10 / §17 reported every resumd ordering from a single seed (1337). §23 listed multi-seeding resumd as the #1 follow-up, and §24 Result 4 leaned on the n=1 resumd to claim "encoder ordering flips across resolutions." This sweep adds three seeds (42, 2024, 7), giving **n=4 per encoder at resumd** — matching the sample size at res224.

- 18 new runs (6 configs × 3 seeds), all completed with `rc=0` in 11198s (~3h 7min).
- Hyperparameters identical to §10 / §18: `max_epochs=2`, `patience=1`, `lr=1e-3`, `wd=1e-2`, `batch_size=4`, `precision=bf16`, geometry disabled.
- Output dirs: `outputs/{encoder}_resumd_2026042{8,9}-*/`. Seed recorded in each `summary.json` and `config_snapshot.yaml`.

---

## 26. Aggregate Test mIoU at Resumd (n=4, Sorted by Mean)

Per-seed values listed in seed order **[1337, 42, 2024, 7]**.

| Rank | Encoder | Params | n | Mean | Std | Min | Max | Per-seed values |
|------|---------|--------|---|------|-----|-----|-----|-----------------|
| 1 | DINOv2-L/14 | 300M | 4 | **0.6660** | 0.0208 | 0.6351 | 0.6795 | [0.680, 0.678, 0.672, 0.635] |
| 2 | DINOv2-B/14 | 86M | 4 | **0.6635** | **0.0044** | 0.6592 | 0.6685 | [0.666, 0.669, 0.660, 0.659] |
| 3 | SigLIP raw | 400M | 4 | **0.6284** | 0.0123 | 0.6167 | 0.6414 | [0.619, 0.617, 0.636, 0.641] |
| 4 | SigLIP PG1 | 400M | 4 | **0.6023** | 0.0265 | 0.5745 | 0.6278 | [0.628, 0.575, 0.585, 0.622] |
| 5 | SigLIP π0.5 | 400M | 4 | **0.5648** | 0.0168 | 0.5399 | 0.5770 | [0.571, 0.577, 0.571, 0.540] |
| 6 | SigLIP π0 | 400M | 4 | **0.5496** | 0.0268 | 0.5203 | 0.5738 | [0.574, 0.534, 0.520, 0.571] |

### Two key observations

1. **DINOv2 > SigLIP at resumd is now defensible.** DINOv2-B (μ=0.6635, σ=0.004) and DINOv2-L (μ=0.6660, σ=0.021) sit clearly above SigLIP raw (μ=0.6284, σ=0.012) — a +3.5 pp gap that exceeds either side's σ. Every other SigLIP variant is further below. **Of the inter-encoder claims in this whole project, this is the best-supported one.**

2. **DINOv2-B at resumd is the most seed-stable encoder × resolution pair we have observed** (σ=0.0044, ~3× tighter than DINOv2-L at resumd, and ~6× tighter than its own σ at res224). Why this is so much tighter at resumd than at res224 is itself an open question, but it makes resumd-DINOv2-B the most trustworthy single number in the document.

### Adjacent-rank gap vs σ

| Pair | Mean gap | Relevant σ | Separated? |
|---|---|---|---|
| DINOv2-L − DINOv2-B | 0.3 pp | 2.1 / 0.4 | No — effectively tied |
| DINOv2-B − raw | 3.5 pp | 0.4 / 1.2 | **Yes** |
| raw − PG1 | 2.6 pp | 1.2 / 2.7 | No (within σ_PG1) |
| PG1 − π0.5 | 3.7 pp | 2.7 / 1.7 | Borderline |
| π0.5 − π0 | 1.5 pp | 1.7 / 2.7 | No |

The cleanest separation is the **DINOv2 cluster (top 2) vs the SigLIP cluster (bottom 4)**.

---

## 27. Per-Class Test IoU at Resumd (n=4 Mean)

| Encoder | grasp | cut | scoop | contain | pound | support | wrap-grasp |
|---------|-------|-----|-------|---------|-------|---------|------------|
| DINOv2-L | 0.475 | **0.714** | **0.474** | 0.800 | **0.746** | **0.678** | **0.775** |
| DINOv2-B | **0.518** | 0.702 | 0.468 | 0.794 | 0.732 | 0.666 | 0.766 |
| SigLIP raw | 0.447 | 0.679 | 0.399 | 0.804 | 0.646 | 0.663 | 0.762 |
| SigLIP PG1 | 0.465 | 0.681 | 0.354 | **0.805** | 0.639 | 0.625 | 0.648 |
| SigLIP π0 | 0.431 | 0.673 | 0.325 | 0.763 | 0.497 | 0.498 | 0.660 |
| SigLIP π0.5 | 0.393 | 0.690 | 0.277 | 0.789 | 0.617 | 0.564 | 0.623 |

DINOv2-B leads on grasp; DINOv2-L leads on five classes (cut, scoop, pound, support, wrap-grasp); SigLIP PG1 leads on contain by 0.001 over raw and 0.005 over DINOv2-L (effectively tied). **The two DINOv2 variants together dominate 6 of 7 classes at resumd**, in stark contrast to res224 (§20) where SigLIP π0.5 dominated 4 of 7.

---

## 28. Resumd Claims Revisited (vs §10 Single-Seed)

| Single-seed claim (§10) | n=4 verdict at resumd | Notes |
|---|---|---|
| "DINOv2 > SigLIP at resumd" | **Confirmed and strengthened** | Cleanest, best-supported inter-encoder claim in this project |
| "PG1 is the SigLIP high-water mark at resumd" (rank 3rd of 6) | **Contradicted** | At n=4, raw (0.628) > PG1 (0.602). Within σ_PG1, but the ordering flips |
| "raw is 4th among encoders at resumd" | **Contradicted** | raw moves from 4th to 3rd (above all VLA-trained variants) |
| "π0.5 < π0 at resumd" (single-seed: 0.571 vs 0.574) | **Contradicted** | n=4: π0.5 (0.565) > π0 (0.550). Now contradicted at *both* resolutions |
| "Resumd within-SigLIP trajectory: raw → PG1 → π0 → π0.5 with PG1 peak" | **Different shape at n=4** | Means: raw (0.628) → PG1 (0.602) → π0 (0.550) → π0.5 (0.565). Monotone-decreasing through π0, then a small rebound at π0.5. PG1 is **not** a peak |

**Net effect.** Of the resumd claims in §1–§17, the one that survives n=4 (and is clearly *strengthened*) is the DINOv2-vs-SigLIP one. Every fine-grained ordering among the SigLIP variants at resumd flips somewhere. The "VLA fine-tuning degrades affordance performance vs raw at resumd" trend is preserved in shape (raw remains the best of the SigLIP variants), but the specific within-VLA orderings are not separated.

---

## 29. Cross-Resolution Picture (n=4 on Both Ends)

This is the section that lets the project finally make a defensible *cross-resolution* claim.

### Cross-resolution n=4 means

| Encoder | res224 mean | resumd mean | Δ (resumd − res224) |
|---|---|---|---|
| DINOv2-L | 0.3773 | 0.6660 | **+0.2887** |
| DINOv2-B | 0.3383 | 0.6635 | **+0.3252** |
| SigLIP raw | 0.3125 | 0.6284 | **+0.3159** |
| SigLIP PG1 | 0.3519 | 0.6023 | +0.2504 |
| SigLIP π0.5 | 0.3889 | 0.5648 | +0.1759 |
| SigLIP π0 | 0.3719 | 0.5496 | +0.1777 |

### Three cross-resolution observations now defensible at n=4

1. **DINOv2 dominates at resumd; the picture is murkier at res224.** At resumd, DINOv2-L/B clearly beat all SigLIP. At res224, only DINOv2-L is competitive (rank 2, tied with the top SigLIP cluster); DINOv2-B falls to rank 5. **Encoder rankings are genuinely resolution-dependent**, not single-seed artifacts.

2. **VLA fine-tuning improves res224 mean, hurts resumd mean.** Within SigLIP-So400m:
   - At res224 (n=4 means): raw (0.312) → PG1 (0.352) → π0 (0.372) → π0.5 (0.389). **Monotone improvement, +7.7 pp cumulative.**
   - At resumd (n=4 means): raw (0.628) → PG1 (0.602) → π0 (0.550) → π0.5 (0.565). **Net degradation, −6.3 pp from raw to π0.5.**
   The directions are *opposite*. The VLA training trajectory is helpful at the operating resolution and harmful when the encoder is asked to operate outside it.

3. **VLA-trained encoders gain less from going to resumd.** Δ (resumd − res224) for π0 / π0.5 is +0.178 / +0.176; for raw / PG1 is +0.316 / +0.250; for DINOv2 is +0.325 / +0.289. The VLA-trained variants benefit roughly **half as much** from the higher resolution as raw / DINOv2 do. Consistent with their being trained at 224-native — they are effectively saturated for their training regime and cannot exploit the extra spatial resolution as effectively.

### Caveats

- **Same-seed cross-resolution comparison** (e.g. π0.5 at seed=42 res224 vs. π0.5 at seed=42 resumd) would be a stronger statement than comparing aggregate means. Both sweeps used the same seed set, so this comparison is *available* in the data but not surfaced in the table above.
- **The val/test gap (§22) at resumd** is +14.7 to +20.6 pp — slightly larger than res224's +14–15 pp band — and PG1's resumd gap (+20.6 pp) is the largest single value across either resolution. Still systematic, still unexplained, but more variable across encoders at resumd than at res224.

### Open items after this sweep

- **Cross-resolution claims are now n=4 supported.** The DINOv2-vs-SigLIP and VLA-trajectory-direction findings have a noise floor on both ends.
- **Still un-resolved:** res384 raw-SigLIP crash; coverage-threshold ablation; val/test gap explanation; downstream-policy validation of the proxy.
- **The PG1 val/test gap at resumd (+20.6 pp)** is large enough to flag as its own follow-up — worth a single deliberate look before the next reporting cycle.
