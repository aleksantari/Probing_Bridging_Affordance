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
