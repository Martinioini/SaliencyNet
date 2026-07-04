# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A visual **saliency prediction** model trained on the SALICON dataset. Given an RGB image, the network predicts where humans look (a continuous saliency map). All code lives in a single notebook, **`ProgDL.ipynb`** — there are no `.py` modules. Work in the notebook cells.

## Environment & data

- PyTorch + torchvision, run in Jupyter. GPU expected (`device` auto-selects CUDA).
- Dataset is **not** in the repo (`datasets/` is gitignored). Download once:
  ```
  kaggle datasets download -d roshan401/salicon -p ./datasets/salicon --unzip
  ```
- SALICON layout: `images/` (raw), `maps/` (continuous saliency, 0–255 PNG), `fixations/` (`.mat` files with per-observer gaze coords). A shared 12-char id links a sample across all three. `train/` and `val/` splits only — there is **no** usable `test/` split with labels.
- One code comment flags a local-run gotcha: set `pin_memory=True` on GPU.

## Architecture (cells run top to bottom)

- **`encoder`** — ResNet18 pretrained on ImageNet, truncated after `layer4`. `forward` returns the four skip features `c1,c2,c3,c4` (avgpool + fc are dropped).
- **`decoder`** — U-Net-style: upsample `c4`, concat with `c3`/`c2`/`c1` skips, conv→BN→dropout→ReLU blocks, upsampling back to 256×256, final 1-channel conv. **Outputs raw logits, not a normalized map.**
- **`SaliencyKLLoss`** — flattens per-image, `log_softmax(pred)` vs sum-normalized target, `kl_div(reduction='batchmean')`. The prediction is treated as a probability distribution over pixels.
- Images are 256×256, ImageNet-normalized; saliency maps are `ToTensor`'d to [0,1]. Train-time augmentation (hflip, small rotation) is applied identically to image/map/fixation.

## Two-phase training (important)

Training is deliberately split and checkpointed:

1. **Phase 1** — freeze encoder (`requires_grad=False`), train decoder only (`lr=1e-3`), ~5 epochs, early-stop patience 3. Best weights → **`phase1.pt`**.
2. **Phase 2** — load `phase1.pt`, unfreeze encoder, train whole net with differential LRs (encoder `1e-5`, decoder `1e-4`), `ReduceLROnPlateau`, up to 50 epochs, patience 7. Best weights → **`best.pt`**.

Checkpoints are dicts: `{"encoder": ..., "decoder": ...}` state_dicts. `best.pt` is the final model used for all evaluation. Both `.pt` files are large (~54 MB) and currently committed/present in the working tree.

## Fixation loading & the val/test split (both are past-bug areas)

- **`load_fixation_map`** builds the binary fixation map **directly at target resolution** by scaling gaze coordinates from each `.mat` — do **not** resize a sparse binary map afterward (that discards ~80% of points). Note the axis convention: gaze `x`→column, `y`→row.
- Since SALICON's `val` is the only labeled split besides train, the notebook **shuffles the val split and carves out `val_idx[:3500]` for validation and the rest for test.** Recent commit history ("fixed serious data leakage") shows test/val must come from held-out `val`, never from `train`. Preserve this when editing the data-prep cell.

## Metrics (subtle input conventions — read before touching)

Standard saliency metrics live in separate cells: **CC** (correlation), **SIM** (histogram intersection), **NSS** (uses the *fixation* map, not the continuous map), **KL**. They are computed on `best.pt` over val and test.

The decoder emits logits, so the eval loop **normalizes before calling each metric**, and each metric expects a *different* normalization:
- `sim`/`nss`: softmax over flattened pixels → reshaped probability map.
- `cc`: that softmax map further divided by its per-image max.
- `kl`: takes the **raw logits** directly (it does its own `log_softmax`).

If you add or change a metric, match this contract or results will be silently wrong. A center-baseline (Gaussian prior) is compared against the model in the final visualization cell.

## Conventions

- Inline comments and print notes are largely in **Italian**; match the surrounding language when editing a cell.
- Seeds are fixed (`numpy`/`random` = 42) for reproducible splits/augmentation.
