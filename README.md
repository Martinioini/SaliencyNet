# SaliencyNet

Visual saliency prediction on the SALICON dataset.

Encoder–decoder architecture with a pretrained ResNet backbone, U-Net skip
connections and multi-head self-attention on the `c3`/`c4` encoder stages.
Trained with KL divergence; evaluated with SIM, CC, NSS and KL, plus a
centre-prior baseline.

---

## Repository layout

| File | Contents |
|---|---|
| `dataset.py` | Path matching, train/val/test split, `ImageDataset`, augmentation, fixation-map construction |
| `models.py` | `encoder18` / `encoder50`, `attention18` / `attention50`, `decoder18` / `decoder50` |
| `losses.py` | `SaliencyKLLoss` (KL divergence over the softmax-normalised prediction) |
| `training.py` | `train_epoch`, `test_epoch`, `run_phase1`, `run_phase2`, checkpoint naming |
| `metrics.py` | SIM, CC, NSS, KL, `evaluate_loader`, `compare_to_center_baseline` |
| `main.py` | Entry point: parses flags, builds the model, runs both phases, evaluates |
| `test.py` | Inspects a checkpoint's attention weights and gates (γ) |

---

## Dataset

SALICON, in the following layout (note the double-nested `images/`, as in the
Kaggle mirror):

```
datasets/salicon/
├── images/images/{train,val}/   # RGB stimuli
├── maps/{train,val}/            # continuous density maps  -> CC, SIM, KL
└── fixations/{train,val}/       # .mat gaze records        -> NSS
```

The root path is set by `ROOT` in `main.py` (default `./datasets/salicon`).

**Split.** 10000 train / 3500 val / 1500 test.
Train is the official SALICON training set. Validation and test are a fixed
random partition of the official 5000-image validation set, because the labels
for the official SALICON test set are not publicly released. The partition is
deterministic: `main.py` seeds Python's `random` before `prepare_paths`, which
performs the shuffle.

**Preprocessing.**
- Images: resize to 256×256, `ToTensor`, ImageNet normalisation
  (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) — required for the
  pretrained backbone to receive inputs with the statistics it was trained on.
- Density maps: resize to 256×256, `ToTensor`, values in `[0, 1]`. No
  normalisation: it is the target, not a network input.
- Fixation maps: **built directly at 256×256 by rescaling the gaze coordinates
  from the `.mat` file**, rather than rasterising at full resolution and
  resizing. Downsampling a sparse binary map discards most of the fixation
  points; scaling the coordinates preserves all of them.

**Augmentation** (training split only, applied jointly to image / density map /
fixation map):
- horizontal flip, p = 0.5
- rotation in ±10°, p = 0.3 — bilinear for the image and the density map,
  **nearest-neighbour for the fixation map**, so that the binary points survive
  the interpolation
- brightness / contrast / saturation jitter in [0.8, 1.2], p = 0.5, image only

---

## Running

Every configuration in the report is a flag combination on `main.py`.
The defaults are the final model, so each flag *removes* a component.

| Model | Backbone | Skip | Attention | Command |
|---|---|---|---|---|
| Base | R18 | ✗ | ✗ | `python main.py --backbone resnet18 --no_skip --no_attention` |
| + skip connections | R18 | ✓ | ✗ | `python main.py --backbone resnet18 --no_attention` |
| + self-attention | R18 | ✓ | ✓ | `python main.py --backbone resnet18` |
| + deeper backbone (**final**) | R50 | ✓ | ✓ | `python main.py` |

Flags:

- `--backbone {resnet18,resnet50}` — encoder (default: `resnet50`)
- `--no_skip` — decoder consumes only the `c4` bottleneck; no concatenation with
  `c1`/`c2`/`c3`
- `--no_attention` — the attention module becomes an identity map on `c3`, `c4`

Reproducing every row of the results tables:

```bash
python main.py --backbone resnet18 --no_skip --no_attention  2>&1 | tee log_base.txt
python main.py --backbone resnet18 --no_attention            2>&1 | tee log_r18_noatt.txt
python main.py --backbone resnet18                           2>&1 | tee log_r18.txt
python main.py                                               2>&1 | tee log_r50.txt
```

---

## Architecture

**Encoder.** ResNet-18 or ResNet-50, ImageNet-pretrained, truncated before
`avgpool` and `fc`. The forward pass exposes the four stage outputs
`c1, c2, c3, c4` for the skip connections. Channel widths differ by a factor of
4 between the two backbones (ResNet-50 uses bottleneck blocks):

| | c1 | c2 | c3 | c4 |
|---|---|---|---|---|
| spatial (input 256×256) | 64×64 | 32×32 | 16×16 | 8×8 |
| ResNet-18 channels | 64 | 128 | 256 | 512 |
| ResNet-50 channels | 256 | 512 | 1024 | 2048 |

**Attention.** `nn.MultiheadAttention` (8 heads) applied to `c3` and `c4`, with
the feature map flattened to a token sequence (`c4` → 64 tokens, `c3` → 256
tokens). Each block is residual and gated by a learnable scalar γ:

```
c_out = c + γ · Attention(c, c, c)
```

γ is initialised at 0.1. The gate makes the attention block *ablatable by
training*: if it were useless the network could drive γ → 0 and recover the
skip-only model exactly, so the attention configuration is guaranteed a
performance floor no worse than the configuration without it. γ is also the
quantity that tells us whether the module is actually used — see below.

**Decoder.** Five bilinear-upsampling stages back to 256×256. Each stage is
`Upsample(×2)` → `Conv3×3` → `BatchNorm` → `Dropout2d(0.2)` → `ReLU`. Bilinear
upsampling is used in place of strided transposed convolutions to avoid
checkerboard artefacts, which are clearly visible on saliency maps.

With `use_skip=True` the first three stages concatenate `c3`, `c2`, `c1`
respectively before the convolution. With `use_skip=False` the decoder receives
only `c4` and the input channels of those convolutions shrink accordingly; this
is the Base model.

The final `Conv3×3` outputs a single channel of **raw logits**. There is no
sigmoid: the loss and the metrics apply their own `softmax` / `log_softmax` over
the flattened map.

---

## Training protocol

Identical across every configuration.

**Phase 1 — encoder frozen.** Only the decoder is optimised.
Adam, lr `1e-3`, weight decay `1e-5`, ≤6 epochs, early-stopping patience 3.
The best-validation checkpoint of this phase is reloaded before phase 2.

**Phase 2 — full network, differential learning rates.**

| Parameter group | Learning rate |
|---|---|
| encoder (pretrained) | `1e-5` |
| decoder | `1e-4` |
| attention | `1e-3` |

Adam, weight decay `1e-5`, `ReduceLROnPlateau` (factor 0.5, patience 3,
min lr `1e-7`), ≤50 epochs, early-stopping patience 5.

The three-way learning-rate split reflects how much each group needs to move:
the pretrained encoder is only nudged, the randomly-initialised decoder is
trained properly, and the attention gates need a rate high enough to leave their
initialisation within the epoch budget.

**Common.** Batch size 16, input 256×256, seed 42, `drop_last=True` on the
training loader only. Model selection on best validation loss; the test set is
evaluated once, at the end, on the selected checkpoint.

---

## Loss

KL divergence between the predicted and the ground-truth saliency distribution.
Both are flattened per image; the prediction is passed through `log_softmax` and
the target is normalised to sum to 1. This is the standard formulation for
saliency, which is a *distribution* over the image rather than a per-pixel
regression target.

---

## Evaluation

Four metrics, computed on the checkpoint selected on validation loss:

| Metric | Ground truth used | Direction |
|---|---|---|
| **SIM** — histogram intersection | density map | ↑ |
| **CC** — Pearson correlation | density map | ↑ |
| **NSS** — normalised scanpath saliency | **binary fixation map** | ↑ |
| **KL** — Kullback–Leibler divergence | density map | ↓ |

NSS is the only metric computed against the binary fixation map: it measures the
standardised saliency value *at the actual human fixation points*. Evaluating it
against the continuous density map instead is a common bug and yields values
that are out of range.

`compare_to_center_baseline` additionally evaluates a fixed centre-prior
Gaussian on the same validation set. Human gaze on SALICON is heavily
centre-biased, so a model that only learned the centre bias would still score
well in absolute terms; the margin over this prior is what shows the network
learned image-dependent structure.

Both `evaluate_loader` and the baseline comparison run automatically at the end
of `main.py`. Qualitative panels (input / GT / prediction / fixations) are
written to `plots/`, and training curves to `plots/phase1_curves.png` and
`plots/phase2_curves.png`.

---

## Checkpoints

Saved to `models/`, named after the configuration, so runs do not overwrite each
other:

```
models/best_model_{18|50}_{att|noatt}_{skip|noskip}.pt
```

Each file holds three state dicts: `encoder`, `decoder`, `attention`.

### Inspecting the attention gates

```bash
python test.py --model models/best_model_50_att_skip.pt
```

Prints γ for `c3` and `c4` together with the weight statistics of the attention
blocks. γ is the diagnostic that matters: a gate near zero means the attention
branch has been switched off by training and the module is a no-op, regardless
of what the metrics say.

---

## Requirements

PyTorch, torchvision, numpy, scipy (for reading the SALICON `.mat` fixation
files), pillow, matplotlib.

Developed on an AMD RX 6700 XT (12 GB) under ROCm. The `hipBLASLt` warning at
startup is benign — PyTorch falls back to `hipblas`.
