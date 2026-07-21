# SaliencyNet

The architecture, preprocessing, training protocol and results are described in
the accompanying report. This file covers how the code is organised and how
to run it.

---

## Running

Everything goes through `main.py`. The defaults are the final model, so each flag
*removes* a component.

Setup first:

```bash
pip install -r requirements.txt
```

and put SALICON where the code expects it (see [Dataset](#dataset)).
`--run_type curves` is the only mode that needs neither.

### Evaluating the final model

The final model's checkpoint is the only one included in the submission; shipping
all four would take too much space. To print its validation and test metrics:

```bash
python main.py --run_type test
```

A few minutes, no training. Evaluation is deterministic (`eval()` mode, no
augmentation on the test split, `shuffle=False`), so the numbers reproduce
exactly on the same hardware.

To read the learned attention gates out of that checkpoint:

```bash
python gamma_checker.py --model models/best_model_50_att_skip.pt
```

Prints γ for `c3` and `c4` plus the weight statistics of the attention blocks.
A gate near zero means training has switched the branch off.

### Flags

| Flag | Effect |
|---|---|
| `--backbone {resnet18,resnet50}` | Encoder. Default `resnet50`. |
| `--no_skip` | Decoder consumes only the `c4` bottleneck; no concatenation with `c1`/`c2`/`c3`. |
| `--no_attention` | The attention module becomes an identity map on `c3`, `c4`. |
| `--run_type {train,test,curves}` | What to do. Default `train`. |

`--run_type` in detail:

- **`train`** — both training phases, then validation and test metrics, then the
  centre-prior comparison. Writes a checkpoint, a curves CSV and a curves PNG.
- **`test`** — loads the checkpoint matching the given flag combination and
  prints validation and test metrics only. No training, no centre-prior
  comparison.
- **`curves`** — reads the per-configuration CSVs and writes the combined
  comparison figure. Returns before the dataset or the model are touched, so it
  needs neither a GPU nor the SALICON files.

### The four configurations

Each row removes one more component from the final model.

| Configuration | Command |
|---|---|
| Final (ResNet50, skip, attention) | `python main.py` |
| − deeper backbone | `python main.py --backbone resnet18` |
| − self-attention | `python main.py --backbone resnet18 --no_attention` |
| − skip connections (Base) | `python main.py --backbone resnet18 --no_skip --no_attention` |

Appending `--run_type test` to any of these evaluates that configuration instead
of training it, provided its checkpoint is in `models/`.

### Reproducing the results table

Training all four from scratch, in the order the rows appear in the report:

```bash
python main.py --backbone resnet18 --no_skip --no_attention
python main.py --backbone resnet18 --no_attention
python main.py --backbone resnet18
python main.py
python main.py --run_type curves
```

Roughly two hours per configuration on an AMD RX 6700 XT (12 GB) under ROCm, so
budget most of a day for all four. The runs are independent processes, so each one
re-seeds from scratch and they can be launched separately, in any order, or in
sequence. The final `--run_type curves` call needs all four CSVs to be present to
produce the full comparison figure.

---

## Repository layout

| File | Contents |
|---|---|
| `main.py` | Entry point. Parses flags, builds the model, trains and/or evaluates according to `--run_type`. All hyperparameters are constants at the top of the file. |
| `dataset.py` | Path matching, train/val/test split, `ImageDataset`, augmentation, fixation-map construction |
| `models.py` | `encoder18` / `encoder50`, `attention18` / `attention50`, `decoder18` / `decoder50` |
| `losses.py` | `SaliencyKLLoss` |
| `training.py` | `train_epoch`, `test_epoch`, `run_phase1`, `run_phase2`, checkpoint naming, `plot_all_curves` |
| `metrics.py` | SIM, CC, NSS, KL, `evaluate_loader`, `compare_to_center_baseline` |
| `gamma_checker.py` | Reads the attention gates (γ) and weight statistics out of a checkpoint |

---

## Dataset

Expected layout (note the double-nested `images/`, as in the Kaggle mirror):

```
datasets/salicon/
├── images/images/{train,val}/   # RGB stimuli
├── maps/{train,val}/            # continuous density maps  -> SIM, CC, KL
└── fixations/{train,val}/       # .mat gaze records        -> NSS
```

The root is set by `ROOT` in `main.py` (default `./datasets/salicon`).

The official SALICON test labels are not public, so the official 5000-image
validation set is partitioned 3500 / 1500 into our validation and test splits.
The partition is deterministic: `main.py` seeds `random` with 42 before calling
`prepare_paths`, which performs the shuffle. Every run therefore sees the same
split.

---

## Outputs

**Checkpoints**, one per configuration, so runs do not overwrite each other:

```
models/best_model_{18|50}_{att|noatt}_{skip|noskip}.pt
```

Each file holds three state dicts: `encoder`, `decoder`, `attention`. Only
`best_model_50_att_skip.pt`, the final model, is included in the submission.

**Files under `plots/`**, where `<tag>` encodes the configuration
(e.g. `resnet50_skip_att` for the final model):

| File | Written by | Contents |
|---|---|---|
| `<tag>_curves.png` | every epoch of training | Train and validation loss for that configuration |
| `<tag>_curves.csv` | end of training | One row per epoch: `epoch, train_loss, val_loss, phase` |
| `fig_curves.pdf` | `--run_type curves` | Validation loss across every configuration that has a CSV |
| `center_baseline.png` | `--run_type train` | Input / ground truth / prediction / fixation panel for the first validation batch |

---

## Requirements

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are not listed in `requirements.txt`, because the right
build depends on the accelerator. Install them separately, following
<https://pytorch.org/get-started/locally/>.

Developed on an AMD RX 6700 XT (12 GB) under ROCm. The `hipBLASLt` warning at
startup is benign — PyTorch falls back to `hipblas`. On a machine without a
CUDA/ROCm device, set `PIN_MEMORY = False` and lower `NUM_WORKERS` in `main.py`.
