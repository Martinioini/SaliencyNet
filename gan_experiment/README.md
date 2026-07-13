# GAN Experiment (SalGAN adversarial) — ABANDONED

Attempt at adversarial training (SalGAN style) on top of the encoder-decoder model:
phases 3-4 (discriminator warmup + joint training). It requires transitioning from the 
**KL loss** (used in phases 1-2 to pretrain the saliency generator) to a **BCE loss** 
for the adversarial steps, where a sigmoid is applied to the generator's raw logits.

**Result:** the discriminator fails to separate the generated maps from the real ones 
(stalling at ~0.5 accuracy). The approach was abandoned; the main pipeline (in the repo root) 
has reverted to the **pure KL** regime, running only phases 1-2 (CC ~0.88).

This folder serves as a **complete, self-contained snapshot** of the GAN regime.

## Contents

- **`models.py`** — `encoder`, `decoder` (outputs raw logits), and the **`discriminator`** class.
- **`training.py`** — contains:
  - `train_epoch`, `test_epoch`
  - `run_phase1`, `run_phase2` (KL loss pretraining)
  - `discriminator_training`, `discriminator_testing`
  - `run_phase3` (discriminator warmup, generator frozen)
  - `adversarial_training`, `adversarial_testing`
  - `run_phase4` (joint adversarial training with BCE and generator alpha balancing)
- **`main.py`** — orchestrates all phases 1-4, defining hyperparameters for `PHASE3_*` / `PHASE4_*`. Run with argument `3` or `4` to execute only the GAN phases.
- **`metrics.py`** — handles evaluations (`SIM`, `CC`, `NSS`, `KL`) over the batches and visualizes the center prior baseline comparison.

## Notes

- The files here are an **archived copy**: they import from each other (`from models import ...`, `from training import ...`) but depend on the shared `dataset.py` and `losses.py` in the repo root.
- The main pipeline in the root NO LONGER contains any GAN logic: it purely uses KL divergence for phases 1-2.
