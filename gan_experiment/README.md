# Esperimento GAN (SalGAN adversarial) — ABBANDONATO

Tentativo di training avversario (stile SalGAN) sopra il modello encoder-decoder:
fasi 3-4 (warmup discriminatore + training congiunto). Ha richiesto di convertire
tutto il regime da **KL loss** a **BCE + sigmoide nel decoder**.

**Esito:** il discriminatore non riesce a separare le mappe generate da quelle reali
(stallo a ~0.5). Approccio abbandonato; il flusso principale (nella root del repo) è
tornato al regime **KL puro**, sole fasi 1-2 (CC ~0.88).

Questa cartella è uno **snapshot completo e autoconsistente** del regime GAN, estratto
dal commit `f77e861` (== `origin/main` al momento del rollback).

## Contenuto

- **`models.py`** — `encoder`, `decoder` (con `sigmoid` finale → output in [0,1]) e la classe
  **`discriminator`**.
- **`training.py`** — oltre a `train_epoch`/`test_epoch`/`run_phase1`/`run_phase2`:
  - `denormalize` (riporta l'immagine ImageNet in [0,1] per il discriminatore)
  - `discriminator_training`, `discriminator_testing`
  - `run_phase3` (warmup discriminatore, generatore congelato)
  - `adversarial_training`, `adversarial_testing`
  - `run_phase4` (training congiunto avversario)
- **`main.py`** — orchestrazione di tutte le fasi 1-4 (regime BCE), con gli iperparametri
  `PHASE3_*` / `PHASE4_*`. Invocare con `run_type` 3 o 4 per le sole fasi GAN.
- **`metrics.py`** — `evaluate_loader` in regime BCE (l'output è già [0,1], niente softmax).
- **`overfit_one_batch.py`** — diagnostica: il discriminatore riesce a overfittare un singolo batch?
- **`phase4_no_warmup.py`** — traiettoria della CC di validation in fase 4 senza warmup del discriminatore.

## Note

- I file qui sono una **copia archiviata**: importano tra loro (`from models import ...`,
  `from training import ...`) e dipendono da `dataset.py` nella root del repo.
- Il flusso principale nella root NON contiene più nulla di GAN: `models.py` emette logit
  grezzi (niente sigmoide), `training.py` ha solo fasi 1-2 KL, `main.py` usa `SaliencyKLLoss`.
- Riferimento storico completo: `git show f77e861:<file>`.
